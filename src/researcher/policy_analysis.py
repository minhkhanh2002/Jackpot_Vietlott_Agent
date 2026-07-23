"""Phân tích policy của Agent RL — đo "sự mê tín" (AI Superstition) trong môi trường nhiễu
thuần tuý: Agent có tự học ra thiên vị số nào đó dù không có tín hiệu thật để học không?

Không phụ thuộc trực tiếp vào Stable-Baselines3 — nhận vào 1 callable
`policy_fn(obs) -> list[int]` (6 số Agent chọn), nên test được với policy giả lập (kể cả
policy suy biến/ngẫu nhiên tự viết tay), không cần model đã train.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import log2
from typing import Callable, Optional

import numpy as np
from scipy import stats

from src.envs.vietlott_gym_env import NUM_POOL, VietlottEnv

PolicyFn = Callable[[np.ndarray], list[int]]


def collect_policy_actions(env: VietlottEnv, policy_fn: PolicyFn, n_episodes: int = 1) -> list[list[int]]:
    """Chạy `policy_fn` qua toàn bộ dữ liệu trong `env` (n_episodes lượt reset), thu lại
    các bộ 6 số Agent chọn ở mỗi bước."""
    all_tickets: list[list[int]] = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        terminated = False
        while not terminated:
            ticket = policy_fn(obs)
            action = np.array([n - 1 for n in ticket])
            obs, _, terminated, _, info = env.step(action)
            all_tickets.append(info["ticket_numbers"])
    return all_tickets


@dataclass
class PolicyEntropyReport:
    entropy_bits: float
    max_entropy_bits: float  # entropy nếu phân phối đều trên num_pool số
    normalized_entropy: float  # entropy_bits / max_entropy_bits, trong [0,1] — 1 = hoàn toàn đều
    chi_square_statistic: float
    chi_square_p_value: float
    number_frequency: dict[int, int]
    n_picks: int


def analyze_policy_entropy(tickets: list[list[int]], num_pool: int = NUM_POOL) -> PolicyEntropyReport:
    """H0 (không "mê tín"): Agent chọn số đồng đều trên 1..num_pool -> entropy = max,
    chi-square không bác bỏ phân phối đều. Entropy thấp / chi-square bác bỏ mạnh nghĩa là
    Agent đã học ra thiên vị dù môi trường không có tín hiệu thật để học (dấu hiệu "mê tín"/
    policy suy biến — collapse về một tập số cố định)."""
    counter = Counter(n for ticket in tickets for n in ticket)
    counts = np.array([counter.get(n, 0) for n in range(1, num_pool + 1)], dtype=float)
    total = counts.sum()
    if total == 0:
        raise ValueError("Không có dữ liệu action nào để phân tích")

    probs = counts / total
    nonzero = probs[probs > 0]
    entropy_bits = float(-np.sum(nonzero * np.log2(nonzero)))
    max_entropy_bits = float(log2(num_pool))

    expected = total / num_pool
    chi_stat, chi_p = stats.chisquare(counts, f_exp=np.full(num_pool, expected))

    return PolicyEntropyReport(
        entropy_bits=entropy_bits,
        max_entropy_bits=max_entropy_bits,
        normalized_entropy=entropy_bits / max_entropy_bits if max_entropy_bits > 0 else 0.0,
        chi_square_statistic=float(chi_stat),
        chi_square_p_value=float(chi_p),
        number_frequency={n: int(c) for n, c in zip(range(1, num_pool + 1), counts)},
        n_picks=int(total),
    )


def classify_bias(report: PolicyEntropyReport) -> str:
    """Entropy và chi-square đo 2 khía cạnh KHÁC NHAU của "mê tín": entropy đo mức độ dàn
    trải tổng thể (thấp = co cụm về ít số — suy biến), chi-square đo mức độ lệch khỏi đều
    một cách chặt chẽ hơn (có thể bác bỏ mạnh dù entropy vẫn cao, nếu lệch dàn trải đều khắp
    nhưng nhất quán, không phải do nhiễu ngẫu nhiên). Cần xét cả 2 mới đủ, không chỉ entropy."""
    if report.normalized_entropy < 0.5:
        return "CÓ dấu hiệu policy SUY BIẾN mạnh (co cụm về một nhóm nhỏ số cố định)"
    if report.chi_square_p_value < 0.05:
        return (
            "CÓ thiên vị thống kê rõ ràng nhưng KHÔNG suy biến hoàn toàn — Agent vẫn dùng gần "
            "hết số lượng số (entropy cao) nhưng phân bố không đều một cách nhất quán, không "
            "phải do nhiễu ngẫu nhiên"
        )
    return "KHÔNG có dấu hiệu thiên vị rõ rệt — policy gần với ngẫu nhiên đều"


@dataclass
class PolicyComparisonReport:
    trained: PolicyEntropyReport
    baseline_random: PolicyEntropyReport
    entropy_gap_bits: float  # baseline.entropy - trained.entropy; dương = trained "mê tín" hơn baseline


def _uniform_random_policy_fn(num_pool: int, rng: np.random.Generator) -> PolicyFn:
    def _policy(obs: np.ndarray) -> list[int]:
        return sorted(rng.choice(np.arange(1, num_pool + 1), size=6, replace=False).tolist())

    return _policy


def compare_to_random_baseline(
    env: VietlottEnv,
    trained_policy_fn: PolicyFn,
    n_episodes: int = 1,
    rng: Optional[np.random.Generator] = None,
) -> PolicyComparisonReport:
    """So sánh entropy/chi-square của policy đã train với baseline chọn số ngẫu nhiên đều
    trên CÙNG một chuỗi dữ liệu (env). Baseline luôn có entropy ~ max theo xây dựng — dùng
    làm mốc đối chiếu xem policy đã train lệch bao xa khỏi "hoàn toàn không mê tín"."""
    rng = rng or np.random.default_rng()

    trained_tickets = collect_policy_actions(env, trained_policy_fn, n_episodes)
    random_tickets = collect_policy_actions(env, _uniform_random_policy_fn(env.num_pool, rng), n_episodes)

    trained_report = analyze_policy_entropy(trained_tickets, env.num_pool)
    random_report = analyze_policy_entropy(random_tickets, env.num_pool)

    return PolicyComparisonReport(
        trained=trained_report,
        baseline_random=random_report,
        entropy_gap_bits=random_report.entropy_bits - trained_report.entropy_bits,
    )
