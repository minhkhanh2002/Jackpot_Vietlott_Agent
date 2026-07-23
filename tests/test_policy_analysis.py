"""Test policy_analysis.py với policy giả lập (không cần model SB3 đã train)."""
import random
from math import log2

import numpy as np
import pytest

from src.envs.prize_calculator import DrawOutcome
from src.envs.vietlott_gym_env import NUM_POOL, VietlottEnv
from src.researcher.policy_analysis import (
    analyze_policy_entropy,
    classify_bias,
    collect_policy_actions,
    compare_to_random_baseline,
)


def _synthetic_draws(count: int, seed: int = 42) -> list[DrawOutcome]:
    rng = random.Random(seed)
    draws = []
    for _ in range(count):
        numbers = sorted(rng.sample(range(1, NUM_POOL + 1), 6))
        draws.append(DrawOutcome(numbers=numbers, jackpot1_amount=15_000_000_000))
    return draws


def test_analyze_policy_entropy_near_max_for_uniform_random_tickets():
    rng = random.Random(1)
    tickets = [sorted(rng.sample(range(1, NUM_POOL + 1), 6)) for _ in range(3000)]

    report = analyze_policy_entropy(tickets)

    assert report.normalized_entropy > 0.98
    assert report.chi_square_p_value > 0.05  # khong bac bo phan phoi deu


def test_analyze_policy_entropy_reduced_for_fixed_ticket():
    # Agent "suy bien": luon chon dung 1 bo 6 so co dinh moi buoc.
    fixed_ticket = [3, 14, 22, 31, 40, 55]
    tickets = [fixed_ticket for _ in range(500)]

    report = analyze_policy_entropy(tickets)

    # Chi phan bo deu tren DUNG 6 so (khong phai 55) -> entropy ~ log2(6), thap hon han max.
    assert report.entropy_bits == pytest.approx(log2(6), abs=1e-6)
    assert report.normalized_entropy < 0.5
    assert report.chi_square_p_value < 0.001


def test_collect_policy_actions_matches_expected_step_count():
    draws = _synthetic_draws(20)
    env = VietlottEnv(draws, window_size=5)

    def always_same_policy(obs: np.ndarray) -> list[int]:
        return [1, 2, 3, 4, 5, 6]

    tickets = collect_policy_actions(env, always_same_policy, n_episodes=2)

    expected_steps_per_episode = len(draws) - env.window_size
    assert len(tickets) == expected_steps_per_episode * 2
    assert all(t == [1, 2, 3, 4, 5, 6] for t in tickets)


def test_compare_to_random_baseline_detects_degenerate_policy():
    draws = _synthetic_draws(60)
    env = VietlottEnv(draws, window_size=5)

    def degenerate_policy(obs: np.ndarray) -> list[int]:
        return [1, 2, 3, 4, 5, 6]

    result = compare_to_random_baseline(env, degenerate_policy, n_episodes=1, rng=np.random.default_rng(0))

    assert result.trained.normalized_entropy < 0.5
    assert result.baseline_random.normalized_entropy > 0.9
    assert result.entropy_gap_bits > 0


def _weighted_biased_tickets(n_tickets: int, seed: int = 5) -> list[list[int]]:
    """Chỉ số 1 được thiên vị nhẹ (trọng số x3), 54 số còn lại đều nhau — mô phỏng
    "mê tín nhẹ": phân bố vẫn dàn trải gần hết cả pool nhưng lệch có ý nghĩa thống kê."""
    rng = random.Random(seed)
    weights = {n: (3.0 if n == 1 else 1.0) for n in range(1, NUM_POOL + 1)}
    tickets = []
    for _ in range(n_tickets):
        pool = list(range(1, NUM_POOL + 1))
        pool_w = [weights[n] for n in pool]
        ticket: list[int] = []
        for _ in range(6):
            chosen = rng.choices(pool, weights=pool_w, k=1)[0]
            idx = pool.index(chosen)
            pool.pop(idx)
            pool_w.pop(idx)
            ticket.append(chosen)
        tickets.append(sorted(ticket))
    return tickets


def test_classify_bias_flags_soft_bias_despite_high_entropy():
    report = analyze_policy_entropy(_weighted_biased_tickets(8000))

    # Tien dieu kien: van dan trai gan het cac so (entropy cao) NHUNG lech co y nghia thong ke.
    assert report.normalized_entropy > 0.9
    assert report.chi_square_p_value < 0.05

    assert "KHÔNG suy biến hoàn toàn" in classify_bias(report)


def test_classify_bias_flags_full_collapse():
    fixed_ticket = [3, 14, 22, 31, 40, 55]
    report = analyze_policy_entropy([fixed_ticket for _ in range(500)])

    assert "SUY BIẾN mạnh" in classify_bias(report)


def test_classify_bias_reports_no_bias_for_uniform_sample():
    rng = random.Random(1)
    tickets = [sorted(rng.sample(range(1, NUM_POOL + 1), 6)) for _ in range(3000)]
    report = analyze_policy_entropy(tickets)

    assert "KHÔNG có dấu hiệu thiên vị" in classify_bias(report)
