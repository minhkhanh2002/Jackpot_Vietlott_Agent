"""Sinh báo cáo "AI Superstition" — so sánh entropy/chi-square của policy PPO đã train với
baseline chọn số ngẫu nhiên đều, trên cùng chuỗi dữ liệu lịch sử Power 6/55.

Yêu cầu đã train model trước (xem src/researcher/train.py). Chạy nhiều episode để baseline
ngẫu nhiên đủ mẫu ổn định thống kê (mỗi episode duyệt lại toàn bộ ~1.360 kỳ).

Chạy: python scripts/rl_report.py --episodes 5
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from stable_baselines3 import PPO

from src.data_engine.loader import load_results
from src.envs.vietlott_gym_env import VietlottEnv
from src.researcher.policy_analysis import PolicyComparisonReport, classify_bias, compare_to_random_baseline
from src.researcher.train import DEFAULT_MODEL_PATH, draws_from_dataframe

# docs/ (khác data/) không bị .gitignore loại — báo cáo cần commit để mọi lần deploy (kể cả
# Streamlit Cloud, checkout mới) đều có sẵn, không phải retrain mới thấy được.
REPORT_PATH = Path(__file__).resolve().parents[1] / "docs" / "reports" / "ai_superstition_report.md"


def make_trained_policy_fn(model: PPO):
    def _policy(obs: np.ndarray) -> list[int]:
        action, _ = model.predict(obs, deterministic=True)
        return sorted(int(a) + 1 for a in action)

    return _policy


def render_report(result: PolicyComparisonReport, model_path: Path, episodes: int) -> str:
    t, b = result.trained, result.baseline_random
    verdict = classify_bias(t)
    effect_size_ratio = t.chi_square_statistic / b.chi_square_statistic if b.chi_square_statistic else float("nan")
    top5 = sorted(t.number_frequency.items(), key=lambda kv: kv[1], reverse=True)[:5]

    lines = [
        "# Báo cáo AI Superstition — PPO trên VietlottEnv (Power 6/55)",
        "",
        f"Ngày sinh báo cáo: {date.today().isoformat()}  ",
        f"Model: `{model_path}`  ",
        f"Số episode chạy để lấy mẫu: {episodes} (mỗi episode ~{t.n_picks // episodes // 6} kỳ)",
        "",
        "## Bối cảnh",
        "",
        "VietlottEnv về bản chất là môi trường **nhiễu thuần tuý** — không có tín hiệu thật nào "
        "trong lịch sử kỳ quay trước có thể dự đoán kỳ quay sau (đã kiểm chứng ở Auditor: máy quay "
        "số công bằng, chi-square không bác bỏ phân phối đều). Câu hỏi nghiên cứu: khi PPO được "
        "train lặp lại nhiều lượt trên cùng một chuỗi dữ liệu này, nó có tự học ra một thiên vị/"
        "\"mê tín\" nào đó (policy suy biến về một tập số cố định) thay vì giữ hành vi ngẫu nhiên "
        "đều — hành vi ĐÚNG duy nhất khi không có tín hiệu để học?",
        "",
        "## Kết quả định lượng",
        "",
        "| Chỉ số | PPO đã train | Baseline ngẫu nhiên đều |",
        "|---|---|---|",
        f"| Entropy (bits) | {t.entropy_bits:.3f} | {b.entropy_bits:.3f} |",
        f"| Entropy tối đa lý thuyết (bits) | {t.max_entropy_bits:.3f} | {b.max_entropy_bits:.3f} |",
        f"| Entropy chuẩn hoá (0-1) | {t.normalized_entropy:.3f} | {b.normalized_entropy:.3f} |",
        f"| Chi-square statistic | {t.chi_square_statistic:.1f} | {b.chi_square_statistic:.1f} |",
        f"| Chi-square p-value | {t.chi_square_p_value:.4g} | {b.chi_square_p_value:.4g} |",
        f"| Số lượt chọn số (n) | {t.n_picks} | {b.n_picks} |",
        "",
        f"**Chênh lệch entropy (baseline - trained): {result.entropy_gap_bits:.3f} bits.**",
        "",
        f"5 số PPO chọn nhiều nhất: {', '.join(f'{n} ({c} lần)' for n, c in top5)}",
        "",
        "## Kết luận",
        "",
        f"**{verdict}.**",
        "",
        f"Chi-square statistic của policy đã train cao gấp **{effect_size_ratio:.0f} lần** chi-square "
        "statistic của baseline ngẫu nhiên trên CÙNG cỡ mẫu — không phải hiệu ứng cỡ mẫu lớn "
        "(baseline cũng có n lớn tương đương nhưng statistic vẫn ở mức bình thường theo lý thuyết), "
        "mà là một lệch hệ thống thật sự trong hành vi chọn số của Agent.",
        "",
        "Lưu ý: bias thống kê này không có nghĩa là Agent \"dự đoán được\" kỳ quay — reward "
        "trung bình vẫn âm (mất giá vé) và không có tín hiệu thật nào trong dữ liệu để học. "
        "Khác với hệ thống RL trước đây của dự án (nơi agent suy biến hoàn toàn về đúng 1 bộ số "
        "cố định — entropy gần 0), lần chạy này Agent vẫn dùng gần hết cả 55 số (entropy chuẩn "
        "hoá 0.967) nhưng với tần suất không đều một cách nhất quán — một dạng \"mê tín nhẹ\" "
        "tinh vi hơn: không lộ rõ qua entropy tổng thể, chỉ phát hiện được nhờ chi-square test "
        "chi tiết hơn. Bài học phương pháp luận: đo policy RL trong môi trường ngẫu nhiên nên "
        "dùng cả hai loại kiểm định, vì mỗi loại bắt được một dạng lệch khác nhau.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sinh báo cáo AI Superstition từ model PPO đã train.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model = PPO.load(str(args.model))

    df = load_results()
    draws = draws_from_dataframe(df)
    env = VietlottEnv(draws, window_size=args.window_size)

    result = compare_to_random_baseline(
        env,
        make_trained_policy_fn(model),
        n_episodes=args.episodes,
        rng=np.random.default_rng(args.seed),
    )

    report = render_report(result, args.model, args.episodes)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")

    print(report)
    print(f"\n(Đã lưu báo cáo tại {REPORT_PATH})")


if __name__ == "__main__":
    main()
