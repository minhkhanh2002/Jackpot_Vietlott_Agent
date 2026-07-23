"""Test statistical_tests.py với DataFrame synthetic (không cần Neon)."""
import random

import pandas as pd
import pytest

from math import comb

from src.auditor.statistical_tests import (
    autocorrelation_of_number_sums,
    chi_square_number_frequency,
    estimate_tickets_from_jackpot_increment,
    hit_rate_by_jackpot_size,
    jackpot_rollover_streaks,
    kolmogorov_smirnov_uniformity,
    predicted_vs_observed_hit_rate,
    validate_jackpot_split_ratio,
)


def _make_draws_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _uniform_random_draws(n: int, num_pool: int, seed: int = 1) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        nums = sorted(rng.sample(range(1, num_pool + 1), 6))
        rows.append(
            {
                "draw_date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=i),
                "num_1": nums[0], "num_2": nums[1], "num_3": nums[2],
                "num_4": nums[3], "num_5": nums[4], "num_6": nums[5],
            }
        )
    return _make_draws_df(rows)


def _biased_draws(n: int) -> pd.DataFrame:
    """Số 1 luôn xuất hiện -> lệch rõ khỏi phân phối đều, chi-square phải bắt được."""
    rows = []
    for i in range(n):
        rows.append(
            {
                "draw_date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=i),
                "num_1": 1, "num_2": 2, "num_3": 3, "num_4": 4, "num_5": 5, "num_6": 6,
            }
        )
    return _make_draws_df(rows)


def test_chi_square_flags_heavily_biased_numbers():
    df = _biased_draws(50)
    result = chi_square_number_frequency(df, num_pool=45)

    assert result.observed[1] == 50  # số 1 xuất hiện MỌI kỳ
    assert result.observed[45] == 0  # số 45 chưa từng xuất hiện
    assert result.p_value < 0.001  # bác bỏ rõ ràng giả thuyết phân phối đều


def test_chi_square_structure_on_uniform_random_sample():
    df = _uniform_random_draws(200, 45)
    result = chi_square_number_frequency(df, num_pool=45)

    assert sum(result.observed.values()) == 200 * 6
    assert len(result.observed) == 45


def test_ks_uniformity_structure():
    df = _uniform_random_draws(200, 45)
    result = kolmogorov_smirnov_uniformity(df, num_pool=45)

    assert 0.0 <= result.statistic <= 1.0
    assert 0.0 <= result.p_value <= 1.0


def test_autocorrelation_detects_alternating_period_2_pattern():
    # tổng 6 số luân phiên cao/thấp mỗi kỳ -> lag=2 phải tương quan dương mạnh, lag=1 âm mạnh
    rows = []
    for i in range(40):
        base = [1, 2, 3, 4, 5, 6] if i % 2 == 0 else [40, 41, 42, 43, 44, 45]
        rows.append(
            {
                "draw_date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=i),
                "num_1": base[0], "num_2": base[1], "num_3": base[2],
                "num_4": base[3], "num_5": base[4], "num_6": base[5],
            }
        )
    df = _make_draws_df(rows)

    result = autocorrelation_of_number_sums(df, max_lag=3)

    assert result[1] < -0.9  # kỳ liền kề luôn trái dấu (cao rồi thấp)
    assert result[2] > 0.9  # cách 2 kỳ luôn cùng pha


def test_jackpot_rollover_streaks_matches_manual_calculation():
    won_pattern = [False, False, True, False, False, False, True, True]
    rows = [
        {"draw_date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=i), "jackpot1_won": w}
        for i, w in enumerate(won_pattern)
    ]
    df = _make_draws_df(rows)

    result = jackpot_rollover_streaks(df)

    assert result.streak_lengths == [2, 3, 0]
    assert result.mean_streak == pytest.approx(5 / 3)


def test_hit_rate_by_jackpot_size_reflects_correlation():
    rows = []
    for i in range(20):
        amount = 10_000_000_000 + i * 1_000_000_000  # tăng dần
        won = i >= 15  # chỉ nổ ở nhóm jackpot cao nhất
        rows.append(
            {
                "draw_date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=i),
                "jackpot1_amount": amount,
                "jackpot1_won": won,
            }
        )
    df = _make_draws_df(rows)

    result = hit_rate_by_jackpot_size(df, n_buckets=2)

    assert len(result.hit_rate_per_bucket) == 2
    assert result.hit_rate_per_bucket[0] < result.hit_rate_per_bucket[-1]
    assert sum(result.count_per_bucket) == 20


def _ticket_estimation_df() -> pd.DataFrame:
    """5 kỳ: 2 đoạn "sạch" tăng đúng tỉ lệ 9:1 (n_tickets=1000 mỗi đoạn), xen giữa 1 kỳ
    trúng Jackpot 1 (jackpot nhảy vọt rồi reset) — 2 đoạn kề kỳ trúng đó phải bị loại."""
    rows = [
        {"draw_date": pd.Timestamp("2020-01-01"), "jackpot1_amount": 30_000_000, "jackpot2_amount": 3_000_000, "jackpot1_won": False, "jackpot2_won": False},
        {"draw_date": pd.Timestamp("2020-01-03"), "jackpot1_amount": 34_950_000, "jackpot2_amount": 3_550_000, "jackpot1_won": False, "jackpot2_won": False},
        {"draw_date": pd.Timestamp("2020-01-05"), "jackpot1_amount": 39_900_000, "jackpot2_amount": 4_100_000, "jackpot1_won": True, "jackpot2_won": False},
        {"draw_date": pd.Timestamp("2020-01-07"), "jackpot1_amount": 30_000_000, "jackpot2_amount": 3_000_000, "jackpot1_won": False, "jackpot2_won": False},
        {"draw_date": pd.Timestamp("2020-01-09"), "jackpot1_amount": 34_950_000, "jackpot2_amount": 3_550_000, "jackpot1_won": False, "jackpot2_won": False},
    ]
    return pd.DataFrame(rows)


def test_validate_jackpot_split_ratio_excludes_segments_touching_a_win():
    df = _ticket_estimation_df()

    result = validate_jackpot_split_ratio(df)

    # chỉ 2 đoạn "sạch" (kỳ 0->1 và kỳ 3->4), cả 2 đều đúng tỉ lệ 900k/100k = 9
    assert result.n_samples == 2
    assert result.ratio_median == pytest.approx(9.0)
    assert result.ratio_min == pytest.approx(9.0)
    assert result.ratio_max == pytest.approx(9.0)


def test_estimate_tickets_from_jackpot_increment_matches_hand_calculation():
    df = _ticket_estimation_df()

    result = estimate_tickets_from_jackpot_increment(df)

    assert len(result) == 4  # len(df) - 1
    # rate = 0.55 * 0.90 = 0.495 ; ticket_price = 10_000 -> 4_950 VNĐ/vé vào quỹ jackpot1
    assert result["n_tickets_est"].iloc[0] == pytest.approx(1000.0)  # (34_950_000-30_000_000)/4950
    assert pd.isna(result["n_tickets_est"].iloc[1])  # kề kỳ trúng thưởng -> loại
    assert pd.isna(result["n_tickets_est"].iloc[2])  # kề kỳ trúng thưởng -> loại
    assert result["n_tickets_est"].iloc[3] == pytest.approx(1000.0)


def test_predicted_vs_observed_hit_rate_single_bucket():
    df = _ticket_estimation_df()

    results = predicted_vs_observed_hit_rate(df, num_pool=10, picks=2, n_buckets=1)

    assert len(results) == 1
    bucket = results[0]
    assert bucket.median_n_tickets == pytest.approx(1000.0)
    assert bucket.n_clean_draws == 2
    assert bucket.observed_count == 5
    assert bucket.observed_hit_rate == pytest.approx(1 / 5)  # 1 kỳ trúng / 5 kỳ

    p_single = 1 / comb(10, 2)
    expected_predicted = 1 - (1 - p_single) ** 1000
    assert bucket.predicted_hit_rate == pytest.approx(expected_predicted)
