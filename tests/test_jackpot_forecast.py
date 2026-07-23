"""Test jackpot_forecast.py với DataFrame synthetic (không cần Neon)."""
import pandas as pd
import pytest

from src.optimizer.jackpot_forecast import forecast_next_jackpot


def _rows(specs: list[tuple]) -> pd.DataFrame:
    """specs: list of (date_offset, jackpot1_amount, jackpot1_won, jackpot2_won)."""
    return pd.DataFrame([
        {
            "draw_date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=i),
            "jackpot1_amount": amount,
            "jackpot1_won": won1,
            "jackpot2_won": won2,
        }
        for i, (amount, won1, won2) in enumerate(specs)
    ])


def test_forecast_uses_median_increment_when_last_draw_not_won():
    # 3 doan sach, muc tang lan luot 1_000_000 / 1_000_000 / 3_000_000 -> median = 1_000_000
    df = _rows([
        (30_000_000, False, False),
        (31_000_000, False, False),
        (32_000_000, False, False),
        (35_000_000, False, False),
    ])

    result = forecast_next_jackpot(df, window=20)

    assert result.last_jackpot1_amount == 35_000_000
    assert result.last_draw_won is False
    assert result.median_recent_increment == pytest.approx(1_000_000)  # median cua [1e6, 1e6, 3e6]
    assert result.predicted_next == pytest.approx(35_000_000 + 1_000_000)
    assert result.predicted_next == pytest.approx(result.predicted_next_if_no_win)


def test_forecast_falls_back_to_floor_when_last_draw_won():
    df = _rows([
        (30_000_000, False, False),
        (34_000_000, False, False),
        (80_000_000, True, False),  # trung o ky cuoi
    ])

    result = forecast_next_jackpot(df, window=20)

    assert result.last_draw_won is True
    assert result.reset_floor_estimate == pytest.approx(30_000_000)  # min quan sat duoc
    assert result.predicted_next == pytest.approx(result.reset_floor_estimate)


def test_forecast_window_limits_number_of_increments_used():
    specs = [(30_000_000 + i * 1_000_000, False, False) for i in range(10)]
    df = _rows(specs)

    result = forecast_next_jackpot(df, window=3)

    assert result.n_increments_used == 3


def test_forecast_requires_at_least_two_draws():
    df = _rows([(30_000_000, False, False)])

    with pytest.raises(ValueError):
        forecast_next_jackpot(df)
