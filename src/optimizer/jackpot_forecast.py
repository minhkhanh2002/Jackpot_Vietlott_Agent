"""Dự báo giá trị Jackpot 1 kỳ tới — baseline đơn giản (trung vị mức tăng gần đây),
dùng chung cơ chế "đoạn sạch" với `auditor.statistical_tests` để mức tăng phản ánh đúng
tích luỹ thực tế (không lẫn các kỳ có người trúng/chạm trần).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.auditor.statistical_tests import JACKPOT1_CAP, clean_jackpot1_transition_mask


@dataclass
class JackpotForecast:
    last_jackpot1_amount: int
    last_draw_won: bool
    predicted_next_if_no_win: float  # dự báo NẾU kỳ gần nhất không có người trúng
    reset_floor_estimate: float  # mức sàn quan sát được (dùng khi kỳ gần nhất CÓ người trúng)
    predicted_next: float  # dự báo cuối cùng, tự chọn 1 trong 2 giá trị trên theo last_draw_won
    median_recent_increment: float
    n_increments_used: int


def forecast_next_jackpot(
    df: pd.DataFrame,
    window: int = 20,
    cap: int = JACKPOT1_CAP,
) -> JackpotForecast:
    """Baseline: nếu kỳ gần nhất KHÔNG có người trúng, dự báo kỳ tới = jackpot hiện tại +
    trung vị mức tăng của tối đa `window` đoạn "sạch" gần nhất. Nếu kỳ gần nhất CÓ người
    trúng, dự báo kỳ tới = mức sàn quan sát được (min jackpot1_amount trong dữ liệu).

    Đây là baseline thống kê đơn giản (không dùng mô hình hồi quy/ML) — đủ dùng làm điểm
    xuất phát, có thể nâng cấp sau bằng cách đưa thêm biến số vé ước lượng.
    """
    d = df.sort_values("draw_date").reset_index(drop=True)
    if len(d) < 2:
        raise ValueError("Cần ít nhất 2 kỳ quay để dự báo")

    clean = clean_jackpot1_transition_mask(d, cap)
    j1 = d["jackpot1_amount"].to_numpy()
    increments = (j1[1:] - j1[:-1])[clean]
    recent = increments[-window:] if len(increments) > 0 else increments
    median_increment = float(np.median(recent)) if len(recent) > 0 else 0.0

    last_amount = int(d["jackpot1_amount"].iloc[-1])
    last_won = bool(d["jackpot1_won"].iloc[-1])
    floor_estimate = float(d["jackpot1_amount"].min())

    predicted_if_no_win = last_amount + median_increment
    predicted = floor_estimate if last_won else predicted_if_no_win

    return JackpotForecast(
        last_jackpot1_amount=last_amount,
        last_draw_won=last_won,
        predicted_next_if_no_win=predicted_if_no_win,
        reset_floor_estimate=floor_estimate,
        predicted_next=predicted,
        median_recent_increment=median_increment,
        n_increments_used=len(recent),
    )
