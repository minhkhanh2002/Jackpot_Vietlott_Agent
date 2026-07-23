"""API đọc dữ liệu Power 6/55 ra pandas DataFrame — dùng chung cho Auditor/Researcher/Optimizer.

`_result_to_record` tách riêng khỏi phần query DB để test được mà không cần kết nối thật.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
from sqlalchemy import Engine, select

from src.data_engine.db import get_engine, get_session_factory
from src.data_engine.models import VietlottResult


def _result_to_record(row: VietlottResult) -> dict:
    n1, n2, n3, n4, n5, n6 = row.numbers
    return {
        "period_id": row.period_id,
        "draw_date": row.draw_date,
        "num_1": n1, "num_2": n2, "num_3": n3, "num_4": n4, "num_5": n5, "num_6": n6,
        "bonus_number": row.bonus_number,
        "jackpot1_amount": row.jackpot1_value,
        "jackpot2_amount": row.jackpot2_value,
        "jackpot1_won": (row.jackpot1_winners or 0) > 0,
        "jackpot2_won": (row.jackpot2_winners or 0) > 0,
    }


def load_results(engine: Optional[Engine] = None) -> pd.DataFrame:
    """DataFrame toàn bộ kỳ quay Power 6/55, sắp xếp theo draw_date tăng dần."""
    eng = engine or get_engine()
    factory = get_session_factory(eng)
    with factory() as session:
        rows = session.scalars(select(VietlottResult).order_by(VietlottResult.draw_date)).all()
        records = [_result_to_record(r) for r in rows]
    return pd.DataFrame.from_records(records)
