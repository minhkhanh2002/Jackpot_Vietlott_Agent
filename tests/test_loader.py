"""Test _result_to_record với object giả lập — không cần kết nối DB thật."""
from dataclasses import dataclass
from datetime import date
from typing import Optional

from src.data_engine.loader import _result_to_record


@dataclass
class _FakeRow:
    period_id: int
    draw_date: date
    numbers: list
    bonus_number: Optional[int]
    jackpot1_value: Optional[int]
    jackpot2_value: Optional[int]
    jackpot1_winners: Optional[int]
    jackpot2_winners: Optional[int]


def test_result_to_record_maps_fields_and_expands_numbers():
    row = _FakeRow(
        period_id=1,
        draw_date=date(2017, 8, 1),
        numbers=[5, 10, 14, 23, 24, 38],
        bonus_number=35,
        jackpot1_value=30_528_493_950,
        jackpot2_value=3_058_721_550,
        jackpot1_winners=0,
        jackpot2_winners=0,
    )

    record = _result_to_record(row)

    assert record["period_id"] == 1
    assert record["num_1"] == 5
    assert record["num_6"] == 38
    assert record["bonus_number"] == 35
    assert record["jackpot1_amount"] == 30_528_493_950
    assert record["jackpot2_amount"] == 3_058_721_550
    assert record["jackpot1_won"] is False
    assert record["jackpot2_won"] is False


def test_result_to_record_derives_won_flags_from_winner_counts():
    row = _FakeRow(
        period_id=1368,
        draw_date=date(2026, 7, 7),
        numbers=[4, 6, 25, 32, 33, 44],
        bonus_number=8,
        jackpot1_value=88_779_791_550,
        jackpot2_value=3_643_241_150,
        jackpot1_winners=0,
        jackpot2_winners=1,
    )

    record = _result_to_record(row)

    assert record["jackpot1_won"] is False
    assert record["jackpot2_won"] is True
