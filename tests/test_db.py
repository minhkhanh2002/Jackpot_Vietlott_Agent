"""Test phần logic thuần của db.py (không cần kết nối Postgres thật).

`vietlott_results` dùng cột ARRAY và upsert kiểu Postgres-native (ON CONFLICT) nên không
test round-trip qua SQLite được nữa — phần ghi DB thực tế đã verify bằng cách chạy thật
trên Neon (xem lịch sử backfill/update). Ở đây chỉ test hàm phụ trợ thuần Python.
"""
from src.data_engine.db import _tier_winner_count
from src.data_engine.scraper import PrizeTier


def test_tier_winner_count_finds_matching_tier():
    tiers = [
        PrizeTier(tier_name="Jackpot 1", winner_count=0, prize_amount=30_528_493_950),
        PrizeTier(tier_name="Jackpot 2", winner_count=2, prize_amount=3_058_721_550),
        PrizeTier(tier_name="Giải Nhất", winner_count=3, prize_amount=40_000_000),
    ]

    assert _tier_winner_count(tiers, "Jackpot 1") == 0
    assert _tier_winner_count(tiers, "Jackpot 2") == 2


def test_tier_winner_count_returns_zero_when_tier_missing():
    tiers = [PrizeTier(tier_name="Giải Ba", winner_count=100, prize_amount=50_000)]

    assert _tier_winner_count(tiers, "Jackpot 1") == 0
