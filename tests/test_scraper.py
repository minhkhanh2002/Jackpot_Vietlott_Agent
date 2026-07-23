"""Test scraper.parse_draw() với fixture HTML thật (đã tải offline), không phụ thuộc mạng."""
from pathlib import Path

from src.data_engine.scraper import parse_draw

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _read_fixture(name: str) -> str:
    return (FIXTURES_DIR / name).read_text(encoding="utf-8")


def test_parse_mega645_draw():
    html = _read_fixture("mega645_draw01000.html")
    result = parse_draw(html, "mega645")

    assert result is not None
    assert result.product == "mega645"
    assert result.draw_id == "01000"
    assert result.draw_date == "2023-02-05"
    assert result.numbers == [13, 15, 23, 29, 31, 34]
    assert result.bonus_number is None
    assert result.jackpot1_amount == 32_791_181_500
    assert result.jackpot2_amount is None

    tiers = {t.tier_name: (t.winner_count, t.prize_amount) for t in result.prize_tiers}
    assert tiers["Jackpot"] == (0, 32_791_181_500)
    assert tiers["Giải Nhất"] == (29, 10_000_000)
    assert tiers["Giải Nhì"] == (1261, 300_000)
    assert tiers["Giải Ba"] == (20_845, 30_000)


def test_parse_power655_draw_has_bonus_number_and_two_jackpots():
    html = _read_fixture("power655_draw00001.html")
    result = parse_draw(html, "power655")

    assert result is not None
    assert result.draw_id == "00001"
    assert result.draw_date == "2017-08-01"
    assert result.numbers == [5, 10, 14, 23, 24, 38]
    assert result.bonus_number == 35
    assert result.jackpot1_amount == 30_528_493_950
    assert result.jackpot2_amount == 3_058_721_550

    tiers = {t.tier_name: (t.winner_count, t.prize_amount) for t in result.prize_tiers}
    assert tiers["Jackpot 1"] == (0, 30_528_493_950)
    assert tiers["Jackpot 2"] == (0, 3_058_721_550)
    assert tiers["Giải Ba"] == (4024, 50_000)


def test_parse_invalid_draw_id_returns_none():
    html = _read_fixture("invalid_draw99999.html")
    result = parse_draw(html, "mega645")

    assert result is None
