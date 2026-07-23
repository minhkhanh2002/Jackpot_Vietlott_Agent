"""Tính tiền thưởng cho 1 vé Power 6/55, dựa trên config/prize_rules.yaml và kết quả 1 kỳ quay.

Không phụ thuộc DB — nhận `DrawOutcome` thuần dữ liệu nên dùng được cả với kết quả synthetic
(unit test) lẫn dữ liệu thật load từ loader.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

DEFAULT_RULES_PATH = Path(__file__).resolve().parents[2] / "config" / "prize_rules.yaml"


@dataclass
class DrawOutcome:
    """Kết quả 1 kỳ quay, đủ để tính thưởng."""

    numbers: list[int]  # 6 số chính
    bonus_number: Optional[int] = None  # "số Power"
    jackpot1_amount: Optional[int] = None
    jackpot2_amount: Optional[int] = None


def load_prize_rules(path: Path = DEFAULT_RULES_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _fixed_tier_amount(tiers: list[dict], match_count: int) -> int:
    """Tra hạng giải có amount cố định (không phải jackpot) khớp số lượng trùng."""
    for tier in tiers:
        if tier["amount"] is not None and tier["match"] == match_count:
            return tier["amount"]
    return 0


def compute_reward(
    ticket_numbers: list[int],
    draw: DrawOutcome,
    rules: dict,
    ticket_bonus_number: Optional[int] = None,
) -> int:
    """Tiền thưởng (VNĐ) trừ giá vé cho 1 vé Power 6/55 so với kết quả `draw`.

    Vé không hợp lệ (không đúng 6 số riêng biệt) coi như bị từ chối: mất tiền vé, không hoàn.

    Lưu ý: coi "5/6 số chính + đúng số Power" là Jackpot 2, "5/6 số chính + sai/không có số
    Power" là Giải Nhất — đơn giản hoá dựa trên cách hiển thị kết quả trên vietlott.vn, CẦN
    đối chiếu lại thể lệ chính thức để xác nhận không có kỳ nào cả hai điều kiện cùng được
    trả thưởng.
    """
    ticket_price = rules["ticket_price"]

    if len(set(ticket_numbers)) != 6:
        return -ticket_price

    match_count = len(set(ticket_numbers) & set(draw.numbers))
    bonus_match = ticket_bonus_number is not None and ticket_bonus_number == draw.bonus_number

    tiers = rules["tiers"]

    if match_count == 6:
        prize = draw.jackpot1_amount or 0
    elif match_count == 5 and bonus_match:
        prize = draw.jackpot2_amount or 0
    else:
        prize = _fixed_tier_amount(tiers, match_count)

    return prize - ticket_price
