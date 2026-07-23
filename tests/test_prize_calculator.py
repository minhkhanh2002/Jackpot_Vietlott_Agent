"""Test tính thưởng Power 6/55 — đối chiếu với dữ liệu thật (kỳ #00001)."""
from src.envs.prize_calculator import DrawOutcome, compute_reward, load_prize_rules

RULES = load_prize_rules()
TICKET_PRICE = RULES["ticket_price"]

# power655 kỳ #00001: numbers=[5,10,14,23,24,38], bonus=35, jackpot1=30.528.493.950, jackpot2=3.058.721.550
POWER_DRAW = DrawOutcome(
    numbers=[5, 10, 14, 23, 24, 38],
    bonus_number=35,
    jackpot1_amount=30_528_493_950,
    jackpot2_amount=3_058_721_550,
)


def test_jackpot1_match():
    reward = compute_reward([5, 10, 14, 23, 24, 38], POWER_DRAW, RULES)
    assert reward == 30_528_493_950 - TICKET_PRICE


def test_jackpot2_five_match_plus_bonus():
    reward = compute_reward(
        [5, 10, 14, 23, 24, 99], POWER_DRAW, RULES, ticket_bonus_number=35
    )
    assert reward == 3_058_721_550 - TICKET_PRICE


def test_giai_nhat_five_match_wrong_bonus():
    reward = compute_reward(
        [5, 10, 14, 23, 24, 99], POWER_DRAW, RULES, ticket_bonus_number=1
    )
    assert reward == 40_000_000 - TICKET_PRICE


def test_giai_ba_three_match():
    reward = compute_reward([5, 10, 14, 1, 2, 3], POWER_DRAW, RULES)
    assert reward == 50_000 - TICKET_PRICE


def test_no_match_loses_ticket_price():
    reward = compute_reward([1, 2, 3, 40, 41, 42], POWER_DRAW, RULES)
    assert reward == -TICKET_PRICE


def test_duplicate_numbers_is_invalid_ticket():
    reward = compute_reward([5, 5, 14, 23, 24, 38], POWER_DRAW, RULES)
    assert reward == -TICKET_PRICE
