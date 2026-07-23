"""Test VietlottEnv (Power 6/55) với dữ liệu synthetic (không cần Neon)."""
import random

import numpy as np
from gymnasium.utils.env_checker import check_env

from src.envs.prize_calculator import DrawOutcome, load_prize_rules
from src.envs.vietlott_gym_env import NUM_POOL, VietlottEnv

RULES = load_prize_rules()
TICKET_PRICE = RULES["ticket_price"]


def _synthetic_draws(count: int, seed: int = 42) -> list[DrawOutcome]:
    rng = random.Random(seed)
    draws = []
    for _ in range(count):
        numbers = sorted(rng.sample(range(1, NUM_POOL + 1), 6))
        draws.append(DrawOutcome(numbers=numbers, jackpot1_amount=15_000_000_000))
    return draws


def test_reset_returns_observation_matching_space():
    draws = _synthetic_draws(20)
    env = VietlottEnv(draws, window_size=5)

    obs, info = env.reset()

    assert obs.shape == env.observation_space.shape
    assert obs.dtype == np.float32
    assert (obs >= 0).all() and (obs <= 1).all()
    assert info == {}


def test_step_reward_matches_prize_calculator_on_exact_match():
    draws = _synthetic_draws(20)
    env = VietlottEnv(draws, window_size=5)
    env.reset()

    next_draw = draws[env._cursor]
    action = np.array([n - 1 for n in next_draw.numbers])  # trùng khớp tuyệt đối kỳ tới

    obs, reward, terminated, truncated, info = env.step(action)

    assert reward == next_draw.jackpot1_amount - TICKET_PRICE
    assert info["match_count"] == 6
    assert terminated is False


def test_step_invalid_duplicate_action_loses_ticket_price():
    draws = _synthetic_draws(20)
    env = VietlottEnv(draws, window_size=5)
    env.reset()

    action = np.array([0, 0, 1, 2, 3, 4])  # trùng số -> vé không hợp lệ
    _, reward, _, _, _ = env.step(action)

    assert reward == -TICKET_PRICE


def test_episode_terminates_after_exhausting_draws():
    draws = _synthetic_draws(10)
    window_size = 5
    env = VietlottEnv(draws, window_size=window_size)
    env.reset()

    expected_steps = len(draws) - window_size
    terminated = False
    steps_taken = 0
    for _ in range(expected_steps):
        action = env.action_space.sample()
        _, _, terminated, _, _ = env.step(action)
        steps_taken += 1

    assert terminated is True
    assert steps_taken == expected_steps


def test_gymnasium_env_checker_passes():
    draws = _synthetic_draws(30)
    env = VietlottEnv(draws, window_size=5)
    check_env(env, skip_render_check=True)
