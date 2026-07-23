"""Test portfolio.py — heuristic độ phổ biến số + gợi ý vé né số phổ biến."""
import random

import pytest

from src.optimizer.portfolio import (
    NUM_POOL,
    average_popularity,
    popularity_score,
    recommend_tickets,
)


def test_popularity_score_decreases_with_number_range():
    assert popularity_score(5) > popularity_score(20) > popularity_score(40)


def test_popularity_score_rejects_out_of_range():
    with pytest.raises(ValueError):
        popularity_score(0)
    with pytest.raises(ValueError):
        popularity_score(56)


def test_recommend_tickets_returns_valid_structure():
    tickets = recommend_tickets(count=10, rng=random.Random(42))

    assert len(tickets) == 10
    for ticket in tickets:
        assert len(ticket) == 6
        assert len(set(ticket)) == 6  # khong trung so trong 1 ve
        assert all(1 <= n <= NUM_POOL for n in ticket)
        assert ticket == sorted(ticket)


def test_recommend_tickets_favors_cold_numbers_over_uniform_baseline():
    population_mean = sum(popularity_score(n) for n in range(1, NUM_POOL + 1)) / NUM_POOL

    tickets = recommend_tickets(count=500, rng=random.Random(7))
    avg_scores = [average_popularity(t) for t in tickets]
    overall_avg = sum(avg_scores) / len(avg_scores)

    # Ve goi y phai "lanh" hon ro ret so voi trung binh quan the (~0.49)
    assert overall_avg < population_mean - 0.1


def test_recommend_tickets_is_reproducible_with_same_seed():
    a = recommend_tickets(count=5, rng=random.Random(123))
    b = recommend_tickets(count=5, rng=random.Random(123))
    assert a == b
