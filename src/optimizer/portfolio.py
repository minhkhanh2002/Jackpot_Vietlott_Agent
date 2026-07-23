"""Cố vấn chọn vé Power 6/55 nhằm giảm rủi ro CHIA GIẢI nếu trúng — KHÔNG làm tăng xác suất
trúng của bất kỳ vé nào. Đã kiểm chứng bằng chi-square trên dữ liệu thật (statistical_tests.py):
máy quay số công bằng, mọi tổ hợp 6/55 có xác suất trúng bằng nhau; module này chỉ tối ưu
"nếu trúng thì đỡ phải chia" bằng cách né các số nhiều người khác cũng hay chọn.

Heuristic "độ phổ biến": Vietlott không công bố dãy số người chơi chọn, nên độ phổ biến ở
đây là MÔ HÌNH HOÁ GIẢ ĐỊNH dựa trên hành vi chọn số phổ biến (ngày/tháng sinh), KHÔNG PHẢI
dữ liệu thực đo — số <=31 (có thể là ngày sinh) giả định phổ biến hơn số >31 ("số lạnh").
"""
from __future__ import annotations

import random
from typing import Optional

NUM_POOL = 55
PICKS = 6


def popularity_score(number: int) -> float:
    """Điểm độ phổ biến giả định trong [0,1] — cao hơn nghĩa là nhiều người có xu hướng
    chọn hơn (giả định dựa trên khả năng trùng ngày/tháng sinh)."""
    if not 1 <= number <= NUM_POOL:
        raise ValueError(f"Số phải trong khoảng 1..{NUM_POOL}")
    if number <= 12:
        return 1.0  # co the la ca ngay lan thang sinh
    if number <= 31:
        return 0.6  # chi co the la ngay sinh
    return 0.15  # "so lanh" gia dinh, ngoai pham vi ngay/thang


def average_popularity(ticket: list[int]) -> float:
    """Điểm phổ biến trung bình của 1 vé — dùng để so sánh vé gợi ý với vé chọn ngẫu nhiên
    hoặc vé người dùng tự chọn."""
    return sum(popularity_score(n) for n in ticket) / len(ticket)


def recommend_tickets(count: int = 1, rng: Optional[random.Random] = None) -> list[list[int]]:
    """Sinh `count` bộ vé 6 số, ưu tiên né các số có popularity_score cao (giả định) để
    giảm rủi ro chia giải NẾU trúng.

    Lấy mẫu có trọng số không hoàn lại: trọng số mỗi số = 1 - popularity_score(number) (sàn
    0.01 để số nào cũng còn cơ hội được chọn) — số càng "lạnh" càng dễ được chọn vào vé.
    """
    rng = rng or random.Random()
    weights = {n: max(1.0 - popularity_score(n), 0.01) for n in range(1, NUM_POOL + 1)}

    tickets: list[list[int]] = []
    for _ in range(count):
        pool = list(range(1, NUM_POOL + 1))
        pool_weights = [weights[n] for n in pool]
        ticket: list[int] = []
        for _ in range(PICKS):
            chosen = rng.choices(pool, weights=pool_weights, k=1)[0]
            idx = pool.index(chosen)
            pool.pop(idx)
            pool_weights.pop(idx)
            ticket.append(chosen)
        tickets.append(sorted(ticket))
    return tickets
