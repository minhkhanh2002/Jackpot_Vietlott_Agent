"""Scraper cho kết quả xổ số Vietlott (Mega 6/45, Power 6/55) từ vietlott.vn.

Nguồn: https://vietlott.vn/vi/trung-thuong/ket-qua-trung-thuong/{645,655}
Mỗi kỳ quay có thể lấy trực tiếp qua query string `?id=<draw_id>&nocatche=1`
(GET tĩnh, không cần session/AjaxPro) — đã xác minh bằng cách khảo sát thực tế
trang này, id không tồn tại sẽ trả về trang không chứa khối kết quả.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Iterator, Optional

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

BASE_URL = "https://vietlott.vn/vi/trung-thuong/ket-qua-trung-thuong/{path}"

PRODUCT_PATHS = {
    "mega645": "645",
    "power655": "655",
}

USER_AGENT = (
    "Mozilla/5.0 (compatible; JackpotVietlottAgent/1.0; "
    "research project, contact via GitHub repo)"
)

REQUEST_TIMEOUT = 20
RETRY_COUNT = 3
RETRY_BACKOFF_SECONDS = 2

_DRAW_HEADER_RE = re.compile(
    r"Kỳ quay thưởng\s*<b>#?(\d+)</b>\s*ngày\s*<b>(\d{2})/(\d{2})/(\d{4})</b>",
    re.IGNORECASE,
)


@dataclass
class PrizeTier:
    tier_name: str
    winner_count: int
    prize_amount: int


@dataclass
class DrawResult:
    product: str
    draw_id: str
    draw_date: str  # ISO format YYYY-MM-DD
    numbers: list[int]
    bonus_number: Optional[int]  # "số Power" — chỉ có ở power655, None với mega645
    jackpot1_amount: Optional[int]
    jackpot2_amount: Optional[int]  # chỉ có ở power655
    prize_tiers: list[PrizeTier] = field(default_factory=list)


def _parse_int(text: str) -> int:
    """Bóc số từ chuỗi kiểu '1.076' hoặc '30.528.493.950' (dấu chấm phân cách nghìn)."""
    return int(re.sub(r"[^\d]", "", text))


def fetch_html(
    product: str,
    draw_id: Optional[str] = None,
    session: Optional[requests.Session] = None,
) -> str:
    """Fetch HTML thô cho 1 kỳ quay. draw_id=None -> kỳ mới nhất hiện tại."""
    if product not in PRODUCT_PATHS:
        raise ValueError(f"Sản phẩm không hợp lệ: {product}")

    url = BASE_URL.format(path=PRODUCT_PATHS[product])
    params: dict[str, object] = {"nocatche": 1}
    if draw_id is not None:
        params["id"] = draw_id

    sess = session or requests
    last_exc: Optional[Exception] = None
    for attempt in range(1, RETRY_COUNT + 1):
        try:
            resp = sess.get(
                url, params=params, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT
            )
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as exc:
            last_exc = exc
            logger.warning(
                "Fetch %s (id=%s) attempt %d/%d failed: %s",
                product, draw_id, attempt, RETRY_COUNT, exc,
            )
            if attempt < RETRY_COUNT:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
    assert last_exc is not None
    raise last_exc


def parse_draw(html: str, product: str) -> Optional[DrawResult]:
    """Parse HTML kết quả 1 kỳ quay. Trả None nếu draw_id không tồn tại (ngoài phạm vi)."""
    soup = BeautifulSoup(html, "html.parser")

    numbers_container = soup.select_one(".day_so_ket_qua_v2")
    if numbers_container is None:
        return None  # id không tồn tại -> trang không render khối kết quả

    header_match = _DRAW_HEADER_RE.search(html)
    if header_match is None:
        return None

    raw_id, dd, mm, yyyy = header_match.groups()
    draw_id = raw_id.zfill(5)
    draw_date = f"{yyyy}-{mm}-{dd}"

    numbers: list[int] = []
    bonus_number: Optional[int] = None
    for span in numbers_container.select(".bong_tron"):
        value = int(span.get_text(strip=True))
        if "active" in span.get("class", []):
            # Power 6/55 đánh dấu riêng "số Power" (số thứ 7) bằng class "active"
            bonus_number = value
        else:
            numbers.append(value)
    numbers.sort()

    # Lưu ý: mega645 chỉ có 1 cặp (h5 nhãn, h3 số tiền) trong .gt_jackpot; power655 có 2 cặp
    # nằm CHUNG một hàng .row (không tách 2 div .gt_jackpot riêng) nên phải ghép theo thứ tự
    # xuất hiện (zip) thay vì lấy phần tử đầu tiên của mỗi loại.
    jackpot1_amount: Optional[int] = None
    jackpot2_amount: Optional[int] = None
    for block in soup.select(".gt_jackpot"):
        labels = block.select("h5")
        amounts = block.select(".so_tien h3")
        for label_el, amount_el in zip(labels, amounts):
            label = label_el.get_text(strip=True)
            amount = _parse_int(amount_el.get_text())
            if "Jackpot 2" in label:
                jackpot2_amount = amount
            else:
                jackpot1_amount = amount

    prize_tiers: list[PrizeTier] = []
    table = soup.select_one("table.table-hover")
    if table is not None:
        for row in table.select("tbody tr"):
            cells = row.find_all("td")
            if len(cells) < 4:
                continue
            prize_tiers.append(
                PrizeTier(
                    tier_name=cells[0].get_text(strip=True),
                    winner_count=_parse_int(cells[2].get_text()),
                    prize_amount=_parse_int(cells[3].get_text()),
                )
            )

    return DrawResult(
        product=product,
        draw_id=draw_id,
        draw_date=draw_date,
        numbers=numbers,
        bonus_number=bonus_number,
        jackpot1_amount=jackpot1_amount,
        jackpot2_amount=jackpot2_amount,
        prize_tiers=prize_tiers,
    )


def fetch_draw(
    product: str,
    draw_id: Optional[str] = None,
    session: Optional[requests.Session] = None,
) -> Optional[DrawResult]:
    html = fetch_html(product, draw_id=draw_id, session=session)
    return parse_draw(html, product)


def get_latest_draw_id(product: str, session: Optional[requests.Session] = None) -> str:
    result = fetch_draw(product, draw_id=None, session=session)
    if result is None:
        raise RuntimeError(f"Không lấy được kỳ quay mới nhất cho {product}")
    return result.draw_id


def iter_draws(
    product: str,
    start_id: int,
    end_id: int,
    session: Optional[requests.Session] = None,
    delay_seconds: float = 0.4,
) -> Iterator[DrawResult]:
    """Duyệt tuần tự draw_id từ start_id đến end_id (bao gồm cả hai đầu).

    Bỏ qua id không tồn tại (out of range) thay vì raise, để backfill chạy hết
    dải id mà không dừng giữa chừng vì vài kỳ bị thiếu.
    """
    sess = session or requests.Session()
    for i in range(start_id, end_id + 1):
        draw_id = f"{i:05d}"
        result = fetch_draw(product, draw_id=draw_id, session=sess)
        if result is not None:
            yield result
        else:
            logger.info("Kỳ quay %s id=%s không tồn tại, bỏ qua", product, draw_id)
        time.sleep(delay_seconds)
