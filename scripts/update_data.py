"""Cron entrypoint: cập nhật các kỳ Power 6/55 mới (nếu có) vào `vietlott_results`.

Chạy: python scripts/update_data.py
Yêu cầu biến môi trường DATABASE_URL (Neon Postgres) — đọc qua .env khi chạy local.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import func, select

from src.data_engine.db import get_engine, get_session_factory, upsert_result
from src.data_engine.models import VietlottResult
from src.data_engine.scraper import fetch_draw, get_latest_draw_id

Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("logs/scraper.log", encoding="utf-8")],
)
logger = logging.getLogger("update_data")

PRODUCT = "power655"


def main() -> None:
    engine = get_engine()
    session_factory = get_session_factory(engine)

    with session_factory() as session:
        last_id = session.scalar(select(func.max(VietlottResult.period_id)))
    last_id_int = last_id or 0

    try:
        latest_id_int = int(get_latest_draw_id(PRODUCT))
    except Exception:
        logger.exception("Không lấy được kỳ quay mới nhất — bỏ qua lần chạy này")
        return

    if latest_id_int <= last_id_int:
        logger.info("Đã cập nhật tới kỳ %05d, chưa có kỳ mới", last_id_int)
        return

    updated = 0
    with session_factory() as session:
        for draw_id_int in range(last_id_int + 1, latest_id_int + 1):
            draw_id = f"{draw_id_int:05d}"
            try:
                result = fetch_draw(PRODUCT, draw_id=draw_id)
            except Exception:
                logger.exception("Lỗi fetch id=%s, bỏ qua kỳ này", draw_id)
                continue
            if result is None:
                logger.warning("id=%s không parse được dữ liệu, bỏ qua", draw_id)
                continue
            upsert_result(session, result)
            updated += 1
            logger.info("Đã cập nhật kỳ %s (%s)", draw_id, result.draw_date)
        session.commit()

    logger.info("Hoàn tất update_data: %d kỳ quay mới được ghi.", updated)


if __name__ == "__main__":
    main()
