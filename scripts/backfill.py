"""Script backfill Power 6/55 — chạy TAY khi cần nạp/vá một khoảng kỳ quay cụ thể
(vd. lấp gap phát hiện được, hoặc khôi phục dữ liệu). Lịch sử đầy đủ đã có sẵn trong
`vietlott_results`; script này KHÔNG cần chạy định kỳ — dùng scripts/update_data.py cho cron.

Ví dụ:
    python scripts/backfill.py --start 1370 --end 1374

Yêu cầu biến môi trường DATABASE_URL (Neon Postgres) — đọc qua .env khi chạy local.
Neon serverless có thể ngắt connection giữa phiên dài; script tự retry bằng cách mở
session mới khi gặp lỗi DB, không dừng giữa chừng.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy.exc import OperationalError

from src.data_engine.db import get_engine, get_session_factory, upsert_result
from src.data_engine.scraper import get_latest_draw_id, iter_draws

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("backfill")

PRODUCT = "power655"
DB_RETRY_COUNT = 3
DB_RETRY_SLEEP_SECONDS = 3
COMMIT_EVERY = 20


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill/vá dữ liệu Power 6/55 vào vietlott_results.")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=None, help="Mặc định: kỳ mới nhất hiện tại")
    parser.add_argument("--delay", type=float, default=0.4, help="Giây nghỉ giữa các request")
    args = parser.parse_args()

    end_id = args.end if args.end is not None else int(get_latest_draw_id(PRODUCT))
    logger.info("Backfill %s từ kỳ %05d đến %05d", PRODUCT, args.start, end_id)

    engine = get_engine()
    session_factory = get_session_factory(engine)

    count = 0
    session = session_factory()
    try:
        for result in iter_draws(PRODUCT, args.start, end_id, delay_seconds=args.delay):
            for db_attempt in range(1, DB_RETRY_COUNT + 1):
                try:
                    upsert_result(session, result)
                    count += 1
                    if count % COMMIT_EVERY == 0:
                        session.commit()
                        logger.info("Đã ghi %d kỳ quay (mới nhất: %s)", count, result.draw_id)
                    break
                except OperationalError:
                    logger.exception(
                        "Lỗi DB khi ghi kỳ %s (lần %d/%d) — mở lại session và thử lại",
                        result.draw_id, db_attempt, DB_RETRY_COUNT,
                    )
                    session.rollback()
                    session.close()
                    time.sleep(DB_RETRY_SLEEP_SECONDS)
                    session = session_factory()
            else:
                logger.error("Bỏ qua kỳ %s sau %d lần lỗi DB liên tiếp", result.draw_id, DB_RETRY_COUNT)
        session.commit()
    finally:
        session.close()

    logger.info("Hoàn tất backfill: %d kỳ quay đã ghi.", count)


if __name__ == "__main__":
    main()
