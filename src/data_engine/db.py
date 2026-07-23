"""Engine, session, và upsert logic cho bảng `vietlott_results` (Postgres/Neon, có sẵn dữ liệu)."""
from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import date
from typing import Iterable, Iterator, Optional

from dotenv import load_dotenv
from sqlalchemy import Engine, create_engine
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session, sessionmaker

from src.data_engine.models import Base, VietlottResult
from src.data_engine.scraper import DrawResult

load_dotenv()


def _normalize_url(url: str) -> str:
    """Neon cấp connection string dạng `postgresql://` trần, SQLAlchemy lại mặc định map
    scheme đó sang driver psycopg2 (không cài trong requirements.txt). Ép rõ driver psycopg
    (v3) đã cài sẵn để không cần người dùng tự sửa URL."""
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


def get_engine(database_url: Optional[str] = None) -> Engine:
    url = database_url or os.environ["DATABASE_URL"]
    return create_engine(_normalize_url(url), pool_pre_ping=True)


def init_db(engine: Engine) -> None:
    """Không dùng để TẠO bảng (vietlott_results đã có sẵn) — chỉ tạo nếu thật sự chưa tồn tại,
    ví dụ khi restore trên một DB Postgres trống."""
    Base.metadata.create_all(engine)


def get_session_factory(engine: Engine) -> sessionmaker:
    return sessionmaker(bind=engine, expire_on_commit=False)


@contextmanager
def session_scope(engine: Engine) -> Iterator[Session]:
    factory = get_session_factory(engine)
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _tier_winner_count(prize_tiers, tier_name: str) -> int:
    for tier in prize_tiers:
        if tier.tier_name.strip() == tier_name:
            return tier.winner_count
    return 0


def upsert_result(session: Session, result: DrawResult) -> None:
    """Idempotent: period_id đã tồn tại thì update tại chỗ, chưa có thì insert mới."""
    stmt = pg_insert(VietlottResult).values(
        period_id=int(result.draw_id),
        draw_date=date.fromisoformat(result.draw_date),
        numbers=result.numbers,
        bonus_number=result.bonus_number,
        jackpot1_value=result.jackpot1_amount,
        jackpot1_winners=_tier_winner_count(result.prize_tiers, "Jackpot 1"),
        jackpot2_value=result.jackpot2_amount,
        jackpot2_winners=_tier_winner_count(result.prize_tiers, "Jackpot 2"),
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=["period_id"],
        set_={
            "draw_date": stmt.excluded.draw_date,
            "numbers": stmt.excluded.numbers,
            "bonus_number": stmt.excluded.bonus_number,
            "jackpot1_value": stmt.excluded.jackpot1_value,
            "jackpot1_winners": stmt.excluded.jackpot1_winners,
            "jackpot2_value": stmt.excluded.jackpot2_value,
            "jackpot2_winners": stmt.excluded.jackpot2_winners,
        },
    )
    session.execute(stmt)


def upsert_results(session: Session, results: Iterable[DrawResult]) -> int:
    count = 0
    for result in results:
        upsert_result(session, result)
        count += 1
    return count
