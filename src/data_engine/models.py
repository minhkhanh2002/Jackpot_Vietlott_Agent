"""SQLAlchemy ORM model cho bảng `vietlott_results` (Power 6/55) — bảng có sẵn từ trước,
dự án chỉ đọc/ghi theo đúng schema hiện có, không tạo lại bằng init_db.
"""
from __future__ import annotations

from datetime import date
from typing import Optional

from sqlalchemy import BigInteger, Date, Integer
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class VietlottResult(Base):
    __tablename__ = "vietlott_results"

    period_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    draw_date: Mapped[date] = mapped_column(Date, nullable=False)
    numbers: Mapped[list[int]] = mapped_column(ARRAY(Integer), nullable=False)  # 6 số chính
    bonus_number: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # "số Power"

    jackpot1_value: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    jackpot1_winners: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    jackpot2_value: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    jackpot2_winners: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
