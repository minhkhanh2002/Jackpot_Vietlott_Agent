"""Vietlott Power 6/55 Agent — Streamlit showcase.

Tích hợp 3 module đã xây dựng: Auditor (kiểm toán tính ngẫu nhiên + điều tra jackpot),
Optimizer (dự báo jackpot + gợi ý vé), Researcher (báo cáo AI Superstition từ PPO đã train).

Chạy local: streamlit run app.py  (cần .env có DATABASE_URL)
Deploy Streamlit Cloud: khai báo DATABASE_URL trong Secrets của app.
"""
from __future__ import annotations

import os
from math import comb
from pathlib import Path

import pandas as pd
import streamlit as st

# Streamlit Cloud dùng st.secrets, không tự inject vào os.environ — bắc cầu trước khi các
# module data_engine đọc DATABASE_URL từ env (xem src/data_engine/db.py:get_engine()).
# st.secrets ném exception nếu không có file secrets.toml (bình thường khi chạy local bằng
# .env) nên phải try/except thay vì kiểm tra "in" trực tiếp.
if "DATABASE_URL" not in os.environ:
    try:
        os.environ["DATABASE_URL"] = st.secrets["DATABASE_URL"]
    except Exception:
        pass

from src.auditor.statistical_tests import (
    chi_square_number_frequency,
    hit_rate_by_jackpot_size,
    jackpot_rollover_streaks,
    predicted_vs_observed_hit_rate,
    validate_jackpot_split_ratio,
)
from src.data_engine.loader import load_results
from src.envs.vietlott_gym_env import NUM_POOL
from src.optimizer.jackpot_forecast import forecast_next_jackpot
from src.optimizer.portfolio import average_popularity, recommend_tickets

RL_REPORT_PATH = Path(__file__).resolve().parent / "data" / "reports" / "ai_superstition_report.md"

st.set_page_config(page_title="Vietlott Power 6/55 Agent", page_icon="🎰", layout="wide")


@st.cache_data(ttl=3600)
def get_data() -> pd.DataFrame:
    return load_results()


def fmt_vnd(n: float) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.2f} tỷ"
    if n >= 1e6:
        return f"{n / 1e6:.1f} triệu"
    return f"{n:,.0f}"


st.title("🎰 Vietlott Power 6/55 Agent")
st.caption(
    "Dự án nghiên cứu học thuật — **không** dự đoán số trúng thưởng (bất khả thi về mặt "
    "toán học). Mục tiêu: kiểm toán tính ngẫu nhiên, cố vấn giảm rủi ro chia giải, và nghiên "
    "cứu hành vi Agent RL trong môi trường nhiễu thuần tuý."
)

try:
    df = get_data()
except Exception as exc:  # pragma: no cover - phu thuoc moi truong trien khai
    st.error(f"Không kết nối được tới database: {exc}")
    st.stop()

tab_auditor, tab_optimizer, tab_researcher = st.tabs(
    ["📊 Auditor — Kiểm toán ngẫu nhiên", "🎯 Optimizer — Cố vấn vé", "🤖 Researcher — AI Superstition"]
)

# ---------------------------------------------------------------- Auditor ----
with tab_auditor:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tổng số kỳ quay", f"{len(df):,}")
    c2.metric("Jackpot 1 hiện tại", fmt_vnd(df["jackpot1_amount"].iloc[-1]) + " đ")
    c3.metric("Jackpot 2 hiện tại", fmt_vnd(df["jackpot2_amount"].iloc[-1]) + " đ")
    c4.metric("Cập nhật tới kỳ", str(df["period_id"].iloc[-1]))

    st.subheader("1. Máy quay số có ngẫu nhiên không?")
    chi = chi_square_number_frequency(df, num_pool=NUM_POOL)
    freq_df = pd.DataFrame({"Số": list(chi.observed.keys()), "Tần suất": list(chi.observed.values())}).set_index("Số")
    st.bar_chart(freq_df)
    verdict = "không có bằng chứng thiên vị" if chi.p_value > 0.05 else "có dấu hiệu lệch, cần điều tra thêm"
    st.caption(f"Chi-square: χ²={chi.statistic:.2f}, p-value={chi.p_value:.3f} → **{verdict}**.")

    st.subheader("2. Điều tra: Jackpot có dễ nổ hơn khi giá trị lớn không?")
    split = validate_jackpot_split_ratio(df)
    st.caption(
        f"Kiểm chứng công thức Vietlott công bố (90/10 giữa Jackpot 1/2): tỉ lệ mức tăng đo được "
        f"= **{split.ratio_median:.2f}** trên {split.n_samples} cặp kỳ sạch — khớp gần tuyệt đối."
    )
    fairness = predicted_vs_observed_hit_rate(df, num_pool=NUM_POOL, picks=6, n_buckets=5)
    fairness_df = pd.DataFrame(
        {
            "Dự đoán (mô hình công bằng)": [b.predicted_hit_rate for b in fairness],
            "Thực tế quan sát": [b.observed_hit_rate for b in fairness],
        },
        index=[f"Nhóm {i+1}" for i in range(len(fairness))],
    )
    st.bar_chart(fairness_df)
    st.caption(
        "Hai đường gần trùng và cùng tăng theo nhóm → tỉ lệ nổ tăng khi jackpot lớn khớp với mô "
        "hình công bằng (nhiều người mua vé hơn khi jackpot hấp dẫn) — không cần giả thuyết gian lận."
    )

    st.subheader("3. Chuỗi kỳ trượt liên tiếp trước khi nổ")
    col1, col2 = st.columns(2)
    for col, won_col, label in [(col1, "jackpot1_won", "Jackpot 1"), (col2, "jackpot2_won", "Jackpot 2")]:
        streaks = jackpot_rollover_streaks(df, won_column=won_col)
        col.metric(f"{label} — trung bình kỳ trượt/lần nổ", f"{streaks.mean_streak:.1f}")
        col.bar_chart(pd.Series(streaks.streak_lengths, name="Số kỳ trượt").value_counts().sort_index())

    with st.expander("Xem dữ liệu thô"):
        st.dataframe(df, width="stretch")

# -------------------------------------------------------------- Optimizer ----
with tab_optimizer:
    st.subheader("Dự báo Jackpot 1 kỳ tới")
    forecast = forecast_next_jackpot(df, window=20)
    fc1, fc2, fc3 = st.columns(3)
    fc1.metric("Jackpot hiện tại", fmt_vnd(forecast.last_jackpot1_amount) + " đ")
    fc1.caption("Kỳ gần nhất " + ("CÓ" if forecast.last_draw_won else "KHÔNG") + " người trúng")
    fc2.metric("Mức tăng trung vị gần đây", fmt_vnd(forecast.median_recent_increment) + " đ")
    fc3.metric("Dự báo kỳ tới", fmt_vnd(forecast.predicted_next) + " đ")

    st.divider()
    st.subheader("Cố vấn chọn vé — giảm rủi ro chia giải")
    st.caption(
        "⚠️ Gợi ý này **không** làm tăng xác suất trúng — mọi tổ hợp 6/55 có xác suất bằng nhau "
        "(đã kiểm chứng ở tab Auditor). Chỉ né các số nhiều người khác cũng hay chọn (giả định "
        "heuristic ngày/tháng sinh, không phải dữ liệu thực đo), để nếu trúng thì đỡ phải chia."
    )
    n_tickets = st.slider("Số bộ vé muốn gợi ý", 1, 10, 3)
    if st.button("Sinh vé"):
        tickets = recommend_tickets(count=n_tickets)
        rows = [
            {"Vé": " - ".join(f"{n:02d}" for n in t), "Điểm phổ biến TB (thấp = lạnh hơn)": round(average_popularity(t), 3)}
            for t in tickets
        ]
        st.table(pd.DataFrame(rows))

# ------------------------------------------------------------- Researcher ----
with tab_researcher:
    st.subheader("Báo cáo AI Superstition")
    st.caption(
        "PPO được train lặp lại trên chuỗi kỳ quay lịch sử (môi trường nhiễu thuần tuý) — câu "
        "hỏi nghiên cứu: Agent có tự học ra thiên vị số nào đó dù không có tín hiệu thật để học?"
    )
    if RL_REPORT_PATH.exists():
        st.markdown(RL_REPORT_PATH.read_text(encoding="utf-8"))
    else:
        st.info(
            "Chưa có báo cáo. Chạy `python -m src.researcher.train` rồi `python scripts/rl_report.py` "
            "để train và sinh báo cáo."
        )

st.divider()
st.caption(
    f"Nguồn dữ liệu: vietlott.vn/vi/trung-thuong/ket-qua-trung-thuong (lưu trên Neon Postgres, "
    f"bảng `vietlott_results`). Xác suất trúng 1 vé Power 6/55: 1/{comb(55, 6):,}."
)
