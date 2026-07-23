"""Kiểm định thống kê trên dữ liệu kết quả Vietlott.

Mọi hàm nhận `pd.DataFrame` đúng schema trả về bởi `data_engine.loader.load_draws()`
(cột num_1..num_6, jackpot1_amount, jackpot1_won, ...) — không tự đọc DB, nên test được
với DataFrame synthetic lẫn dữ liệu thật.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

NUMBER_COLUMNS = ["num_1", "num_2", "num_3", "num_4", "num_5", "num_6"]

# Công thức chính thức Vietlott (vietlott.vn/vi/tin-tuc/8851-cach-xac-dinh-gia-tri-jackpot-1-va-jackpot-2):
# Jackpot1(kỳ N) = Jackpot1(kỳ N-1) + 90% x (doanh thu dành cho jackpot trong kỳ N);
# doanh thu dành cho jackpot = 55% tổng doanh thu bán vé; Jackpot2 nhận 10% còn lại của phần đó.
# => mức tăng jackpot1 giữa 2 kỳ liên tiếp (khi không có ai trúng) tỉ lệ thuận với số vé bán ra.
TICKET_PRICE_DEFAULT = 10_000
JACKPOT_REVENUE_SHARE = 0.55
JACKPOT1_SHARE_OF_POOL = 0.90
JACKPOT1_CAP = 300_000_000_000  # mốc "khoá trần" — phần vượt chuyển sang Jackpot2, phá công thức cộng dồn tuyến tính


@dataclass
class ChiSquareResult:
    statistic: float
    p_value: float
    observed: dict[int, int]
    expected_per_number: float


def chi_square_number_frequency(df: pd.DataFrame, num_pool: int) -> ChiSquareResult:
    """H0: mỗi số 1..num_pool có xác suất được quay ra bằng nhau."""
    counts = pd.Series(df[NUMBER_COLUMNS].to_numpy().ravel()).value_counts()
    observed = np.array([counts.get(n, 0) for n in range(1, num_pool + 1)], dtype=float)
    expected = observed.sum() / num_pool
    statistic, p_value = stats.chisquare(observed, f_exp=np.full(num_pool, expected))
    return ChiSquareResult(
        statistic=float(statistic),
        p_value=float(p_value),
        observed={n: int(c) for n, c in zip(range(1, num_pool + 1), observed)},
        expected_per_number=float(expected),
    )


@dataclass
class KSResult:
    statistic: float
    p_value: float


def kolmogorov_smirnov_uniformity(df: pd.DataFrame, num_pool: int) -> KSResult:
    """H0: các số quay ra phân bố đều trên [1, num_pool].

    Lưu ý: K-S test giả định phân phối liên tục, ở đây áp dụng gần đúng cho dữ liệu số
    nguyên rời rạc — đủ dùng để phát hiện lệch rõ rệt, không thay thế chi-square test ở trên.
    """
    sample = df[NUMBER_COLUMNS].to_numpy().ravel()
    statistic, p_value = stats.kstest(sample, "uniform", args=(1, num_pool - 1))
    return KSResult(statistic=float(statistic), p_value=float(p_value))


def autocorrelation_of_number_sums(df: pd.DataFrame, max_lag: int = 10) -> dict[int, float]:
    """H0 (không có chu kỳ): autocorrelation ~ 0 ở mọi lag.

    Dùng tổng 6 số mỗi kỳ (sắp theo draw_date) làm chuỗi thời gian vô hướng đơn giản.
    """
    series = df.sort_values("draw_date")[NUMBER_COLUMNS].sum(axis=1).to_numpy(dtype=float)
    n = len(series)
    mean = series.mean()
    var = np.sum((series - mean) ** 2)

    result: dict[int, float] = {}
    for lag in range(1, max_lag + 1):
        if lag >= n:
            break
        cov = np.sum((series[:-lag] - mean) * (series[lag:] - mean))
        result[lag] = float(cov / var) if var > 0 else 0.0
    return result


@dataclass
class RolloverStreaks:
    streak_lengths: list[int]  # số kỳ trượt liên tiếp TRƯỚC mỗi lần jackpot nổ
    mean_streak: float
    implied_hit_probability: float  # ước lượng xác suất nổ/kỳ từ dữ liệu (≈ 1/(mean_streak+1))


def jackpot_rollover_streaks(df: pd.DataFrame, won_column: str = "jackpot1_won") -> RolloverStreaks:
    """Tính chuỗi số kỳ trượt liên tiếp trước mỗi lần Jackpot nổ (mục 2.1 kế hoạch dự án).

    KHÔNG kết luận gian lận hay không — chỉ mô tả phân phối thực tế để đối chiếu với phân phối
    hình học kỳ vọng (nếu xác suất nổ mỗi kỳ là hằng số, độc lập thời gian/memoryless). Vietlott
    không công bố số vé bán mỗi kỳ nên không thể tách "gian lận" khỏi "jackpot lớn -> nhiều
    người mua vé hơn -> tự nhiên dễ có người trúng hơn" chỉ bằng dữ liệu công khai này — báo cáo
    dùng hàm này PHẢI nêu rõ giới hạn trên, không kết luận thay dữ liệu.
    """
    won = df.sort_values("draw_date")[won_column].to_numpy()
    streaks: list[int] = []
    current = 0
    for w in won:
        if bool(w):
            streaks.append(current)
            current = 0
        else:
            current += 1

    mean_streak = float(np.mean(streaks)) if streaks else float("nan")
    implied_p = 1.0 / (mean_streak + 1) if streaks else float("nan")
    return RolloverStreaks(streak_lengths=streaks, mean_streak=mean_streak, implied_hit_probability=implied_p)


@dataclass
class HitRateByJackpotBucket:
    bucket_labels: list[str]
    hit_rate_per_bucket: list[float]
    count_per_bucket: list[int]


def hit_rate_by_jackpot_size(
    df: pd.DataFrame,
    won_column: str = "jackpot1_won",
    amount_column: str = "jackpot1_amount",
    n_buckets: int = 4,
) -> HitRateByJackpotBucket:
    """Chia các kỳ theo bucket giá trị jackpot, tính tỉ lệ kỳ có người trúng trong mỗi bucket.

    Hit-rate tăng rõ theo bucket là tín hiệu ĐÁNG điều tra thêm, nhưng xem giới hạn dữ liệu
    ghi ở docstring của `jackpot_rollover_streaks()` trước khi kết luận.
    """
    valid = df.dropna(subset=[amount_column])
    buckets = pd.qcut(valid[amount_column], q=n_buckets, duplicates="drop")
    grouped = valid.groupby(buckets, observed=True)[won_column]

    hit_rate = grouped.mean()
    counts = grouped.count()
    return HitRateByJackpotBucket(
        bucket_labels=[str(idx) for idx in hit_rate.index],
        hit_rate_per_bucket=[float(x) for x in hit_rate.to_numpy()],
        count_per_bucket=[int(x) for x in counts.to_numpy()],
    )


def clean_jackpot1_transition_mask(df: pd.DataFrame, cap: int) -> np.ndarray:
    """True tại vị trí k nếu đoạn (k -> k+1) là cộng dồn "sạch": không ai trúng Jackpot 1
    hay Jackpot 2 ở cả 2 đầu đoạn, và jackpot1 chưa chạm mốc khoá trần — chỉ những đoạn này
    mới tuân theo công thức cộng dồn tuyến tính đơn giản."""
    j1 = df["jackpot1_amount"].to_numpy()
    won1 = df["jackpot1_won"].to_numpy()
    won2 = df["jackpot2_won"].to_numpy()
    return (
        (~won1[:-1]) & (~won1[1:]) & (~won2[:-1]) & (~won2[1:]) & (j1[:-1] < cap - 1_000_000_000)
    )


@dataclass
class SplitRatioCheck:
    """Kiểm chứng tỉ lệ mức tăng Jackpot1/Jackpot2 có đúng 90/10 như Vietlott công bố không."""

    ratio_median: float
    ratio_min: float
    ratio_max: float
    n_samples: int


def validate_jackpot_split_ratio(df: pd.DataFrame, cap: int = JACKPOT1_CAP) -> SplitRatioCheck:
    d = df.sort_values("draw_date").reset_index(drop=True)
    j1 = d["jackpot1_amount"].to_numpy()
    j2 = d["jackpot2_amount"].to_numpy()
    clean = clean_jackpot1_transition_mask(d, cap)

    inc1 = (j1[1:] - j1[:-1])[clean]
    inc2 = (j2[1:] - j2[:-1])[clean]
    ratio = inc1 / inc2
    return SplitRatioCheck(
        ratio_median=float(np.median(ratio)) if len(ratio) else float("nan"),
        ratio_min=float(ratio.min()) if len(ratio) else float("nan"),
        ratio_max=float(ratio.max()) if len(ratio) else float("nan"),
        n_samples=int(clean.sum()),
    )


def estimate_tickets_from_jackpot_increment(
    df: pd.DataFrame,
    ticket_price: int = TICKET_PRICE_DEFAULT,
    jackpot_revenue_share: float = JACKPOT_REVENUE_SHARE,
    jackpot1_share_of_pool: float = JACKPOT1_SHARE_OF_POOL,
    cap: int = JACKPOT1_CAP,
) -> pd.DataFrame:
    """Ước lượng số vé bán ra mỗi kỳ từ mức tăng Jackpot1 giữa 2 kỳ liên tiếp (chỉ tính được
    tại các đoạn "sạch" — xem `clean_jackpot1_transition_mask`). Trả về DataFrame với
    `draw_date`, `jackpot1_amount` (giá trị ở ĐẦU đoạn — cái thu hút người mua vé kỳ đó),
    `n_tickets_est` (NaN nếu đoạn không "sạch", không ước lượng được).
    """
    d = df.sort_values("draw_date").reset_index(drop=True)
    j1 = d["jackpot1_amount"].to_numpy()
    clean = clean_jackpot1_transition_mask(d, cap)

    inc1 = j1[1:] - j1[:-1]
    rate = jackpot_revenue_share * jackpot1_share_of_pool
    n_tickets = np.where(clean, inc1 / (rate * ticket_price), np.nan)

    return pd.DataFrame({
        "draw_date": d["draw_date"].iloc[:-1].to_numpy(),
        "jackpot1_amount": d["jackpot1_amount"].iloc[:-1].to_numpy(),
        "n_tickets_est": n_tickets,
    })


@dataclass
class FairnessCheckBucket:
    """1 dòng so sánh tỉ lệ nổ Jackpot1 DỰ ĐOÁN (mô hình công bằng + số vé ước lượng) với
    tỉ lệ nổ THỰC TẾ quan sát, trong 1 khoảng giá trị jackpot."""

    bucket_label: str
    median_n_tickets: float
    n_clean_draws: int
    predicted_hit_rate: float
    observed_hit_rate: float
    observed_count: int


def predicted_vs_observed_hit_rate(
    df: pd.DataFrame,
    num_pool: int = 55,
    picks: int = 6,
    n_buckets: int = 5,
    ticket_price: int = TICKET_PRICE_DEFAULT,
    jackpot_revenue_share: float = JACKPOT_REVENUE_SHARE,
    jackpot1_share_of_pool: float = JACKPOT1_SHARE_OF_POOL,
    cap: int = JACKPOT1_CAP,
) -> list[FairnessCheckBucket]:
    """So sánh tỉ lệ nổ Jackpot1 dự đoán bởi mô hình "công bằng, số vé tỉ lệ thuận với độ hot
    của jackpot" với tỉ lệ nổ thực tế quan sát, theo từng khoảng giá trị jackpot.

    Dự đoán = 1 - (1-p)^N, với p = 1/C(num_pool, picks) (xác suất 1 vé trùng 6/6) và N =
    số vé ước lượng trung vị của các kỳ "sạch" trong bucket đó (từ
    `estimate_tickets_from_jackpot_increment`). Nếu dự đoán và quan sát khớp nhau (cùng
    chiều tăng, cùng bậc độ lớn) thì pattern "jackpot lớn dễ nổ hơn" giải thích được hoàn
    toàn bởi số vé bán tăng theo jackpot — không cần giả thuyết gian lận.
    """
    d = df.sort_values("draw_date").reset_index(drop=True)
    p_single = 1 / comb(num_pool, picks)

    bucketed_all, bin_edges = pd.qcut(d["jackpot1_amount"], q=n_buckets, duplicates="drop", retbins=True)
    observed = d.groupby(bucketed_all, observed=True)["jackpot1_won"].agg(["mean", "count"])

    tickets_df = estimate_tickets_from_jackpot_increment(
        d, ticket_price, jackpot_revenue_share, jackpot1_share_of_pool, cap
    )
    clean_tickets = tickets_df.dropna(subset=["n_tickets_est"]).copy()
    clean_tickets["bucket"] = pd.cut(clean_tickets["jackpot1_amount"], bins=bin_edges, include_lowest=True)

    predicted = clean_tickets.groupby("bucket", observed=True).agg(
        median_n_tickets=("n_tickets_est", "median"),
        n_clean_draws=("n_tickets_est", "count"),
    )
    predicted["predicted_hit_rate"] = 1 - (1 - p_single) ** predicted["median_n_tickets"]

    combined = predicted.join(observed)
    results: list[FairnessCheckBucket] = []
    for label, row in combined.iterrows():
        results.append(
            FairnessCheckBucket(
                bucket_label=str(label),
                median_n_tickets=float(row["median_n_tickets"]) if pd.notna(row["median_n_tickets"]) else float("nan"),
                n_clean_draws=int(row["n_clean_draws"]) if pd.notna(row["n_clean_draws"]) else 0,
                predicted_hit_rate=float(row["predicted_hit_rate"]) if pd.notna(row["predicted_hit_rate"]) else float("nan"),
                observed_hit_rate=float(row["mean"]) if pd.notna(row["mean"]) else float("nan"),
                observed_count=int(row["count"]) if pd.notna(row["count"]) else 0,
            )
        )
    return results
