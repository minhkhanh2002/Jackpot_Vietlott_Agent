# Jackpot Vietlott Agent — Power 6/55

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![Stable Baselines3](https://img.shields.io/badge/RL-Stable%20Baselines%203-purple)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)
![License](https://img.shields.io/badge/license-MIT-green)

> **Disclaimer:** Dự án phục vụ mục đích **nghiên cứu học thuật**, không dự đoán hay đảm bảo số trúng thưởng — xổ số là hệ thống ngẫu nhiên thật (True Random Number Generator).

**Jackpot Vietlott Agent** là hệ thống phân tích dữ liệu và AI cho xổ số **Power 6/55** (đã thu hẹp scope từ dự định ban đầu gồm cả Mega 6/45), gồm 3 hướng nghiên cứu:

1. **Kiểm toán tính ngẫu nhiên (Auditor):** Chi-square/K-S test trên số quay ra; điều tra định lượng nghi vấn "jackpot lớn dễ nổ hơn" bằng cách suy ngược số vé bán ra từ chính công thức tính Jackpot mà Vietlott công bố, rồi so sánh tỉ lệ nổ dự đoán (mô hình công bằng) với tỉ lệ nổ thực tế quan sát được.
2. **Nghiên cứu AI (Researcher):** Train PPO trên môi trường Gymnasium mô phỏng việc mua vé qua lịch sử thật; đo entropy + chi-square của phân phối số Agent chọn so với baseline ngẫu nhiên để phát hiện "mê tín" (bias tự học ra dù môi trường không có tín hiệu thật).
3. **Cố vấn vé (Optimizer):** Dự báo giá trị Jackpot kỳ tới; gợi ý vé né các số "dễ trùng ngày/tháng sinh" (giả định heuristic, không phải dữ liệu thực đo) để giảm rủi ro chia giải — **không** làm tăng xác suất trúng.

---

## Kiến trúc

```text
src/
├── data_engine/    # scraper (vietlott.vn) + Neon Postgres (bảng vietlott_results) + loader
├── envs/            # VietlottEnv (Gymnasium) + tính thưởng theo cơ cấu giải chính thức
├── auditor/         # chi-square, K-S, autocorrelation, ước lượng số vé bán ra, fairness check
├── optimizer/        # dự báo jackpot + gợi ý vé
└── researcher/       # train PPO + phân tích entropy/chi-square policy
scripts/              # backfill.py, update_data.py (cron), rl_report.py
app.py                # Streamlit showcase
```

Dữ liệu lưu trên **Neon (Postgres serverless, free tier)**, bảng `vietlott_results` — không tự tạo schema, dùng đúng bảng có sẵn từ trước (đã bổ sung kỳ thiếu, không backfill lại từ đầu).

---

## Lộ trình (đã hoàn thành)

- [x] **Phase 1:** Data Engine — scraper + Neon Postgres + cron cập nhật hàng ngày.
- [x] **Phase 2:** `VietlottEnv` (Gymnasium) + Auditor (thống kê + điều tra jackpot).
- [x] **Phase 3:** Optimizer — dự báo jackpot + cố vấn chọn vé.
- [x] **Phase 4:** RL Researcher — train PPO + báo cáo AI Superstition.
- [x] **Phase 5:** Streamlit showcase (`app.py`).

## Tech Stack

- **Ngôn ngữ:** Python 3.10+
- **Data/DB:** Pandas, NumPy, SciPy, SQLAlchemy, psycopg (Postgres), Neon
- **RL:** Gymnasium, Stable-Baselines3, PyTorch (PPO — DQN không tương thích vì action space `MultiDiscrete`)
- **UI:** Streamlit
- **Automation:** GitHub Actions, BeautifulSoup

---

## Cài đặt & Sử dụng

```bash
git clone <repo-url>
cd Jackpot_Vietlott_Agent

python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash; dùng .venv\Scripts\activate.bat cho cmd

pip install -r requirements.txt

cp .env.example .env
# Điền DATABASE_URL (connection string Neon Postgres) vào .env
```

Chạy test:

```bash
pytest
```

Chạy web app:

```bash
streamlit run app.py
```

Train agent RL và sinh báo cáo AI Superstition:

```bash
python -m src.researcher.train --timesteps 150000
python scripts/rl_report.py --episodes 5
```

Cập nhật dữ liệu thủ công (bình thường chạy tự động qua `.github/workflows/update_data.yml`):

```bash
python scripts/update_data.py
```

### Deploy

- **Neon:** tạo project free tại [neon.tech](https://neon.tech), lấy connection string.
- **GitHub Actions:** thêm repo secret `DATABASE_URL` để `update_data.yml` chạy cron hàng ngày.
- **Streamlit Community Cloud:** deploy từ repo, khai báo `DATABASE_URL` trong mục Secrets của app (`st.secrets`).

---
*Dự án nghiên cứu & xây dựng portfolio Data Engineering / Data Science / RL.*
