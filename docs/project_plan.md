# Kế hoạch Dự án: Jackpot Vietlott Agent (Vietlott Omniverse)

## 1. Tổng quan Dự án (Project Overview)
Dự án **Jackpot Vietlott Agent** là một hệ thống phân tích dữ liệu và AI toàn diện (End-to-End Analytics System) xoay quanh dữ liệu xổ số Vietlott. 
Thay vì cố gắng dự đoán chính xác dãy số trúng thưởng (một việc bất khả thi về mặt toán học), dự án này tập trung vào 3 hướng nghiên cứu học thuật và thực tiễn:
1.  **Kiểm toán tính ngẫu nhiên (Randomness Auditor):** Sử dụng thống kê để kiểm định độ tin cậy của cỗ máy quay số.
2.  **Nghiên cứu AI (RL Researcher):** Phân tích "sự mê tín" của các mô hình Học tăng cường (Reinforcement Learning) khi hoạt động trong môi trường hoàn toàn ngẫu nhiên (pure noise).
3.  **Tối ưu hóa Kỳ vọng (Game Theory Optimizer):** Ứng dụng lý thuyết trò chơi và phân tích hành vi đám đông để tìm ra các dãy số có giá trị kỳ vọng (Expected Value - EV) cao nhất, nhằm giảm thiểu rủi ro phải chia giải khi trúng.

Mục tiêu chính của dự án là xây dựng một portfolio xuất sắc để "showcase" các kỹ năng về Data Engineering, Data Science, Machine Learning và Reinforcement Learning.

## 2. Kiến trúc Hệ thống (System Architecture)

Hệ thống được chia thành 4 module độc lập nhưng có tính liên kết chặt chẽ về mặt dữ liệu:

### 2.1. Module Nền tảng: Data Engine & Gym Environment
*   **Data Scraper:** Bot tự động thu thập kết quả Vietlott (Mega 6/45, Power 6/55) từ website chính thức hoặc các nguồn dữ liệu đáng tin cậy.
*   **Database:** Lưu trữ dữ liệu lịch sử bằng SQLite/PostgreSQL. Cung cấp API hoặc function để truy xuất dữ liệu dạng Pandas DataFrame.
*   **Vietlott Gym Env:** Một môi trường chuẩn `OpenAI Gymnasium`. 
    *   *State:* Kết quả các kỳ quay trước, giá trị Jackpot hiện tại.
    *   *Action:* Chọn một bộ 6 số (hoặc mua nhiều vé).
    *   *Reward:* Trả về số tiền trúng thưởng (dựa trên tỷ lệ thực tế hoặc dữ liệu lịch sử) trừ đi chi phí mua vé.

### 2.2. Module Kiểm toán (The Auditor)
*   **Chức năng:** Thực hiện các kiểm định thống kê (Hypothesis Testing) trên tập dữ liệu lịch sử.
*   **Kỹ thuật:** 
    *   Chi-square test, Kolmogorov-Smirnov (K-S) test.
    *   Phân tích chuỗi thời gian (Autocorrelation) để tìm chu kỳ (nếu có).
*   **Đầu ra:** Báo cáo tự động minh họa tính phân bố đồng đều của các quả bóng.

### 2.3. Module AI Nghiên cứu (The Researcher)
*   **Chức năng:** Huấn luyện các Agent RL trong `Vietlott Gym Env` và phân tích hành vi của chúng.
*   **Kỹ thuật:** 
    *   Cài đặt PPO (Proximal Policy Optimization), DQN (Deep Q-Network).
    *   Phân tích chính sách (Policy Evaluation) để xem liệu Agent có tự động học ra các "con số phong thủy" hay không (hiện tượng overfitting trong môi trường nhiễu).
*   **Đầu ra:** Báo cáo phân tích hành vi AI (AI Superstition).

### 2.4. Module Tối ưu hóa (The Optimizer)
*   **Chức năng:** Cố vấn mua vé dựa trên Lý thuyết trò chơi.
*   **Kỹ thuật:**
    *   Dự đoán quy mô giải Jackpot kỳ tiếp theo (Time-series Forecasting).
    *   Phân tích cụm (Clustering) các dãy số ít người chọn (dựa trên phân phối ngày sinh, số đẹp/xấu theo văn hóa).
    *   Thuật toán Portfolio Optimization để sinh ra bộ vé tối đa hóa EV.
*   **Đầu ra:** Hệ thống Recommender gợi ý bộ số nên mua khi Jackpot đạt mức hấp dẫn.

## 3. Cấu trúc Thư mục (Directory Structure)

```text
Jackpot_Vietlott_Agent/
│
├── data/                  # Chứa file SQLite, CSV
├── docs/                  # Tài liệu dự án (Kế hoạch, Báo cáo)
├── src/                   # Mã nguồn chính
│   ├── data_engine/       # Scraper, Database connectors
│   ├── envs/              # vietlott_gym_env.py
│   ├── auditor/           # Module Thống kê
│   ├── researcher/        # Module AI/RL
│   └── optimizer/         # Module Game Theory
│
├── notebooks/             # Các file Jupyter (EDA, thử nghiệm)
├── scripts/               # Script chạy tự động (cronjobs)
├── tests/                 # Unit test
│
├── app.py                 # Giao diện web bằng Streamlit
├── requirements.txt       # Các thư viện phụ thuộc
└── README.md              # Tài liệu giới thiệu tổng quan
```

## 4. Lộ trình Triển khai (Roadmap)

Dự án được triển khai theo các giai đoạn (Phase) sau:

*   **Phase 1: Xây dựng Nền tảng Dữ liệu (Foundation)**
    *   Viết Data Scraper để lấy dữ liệu Mega 6/45 và Power 6/55.
    *   Lưu trữ vào SQLite và viết hàm load dữ liệu ra Pandas.
    *   Thực hiện Exploratory Data Analysis (EDA) cơ bản trên Notebook.

*   **Phase 2: Môi trường & Thống kê (Environment & Statistics)**
    *   Xây dựng và test `Vietlott Gym Env`.
    *   Hoàn thiện `Module Auditor`, chạy các kiểm định thống kê cơ bản và xuất biểu đồ (Heatmap).

*   **Phase 3: Tối ưu hóa Thực tiễn (Game Theory)**
    *   Xây dựng mô hình dự đoán giá trị Jackpot.
    *   Phân tích các con số "lạnh" (ít người quan tâm).
    *   Hoàn thiện `Module Optimizer` sinh ra bộ vé tối đa hóa EV.

*   **Phase 4: Nghiên cứu AI chuyên sâu (RL Research)**
    *   Tích hợp PyTorch / Stable Baselines 3.
    *   Huấn luyện DQN/PPO trên `Vietlott Gym Env`.
    *   Trích xuất policy và phân tích "sự mê tín" của Agent.

*   **Phase 5: Đóng gói và Trình diễn (Showcase)**
    *   Xây dựng web app bằng Streamlit (`app.py`) tích hợp các biểu đồ và tính năng cố vấn vé.
    *   Hoàn thiện `README.md` chuyên nghiệp.
    *   Cài đặt GitHub Actions để tự động cập nhật dữ liệu.

## 5. Tech Stack
*   **Ngôn ngữ:** Python 3.10+
*   **Data/Stats:** Pandas, NumPy, SciPy, Scikit-learn
*   **RL/AI:** PyTorch, Stable Baselines 3, Gymnasium
*   **Web/Visualization:** Streamlit, Plotly, Matplotlib, Seaborn
*   **Database:** SQLite / SQLAlchemy
*   **Automation:** GitHub Actions, BeautifulSoup (Scraping)
