# Jackpot Vietlott Agent (Omniverse)

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![Stable Baselines3](https://img.shields.io/badge/RL-Stable%20Baselines%203-purple)
![License](https://img.shields.io/badge/license-MIT-green)

> **Disclaimer:** This project is for **academic and research purposes only**. It does not guarantee or claim to predict winning lottery numbers. The lottery is a True Random Number Generator system.

**Jackpot Vietlott Agent** là một hệ thống phân tích dữ liệu và AI toàn diện (End-to-End Analytics System) nghiên cứu về xổ số Vietlott (Mega 6/45 & Power 6/55). 

Dự án tiếp cận bài toán xổ số dưới lăng kính của Khoa học Dữ liệu (Data Science), Học tăng cường (Reinforcement Learning) và Lý thuyết Trò chơi (Game Theory), được chia làm 3 hướng nghiên cứu chính:

1. **Kiểm toán tính ngẫu nhiên (Randomness Auditor):** Kiểm định thống kê độ tin cậy của cỗ máy quay số.
2. **Nghiên cứu AI (RL Researcher):** Phân tích hành vi (đặc biệt là sự "mê tín") của các mô hình RL trong môi trường hoàn toàn ngẫu nhiên.
3. **Tối ưu hóa Kỳ vọng (Game Theory Optimizer):** Phân tích hành vi đám đông để tìm ra các dãy số có giá trị kỳ vọng (Expected Value) cao nhất, giảm thiểu rủi ro chia giải.

---

## Kiến trúc Hệ thống (Modules)

Dự án được cấu trúc thành 4 module độc lập và liên kết với nhau qua một nền tảng dữ liệu chung:

*   **1. Data Engine & Gym Environment:** Scraper tự động cập nhật dữ liệu hàng ngày vào CSDL. Cung cấp môi trường `Vietlott Gym Env` (chuẩn OpenAI Gymnasium) cho các Agent luyện tập.
*   **2. The Auditor:** Chạy các bài test thống kê phức tạp (Chi-square, K-S test) để chứng minh tính phân bố đồng đều của dữ liệu lịch sử.
*   **3. The Researcher:** Huấn luyện DQN/PPO trên Gym Env và trích xuất Policy để phân tích khả năng AI bị "overfit" với tập dữ liệu nhiễu.
*   **4. The Optimizer:** Dự báo quy mô Jackpot và tối ưu hóa Portfolio vé (chọn các bộ số "lạnh" ít người mua) để tối đa hóa EV.

---

## Lộ trình Phát triển (Roadmap)

- [ ] **Phase 1:** Xây dựng Data Scraper & Database (SQLite).
- [ ] **Phase 2:** Phát triển `Vietlott Gym Env` và chạy Thống kê cơ bản.
- [ ] **Phase 3:** Xây dựng mô hình Game Theory & Dự đoán Jackpot.
- [ ] **Phase 4:** Tích hợp PyTorch, train tác tử RL và phân tích.
- [ ] **Phase 5:** Ra mắt Web Dashboard bằng Streamlit.

---

## Tech Stack

*   **Ngôn ngữ:** Python 3.10+
*   **Data Science:** Pandas, NumPy, SciPy, Scikit-learn
*   **AI/RL:** PyTorch, Stable Baselines 3, Gymnasium
*   **Visualization:** Streamlit, Plotly, Seaborn
*   **Khác:** SQLite, GitHub Actions, BeautifulSoup

---

## Cài đặt & Sử dụng (Sắp ra mắt)

*Hướng dẫn cài đặt chi tiết sẽ được cập nhật khi hoàn thành Phase 1.*

```bash
# Clone the repository
git clone https://github.com/your-username/Jackpot_Vietlott_Agent.git

# Install dependencies
pip install -r requirements.txt
```

---
*Dự án được phát triển nhằm mục đích nghiên cứu & xây dựng Portfolio chuyên nghiệp.*