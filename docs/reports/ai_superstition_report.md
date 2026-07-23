# Báo cáo AI Superstition — PPO trên VietlottEnv (Power 6/55)

Ngày sinh báo cáo: 2026-07-24  
Model: `D:\Jackpot_Vietlott_Agent\data\models\ppo_power655.zip`  
Số episode chạy để lấy mẫu: 5 (mỗi episode ~1364 kỳ)

## Bối cảnh

VietlottEnv về bản chất là môi trường **nhiễu thuần tuý** — không có tín hiệu thật nào trong lịch sử kỳ quay trước có thể dự đoán kỳ quay sau (đã kiểm chứng ở Auditor: máy quay số công bằng, chi-square không bác bỏ phân phối đều). Câu hỏi nghiên cứu: khi PPO được train lặp lại nhiều lượt trên cùng một chuỗi dữ liệu này, nó có tự học ra một thiên vị/"mê tín" nào đó (policy suy biến về một tập số cố định) thay vì giữ hành vi ngẫu nhiên đều — hành vi ĐÚNG duy nhất khi không có tín hiệu để học?

## Kết quả định lượng

| Chỉ số | PPO đã train | Baseline ngẫu nhiên đều |
|---|---|---|
| Entropy (bits) | 5.593 | 5.780 |
| Entropy tối đa lý thuyết (bits) | 5.781 | 5.781 |
| Entropy chuẩn hoá (0-1) | 0.967 | 1.000 |
| Chi-square statistic | 12409.8 | 58.2 |
| Chi-square p-value | 0 | 0.3235 |
| Số lượt chọn số (n) | 40920 | 40920 |

**Chênh lệch entropy (baseline - trained): 0.187 bits.**

5 số PPO chọn nhiều nhất: 31 (2125 lần), 11 (1910 lần), 32 (1775 lần), 45 (1650 lần), 28 (1340 lần)

## Kết luận

**CÓ thiên vị thống kê rõ ràng nhưng KHÔNG suy biến hoàn toàn — Agent vẫn dùng gần hết số lượng số (entropy cao) nhưng phân bố không đều một cách nhất quán, không phải do nhiễu ngẫu nhiên.**

Chi-square statistic của policy đã train cao gấp **213 lần** chi-square statistic của baseline ngẫu nhiên trên CÙNG cỡ mẫu — không phải hiệu ứng cỡ mẫu lớn (baseline cũng có n lớn tương đương nhưng statistic vẫn ở mức bình thường theo lý thuyết), mà là một lệch hệ thống thật sự trong hành vi chọn số của Agent.

Lưu ý: bias thống kê này không có nghĩa là Agent "dự đoán được" kỳ quay — reward trung bình vẫn âm (mất giá vé) và không có tín hiệu thật nào trong dữ liệu để học. Khác với hệ thống RL trước đây của dự án (nơi agent suy biến hoàn toàn về đúng 1 bộ số cố định — entropy gần 0), lần chạy này Agent vẫn dùng gần hết cả 55 số (entropy chuẩn hoá 0.967) nhưng với tần suất không đều một cách nhất quán — một dạng "mê tín nhẹ" tinh vi hơn: không lộ rõ qua entropy tổng thể, chỉ phát hiện được nhờ chi-square test chi tiết hơn. Bài học phương pháp luận: đo policy RL trong môi trường ngẫu nhiên nên dùng cả hai loại kiểm định, vì mỗi loại bắt được một dạng lệch khác nhau.
