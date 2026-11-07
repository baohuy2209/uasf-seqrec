## 1 Chẩn đoán B

| Nguồn | Early Test mean ± SD | Late Test mean ± SD |
|---|---|---|
| ID | Chưa đủ 3 seed | Chưa đủ 3 seed |
| C | Chưa đủ 3 seed | Chưa đủ 3 seed |
| B | Chưa đủ 3 seed | Chưa đủ 3 seed |
| T | Chưa đủ 3 seed | Chưa đủ 3 seed |
| CB | Chưa đủ 3 seed | Chưa đủ 3 seed |
| CT | Chưa đủ 3 seed | Chưa đủ 3 seed |
| BT | Chưa đủ 3 seed | Chưa đủ 3 seed |
| CBT | Chưa đủ 3 seed | Chưa đủ 3 seed |
| Add / Drop C | Chưa có | Chưa có |
| Add / Drop B | Chưa có | Chưa có |
| Add / Drop T | Chưa có | Chưa có |
Add = một nguồn − ID; Drop = CBT − bỏ nguồn. SD mẫu qua seed; không gán ý nghĩa seed-level cho CI người dùng.

## 2 Đối chứng khi huấn luyện

| Nguồn thay thế | Test mean ± SD | Δ đối chứng − thật [CI95%] |
|---|---|---|
| category shuffled | Chưa đủ 3 seed | Chưa có |
| category random | Chưa đủ 3 seed | Chưa có |
| brand shuffled | Chưa đủ 3 seed | Chưa có |
| brand random | Chưa đủ 3 seed | Chưa có |
| title shuffled | Chưa đủ 3 seed | Chưa có |
| title random | Chưa đủ 3 seed | Chưa có |
Random V2: C/B đều trên từ vựng; Title Gaussian cùng mean/covariance thật. CI percentile ghép cặp người dùng, 1.000 lần; chưa hiệu chỉnh nhiều so sánh.

## 3 Nguyên nhân hợp nhất muộn

| Mục | Số liệu seed 42 CBT |
|---|---|
| K1 | Chưa có diagnostics đã nhập |
| K2 | Chưa có |
| K3 | Chưa có |
| K4 | Chưa có |
†Best epoch ≤5; đối chiếu ID và late_reference_42 trong results.csv. K3_validation_epochs.csv chứa toàn bộ đường Val; K4 nhiều seed được ghi riêng, không suy SD từ một seed.

## 4 Lựa chọn theo nhóm

| Cơ chế | Global Test | Nhóm Val → Test | Oracle Test |
|---|---|---|---|
| early | Chưa đủ 3 seed | Chưa đủ 3 seed | Chưa đủ 3 seed |
| late | Chưa đủ 3 seed | Chưa đủ 3 seed | Chưa đủ 3 seed |
Cùng nhóm lịch sử được khóa trên Val cho cả hai cơ chế. Chọn subset bằng Val; oracle Test chỉ là trần mô tả, không dùng chọn backbone.

## 5 Căn chỉnh utility và E0 E1

| Nguồn | ΔCE Train-meta | Δrank Train-meta | ΔNDCG Val cùng model |
|---|---|---|---|
| category | +0.061546 | +0.097452 | +0.003750 |
| brand | +0.169126 | +0.276623 | +0.012978 |
| title | +0.037094 | +0.068587 | +0.001727 |
Nhãn: CE; E0/E1: GO; phạm vi chẩn đoán, không cho phép S3. Hai hệ độc lập, cohort eligible-only cho CE; Val giữ OOV miss.

## 6 Source dropout

| Chế độ S1 | Validation mean ± SD | Test mean ± SD |
|---|---|---|
| CBT | Chưa đủ 3 seed | Chưa đủ 3 seed |
| BT | Chưa đủ 3 seed | Chưa đủ 3 seed |
| CT | Chưa đủ 3 seed | Chưa đủ 3 seed |
| CB | Chưa đủ 3 seed | Chưa đủ 3 seed |
| ID | Chưa đủ 3 seed | Chưa đủ 3 seed |
| ID riêng | Chưa đủ 3 seed | Chưa đủ 3 seed |
| Early CBT không SD | Chưa đủ 3 seed | Chưa đủ 3 seed |
ID trong S1 = tắt mọi nguồn, không phải ID-only riêng. Full S1 phải vượt ID, giữ ≥98% early không SD; all-off phải giữ ≥95% ID (validation).

## 7 Đề xuất và trạng thái

| Mục | Nội dung |
|---|---|
| Backbone | PENDING |
| Kiểm tra validation | {} |
| Framing | Chưa chốt; chỉ dùng validation |
| UASF Test | Chưa mở; cần freeze/release-test sau mốc 13/11 |
| Tái lập | Số liệu tính từ results.csv/ranks; source, cohort, checkpoint SHA256 được ràng buộc; output cũ giữ nguyên. |

