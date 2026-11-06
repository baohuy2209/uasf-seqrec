# Báo cáo mini checkpoint ngày 06 tháng 11 năm 2026

**Báo cáo mini checkpoint ngày 06 tháng 11 năm 2026**

**1 Tích hợp sớm B**

| Tập nguồn | Seed 42 | Seed 43 | Seed 44 | Mean ± SD |
|---|---|---|---|---|
| ID riêng | 0.047099 | 0.047842 | 0.048611 | 0.047851 ± 0.000756 |
| C | Chưa có | Chưa có | Chưa có | Chưa đủ 3 seed |
| B | Chưa có | Chưa có | Chưa có | Chưa đủ 3 seed |
| T | Chưa có | Chưa có | Chưa có | Chưa đủ 3 seed |
| CB | Chưa có | Chưa có | Chưa có | Chưa đủ 3 seed |
| CT | Chưa có | Chưa có | Chưa có | Chưa đủ 3 seed |
| BT | Chưa có | Chưa có | Chưa có | Chưa đủ 3 seed |
| CBT † | 0.053815 | Chưa có | Chưa có | Chưa đủ 3 seed |
| Nguồn | Add = nguồn − ID |  | Drop = CBT − bỏ nguồn |  |
| Category | Chưa đủ lưới early |  | Chưa đủ lưới early |  |
| Brand | Chưa đủ lưới early |  | Chưa đủ lưới early |  |
| Title | Chưa đủ lưới early |  | Chưa đủ lưới early |  |
| Beauty; SD mẫu qua seed 42/43/44. ID: CP2 mới. † CBT42: tham chiếu CP1, chưa có source dropout; không tự ghép vào lưới early CP2. |  |  |  |  |

**2 Đối chứng khi huấn luyện trên tích hợp sớm**

| Nguồn | Thật CBT | Hoán vị | Ngẫu nhiên | Δ và CI ghép cặp |
|---|---|---|---|---|
| Category | Chưa đủ 3 seed | Chưa có | Chưa có | Chưa có |
| Brand | Chưa đủ 3 seed | Chưa có | Chưa có | Chưa có |
| Title | Chưa đủ 3 seed | Chưa có | Chưa có | Chưa có |
| Đã có tệp can thiệp; chưa có kết quả của 18 lượt train đối chứng. CI cần ghép đúng user và seed với CBT thật cùng giao thức. |  |  |  |  |

**3 Nguyên nhân hợp nhất muộn**

| Mã | Kết quả kiểm tra | Đối chiếu |
|---|---|---|
| K1 | ΔCE(late − ID): base -0.368857; meta -0.184443. | CE giảm cả ngoài mẫu; NDCG meta 0.064812 → 0.061543. |
| K2 | Val/Test 0.051890 / 0.041375; fit 11.120 user Train-meta. | Test -12.15% so ID; stacking chưa phục hồi điểm. |
| K3 | Val/Test 0.054235 / 0.042370; e1 − initial -0.005476. | Ref e1 − initial -0.004518; mức rơi chưa nhỏ hơn. |
| K4 | Val/Test 0.058125 / 0.046375; best epoch 5. | Gói joint + source dropout gần ID nhất; Test -1.54% so ID. |
| Seed42 CBT; ID Val/Test 0.059711/0.047099; late ref 0.056766/0.044506. K1 chưa xác nhận cải thiện chỉ trong mẫu; K4 thu hẹp khoảng cách nhưng chưa cô lập riêng hiệu ứng đóng băng ID. |  |  |

**4 Lựa chọn theo nhóm**

| Cơ chế | Toàn cục chọn Val | Nhóm Val → Test | Oracle nhóm Test |
|---|---|---|---|
| Late † | ID; 0.047851 ± 0.000756 | 0.047830 ± 0.000836 | 0.047898 ± 0.000769 |
| Early | Chưa đủ 8 tổ hợp | Chưa có | Chưa có |
| † Chẩn đoán CP1 theo quy tắc Minh, đã tái tính: global chọn ID cả 3 seed; nhóm −0,00002094 so ID; oracle +0,00004713 (+0,0985%). Ngưỡng fit từ lịch sử Train-meta; oracle chỉ là trần 10 nhóm × 8 tập nguồn. |  |  |  |

**5 Nhãn utility của late CBT lịch sử**

| Nguồn | Mean ΔCE | Mean Δrank | ΔNDCG Val | Khớp CE/rank |
|---|---|---|---|---|
| Category | +0.029994 | +0.022768 | -0.000236 | Không / Không |
| Brand | -0.019650 | -0.022129 | -0.000983 | Có / Có |
| Title | +0.027405 | -0.049555 | -0.001543 | Không / Có |
| Ref CP1 42/43; ΔCE tái tính từ utility, Δrank và ΔNDCG theo báo cáo Minh 05/11. CE và log-rank đều chưa khớp đủ 3 nguồn: chưa chọn nhãn mới; Category còn lệch dấu. Giữ nguyên CONDITIONAL lịch sử. |  |  |  |  |

**6 Tích hợp sớm có source dropout**

| Chế độ | NDCG@10 mean ± SD |
|---|---|
| Đủ C B T | Chưa có |
| Tắt Category | Chưa có |
| Tắt Brand | Chưa có |
| Tắt Title | Chưa có |
| Tắt hết trong S1 | Chưa có |
| ID-only riêng CP2 | 0.047851 ± 0.000756 |
| Chưa có 3 seed S1 early và ablation. Tắt hết nguồn là chế độ của S1 đã học; cần đối chiếu với ID riêng cùng giao thức. |  |

**7 Đề xuất kiến trúc nền cho UASF**

| Mục | Đề xuất và lý do |
|---|---|
| Kiến trúc nền | Chưa chốt. K4 thu hẹp mức giảm Test từ 5,51% xuống 1,54% so ID, nhưng chưa vượt ID. Cần đối chiếu với early có source dropout qua 3 seed. |
| Ưu tiên | Hoàn tất early B, đối chứng train và S1 source dropout. Early CBT42 lịch sử tăng 14,26% so ID42 là tín hiệu một seed; K1–K4 cũng chỉ seed42, chưa chứng minh độ bền seed. |
| S2 và S3 | Sau khi chọn nền bằng validation, sinh lại S2 và E1 đầy đủ ψ; chọn nhãn đúng quy tắc rồi mới xét S3. Không chạy S3 trên late đóng băng cũ. |
