# Mini checkpoint 06/11/2026

**1 Tích hợp sớm B**

| Nguồn | Seed 42 | Seed 43 | Seed 44 | Mean ± SD |
|---|---|---|---|---|
| ∅ ID riêng | 0.047099 | 0.047842 | 0.048611 | 0.047851 ± 0.000756 |
| C | Chưa có | Chưa có | Chưa có | Chưa có |
| B | Chưa có | Chưa có | Chưa có | Chưa có |
| T | Chưa có | Chưa có | Chưa có | Chưa có |
| CB | Chưa có | Chưa có | Chưa có | Chưa có |
| CT | Chưa có | Chưa có | Chưa có | Chưa có |
| BT | Chưa có | Chưa có | Chưa có | Chưa có |
| CBT | Chưa có | Chưa có | Chưa có | Chưa có |
| C/B/T = Category/Brand/Title. SD mẫu qua 3 seed; không suy SD từ một seed. | | | | |

| Nguồn early fusion | Add = nguồn − ID | Drop = CBT − bỏ nguồn |
|---|---|---|
| Category | Chưa có | Chưa có |
| Brand | Chưa có | Chưa có |
| Title | Chưa có | Chưa có |

**2 Đối chứng khi huấn luyện trên early fusion**

| Nguồn | Thật CBT | Hoán vị Δ [CI95] | Ngẫu nhiên Δ [CI95] |
|---|---|---|---|
| Category | Chưa có | Chưa có | Chưa có |
| Brand | Chưa có | Chưa có | Chưa có |
| Title | Chưa có | Chưa có | Chưa có |
| Δ = đối chứng − thật; bootstrap ghép cặp theo người dùng, giữ chung dự đoán 3 seed; không phải CI biến thiên seed. | | | |

**3 Nguyên nhân hợp nhất muộn**

| Mã | Kết quả K1 đến K4 |
|---|---|
| K1 | CE late−ID: base -0.3689; meta -0.1844 |
| K2 | Val 0.051890; Test 0.041375; early_stopping |
| K3 | Val 0.054235; Test 0.042370; early_stopping |
| K4 | Val 0.058125; Test 0.046375; early_stopping |
| K1 chưa xác nhận mẫu hình học phần dư trong mẫu; K2–K4 mô tả các đối chứng dữ liệu, điều chuẩn và gói huấn luyện chung. | |

**4 Lựa chọn theo nhóm**

| Cơ chế | ID riêng | Subset toàn cục | Nhóm Val → Test | Oracle nhóm Test |
|---|---|---|---|---|
| Late fusion | Chưa có | Chưa có | Chưa có | Chưa có |
| Early fusion | Chưa có | Chưa có | Chưa có | Chưa có |
| Ranh giới nhóm fit trên lịch sử validation và giữ nguyên khi áp dụng test; oracle dùng nhãn test chỉ là trần mô tả. | | | | |

**5 Nhãn utility**

| Nguồn | Mean ΔCE | Mean Δrank | ΔNDCG Val cùng model |
|---|---|---|---|
| Category | Chưa có | Chưa có | Chưa có |
| Brand | Chưa có | Chưa có | Chưa có |
| Title | Chưa có | Chưa có | Chưa có |
| Nhãn chọn: PENDING; CE nếu khớp dấu cả ba nguồn, nếu không xét log-rank; không chọn bằng test. | | | |

**6 Early fusion có source dropout**

| Chế độ | NDCG@10 mean ± SD |
|---|---|
| Đủ C B T | Chưa có |
| Tắt Category | Chưa có |
| Tắt Brand | Chưa có |
| Tắt Title | Chưa có |
| Tắt hết trong S1 | Chưa có |
| ID-only riêng | 0.047851 ± 0.000756 |
| Tắt hết là chế độ của S1 đã học, không phải checkpoint ID-only riêng. Cùng mask nguồn ở lịch sử và ứng viên. | |

**7 Đề xuất kiến trúc nền cho UASF**

| Mục | Đề xuất và lý do |
|---|---|
| Kiến trúc nền | Chưa chốt kiến trúc nền; không tự chạy S3 trên late fusion cũ. |
| Trạng thái | Chưa đủ kết quả; các ô thiếu cần hoàn tất trước khi chốt |
| S2 và S3 | Chọn nền bằng validation; sinh lại S2/E1 đầy đủ ψ; chỉ chạy S3 khi E0/E1 đạt đúng chế độ. Không tự chạy S3 trên late fusion cũ. |
