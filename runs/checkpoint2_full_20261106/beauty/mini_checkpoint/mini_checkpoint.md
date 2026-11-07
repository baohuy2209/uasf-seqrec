# Mini checkpoint 06/11/2026

**1 Tích hợp sớm B**

| Nguồn | Seed 42 | Seed 43 | Seed 44 | Mean ± SD |
|---|---|---|---|---|
| ∅ ID riêng | 0.047099 | 0.047842 | 0.048611 | 0.047851 ± 0.000756 |
| C | 0.045759 | 0.048239 | 0.045332 | 0.046443 ± 0.001569 |
| B | 0.053966 | 0.051804 | 0.050824 | 0.052198 ± 0.001608 |
| T | 0.050941 | 0.050390 | 0.050715 | 0.050682 ± 0.000277 |
| CB | 0.054446 | 0.055301 | 0.055994 | 0.055247 ± 0.000775 |
| CT | 0.048742 | 0.048898 | 0.048164 | 0.048601 ± 0.000386 |
| BT | 0.054110 | 0.055846 | 0.054322 | 0.054759 ± 0.000947 |
| CBT | 0.055009 | 0.055374 | 0.055112 | 0.055165 ± 0.000188 |
| C/B/T = Category/Brand/Title. SD mẫu qua 3 seed; không suy SD từ một seed. | | | | |

| Nguồn early fusion | Add = nguồn − ID | Drop = CBT − bỏ nguồn |
|---|---|---|
| Category | -0.001407 ± 0.001839 | 0.000405 ± 0.000762 |
| Brand | 0.004347 ± 0.002351 | 0.006564 ± 0.000348 |
| Title | 0.002831 ± 0.000903 | -0.000082 ± 0.000735 |

**2 Đối chứng khi huấn luyện trên early fusion**

| Nguồn | Thật CBT | Hoán vị Δ [CI95] | Ngẫu nhiên Δ [CI95] |
|---|---|---|---|
| Category | 0.055165 ± 0.000188 | 0.053644 ± 0.000323; Δ -0.00152 [-0.00235, -0.00064] | 0.053342 ± 0.000811; Δ -0.00182 [-0.00277, -0.00092] |
| Brand | 0.055165 ± 0.000188 | 0.047428 ± 0.000731; Δ -0.00774 [-0.00911, -0.00638] | 0.048006 ± 0.000759; Δ -0.00716 [-0.00858, -0.00583] |
| Title | 0.055165 ± 0.000188 | 0.055176 ± 0.000606; Δ +0.00001 [-0.00069, +0.00076] | 0.055049 ± 0.000577; Δ -0.00012 [-0.00085, +0.00063] |
| Δ = đối chứng − thật; bootstrap ghép cặp theo người dùng, giữ chung dự đoán 3 seed; không phải CI biến thiên seed. | | | |

**3 Nguyên nhân hợp nhất muộn**

| Mã | Kết quả K1 đến K4 |
|---|---|
| K1 | Chưa có kết quả đối chứng |
| K2 | Chưa có kết quả đối chứng |
| K3 | Chưa có kết quả đối chứng |
| K4 | Chưa có kết quả đối chứng |
| Chưa đủ K1–K4 để chốt nguyên nhân. | |

**4 Lựa chọn theo nhóm**

| Cơ chế | ID riêng | Subset toàn cục | Nhóm Val → Test | Oracle nhóm Test |
|---|---|---|---|---|
| Late fusion | 0.047851 ± 0.000756 | ID; 0.047851 ± 0.000756 | 0.047409 ± 0.001042 | 0.047898 ± 0.000793 |
| Early fusion | 0.047851 ± 0.000756 | CB, CBT; 0.054977 ± 0.000479 | 0.054992 ± 0.001809 | 0.056645 ± 0.000217 |
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
| Đủ C B T | 0.055772 ± 0.000171 |
| Tắt Category | 0.052260 ± 0.000160 |
| Tắt Brand | 0.044480 ± 0.000626 |
| Tắt Title | 0.054210 ± 0.000310 |
| Tắt hết trong S1 | 0.039632 ± 0.000547 |
| ID-only riêng | 0.047851 ± 0.000756 |
| Tắt hết là chế độ của S1 đã học, không phải checkpoint ID-only riêng. Cùng mask nguồn ở lịch sử và ứng viên. | |

**7 Đề xuất kiến trúc nền cho UASF**

| Mục | Đề xuất và lý do |
|---|---|
| Kiến trúc nền | Chưa chốt kiến trúc nền; không tự chạy S3 trên late fusion cũ. |
| Trạng thái | Chưa đủ kết quả; các ô thiếu cần hoàn tất trước khi chốt |
| S2 và S3 | Chọn nền bằng validation; sinh lại S2/E1 đầy đủ ψ; chỉ chạy S3 khi E0/E1 đạt đúng chế độ. Không tự chạy S3 trên late fusion cũ. |
