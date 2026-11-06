# Kết quả phân tích lựa chọn nguồn và kiểm tra nhãn utility của Minh ngày 05/11/2026

## 1. Mục tiêu và phạm vi

Báo cáo ba nhiệm vụ theo chỉ đạo ngày 03/11: đánh giá lựa chọn tổ hợp nguồn theo nhóm người dùng; kiểm tra nhãn utility CE và log-rank với NDCG@10; chuẩn bị E1 với đầy đủ đặc trưng lịch sử ψ. Dữ liệu là Beauty, bộ `checkpoint1_version1`. Task 1 dùng tám cấu hình late fusion và seed 42/43/44; Task 2 dùng cùng reference late CBT đã đóng băng ở seed 42/43; Task 3 chỉ đặc tả và kiểm tra mức sẵn sàng. Không huấn luyện thêm backbone hoặc gate UASF.

## 2. Kiểm tra dữ liệu trước phân tích

Đã kiểm tra repository, source lock, protocol, lineage, ranks, checkpoint, utility cohort và metadata mapping; tái tính NDCG từ ranks và ghi readiness trước khi chạy task. Mapping item của pilot cũ khác mapping của grid hiện tại, nên phân tích dùng mapping tương thích với grid, không ghép pilot vào kết quả này.

| Hạng mục | Kết quả |
|---|---:|
| Người dùng mỗi split validation/test | 22.363 |
| Sản phẩm trong catalog Train-base | 11.998 |
| Ngữ cảnh Train-meta hợp lệ cho utility | 22.240 |
| Target OOV loại khỏi utility | 123 |
| Dòng utility bổ sung | 133.440 |
| Dòng validation ablation | 134.178 |

Bằng chứng: [readiness ban đầu](provenance/initial_report.md), [audit](provenance/audit.json), [hash đầu vào](provenance/input_lineage_sha256.json), [manifest thực thi](provenance/execution_manifest.json).

## 3. Task 1 — Lựa chọn tổ hợp nguồn theo nhóm

### 3.1. Phương pháp

NDCG@10 được tính từ per-user rank của tám cấu hình ID, C, B, T, CB, CT, BT, CBT. Thiết kế 10 nhóm bằng năm mức độ dài lịch sử × hai mức tỷ lệ item tail. Độ phổ biến lấy từ Train-base; ngưỡng nhóm fit trên lịch sử terminal Train-base và áp dụng cố định cho validation/test. Mỗi seed chọn cấu hình riêng, rồi tổng hợp theo số người dùng. Quy tắc nhóm là lựa chọn chẩn đoán công bố trước khi tính, chưa có xác nhận phê duyệt của thầy.

| Phương án | Quy tắc |
|---|---|
| ID-only | Dùng ID cho mọi người dùng |
| Best subset toàn cục | Chọn bằng validation cho toàn cohort |
| Best subset theo nhóm | Chọn bằng validation riêng cho từng nhóm |
| Oracle theo nhóm | Chọn bằng test, chỉ làm trần hậu kiểm |

### 3.2. Kết quả

| Phương án | Test NDCG@10, mean ± SD qua ba seed |
|---|---:|
| ID-only | 0,047850699 ± 0,000755962 |
| Best subset toàn cục | 0,047850699 ± 0,000755962 |
| Best subset theo nhóm | 0,047829754 ± 0,000836088 |
| Oracle theo nhóm | 0,047897828 ± 0,000769039 |

Best subset toàn cục chọn ID ở cả ba seed. Chọn theo nhóm thấp hơn ID khoảng 0,044% tương đối; oracle cao hơn ID khoảng 0,0985% tương đối, tương ứng 0,000047129 NDCG@10. Chưa có kiểm định ý nghĩa thống kê.

### 3.3. Diễn giải và giới hạn

Với đúng cách chia nhóm và tám cấu hình late fusion này, dư địa oracle quan sát được rất nhỏ; **không đáng ưu tiên phát triển gate chọn subset theo nhóm trên cơ chế này**. Kết luận này không áp dụng cho mọi gate hoặc early fusion.

Thiết kế có 10 nhóm, nhưng mỗi seed chỉ có tám nhóm chứa người dùng trên validation (nhóm 2–9) và sáu nhóm trên test (nhóm 4–9). Hai nhóm test nhỏ nhất chỉ có lần lượt 5 và 56 người dùng. Phân bố nhóm đổi giữa hai split vì lịch sử dài thêm trong khi ngưỡng giữ cố định. Oracle thực tế chỉ chọn trên sáu nhóm test có dữ liệu, không phải 10 nhóm đều có mẫu.

Bằng chứng: [quy tắc nhóm](task1/grouping_rules.json), [cấu hình đã chọn và cỡ nhóm](task1/selected_configurations.csv), [kết quả từng seed](task1/summary_per_seed.csv), [tổng hợp](task1/summary.csv).

## 4. Task 2 — Căn chỉnh nhãn utility với NDCG@10

### 4.1. Phương pháp và kiểm tra

Trên cùng reference late CBT đã đóng băng, tắt lần lượt từng nguồn tại suy luận; không thay bằng mô hình subset huấn luyện riêng. Với nguồn m, tính:

```text
ΔCE   = CE(khi bỏ nguồn m) − CE(đủ nguồn)
Δrank = log₂(1 + rank khi bỏ nguồn m) − log₂(1 + rank đủ nguồn)
Dval  = NDCG@10(đủ nguồn) − NDCG@10(khi bỏ nguồn m)
```

Cả ba quy ước dương là nguồn có ích. So dấu utility trung bình trên Train-meta với chênh NDCG validation; tổng hợp hai reference seed theo quy tắc đã công bố trước. Full-source validation ranks được tái lập chính xác; sai lệch CE replay tối đa khoảng 4,44 × 10⁻¹⁶. Đã kiểm tra khóa duy nhất, cohort, công thức và tổng hợp.

### 4.2. Kết quả căn chỉnh

| Nguồn | Mean ΔCE | Mean Δlog-rank | Dval | Nhận xét |
|---|---:|---:|---:|---|
| Category | +0,029994 | +0,022768 | −0,000236 | Cả hai nhãn sai dấu |
| Brand | −0,019650 | −0,022129 | −0,000983 | Cả hai khớp ở mức tổng hợp |
| Title | +0,027405 | −0,049555 | −0,001543 | Log-rank khớp; CE sai dấu |

Dấu mean ΔCE của Brand khác nhau giữa hai seed; dấu trung bình không chứng minh CE ổn định qua seed. Theo quy tắc yêu cầu khớp cả ba nguồn, CE không đạt và log-rank cũng không đạt do Category. **Chưa chọn nhãn utility.** Phần tính toán và đối chiếu đã hoàn tất; quyết định nhãn đang chờ xử lý điều kiện nghiên cứu. Không đổi ngưỡng, bỏ Category hoặc chọn log-rank theo số nguồn khớp.

Bằng chứng: [bảng dấu](task2/sign_alignment.csv), [từng seed](task2/alignment_per_seed.csv), [quyết định nhãn](task2/label_decision.json), [kiểm tra ablation](task2/validation_same_reference_ablation.csv).

## 5. Task 3 — Chuẩn bị E1 với đầy đủ ψ

Đã lập đặc tả bảy nhóm đặc trưng, đối chiếu với code và kiểm tra leakage cùng ý nghĩa phép tính.

| Nhóm đặc trưng | Trạng thái |
|---|---|
| Độ dài lịch sử | Có: code dùng `log(L)` với L ≥ 1 |
| Coverage nguồn | Có: tính trên item lịch sử đã biết, loại OOV |
| Entropy | Có; Title dựa trên danh tính vector dense chính xác, cần xác nhận ý nghĩa |
| Recency | Có: độ khả dụng trong tối đa ba item gần nhất, không phải thời gian đã trôi qua |
| Popularity lịch sử | Có dữ liệu Train-base, thiếu công thức được xác nhận |
| Tóm tắt trạng thái chuỗi | Thiếu định nghĩa cụ thể |
| Đồng thuận giữa các nhánh | Thiếu định nghĩa cụ thể |

Chưa fit E1 mới vì thiếu kiến trúc reference được chấp thuận, nhãn utility đạt quy tắc Task 2 và đặc tả ψ đầy đủ được xác nhận. Chưa có OOF predictions, AUC hoặc CI E1 mới. E1 lịch sử dùng CE với bốn đặc trưng không thay thế E1 đầy đủ ψ. Khi đủ điều kiện, cần giữ user-grouped OOF và fit các phép tiền xử lý học từ dữ liệu chỉ trong training fold.

Bằng chứng: [đặc tả ψ](task3/psi_spec.csv), [readiness](task3/readiness.json).

## 6. Trạng thái và bước tiếp theo

| Task | Đã hoàn thành | Việc còn lại |
|---|---|---|
| 1 | Phân tích late fusion, ba seed | Xác nhận quy tắc nhóm; phân tích early khi đủ 8 cấu hình × 3 seed |
| 2 | Tính utility và kiểm tra dấu | Quyết định cách xử lý khi cả CE và log-rank đều không đạt |
| 3 | Đặc tả ψ, kiểm tra leakage/readiness | Chốt reference, nhãn và công thức đặc trưng trước khi fit |

Cần trao đổi với thầy: có giữ cách chia nhóm cho early fusion khi một số nhóm rỗng hoặc rất nhỏ; xử lý nhãn CE/log-rank không khớp đủ ba nguồn; chọn kiến trúc reference cho S2/E1; xác nhận công thức popularity, state summary và branch agreement. Các thí nghiệm early fusion còn thiếu, nên chưa có kết luận Task 1 cho early.

## 7. Bằng chứng và khả năng tái lập

[Manifest thực thi](provenance/execution_manifest.json), [hash đầu vào](provenance/input_lineage_sha256.json), [hash đầu ra](output_checksums.json), [source đã chạy](provenance/executed_source.py), [môi trường](provenance/environment_freeze.txt), [kiểm tra cuối](provenance/verification.json), [readiness ban đầu](provenance/initial_report.md) và [báo cáo thực thi gốc](provenance/execution_report_original.md) lưu trong thư mục này. `verification.json` ghi PASS cho các kiểm tra đã chạy. Tên đường dẫn `reports/minh_20261105_221459` trong manifest và báo cáo gốc là đường dẫn **tại thời điểm thực thi**; thư mục hiện được đặt tên mô tả nội dung để nhóm dễ tìm.

Dữ liệu gốc Checkpoint 1 nằm trong `runs/checkpoint1_version1`; các phân tích bổ sung ngày 05/11 nằm trong thư mục báo cáo này. Snapshot và checksum hỗ trợ truy vết, nhưng gói xuất này chưa xác nhận đã hoàn thành toàn bộ yêu cầu tag Git và backup chung của nhóm.

