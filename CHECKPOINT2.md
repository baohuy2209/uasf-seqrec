# Checkpoint 2 theo feedback ngày 03 tháng 11 năm 2026

Phần bổ sung này tích hợp merge của Giang và cung cấp các stage tái lập để chạy trên máy RTX 4060. Việc import module, gọi `--help` hoặc gọi CLI mặc định không huấn luyện hay suy luận. Không có kết quả thực nghiệm mới đi kèm thay đổi code này. Kết quả lịch sử ở `runs/checkpoint1_version1` được giữ nguyên.

## Môi trường và dữ liệu

Giữ môi trường của bundle Checkpoint 1: Python 3.12.16, PyTorch 2.13.0+cu130, NumPy 2.5.1, SciPy 1.18.0, scikit-learn 1.9.0, pandas 3.0.5, RTX 4060. Không chạy lệnh cài đặt hoặc nâng cấp để tự khắc phục lỗi kiểm tra môi trường. Nếu máy thực tế khác hồ sơ này, xác minh với nhóm trước khi khóa cấu hình mới trong `configs/experiments/checkpoint2_0611.yaml` và dùng output mới.

Khôi phục reviews thật và toàn bộ `artifacts/metadata/{beauty,sports,toys}` theo manifest. Các file random Title `.npy` của phần merge có thể không có trong clone. Stage `prepare` sẽ tạo đủ bộ đối chứng ở **output mới**, dùng kích thước Title từ tensor thật và không sửa artifact của Giang. Cùng phép hoán vị/ngẫu nhiên được áp dụng ở lịch sử và phía ứng viên; missingness gốc giữ nguyên.

Kiểm tra thay đổi rồi commit trước khi chạy thật. Guard sẽ từ chối working tree chưa sạch, sai GPU/library, metadata sai checksum hoặc cache khác cấu hình. Không dùng `git add --renormalize .` trên các artifact lịch sử. Quy tắc LF mới áp dụng cho code; `runs/**` và `reports/**` giữ chính sách Git cũ để không làm hàng loạt output CRLF bị báo thay đổi. Metadata vẫn dùng `-text`. Thay đổi code này không sửa byte output lịch sử; ZIP và manifest SHA256 bảo toàn chính xác byte của các output mới.

Sau commit đã review, gắn tag local cho code của bộ kết quả, ví dụ `git tag checkpoint2-code-20261106`. Không đổi tag để trỏ sang code khác khi đã chạy; dùng output và tag mới cho giao thức mới. `protocol.lock.json` lưu commit, cấu hình, môi trường và checksum dữ liệu đi cùng bộ kết quả.

## Các lệnh trên máy 4060

Các ví dụ dưới dùng môi trường đang có của máy Lab. Thay `python` bằng đường dẫn interpreter của môi trường đó, không tạo hoặc nâng cấp môi trường mới.

```bash
# Chỉ in kế hoạch; không đọc dữ liệu thật hay tạo model.
python scripts/run_checkpoint2.py

# Đọc reviews/metadata và kiểm tra catalog/checksum trước khi huấn luyện.
# Không tạo model, không dùng GPU và không ghi output; không cần cờ thực thi.
python scripts/run_checkpoint2.py --stage check-data --datasets beauty sports toys

# Sau khi review và commit code: chuẩn bị artifact và khóa quy tắc utility.
python scripts/run_checkpoint2.py --stage prepare --datasets beauty --execute-inference

# Toàn bộ phần mini-checkpoint trên Beauty, có resume theo job/checksum.
python scripts/run_checkpoint2.py --stage all --datasets beauty --execute-training
```

`all` thực hiện grid early, đối chứng huấn luyện, K1–K4, early có source dropout, phân tích nhóm, căn chỉnh nhãn và sinh báo cáo. Nếu chưa có quyết định backbone thì dừng ở mức báo cáo mini, không tự chọn kiến trúc. Nếu đã khóa backbone, chạy tiếp S2/E1 theo quyết định đó; **không tự chạy S3**.

Các stage có thể chạy riêng theo đúng thứ tự:

```bash
python scripts/run_checkpoint2.py --stage early-grid --execute-training
python scripts/run_checkpoint2.py --stage controls --execute-training
python scripts/run_checkpoint2.py --stage diagnostics --execute-training
python scripts/run_checkpoint2.py --stage source-dropout --execute-training
python scripts/run_checkpoint2.py --stage groups
python scripts/run_checkpoint2.py --stage utility-alignment --execute-inference
python scripts/run_checkpoint2.py --stage report
```

Grid mới sử dụng cùng training loop và quy tắc shuffle cho các cấu hình. Vì vòng lặp đã được chuẩn hóa, không âm thầm coi early CBT seed42 lịch sử là một job hoàn tất của giao thức mới. Kết quả cũ vẫn là tham chiếu riêng; các job mới hoàn tất sẽ được dùng lại khi chạy tiếp. Initial validation luôn đứng riêng; các epoch đã cập nhật bắt đầu từ 1, và chỉ những epoch này được xét chọn.

Catalog kiểm tra từ `metadata.csv` và phần Train-base của `item_map.csv`; các hàng OOV trong mapping có model ID 0 được giữ đúng thiết kế. Kích thước embedding của đối chứng ngẫu nhiên lấy từ catalog gốc, tránh đổi kiến trúc khi một ID không được lấy mẫu. So dấu utility dùng cùng tập reference trên Train-meta và validation, giữ target OOV ở mẫu số NDCG validation (CE của OOV để trống).

Sau cập nhật code, nếu output cũ đã có `protocol.lock.json`, giữ nguyên thư mục đó và dùng một output mới như `--output runs/checkpoint2_20261106_audit`. Không xóa khóa để ghép kết quả khác phiên bản code. Lượt lỗi catalog dừng trước khi tạo protocol lock; kiểm tra trạng thái thư mục trước khi chọn output.

## Các đối chứng nguyên nhân

| Mục | Cách thực hiện trong code |
|---|---|
| K1 | ID và late CBT cùng context Train-base/Train-meta; lưu CE từng context và chênh lệch. CE dùng catalog chưa mask như giao thức cũ. |
| K2 | Đóng băng ID; chia user của Train-meta seed42 thành hai nửa, residual chỉ học nửa fit; ghi nửa held-out riêng, không dùng nửa fit làm S2 ngoài mẫu. |
| K3 | Residual lr=1e-4, weight decay=1e-4, encoder dropout=0.5, source dropout theo mẫu; không thêm gói 20% all-off của K4. |
| K4 | ID và nhánh metadata khởi tạo mới, train chung; bỏ freeze/no_grad/eval cố định ở ID; source dropout theo mẫu và khoảng 20% mẫu all-off. Đây là đối chứng gói joint+dropout, không tách riêng mọi tác động. |

K1–K4 dùng đồng hồ tổng tối đa 36 giờ tính từ lần bắt đầu diagnostics. Không tự đặt lại đồng hồ khi resume. Mô hình dừng do hết thời gian được ghi `diagnostic_time_limit`, không giả làm một lượt early stopping hoàn chỉnh. Giữ mọi failed/interrupted attempt để kiểm tra, rồi tạo attempt mới khi chạy lại. Lock còn sót sau crash phải được kiểm tra trước khi xóa thủ công; không xóa lock của tiến trình đang chạy.

Resume tái sử dụng **job đã hoàn tất** và cache suy luận có checksum khớp, gồm CE K1, ablation, utility và features của gate. Một lượt bị ngắt giữa chừng chưa có marker hoàn tất sẽ chạy lại trong attempt mới; đây không phải resume optimizer từ giữa epoch. Không đưa cache Checkpoint 1 khác giao thức vào output mới.

## Nhóm người dùng và utility

Head/tail chỉ tính từ sự kiện Train-base, có tie-break cố định bằng catalog ID. Ranh giới độ dài và tail-ratio fit một lần trên lịch sử validation và dùng nguyên cho test. Hai context giống nhau thuộc cùng nhóm; không tách tie theo thứ tự user để ép cân bằng. Nhóm trống được báo rõ; nhóm test chưa có ở validation dùng global subset đã chọn từ validation. Oracle chọn bằng test chỉ là trần mô tả, không dùng chọn mô hình triển khai.

S2 chạy lại đầy đủ forward khi bỏ nguồn trên early fusion, tắt cùng nguồn ở cả đầu vào lẫn ứng viên. Không dùng phép trừ điểm residual của late fusion cho early fusion. Rank dùng đúng `input_history_strict`, tie policy và catalog của evaluator; OOV giữ trong mẫu số đánh giá, chỉ loại khỏi CE khi target không thuộc catalog.

Quy tắc nhãn ghi trước ở `utility_label_rule.json`: CE nếu dấu mean delta khớp chênh NDCG validation ở cả ba nguồn; nếu không thì log-rank khi thỏa cùng điều kiện; còn lại UNRESOLVED. Dấu zero là chưa đủ bằng chứng. Không chọn nhãn theo test.

Psi gồm log độ dài, popularity trung bình/item cuối và tail-ratio từ Train-base, coverage/entropy/recency theo nguồn, trạng thái reference **ungated** và độ đồng thuận top10/điểm giữa full và ablation. Không đưa target, target rank/loss/utility vào psi. Đặc trưng luôn lấy từ reference seed42 cố định và dùng chung cho nhãn của hai reference, tránh thay features theo reference seed hoặc tạo vòng phụ thuộc gate.

## Chọn kiến trúc rồi mới chạy S2 và S3

Xem bảng validation, đối chứng, các chế độ all-off và phân tích nhóm trước khi chọn. Ví dụ chỉ khi bằng chứng thực tế ủng hộ early SD:

```bash
python scripts/run_checkpoint2.py --stage choose --backbone early_sd --reason "Lý do dựa trên validation và các đối chứng đã xem"
python scripts/run_checkpoint2.py --stage s2-e1 --execute-inference
```

Nếu chọn `late_joint`, reference thứ hai seed43 cần được huấn luyện mới; chạy `s2-e1 --execute-training` cho phép bước này. Code không tự biến cờ inference thành quyền training.

Sau khi xem E0/E1, đúng fallback theo từng nguồn và trần lựa chọn nhóm, mới yêu cầu S3:

```bash
python scripts/run_checkpoint2.py --stage s3 --execute-training
```

S3 không chạy nếu GO/CONDITIONAL chưa đủ điều kiện. Soft-supervision không tự định tuyến thành group gate. S1 đóng băng nhưng forward của early fusion vẫn giữ gradient theo alpha, vì gate phải tác động lên cả context encoder và ứng viên. Các nguồn chưa đủ E0/E1 dùng trọng số global thay vì tuyên bố đã có bằng chứng contextual utility. Nhiệt độ soft-label và các lambda được khóa trong YAML trước khi tính, không chỉnh từ test.

Baseline lồng nhau dùng cùng S1: all-fixed, subset toàn cục, subset theo nhóm, trọng số global, gate không giám sát và UASF; thêm ID-only riêng và early không source dropout. Tắt hết nguồn là chế độ của S1 đã học, không khẳng định bằng checkpoint ID độc lập. Bảy so sánh dùng paired-user bootstrap, Holm và TOST với biên tương đối 1% đã khóa. Phần S3 hiện báo điều kiện theo seed42, không nhận là bằng chứng độ bền qua ba seed huấn luyện S3.

Trong S2/S3 và các baseline lồng nhau, all-fixed dùng alpha=1 cho mọi nguồn được cấu hình, đúng feedback. Subset đặt alpha của nguồn bị tắt bằng 0. Missingness tự nhiên vẫn được mask theo từng item ở cả lịch sử và ứng viên; coverage lịch sử bằng 0 không tự tắt metadata của toàn bộ ứng viên. Psi lấy từ reference cố định alpha=1, không lấy từ encoder đã nhận đầu ra của gate. Các baseline này dùng cùng cache `ablations/` của S1.

## Sports và Toys ở tiến trình thứ hai

```bash
# Terminal thứ hai: riêng B late fusion theo Q7, không chen vào job Beauty.
python scripts/run_checkpoint2.py --stage transfer-late --datasets sports toys --execute-training

# Early B của hai dataset khi đến giai đoạn tiếp theo.
python scripts/run_checkpoint2.py --stage early-grid --datasets sports toys --execute-training
```

Các tiến trình ghi job theo dataset/key riêng. Không khởi chạy cùng một job ở hai tiến trình. Hai tiến trình GPU là gợi ý từ feedback, cần kiểm tra bộ nhớ thực tế trên 4060; không tự giảm batch size hoặc đổi giao thức khi gặp OOM. `transfer` chạy cả late và early nếu muốn thực hiện tuần tự một lệnh.

## Báo cáo và bản lưu chung

```bash
python scripts/run_checkpoint2.py --stage report --datasets beauty
python scripts/run_checkpoint2.py --stage archive --archive-dir /duong/dan/thu_muc_chung
```

Báo cáo nằm ở `runs/checkpoint2_20261106/beauty/mini_checkpoint/`: JSON số gốc, Markdown và Word hai trang chỉ gồm bảy nhóm bảng. Ô thiếu ghi PENDING/Chưa có, không thay bằng 0 hoặc output late inference-time. Mean ± SD chỉ có khi đủ ba seed. CI đối chứng ghép cặp theo user và giữ ba dự đoán seed cùng user; không gọi đó là CI biến thiên seed. Xem lại số trang trong Word sau khi thêm ghi chú tùy ý.

Archive ZIP gồm checkpoints, scalar cache, ranks, contrasts, cấu hình, source snapshot, log và manifest SHA256. Nó không gom reviews thô. Lệnh không upload hoặc gửi tin nhắn tự động. Chọn một đường dẫn bản lưu chung của nhóm hoặc thư mục đồng bộ Drive; không ghi đè archive cũ. Nếu máy tắt giữa chừng, giữ output và đọc state/lock trước khi resume.
