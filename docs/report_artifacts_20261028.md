# Dữ liệu lưu để lập báo cáo TAIMMA

Căn cứ: memo 20/10 mục 4.1–4.3 và mục 6; memo 26/10 mục 4–6. Phạm vi
hiện tại là hoàn thiện cơ chế lưu; không chạy lại pilot thật hoặc tạo số nghiên cứu.

## Mỗi lượt pilot mới lưu gì

Trong `runs/<run_id>/`:

| Tệp | Dùng cho báo cáo |
| --- | --- |
| `config.json` | Cấu hình thực sự có hiệu lực, seed, split/protocol, nguồn dữ liệu |
| `run_manifest.json` | Commit/dirty status, CPU/GPU, phiên bản thư viện, metric |
| `source_snapshot.zip` và `.json` | Bản mã/config chính xác cùng SHA256, kể cả khi chưa commit |
| `decision_log_snapshot.md` | Bản nhật ký quyết định tại thời điểm chạy, không tự đánh dấu quyết định là đã duyệt |
| `dataset_statistics.json` | Số user/item/tương tác, sparsity, độ dài chuỗi, OOV, coverage và vocab |
| `metrics.json` | Toàn bộ Recall/NDCG@10/20 validation và test của checkpoint được chọn |
| `baseline_metrics.csv` | Random, Popularity, SASRec trên cả validation/test; diagnostic shuffled test |
| `baselines/<method>/per_user_<split>.*` | Rank và metric từng user của các baseline, có thể kiểm định lại |
| `per_user_val.*`, `per_user_test.*` | Rank, target, history length, OOV, tần suất Train-base, nhóm phân tích |
| `training_history.json`, `epochs.csv` | Loss/metric và thời gian từng epoch, best epoch, lý do dừng, tài nguyên |
| `report_data.json` | Gom các trường cho 6 mục báo cáo pilot trong memo 20/10 |
| `group_metrics.csv` | Thống kê mô tả theo quintile độ dài và target head/tail, chưa phải kiểm định ý nghĩa |
| `compute_budget.csv` | Số lượt dự kiến theo memo; GPU-giờ chỉ ước tính khi có phép đo GPU tương ứng |
| `run.log`, `run_state.json` | Console/traceback và trạng thái thực thi; có cả khi chạy bị lỗi |
| `artifact_index.json` | Danh sách tệp, kích thước và checksum sau khi kết thúc |

Nếu dùng `--verify_reproducibility`, thư mục `repeat/` lưu checkpoint, metric,
per-user, lịch sử và số liệu báo cáo riêng của lượt chạy lại. Log chung ghi cả hai lượt.

Các nhóm có tiền tố `analysis_` phục vụ phân tích offline. Nhóm theo target popularity
không được đưa vào gate lúc suy luận; gate không biết item mục tiêu tương lai.

## Registry cho báo cáo tuần

`results/results.csv` được bổ sung các trường: commit/dirty, data mode, mechanism,
sources, control, missing rate, đủ metric validation/test, epoch/best epoch, thời gian,
CPU/GPU và RAM/GPU peak, cùng đường dẫn đến config/manifest/report/per-user.

Khi lần chạy mới gặp header cũ, code giữ nguyên mọi giá trị lịch sử, để trống trường
chưa có bằng chứng, tạo bản sao byte-for-byte `results.csv.schema-before.*.bak`, rồi
nâng header và ghi nguyên tử có lock. Không tự tạo metric hay đổi status cũ. Hai tiến
trình ghi đồng thời được phát hiện qua lock thay vì trộn dòng. Run ID trùng bị từ chối.
Việc nâng schema chỉ diễn ra lúc đăng ký run mới; lượt sửa code không sửa registry thật.

## Đơn vị và giới hạn phép đo

- Thời gian từng epoch có cả train-only và wall time gồm validation/checkpoint.
- Training wall time và thời gian đến bước ghi báo cáo được phân biệt. Tổng thời gian
  thực thi đầy đủ, gồm ghi artifact, ở `run_state.json`.
- RAM peak là peak của process trong vòng đời process, không phải peak đã reset riêng cho epoch.
- GPU peak là bộ nhớ allocator PyTorch đã cấp phát/reserve, reset trước mỗi lượt fit;
  không đại diện cho tất cả process hay allocation ngoài PyTorch trên card.
- Chạy CPU thì trường GPU là null/not-applicable; không đổi thời gian CPU thành GPU-giờ.
- Thời gian pilot ID-only không được coi là thời gian đo của fusion/UASF/ablation.
  Các khối chưa có phép đo phù hợp để null/pending trong bảng ngân sách.
- Không coi chạm max_epochs là đã hội tụ. `stop_reason` ghi `max_epochs` hoặc `early_stopping`.

## Bảng 1 cho ba dataset, không huấn luyện

Từ repository root, sau khi có các file reviews thật:

```powershell
.\.venv\Scripts\python.exe notebooks/taimma/scripts/collect_dataset_statistics.py --output_dir data/taimma/reports/table1_NEW_RUN
```

Thư mục output phải mới. Script chỉ đọc reviews/metadata, tạo JSON từng dataset,
`table1.csv` và manifest checksum nguồn; không train model, không thay split cũ.
Coverage/vocab khi thiếu file metadata là null, không phải số 0 đã đo.

## CSV và Parquet

Tham số mới `--rank_format csv|parquet|both` điều khiển định dạng per-user.
CSV lưu đầy đủ thông tin để tính nhóm/bootstrap/paired statistics. Muốn dùng đúng
container Parquet như ví dụ trong memo, môi trường chạy cần `pyarrow` hoặc `fastparquet`.
Nếu yêu cầu Parquet nhưng thiếu engine, script dừng trước training, không đổi đuôi CSV
thành Parquet và không tự chuyển định dạng.

Trong `.venv` Windows đã kiểm tra, chưa có engine Parquet. Lượt hoàn thiện này không
tự cài dependency. Trên WSL/Colab hoặc môi trường đã có engine, có thể dùng `--rank_format both`.

Đối với S2, helper `save_utility_artifact` trong `evaluation/utility.py` lưu dataframe delta
thật sang CSV hoặc Parquet, kèm sidecar chứa hash dữ liệu/split manifest và từng checkpoint
tham chiếu theo seed. Đường dẫn hoặc lineage thiếu sẽ báo lỗi; không có nhãn thay thế.

## Dữ liệu chưa có vẫn phải được báo thiếu

Cơ chế lưu không thay thế metadata thật, Title embeddings/encoder version, benchmark
tham chiếu, lưới Diagnostic B, delta hai seed, S3, ablation hay kết quả 3 dataset × 5 seed.
Những kết quả chưa chạy không được thêm vào báo cáo dưới dạng số ước đoán như đã đo.
Các fixture kiểm thử có data mode rõ ràng và không dùng làm kết quả Amazon.
