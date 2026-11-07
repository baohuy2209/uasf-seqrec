# Hoàn tất checkpoint theo góp ý 03 và 06 tháng 11

Entry point mới: `scripts/complete_checkpoint2.py`. Chỉ in kế hoạch nếu không
chọn stage; không tự train hay inference khi import. Giữ nguyên các thư mục
output cũ. Dùng môi trường RTX 4060 đã khóa, không cài/nâng thư viện.

## Checkpoint ở server

`.pt` không cần đưa lên Git hoặc tải về Windows. Trên server phải còn đúng
checkpoint gốc trong `jobs/<job>/attempt_*/checkpoint.pt`, cùng `completed.json`,
`config.json`, `source_snapshot.json`, `history.json`, ranks và manifest.
Ba input root mặc định được ghi trong `configs/experiments/completion_0711.yaml`.
Có thể đổi bằng `--early-root`, `--late-root`, `--diagnostics-root`.

Code xác minh hash, nguồn model/trainer và chữ ký cohort trước khi copy sang
output mới; không sửa, di chuyển hay ghi đè dữ liệu cũ. Chỉ chấp nhận khác biệt
CRLF/LF ở text. Checkpoint khác hash hoặc model/trainer khác phải dừng, không
tự coi là cache hợp lệ. Thiếu `.pt` sẽ báo đường dẫn cần bổ sung; ranks không thể
khôi phục CE. Lượt bị ngắt chưa hoàn tất chạy lại trong attempt mới, không phải
resume optimizer giữa epoch. Chỉ xóa execution lock sau khi xác nhận không có
process đang giữ nó.

## Checkout riêng trong lúc các lượt cũ còn chạy

Commit/push bản sửa từ Windows trước. Đừng pull hoặc checkout vào thư mục đang
train. Tạo worktree riêng trên server (có thể thay origin/main bằng hash commit
đã kiểm tra để cố định phiên bản):

```bash
cd /home/ubuntu/projects/uasf-seqrec
git fetch origin
git worktree add -b completion-20261107 /home/ubuntu/projects/uasf-seqrec-completion-20261107 origin/main
cd /home/ubuntu/projects/uasf-seqrec-completion-20261107
mkdir -p data
ln -s /home/ubuntu/projects/uasf-seqrec/data/raw data/raw
source /home/ubuntu/projects/uasf-seqrec/.venv/bin/activate
python scripts/complete_checkpoint2.py --stage check --datasets beauty
```

Một checkout riêng không cung cấp thêm GPU. Nếu cùng một RTX 4060, chờ job cũ
xong trước khi chạy thêm train/inference.

## Chỉ điền utility từ early SD đã có

```bash
python -u scripts/complete_checkpoint2.py --stage utility --datasets beauty \
  --execute-inference
```

Lệnh này không train lại ID/early/late. Nó tái sử dụng checkpoint early SD 42/43,
tính ΔCE, Δlog-rank trên Train-meta có target hợp lệ, chênh NDCG validation trên
cùng hệ tham chiếu, đầy đủ ψ, ổn định nhãn và E0/E1. Kết quả CE và LOG_RANK đều
được xuất như kiểm tra chẩn đoán; quy tắc đã khóa mới được chọn nhãn chính.
Nếu UNRESOLVED thì giữ nguyên trạng thái đó. Early SD chưa đủ điều kiện backbone
vẫn được chẩn đoán, nhưng không được dùng kết quả này để tự cho phép S3.

## Một lệnh hoàn tất bộ thí nghiệm còn thiếu và báo cáo

```bash
python -u scripts/complete_checkpoint2.py --stage all --execute-training \
  --publish-dir results/completion_20261107
```

Trình tự:

1. Nhập các job hoàn tất hợp lệ từ ba input root; không train lại job đã nhập.
2. Hoàn tất hai grid × tám tổ hợp × ba seed trên Beauty/Sports/Toys.
3. Đối chứng Beauty: giữ phép hoán vị gốc để tái sử dụng 9 job shuffled nếu còn
   đủ bằng chứng; tạo random V2 đúng yêu cầu và chạy 9 job random mới.
4. Kiểm tra early SD theo validation trung bình ba seed. Nếu không đạt, bổ sung
   K4 seed 43/44 để đối chiếu đúng ba seed; không thay đổi hyperparameter để cứu.
5. Tính utility/E0/E1 chẩn đoán, chọn backbone bằng quy tắc 06/11 trên Beauty;
   áp dụng cùng lựa chọn đó cho các tập còn lại.
6. Chỉ khi backbone và E0/E1 đủ điều kiện: UASF, bảy đối chứng và năm ablation,
   mặc định năm seed cả ba tập; chỉ đánh giá UASF validation. Chỉ từ tối 12/11,
   nếu không kịp, mới đặt Sports/Toys còn 3 seed và bật
   `allow_reduced_transfer_seeds: true` theo ngoại lệ thầy cho; thay cấu hình
   khoa học phải dùng output mới, không sửa lock cũ.
   Các đối chứng học thêm Train-meta được train riêng, không thay hệ tham chiếu.
7. Phân tích nhóm dùng cùng ranh giới lịch sử validation, độ bền thiếu metadata
   0/20/40/60/80% cho early/late, cờ best_epoch ≤5 và kiểm chứng số từ ranks.
8. Xuất registry/results.csv, các CSV utility, report_tables.csv, bảy bảng Word/
   Markdown và manifest SHA256; copy các tệp nhẹ sang publish-dir khi tất cả
   tính toán đã xong. Kết quả không đủ điều kiện được ghi F2/skipped, không giả số.

`utility` và `all` dùng cùng output/config và tái sử dụng cache phù hợp. Phải giữ
nguyên ba input root, cấu hình khoa học, source và môi trường giữa các lần.
Nếu đã publish kết quả vào Git working tree, commit chúng trước lượt tính toán
tiếp theo để qua clean-tree guard. Commit chỉ thay kết quả không làm đổi cache
theo source. Output runtime mới `/runs/completion_*/` được ignore để tránh guard
bị chính logs chặn; các tệp nhẹ đã publish trong `results/` vẫn được Git nhận.

## Khóa số liệu và mở test UASF riêng

Trước 13/11, `all` chỉ train/chọn gate bằng validation. Test của chẩn đoán B đã
có vẫn được giữ riêng. Sau mốc khóa, kiểm tra kết quả và commit/tag bộ dùng cho
bài rồi chạy:

```bash
git add results/completion_20261107
git commit -m "results: freeze reviewed checkpoint2 tables"
git tag checkpoint2_frozen_20261113
python scripts/complete_checkpoint2.py --stage freeze \
  --freeze-tag checkpoint2_frozen_20261113
python -u scripts/complete_checkpoint2.py --stage release-test --execute-inference \
  --publish-dir results/completion_20261107
```

Freeze ghi hash source/checkpoint/quyết định validation. Release kiểm tra lại
manifest trước khi đánh giá gate; test đã hoàn tất hợp lệ được đọc lại từ cache,
không dùng test để chọn lại. So sánh xác nhận lấy trung bình từng người dùng qua
seed, paired-user bootstrap, Holm bảy so sánh và TOST ±1% tương đối. Nếu không có
backbone/E1 đạt, không có S3 để mở; đây là kết quả thực nghiệm hợp lệ, không phải
phần có thể điền bằng suy đoán.

Checkout mới dùng nhánh `completion-20261107`; có thể commit kết quả nhẹ và
`git push -u origin completion-20261107` để người khác lấy về phân tích, không
cần đổi nhánh hoặc pull thư mục đang chạy các thí nghiệm cũ.

## Các output cần xem

- `runs/completion_20261107/beauty/diagnostic_early_sd/alignment.json` và
  `utility_both_labels.csv`: số cho bảng 5.
- `diagnostic_early_sd/e0_e1_CE.json`, `e0_e1_LOG_RANK.json`, `status.json`:
  chẩn đoán hai nhãn và điều kiện áp dụng.
- `backbone_decision.json`, `framing_decision.json`: bằng chứng validation.
- `results.csv`, `utility_summary.csv`, `dropout_profile.csv`, `robustness.csv`,
  `group_analysis.json`, `report_tables.csv`, `mini_checkpoint.docx/.md`.
- `gates/` và `gate_validation.csv` chỉ tồn tại khi S3 đủ điều kiện; `gate_test.csv`
  và `baseline_comparisons.json` chỉ xuất sau freeze/release.
- `.pt`, snapshot và cache lớn giữ ở server/kho chung cùng SHA256, không push Git.

`--publish-dir` cũng chép ranks Val/Test và logs/JSON theo từng job để phân tích
trên Windows không cần `.pt`; không chép feature cache rộng, mảng numpy hay weights.
Để lưu bundle đầy đủ ra kho chung sau khi tính toán đã kết thúc:

```bash
python scripts/run_checkpoint2.py --stage archive \
  --output runs/completion_20261107 \
  --archive-dir /home/ubuntu/projects/taimma-shared
```

Không có lựa chọn ép GO/F1 hoặc bỏ qua thiếu checkpoint. Các test local dùng
fixtures CPU và checkpoint giả; không thay thế việc kiểm tra thực thi trên 4060.
