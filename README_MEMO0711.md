# TAIMMA — hoàn tất thực nghiệm theo memo 07/11, bản kiểm định v3 ngày 10/11

Entry point: `scripts/run_memo0711.py`. Config: `configs/experiments/memo0711.yaml`.
Mặc định chỉ in plan; không train/inference trên import. Không chạy lại runner
`complete_checkpoint2.py --stage all` cũ cho protocol này.

## Sửa một lượt memo v1 đã hoàn tất trên server

Giữ nguyên `runs/memo0711_v1` và toàn bộ checkpoint gốc. Commit/review source mới
rồi cập nhật server khi không còn process train dùng checkout này. Không nâng
thư viện, thay dữ liệu hoặc sửa lock v1. Dùng output mới `runs/memo0711_v3`.

```bash
cd /home/ubuntu/projects/uasf-seqrec
source .venv/bin/activate

COMMON=(
  --output runs/memo0711_v3
  --reuse-memo runs/memo0711_v1
)

python scripts/run_memo0711.py --stage plan
python scripts/run_memo0711.py --stage check "${COMMON[@]}"

# Nhận lại các checkpoint hợp lệ: hash/data/environment/model source phải khớp.
# Flag inference dùng cho guard môi trường; adopt không chạy dự đoán mô hình.
python -u scripts/run_memo0711.py --stage adopt \
  --reuse-memo runs/memo0711_v1 --execute-inference "${COMMON[@]}"

# Hoàn thành các phép tính bị ảnh hưởng, cache những phần được nhận lại hợp lệ.
python -u scripts/run_memo0711.py --stage all \
  --reuse-memo runs/memo0711_v1 --execute-training "${COMMON[@]}"

# all tự xuất OOF E1. Có thể chạy riêng CPU nếu chỉ còn thiếu bằng chứng này:
python -u scripts/run_memo0711.py --stage e1-evidence --datasets beauty "${COMMON[@]}"

# Kiểm định bằng chứng CPU, không ghi vào runtime, không mở mô hình.
python scripts/run_memo0711.py --stage audit "${COMMON[@]}"

# Tạo bảng số liệu và gói publish sau khi compute hoàn tất.
python scripts/run_memo0711.py --stage report \
  --publish-dir results/memo0711_v3 "${COMMON[@]}"
```

Các input roots được lấy tự động từ protocol của memo v1 để giữ đúng config và khả năng dùng lại gate. Chỉ truyền `--late-root` rõ ràng nếu đường dẫn gốc đã chuyển. Flag CLI chỉ override
YAML khi truyền rõ; đường YAML không còn bị default argparse ghi đè. Giữ cùng
roots/output trên mọi lệnh. Windows thiếu .pt không thay thế được bước nhận lại
server: checkpoint phải tồn tại và khớp hash; thiếu sẽ dừng rõ thay vì train thay.

Khi dùng `--reuse-memo`, job S1 bắt buộc phải được nhận lại từ kết quả có thật.
Không dùng `all` không có reuse để sửa v1: đó là cold workflow có thể fit S1 mới.
Không sửa fingerprint hoặc lock để vượt qua lỗi cache.

### Phần phải tính lại

- Robustness v1: mask từng được gán lên wrapper thay vì scoring adapter. Bản sửa
  chuyển mask thật vào history/candidate scoring, version hóa binding và chỉ
  chạy lại inference. Không train lại ID, early, late, SD hoặc fairness vì lỗi này.
- Add-one v1: `no_grad()` không tắt dropout. Bản sửa đặt S1/fallback eval rồi tạo
  lại raw deltas/noise/labels và gate add-one (Beauty năm seed nếu đủ S2).
- Gate khác chỉ nhận lại khi model/config/label/feature/source dependency được
  chứng minh giống nhau. Manifest nhận lại ghi rõ gate nào được dùng lại; nếu
  không chứng minh được thì chỉ fit lại gate đó, không huấn luyện lại S1.
- Group/Title/rank summary và registry được tái tạo từ dữ liệu đã xác minh.

Các S1 Beauty45/46, Sports/Toys SD, fairness Train-base+Train-meta được nhận lại
đúng cohort/fingerprint. ID-only fallback vẫn là ID Train-base cùng seed;
không dùng ID fairness làm fallback. Các run cũ giữ nguyên làm provenance.

## Protocol được giữ theo thầy

- EarlySD là S1 được chọn. Quy tắc all-off phải đạt95% ID đã được thay bằng
  fallback ID-only riêng. Thêm ablation fallback S1 all-off vs ID-only riêng.
- Dừng K4; không thêm seed. Không fit lại chín random controls. Gọi đúng random
  theo phân phối biên: C/B theo tần suất, Title Gaussian unit-norm.
- Chỉ late mới `checkpoint2_late_20261106` vào bảng chính; Checkpoint1 là lịch sử.
- Utility LOO và feature diagnostic dùng cùng S1. Nhãn gate thuộc S1 của chính
  seed; noise/regime khóa từ reference42/43. Add-one có noise riêng.
- Sigmoid threshold và softmax-ID/null routing được định nghĩa rõ; all-off
  trả về ID riêng, còn nguồn dùng early S1. Không cắt gradient alpha trong S3.
- B sử dụng45 job/cohort (ID dùng chung cho hai ô cơ chế), tương ứng48 ô bảng:
  early/late ×8subset ×3seed. Không train hai lần cùng ID chỉ để đủ ô bảng.
- F1 Beauty cần năm seed42–46, UASF Val mean cao hơn best nested baseline và
  paired-user bootstrap1.000 lần có cận dưới95%CI chưa làm tròn >0. Kết luận chính giữ family mười đối chứng của memo v1, gồm fairness; sáu selector
  là so sánh phụ, không dùng để đổi F2 thành F1 sau khi đã thấy số. Quy tắc này
  được ghi bằng framing_policy và historic_decisions giữ nguyên quyết định cũ.
- Utility E1 Spearman giữa dự đoán và delta khác rho ổn định nhãn giữa hai seed.

## Các stage riêng khi cần kiểm tra hoặc bù thiếu

```bash
# Chỉ dùng bù thiếu sau kiểm kê; với reuse thiếu S1 sẽ dừng, không fit thay.
python -u scripts/run_memo0711.py --stage b --datasets sports toys \
  --reuse-memo runs/memo0711_v1 --execute-training "${COMMON[@]}"
python -u scripts/run_memo0711.py --stage beauty --datasets beauty \
  --reuse-memo runs/memo0711_v1 --execute-training "${COMMON[@]}"
python -u scripts/run_memo0711.py --stage robustness --datasets beauty sports toys \
  --reuse-memo runs/memo0711_v1 --execute-inference "${COMMON[@]}"
python scripts/run_memo0711.py --stage rank "${COMMON[@]}"
```

`robustness` cần SD/noSD/ID foundations đã có;90 dòng/cohort,270 dòng ba cohort
cho2family ×3nguồn ×5mức0–80% ×3seed. Không buộc NDCG giảm đơn điệu.
Mask, coverage, history/candidate exposure và paired logits/ranks được lưu theo
case. Chỉ đánh dấu hoàn tất sau inspect_evidence; xem mục kiểm định v3 bên dưới.
`rank` chỉ đọc ranks CPU: median/top10/100/1000. Không suy thành phần Brand
top10 từ target ranks; tùy chọn đó cần topK item IDs.

## F1/F2, transfer và Test

Đọc `beauty/framing_decision.json` sau Beauty validation. F2 không cần transfer
UASF Sports/Toys; vẫn hoàn tất B/robustness ba tập. Chỉ F1 và từ12/11:

```bash
python -u scripts/run_memo0711.py --stage transfer --datasets sports toys \
  --execute-training "${COMMON[@]}"
```

Transfer dùng S1 có sẵn, ba seed và các gate chính; không tự mở rộng fairness
hoặc toàn bộ ablation Beauty thành các đợt train Sports/Toys không được yêu cầu.
Nếu transfer không đạt E0/E1, ghi `TRANSFER_NO_ELIGIBLE` kèm lý do và SHA của status S2; không huấn luyện gate rồi che trạng thái thiếu.
Không mở Test UASF trước freeze. Các Test chẩn đoán B/Title/ranks được ghi riêng.

Sau13/11, review code và dữ liệu, commit sạch, không process giữ lock:

```bash
git tag checkpoint2_0711_frozen_v3
python scripts/run_memo0711.py --stage freeze \
  --freeze-tag checkpoint2_0711_frozen_v3 "${COMMON[@]}"
python -u scripts/run_memo0711.py --stage release-test --datasets beauty \
  --execute-inference "${COMMON[@]}"
```

Với F1 đã hoàn tất transfer Val, thêm `sports toys` vào danh sách release.
Freeze kiểm tra coverage/seed/eligibility/hash và robustness binding bản sửa;
tag phải tồn tại và trỏ HEAD đã review. Không fit sau freeze. Test được mở theo
giao thức rồi dùng cache để tái lập bảng; không chọn lại gate theo Test.

## Báo cáo, kho chung và Git

`--stage report` không chạy mô hình. Registry ở runtime; bộ bảng đầy đủ ngoài
runtime tại `results/memo0711_v3_tables/` để không thay artifact đã freeze:

- B ba tập, Add/Drop từng seed, kiểm kê nguồn/số seed và fairness summary;
- gate/baseline Val/Test, mean±SD, evidence framing/paired comparisons;
- utility/E0/E1, label rho và OOF Spearman/AUC; robustness, Title, ranks;
- controls CBT historical với paired-user CI; K1–K4 historical và group evidence;
- gaps, source hashes, shared-feature manifest, package manifest;
- đầu vào bảng LaTeX `.tex` và hình B/robustness `.svg`/`.pdf` từ các nhóm số liệu đủ seed. Hình có trục, đơn vị, legend, nguồn và SD mẫu; không vẽ ô thiếu thành0.

Matplotlib chỉ dùng lúc report; nếu thiếu dependency, `plot_status.json` ghi rõ MISSING_PLOTTING_DEPENDENCY thay vì tạo hình giả. Không tự cài/nâng thư viện trong môi trường đã khóa. LaTeX là table input để ghép/chỉnh layout bài, không phải PDF bài đã biên dịch.

Ô thiếu hoặc Test chưa mở để trống, không gán0. Hash bundle thiếu checkpoint
không được gọi là checkpoint server đã xác minh. Report không xuất robustness
v1 thiếu binding sửa lỗi hoặc add-one thiếu evaluation_version2.

Publish nhỏ copy JSON/CSV/MD/log dưới10MB, không .pt. File lớn không mất dấu:
`large_artifacts_manifest.json` ghi path/size/SHA256 và yêu cầu copy kho chung.
Đặc biệt chuyển nguyên `features_train_meta.csv` và `validation_same_model.csv`
theo manifest vào kho chung; weights giữ server/kho chung có SHA, ngoài Git thường.
Chỉ publish/commit khi compute đã dừng; giữ working tree sạch trước lần compute.

Không đưa lại v1 hoặc checkpoint vào Git chỉ để chạy bước sửa này. User có thể
chuyển metadata/log nhỏ để review; code nhận lại thực sự chạy trên server có .pt.


## Kiểm định v3 và giữ lineage khoa học

`--stage audit` chỉ đọc metadata/ranks và in JSON, không sửa runtime/registry,
không load checkpoint/model, không dùng GPU. Exit0 là đủ các kiểm tra CPU được
thực hiện; exit2 là INCOMPLETE, xem checks. Audit không thay thế freeze guard.
`--stage report` cũng chỉ đọc runtime, ghi gói bảng vào `--publish-dir` hoặc
mặc định `results/memo0711_v3_tables`; không cập nhật registry trong output đã khóa.

`--reuse-memo` tự đọc ba input_roots từ protocol.lock.json Beauty của nguồn,
trừ khi truyền CLI override rõ. Cách này tránh thay đường dẫn ngoài ý muốn làm
cache gate không tương thích. Checkpoint nguồn vẫn phải có trên server.

### Bằng chứng robustness

- `robustness_evidence/`: item mask CSV, manifest, probe history/candidate,
  paired fixed-batch logits mức0/80 và per-user ranks; mọi file có SHA256.
- `robustness.csv`: eligible/removed/remaining item, coverage gốc/effective,
  số token/ngữ cảnh lịch sử thực sự mất metadata, candidate exposure và hashes.
- `robustness_diagnostics.csv`: đổi logits, số user đổi rank, chênh metric,
  cờ đường phẳng. Curve phẳng không tự đồng nghĩa lỗi hoặc miễn nhiễm.
- `robustness_binding.json`: evaluation_version3 và toàn bộ evidence hashes.
  Không nhận lại robustness v1/v2 như bằng chứng đã kiểm định.
- Nếu effective mask đúng nhưng logits vẫn không đổi, trạng thái REVIEW_REQUIRED
  chặn khóa. Chỉ có thể chấp nhận bằng giải thích cụ thể cho từng case trong
  `robustness_review.json`, gắn SHA diagnostic, tên reviewer và thời điểm ISO có timezone; trạng thái là
  HUMAN_REVIEWED_FLAT, không phải tự động VERIFIED. Không sửa metric hoặc ép đường giảm.
  Report giữ trạng thái chưa xác minh và không vẽ curve như kết quả đã xác nhận.

### Nhóm, utility và framing

`group_lineage.json` gắn groups.csv/groups.json/rule, checkpoint/model, rank CSV,
context hash, lựa chọn từng seed và source implementation. Bảng nhóm luôn có
lineage_id và nguồn. Không lấy số07/11 và09/11 khác rule rồi gộp cùng một dòng.
Global/group chọn bằng Val; oracle Test chỉ mô tả. Bản historical giữ riêng.

`framing_decision.json` chính so toàn bộ mười baseline như memo v1; trường
`secondary_six_selectors` báo riêng phép so cùng S1. Các file trong
`historic_decisions/<dataset>/` giữ số/kết luận F2, hash nguồn và family từng
được ghi, không tự gắn lại nhãn. Khác kết quả chính/phụ phải hiện trong báo cáo.
Không đổi threshold, seed, objective hoặc family theo Test.

`feature_definitions.csv` gắn cột psi, nguồn feature CSV và mã định nghĩa.
Đặc trưng dùng lịch sử và score của frozen reference ungated; target chỉ dùng
cho label/audit. Recommendation loss S3 thực tế là full-catalog cross-entropy,
cộng utility supervision/sparsity đã khai báo; không gọi là pairwise BPR.
Spearman E1-prediction-vs-delta luôn tách khỏi rho ổn định nhãn giữa seed.

Bảng thống kê dataset/coverage lấy artifact metadata thật. File thiếu giữ
missing, Test chưa mở không điền0. Bảng CSV/LaTeX và hình SVG/PDF được tạo từ
số đã đo; REVIEW_REQUIRED không được dùng để tuyên bố kết quả RQ3 cuối.


### Bằng chứng dự đoán E1 theo từng context

`--stage e1-evidence` là phân tích CPU: tái fit các predictor OOF LR/tree/length
trên đúng input utility, feature order, seed và user-held-out folds của E1 cũ.
Nó không train S1/gate, không mở Test, không đổi quyết định E0/E1 đã báo cáo.
`all` tự làm bước này sau compute/rank cho những cohort đã có selected S2.

`e1_oof_evidence/<CE hoặc LOG_RANK>/` lưu `paired_oof_predictions.csv`,
`replay_summary.json` và manifest SHA input/report/engine/loader/features.
AUC và Spearman dự đoán-delta được tái tính và so với JSON gốc; khác vượt
sai số cho phép thì dừng, không sửa JSON cũ cho vừa. Audit/report kiểm tra
OOF evidence; báo cáo có `E1_OOF_replay_summary` và `E1_OOF_artifacts`.
Chỉ replay trước freeze; sau freeze chỉ đọc cache hoàn tất đúng binding.
