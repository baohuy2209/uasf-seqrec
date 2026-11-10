# TAIMMA — hoàn tất thực nghiệm theo memo 07/11, bản sửa ngày 10/11

Entry point: `scripts/run_memo0711.py`. Config: `configs/experiments/memo0711.yaml`.
Mặc định chỉ in plan; không train/inference trên import. Không chạy lại runner
`complete_checkpoint2.py --stage all` cũ cho protocol này.

## Sửa một lượt memo v1 đã hoàn tất trên server

Giữ nguyên `runs/memo0711_v1` và toàn bộ checkpoint gốc. Commit/review source mới
rồi cập nhật server khi không còn process train dùng checkout này. Không nâng
thư viện, thay dữ liệu hoặc sửa lock v1. Dùng output mới `runs/memo0711_v2`.

```bash
cd /home/ubuntu/projects/uasf-seqrec
source .venv/bin/activate

COMMON=(
  --output runs/memo0711_v2
  --early-root /home/ubuntu/projects/uasf-seqrec/runs/checkpoint2_full_20261106
  --late-root /home/ubuntu/projects/uasf-seqrec-late-20261106/runs/checkpoint2_late_20261106
  --diagnostics-root /home/ubuntu/projects/uasf-seqrec/runs/checkpoint2_20261106
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

# Tạo bảng số liệu và gói publish sau khi compute hoàn tất.
python scripts/run_memo0711.py --stage report \
  --publish-dir results/memo0711_v2 "${COMMON[@]}"
```

Đổi `--late-root` nếu kết quả late thật ở checkout chính. Flag CLI chỉ override
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
  paired-user bootstrap1.000 lần có cận dưới95%CI chưa làm tròn >0. Sáu selector
  lồng nhau được tách khỏi ID/noSD/fairness bổ sung; không chọn family theo Test.
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
git tag checkpoint2_0711_frozen_v2
python scripts/run_memo0711.py --stage freeze \
  --freeze-tag checkpoint2_0711_frozen_v2 "${COMMON[@]}"
python -u scripts/run_memo0711.py --stage release-test --datasets beauty \
  --execute-inference "${COMMON[@]}"
```

Với F1 đã hoàn tất transfer Val, thêm `sports toys` vào danh sách release.
Freeze kiểm tra coverage/seed/eligibility/hash và robustness binding bản sửa;
tag phải tồn tại và trỏ HEAD đã review. Không fit sau freeze. Test được mở theo
giao thức rồi dùng cache để tái lập bảng; không chọn lại gate theo Test.

## Báo cáo, kho chung và Git

`--stage report` không chạy mô hình. Registry ở runtime; bộ bảng đầy đủ ngoài
runtime tại `results/memo0711_v2_tables/` để không thay artifact đã freeze:

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
