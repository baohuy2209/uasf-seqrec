
# Phiên kiểm tra Minh — 2026-11-05T22:14:59.954307+07:00 (Asia/Ho_Chi_Minh, UTC+07:00)

## 1. Trạng thái và phạm vi kiểm tra chỉ đọc
- Root: /home/ubuntu/projects/uasf-seqrec; branch main; HEAD 7bb443431771aaff98b726fccb043a888eec2163.
- Working tree ban đầu sạch (git status --short không có dòng). Không commit/reset/checkout/stash.
- Môi trường: {"python": "3.12.16 (main, Oct 26 2026, 14:08:45) [Clang 23.1.3 ]", "platform": "Linux-6.8.0-147-generic-x86_64-with-glibc2.43", "torch": "2.13.0+cu130", "cuda": true}. Venv chưa cài editable package; script dùng src của chính checkout qua sys.path, không cài thư viện.
- Không có AGENTS.md tại repo/các thư mục cha áp dụng hoặc trong repo (đã tìm); không có chỉ dẫn agent bổ sung.
- Sandbox bwrap không khởi chạy; các lệnh đọc/chạy trong workspace được thực thi qua escalation đã duyệt.
- Artifact kiểm tra chi tiết: reports/minh_20261105_221459/audit.json. Audit chỉ đọc dữ liệu gốc và tái tính metric, chưa phân tích nhóm, tạo nhãn hay fit E1.

## 2. Nguồn chỉ đạo và mức độ thẩm quyền
- Yêu cầu đính kèm phiên này là chỉ đạo trực tiếp hiện hành: ba seed cho Task 1; CE/log-rank cùng reference; ưu tiên CE nếu khớp dấu cả C/B/T, nếu không thì xét log-rank; không tự chọn nếu cả hai không đạt; E1 đầy đủ ψ có điều kiện.
- docs/decision_log.md: memo 20/10, 26/10 và 29/10 được ghi nhận; 01/11 khóa strict input-history masking; 03/11 yêu cầu chọn checkpoint chỉ trong epoch đã train. Không tìm thấy chỉ đạo được phê duyệt mới hơn trong repo. Bản đầy đủ chỉ đạo nghiên cứu 03/11 và đề xuất ψ hiện hành không có; phần nội dung trong yêu cầu hiện tại được dùng làm mốc, không suy diễn toàn bộ memo.
- runs/checkpoint1_version1/protocol.lock.json: seeds 42/43/44, max_epochs 100, patience 20, full catalog, strict history masking, trained_epochs_only; toàn bộ source/config hash hiện tại khớp lock.
- docs/decision_log.md khóa epsilon=median chênh lệch tuyệt đối giữa hai reference seed; rho >=0.3 contextual, [0.1,0.3) soft/averaging, <0.1 group; E1 user-grouped OOF AUC>=0.55, CI dưới>0.5 và vượt length-only. Không đổi các ngưỡng này.
- README và code là bằng chứng triển khai, không tự coi mọi mô hình có code là đã chạy hoặc kiến trúc đã được duyệt.
- e0_e1.json là kết quả lịch sử CE với 4 đặc trưng, không chứng minh E1 đầy đủ ψ của yêu cầu này.

## 3. Chọn bộ mới nhất tương thích
Đã kiểm tra registry results/results.csv và runs/checkpoint1_version1/results.csv, manifests/state/completed/source của các run hiện có. Bộ version1 có 27 run hoàn tất, bao gồm grid late 24 run, ID repeat, LOO và early CBT seed42. Pilot 28/10 chỉ ID seed42 và repeat, protocol/lineage cũ; không ghép vào grid. Tính mới dựa vào run timestamps + trạng thái hoàn tất + source lock + checksum + đủ grid, không dựa tên version1 hay mtime.
- Task 1: Beauty four-segment, SASRec d64 + residual late, ID/C/B/T/CB/CT/BT/CBT, seeds42/43/44 của version1.
- Task 2: cùng bộ, chỉ reference late_CBT seeds42/43 gắn với utility.csv; sử dụng chính checkpoint tương ứng để suy luận bổ sung; không lấy subset train riêng làm ablation.
- Task 3: bộ trên chỉ phù hợp chẩn đoán late lịch sử. Chưa có căn cứ kiến trúc UASF cuối, nhãn được chọn hoặc đầy đủ định nghĩa ψ.

## 4. Inventory artifact
| Đường dẫn | Vai trò | Dataset / kiến trúc | Nguồn / seed | Trạng thái | Lineage |
|---|---|---|---|---|---|
| runs/checkpoint1_version1/results.csv, progress.json, completed/*.json | Registry/trạng thái | Beauty SASRec d64 + late | 8 cấu hình ×42/43/44; repeat42 và LOO riêng | COMPLETED | source commit 5fd0ce8 với dirty snapshot; hash hiện tại khớp lock |
| runs/checkpoint1_version1/protocol.lock.json, source_snapshot.json | Protocol/source | Beauty | grid | VERIFIED | hash src/scripts/configs khớp |
| runs/checkpoint1_version1/runs/*/per_user_val.csv, per_user_test.csv | Ranks | Beauty four-segment (LOO tách riêng) | 8×3 | VERIFIED | artifact_index/checkpoint/manifest mỗi run |
| runs/checkpoint1_version1/runs/*/checkpoint.pt | Checkpoint ngoài Git, có trên đĩa | Beauty late/ID/early | như trên | có thật | SHA256 của toàn bộ completed late/ID/LOO khớp |
| runs/checkpoint1_version1/runs/early_CBT_seed42_*/ | Early run | Beauty early CBT | 42 | COMPLETED, thiếu grid còn lại | progress + run_state + registry |
| runs/checkpoint1_version1/utility.csv và .manifest.json | Train-meta CE | Beauty late CBT | C/B/T ×42/43 | VERIFIED, thiếu rank | 133440 dòng; checkpoint+metadata+cohort hash |
| runs/checkpoint1_version1/utility_cohort.csv/json | Cohort | Beauty | paired reference42/43 | 22240 included /123 OOV excluded | cohort checksum |
| runs/checkpoint1_version1/e0_e1.json | E0/E1 lịch sử 4 ψ | Beauty late CBT |42/43 | hoàn tất lịch sử | CE chưa được căn chỉnh NDCG theo task mới |
| runs/checkpoint1_version1/runs/id_seed*/s_ID_*.npy; score_cache_integrity/*.json | ID scores unmasked | Beauty ID |42/43/44, meta/val/test | file có, shape22363×11998 | chỉ ID, chưa đủ residual ablation; chưa hash lại cache lớn |
| data/raw/Beauty/Beauty_5.json; artifacts/metadata/beauty/manifest.json và item_map.csv | Lịch sử/catalog/popularity | Amazon2014 Beauty 5-core | Train-base | raw hash khớp processed manifest | split tái dựng bằng source đã khóa |
| data/processed/beauty_2014/*/ | Pilot mappings | Beauty ID pilot |42 | KHÔNG dùng mapping item cho grid | user map khớp; item map khác mapping sorted hiện tại |
| artifacts/metadata/beauty/metadata.csv, availability_mask.csv, title_embeddings.npy | Metadata | Beauty catalog11998 | C/B/T | có trên đĩa | manifest v2, kiểm tra bổ sung trước inference |
| runs/pilot_beauty_real_20261028_*/ | Pilot cũ | Beauty ID |42+repeat | archival | không ghép protocol với grid |

Inventory từng run với đường dẫn/config/state/seed/hash nằm trong audit.json (26 completed marker; early lưu ở progress).

## 5. Chất lượng và giới hạn
- Split tái dựng: 22363 user, 11998 catalog items, 109050 training samples; OOV meta123/val178/test479.
- Toàn bộ 26 run có completed marker: 52 file ranks không null/duplicate user, integer rank>=1; main rank<=11999. User/target của main khớp đúng split tái dựng. OOV giữ rank11999 làm miss.
- NDCG@10 tái tính từ ranks khớp cột từng user (sai số<=5.6e-17) và registry (sai số<1e-15). Metrics.json được round 6 chữ số; không dùng số round để chọn cấu hình.
- Artifact_index tất cả 26 run khớp toàn bộ hash. Source commit lịch sử 5fd0ce8 có git_dirty=true; sử dụng snapshot/lock hash, không đồng nhất source bằng commit ID đơn lẻ.
- utility không null/trùng khóa (context_id,source,seed); mỗi source/seed có22240 user giống cohort; context hash khớp history/target tái dựng; delta CE khớp hiệu CE tới 3e-8. Manifest utility/reference checkpoint/cohort/metadata khớp.
- Pilot processed item2idx KHÁC sorted mapping của grid. Không dùng lại; sẽ đối chiếu metadata item_map và raw ASIN với split hiện tại trước tính. Không sửa mapping lịch sử.
- Tie evaluator: score cao trước, bằng score thì catalog index nhỏ trước. Mask chính strict input history; OOV/target bị mask là miss N+1. CE lịch sử tính logits full catalog chưa mask; rank cần evaluator strict. Cùng reference/cohort nhưng hai mục tiêu khác nhau.
- Không truy cập dịch vụ/vị trí ngoài những đường dẫn repo ghi nhận; không giả định thiếu checkpoint chỉ vì Git ignore.
- Chưa có branch scores/state summaries trong utility; cache hiện tại chỉ s_ID. Không đánh giá một phần đang ghi là kết quả hoàn tất.

## 6. Readiness BAN ĐẦU (giữ nguyên để truy vết)
| Task | Đầu vào bắt buộc | Đã có | Thiếu | Trạng thái | Hành động |
|---|---|---|---|---|---|
|1 late|8 configs×3 seeds, aligned ranks/history/popularity, group rules|grid đã xác minh, raw history|chưa có group rules được thầy khóa|PARTIAL|chạy phân tích chẩn đoán với quy tắc agent công bố trước ở mục7|
|1 early|8 configs×3 seeds tương thích|CBT seed42 + ID|6 subset, CBT42 ngoài seed42 và grid seed43/44|BLOCKED|không train bổ sung trong tác vụ này|
|2|same frozen CBT, CE+rank meta, masked val ablations, lineage|CE paired seeds42/43 + checkpoint CUDA + lịch sử|rank meta và val same-model ablations chưa lưu|PARTIAL|suy luận bổ sung không training; nếu tái lập ranks/CE đạt thì xét quy tắc dấu|
|3|kiến trúc duyệt, nhãn Task2, đầy đủ ψ, stability/OOF/CI|4 ψ code, historical CE E1, locked criteria|quyết định kiến trúc, đề xuất ψ đầy đủ/công thức state+agreement+popularity; nhãn mới chưa xét|PARTIAL|lập đặc tả/gap/leakage; chưa fit hay tuyên bố hoàn tất E1|

## 7. Lựa chọn triển khai agent công bố TRƯỚC tính
Đây là quy tắc chẩn đoán đề xuất, không gắn nhãn yêu cầu đã được thầy duyệt.
- Task1 popularity=count mọi interaction Train-base (kể cả item đầu chuỗi, không dùng riêng training target samples vì bỏ sót item đầu). Catalog Train-base; head là top20% item theo tần suất: lấy count biên của ceil(0.2*N) item đầu sau sort count giảm, ID tăng, và đưa toàn bộ ties biên vào head. Tail là phần còn lại.
- Grouping 5 nhóm length ×2 tail: lịch sử cửa sổ50 đã quan sát; length=số item ID biết; tail ratio=(số observed item thuộc tail hoặc OOV)/(số observed item gồm OOV). Không dùng padding/target/future.
- Fit 4 quantile length [20,40,60,80]% trên lịch sử terminal Train-base (history của Train-meta), method=linear; ties bằng ngưỡng vào nhóm cao qua searchsorted(side=right). Giữ nhóm rỗng nếu quantile trùng.
- Fit ngưỡng tail bằng median trong từng nhóm length trên cùng terminal Train-base, bằng ngưỡng vào mức high. Nhóm length rỗng fallback median toàn cohort training. Cố định trước validation/test; không fit bằng test. Gán nhóm riêng tại thời điểm val và test bằng cùng ngưỡng.
- Mỗi seed chọn cấu hình theo mean validation NDCG chính xác. Tie |diff|<=1e-12: ưu tiên thứ tự ID,C,B,T,CB,CT,BT,CBT; nhóm validation rỗng fallback global validation winner.
- Global và per-group chọn validation; oracle chọn test chỉ là trần hậu kiểm của 10 nhóm và 8 cấu hình này. Report từng seed, mean và SD mẫu ddof=1, không tính repeat42 là seed mới; tổng hợp từng user (weighted group).
- Task2 giữ paired reference42/43 đúng utility manifest. Equal-weight mean hai seed cho mỗi source, đồng thời công bố từng seed. Không đổi seed sau thấy dấu. Sign(x)=0 nếu |x|<=1e-6 cho CE và log-rank; |Dval|<=1e-8 coi0. So dấu tổng hợp của cả3 nguồn; zero được báo riêng là không có hướng hữu ích, không coi zero là khớp định hướng để chọn nhãn.
- Rank utility dùng full-catalog evaluator strict như validation, context/history+catalog nhất quán; positive=without worse. CE giữ full-catalog unmasked như artifact hiện hành. Mọi bất đồng seed báo rõ.
- Chỉ dùng cùng checkpoint late_CBT, subtract nhánh đã availability-mask để tắt nguồn. Không thay bằng model subset train riêng. Tái lập ranks all trên validation và CE lưu trước chấp nhận bổ sung.
- Task3 chưa tự định nghĩa state summaries, agreement hay popularity feature thay cho bản đề xuất thiếu. Không gọi bốn feature hiện hành là đầy đủ ψ; không chọn architecture thay người duyệt.

## 8. Thứ tự thực thi và đầu ra
1. Báo cáo này được ghi và đọc lại trước khi task chạy.
2. Xác minh catalog metadata mapping/hash, thực hiện Task1 late scope chẩn đoán: grouping_rules.json, user_context_groups.csv, group_metrics.csv, selected_configurations.csv, summary_per_seed.csv, summary.csv.
3. Suy luận Task2 cùng CBT42/43: meta ranks, validation ablation ranks, sign_alignment.csv, label_decision.json, input lineage/commands/config/source hash. Không training backbone hay sửa lịch sử.
4. Task3 lập psi_spec.csv + leakage/gap notes và code kiểm tra độc lập nếu thiếu quyết định; chỉ fit nếu nhãn/kiến trúc/ψ đầy đủ.
5. Cập nhật báo cáo sau từng task, bảo tồn readiness ban đầu. Tất cả output mới tại reports/minh_20261105_221459; không ghi đè artifact gốc.


## Catalog preflight sau báo cáo
Metadata item_map có12101 raw items gồm11998 Train-base catalog +103 OOV (model ID0). Sau lọc in_train_base, mapping khớp chính xác split hiện tại; OOV đều ID0. Kiểm tra dict toàn bảng ban đầu dừng ở assertion vì chứa cả103 item ngoài catalog; chưa task nào thực thi ở lần kiểm tra đó. Toàn bộ metadata/raw hash khớp manifest; không dùng pilot item2idx.


## Cập nhật 2026-11-05T22:17:56.582420+07:00
Task1 late: hoàn tất phạm vi chẩn đoán bằng quy tắc agent đã công bố; early vẫn BLOCKED. Output: reports/minh_20261105_221459/task1. Mean ± SD mẫu qua seeds42/43/44:

- ID: 0.047850699 ± 0.000755962
- global_validation_subset: 0.047850699 ± 0.000755962
- group_test_oracle: 0.047897828 ± 0.000769039
- group_validation_subset: 0.047829754 ± 0.000836088
Không suy luận ý nghĩa thống kê từ mean/SD. Oracle chỉ hậu kiểm 10 nhóm ×8 cấu hình. Nhóm val/test dùng cùng ngưỡng nhưng gán theo lịch sử của từng thời điểm.


## Cập nhật 2026-11-05T22:18:19.461840+07:00
Task2: đã suy luận không training, cùng frozen late_CBT42/43. Output reports/minh_20261105_221459/task2. CE replay max error=4.44e-16; all validation ranks tái lập chính xác. Nhãn được chọn: CHƯA ĐẠT — cả CE và log-rank đều không khớp đủ ba nguồn. Dấu dựa mean equal-weight hai seed, tolerance công bố trước; từng seed và bất đồng lưu riêng. Log-rank không tương đương hoàn toàn NDCG@10.

- brand: mean ΔCE=-0.019649979; mean Δlogrank=-0.0221292459; Dval=-0.000983157128.
- category: mean ΔCE=0.0299940659; mean Δlogrank=0.0227676931; Dval=-0.000235602449.
- title: mean ΔCE=0.0274050809; mean Δlogrank=-0.049555034; Dval=-0.00154264014.


## Cập nhật 2026-11-05T22:18:19.485664+07:00
Task3 PARTIAL: hoàn tất bảng ψ và leakage tại reports/minh_20261105_221459/task3/psi_spec.csv, readiness.json. Chưa fit E1/OOF mới. Thiếu: approved reference architecture absent; full current psi proposal with formulas absent; Task2 no label satisfies all3 sign rule. Không dùng E1 CE bốn feature lịch sử thay cho E1 đầy đủ ψ, không tự định nghĩa các feature còn thiếu. Khi đủ đầu vào phải tính lại epsilon/stability trên thang nhãn đã chọn, giữ OOF theo user và fit preprocessing trong train fold.


## Kết luận thực thi và readiness cuối — 2026-11-05T22:20:43.892410+07:00

| Task | Trạng thái cuối | Đã hoàn thành | Đầu vào/quyết định còn thiếu |
|---|---|---|---|
|1|PARTIAL|Late8 cấu hình×3 seed; bảng nhóm/metric/chọn cấu hình/summary; quy tắc chẩn đoán công bố trước|Early thiếu C/B/T/CB/CT/BT cho cả3 seed và CBT43/44; quy tắc nhóm chưa được thầy phê duyệt|
|2|BLOCKED ở bước chọn nhãn; tính toán đã hoàn tất|133440 utility bổ sung,134178 validation ablation records; tái lập CE và full-source validation ranks, đối chiếu dấu|Cả CE và log-rank không đạt điều kiện khớp cả3 nguồn. Cần quyết định khoa học tiếp theo; không tự chọn nhãn hoặc đổi protocol|
|3|PARTIAL ở chuẩn bị; E1 fit BLOCKED|Bảng7 nhóm đặc trưng ψ, công thức4 feature thực có, kiểm tra leakage và gap, protocol readiness|Nhãn Task2 đủ điều kiện; kiến trúc reference được chấp thuận; đề xuất đầy đủ/công thức popularity, sequence state, branch agreement|

Nhận xét Task1: global validation chọn ID ở cả3 seed. Chọn theo nhóm giảm mean NDCG@10 so ID 0.000020945; oracle tăng 0.000047129 (xấp xỉ0.0985% tương đối). Với cách chia nhóm và 8 tổ hợp late fusion đã thử, lợi ích tối đa quan sát được quá nhỏ; không đáng ưu tiên phát triển gate chọn subset theo nhóm trên cơ chế này. Đây là quyết định ưu tiên nghiên cứu, không phải kiểm định ý nghĩa thống kê hay kết luận về mọi dạng gate.
Thiết kế có 10 nhóm, nhưng mỗi seed chỉ có 8 nhóm chứa người dùng trên validation (group 2–9) và 6 nhóm chứa người dùng trên test (group 4–9). Cả hai split vẫn gồm đủ 22363 người dùng mỗi seed. Nhóm rỗng giữ nguyên theo quy tắc: val group0/1 (length<2); test group0/1/2/3 (length<3). Do ngưỡng fit trên terminal Train-base trong khi mỗi thời điểm sau có thêm lịch sử; không refit để lấp nhóm rỗng. Oracle thực tế chỉ chọn trên 6 nhóm test có dữ liệu; tổng hợp đã kiểm tra weighted theo22363 user, không mean nhóm không trọng số.
Task2: Brand có dấu mean CE khác nhau giữa hai reference seed; log-rank/Dval cùng âm. Category CE và log-rank cùng dương nhưng Dval âm ở cảhai seed; Title CE dương, log-rank và Dval âm. Bảng từng seed tại task2/alignment_per_seed.csv. Seed aggregation/tolerance không đổi sau xem kết quả.
Task3: metadata availability của target chỉ là audit cũ, không làm predictor; context ID có target trong hash chỉ là khóa ghép, không là feature. Không có OOF predictions/metric E1 mới vì chưa đủ điều kiện.

Xác minh cuối: verification.json PASS (full cohort/unique keys, weighted aggregation, SD mẫu, công thức utility, miền rank/OOV, NDCG và Dval tái tính độc lập); early_audit.json xác nhận riêng CBT42 hoàn tất, hash và metrics hợp lệ. Original artifact_index của26 completed run vẫn khớp sau thực thi; source lock không đổi. Initial report giữ nguyên nội dung trong initial_report.md và trong lịch sử báo cáo.
Không cần cài/nâng thư viện. Venv không có pip nên pip freeze không chạy; environment_freeze.txt được ghi bằng importlib.metadata, có chú thích nguồn, kèm environment_packages.json (direct_url nếu có).
Tái lập: .venv/bin/python -u analysis/minh_tasks.py --output <thư mục reports mới đã chứa initial_report.md> --device cuda. Lệnh thực tế và input hashes tại execution_manifest.json; executed_source.py giữ nguyên code đã chạy; input_lineage_sha256.json pin từng rank/checkpoint/config/source/metadata dùng. Không chạy lại trong thư mục cũ vì stage outputs được bảo vệ.
Output chính: reports/minh_20261105_221459/. Không có training, S3/UASF, commit hoặc ghi đè artifact lịch sử.
