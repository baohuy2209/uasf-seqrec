# Thực hiện quyết định TAIMMA ngày 07 tháng 11

Entry point mới `scripts/run_memo0711.py`, config `configs/experiments/memo0711.yaml`.
Không chạy lệnh `complete_checkpoint2.py --stage all` cũ cho protocol mới.
Không nâng thư viện, không thay dữ liệu, không sửa các run đã báo cáo.
Phiên bản này chọn early SD theo quyết định thầy, fallback ID-only Train-base cùng seed.
Không thêm K4; không fit random V2; giữ random cũ và mô tả đúng.

## Trước khi chạy server

Commit/push toàn bộ patch từ Windows, rồi cập nhật server khi không còn job dùng source checkout cũ.
Giữ môi trường RTX 4060 và các file .pt gốc trên server. Windows không cần có weights.
Mặc định input root trong chính repo: runs/checkpoint2_full_20261106 và runs/checkpoint2_late_20261106.
Nếu late thật ở checkout khác, dùng `--late-root /duong/dan/thuc/` nhất quán trên mọi lệnh.
Roots sai hoặc Beauty job lịch sử thiếu sẽ dừng; không âm thầm train thay.

```bash
cd /home/ubuntu/projects/uasf-seqrec
source .venv/bin/activate
python scripts/run_memo0711.py --stage plan
python scripts/run_memo0711.py --stage check
```

`plan` không load dữ liệu/model, không train. `check` đọc dữ liệu/catalog/hash, chưa inference.
Runtime mới `runs/memo0711_v1/` được ignore; outputs cũ giữ nguyên.
Code/config/data/source bị đổi thì dùng output phiên bản mới, không chỉnh protocol.lock.json.

## Chạy riêng theo người phụ trách

B Sports/Toys (không chạy UASF hoặc K4):

```bash
python -u scripts/run_memo0711.py --stage b --datasets sports toys --execute-training
```

Beauty UASF năm seed, S2 từng seed, baseline, fairness và ablation; chỉ Val của gate:

```bash
python -u scripts/run_memo0711.py --stage beauty --datasets beauty --execute-training
```

Nhập lại ID/early CBT/early SD Beauty 42–44 đã có; chỉ fit các foundation seed còn thiếu 45/46.
Fairness là job khác: ID/early học Train-base+Train-meta, không thay thế weights ID fallback.
Hai stage khác dataset dùng lock riêng nên có thể chạy song song trên cùng checkout sạch.
Hai process cùng GPU không bảo đảm nhanh hơn; theo dõi thông lượng thực tế.
Không sửa source hoặc publish/commit kết quả vào working tree trong khi process khác đang chạy.

Các mục bổ sung có lệnh riêng:

```bash
python -u scripts/run_memo0711.py --stage fairness --datasets beauty --execute-training
python -u scripts/run_memo0711.py --stage robustness --datasets beauty --execute-inference
python -u scripts/run_memo0711.py --stage title --datasets beauty --execute-training
python scripts/run_memo0711.py --stage rank
```

Robustness trên Sports/Toys cần có early SD; nếu còn thiếu foundation thì phải cấp
`--execute-training` rõ ràng. Không có quyền đó sẽ dừng, không fit tự động.
Title dùng lại ba T-only thật nếu đúng protocol, chỉ train ba shuffled T-only.
Rank diagnostics chỉ dùng target ranks; không suy thành phần Brand top-10 từ target rank.
Phân tích Brand top-10 là tùy chọn cần top-K item IDs, chưa được tạo bởi stage rank.

## Một lệnh thực hiện toàn bộ phạm vi trước Checkpoint 2

```bash
python -u scripts/run_memo0711.py --stage all --execute-training
```

Lệnh này là công việc lớn: B ba tập, Beauty UASF/fairness, robustness, Title và rank.
Có fit các job được khai báo còn thiếu; không phải lệnh chỉ tính báo cáo. Không tự chạy
UASF Sports/Toys trước quyết định F1, không mở gate Test. Ưu tiên stage riêng nếu muốn
chạy Giang và Huy song song. Cache hoàn tất đúng hash/config được dùng lại; interrupted
fit bắt đầu attempt mới, không resume optimizer giữa epoch.

## Quy tắc phương pháp được ghi rõ

- Sigmoid: beta<kappa bị tắt khi đánh giá. Softmax-null: null thắng hoặc mọi nguồn dưới kappa thì chuyển ID.
- All-off dùng ID riêng; còn nguồn dùng early S1. Training gate vẫn dùng soft alpha để có gradient.
- Fallback S1 ablation chỉ thay cách route all-off; các điều kiện khác giữ nguyên.
- Epsilon/regime khóa từ hai hệ tham chiếu; labels S3 lấy utility của chính seed S1.
- Add-one tính noise từ hai hệ add-one tương ứng, không dùng epsilon LOO thay thế.
- Họ baseline khóa gồm sáu selector trong memo, ID riêng, early no-SD và hai fairness controls.
  F1 được đánh giá bảo thủ so với baseline mạnh nhất trong toàn bộ danh sách này; family Holm cập nhật theo số thực tế.
- Framing Beauty dùng năm seed, bootstrap ghép cặp user 1.000 lần trên mean qua seed;
  dùng CI chưa làm tròn, cận dưới >0. Không chọn theo Test.

## Framing và Test

Sau Beauty Val, đọc `beauty/framing_decision.json`. Nếu F1 và đã đến 12/11:

```bash
python -u scripts/run_memo0711.py --stage transfer --datasets sports toys --execute-training
```

Nếu F2 không chạy transfer UASF. Giữ B/robustness ba tập và UASF Beauty.
Sau 13/11, review kết quả, commit/tag sạch trước freeze; không có process đang giữ lock:

```bash
python scripts/run_memo0711.py --stage freeze --freeze-tag checkpoint2_0711_frozen
python -u scripts/run_memo0711.py --stage release-test --datasets beauty --execute-inference
```

Tag phải tồn tại và trỏ HEAD đã review. Với F1 thêm `sports toys` vào release sau khi
transfer Val hoàn tất. Gate Test có binding cache và chỉ được mở sau freeze.

## Báo cáo và lưu trữ

Sau các process hoàn tất, xuất số thực đã có, không chạy mô hình:

```bash
python scripts/run_memo0711.py --stage report --publish-dir results/memo0711_v1
```

Registry results.csv, summary.csv, report.md và các JSON/CSV group/utility/framing/ranks
được tạo từ outputs. Test chưa mở để trống, không gán số 0. Files dưới 10 MB được copy
sang results, kèm published_manifest.json; weights không publish lên Git.
Feature/utility CSV lớn vẫn ở runtime: chép nguyên bundle vào kho chung (có SHA256),
đặc biệt features_train_meta.csv và validation_same_model.csv thầy yêu cầu.
Commit kết quả publish trước lượt compute tiếp theo để qua clean-tree guard.

Các thí nghiệm random cũ và K1–K4 là bằng chứng lịch sử ở input roots; runner mới không
train lại chúng hoặc sửa số. Nhật ký late mới-vs-cũ phải mô tả khác biệt có bằng chứng
từ config/snapshot, không đoán nguyên nhân chỉ vì metric khác.
