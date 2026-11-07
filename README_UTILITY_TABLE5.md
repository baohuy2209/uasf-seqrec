# Chỉ bổ sung bảng 5: utility và E0/E1

Không dùng `complete_checkpoint2.py --stage all` cho việc này.

Entry point: `scripts/complete_utility_table5.py`.
Chỉ dùng hai checkpoint Beauty `early_sd_42` và `early_sd_43` đã có trong
`runs/checkpoint2_full_20261106`. Không cần checkpoint late. Không chạy grid,
random controls, K4, UASF, ablation hoặc train lại ID/early/late.
E0/E1 vẫn thực hiện các phép fit chẩn đoán theo protocol của nó.

Commit/push các file mới từ Windows rồi cập nhật server khi không còn process
thí nghiệm đang chạy trong checkout. Không xóa output cũ. Giữ nguyên môi trường.

Kiểm tra hai checkpoint gốc (chưa inference):

```bash
python scripts/complete_utility_table5.py
```

Tính đúng phần còn thiếu và xuất kết quả:

```bash
python -u scripts/complete_utility_table5.py --execute-inference
```

Có thể chỉ rõ `--early-root /duong/dan/checkpoint2_full_20261106`.
Thiếu checkpoint/manifest hoặc sai hash thì dừng; không tự train thay thế.
Script không chấp nhận `--stage all` hoặc `--execute-training`.

Runtime/cache riêng: `runs/completion_utility_table5_20261107/beauty/`.
Thư mục này cần được ignore bởi `/runs/completion_*/` trong `.gitignore`.
Không dùng runtime của lượt `all` đang dở để tránh JSON đã commit thiếu weights.
Cache chỉ được dùng lại khi model/data/config/source bindings còn phù hợp.

Kết quả để commit: `results/utility_table5_20261107/beauty/` gồm:

- `table5_utility.csv`, `table5_utility.md`: ba dòng C/B/T, đủ ba cột số.
- `table5_inputs.json`, `table5_manifest.json`: checkpoint nguồn và hash.
- `diagnostic_early_sd/alignment.json`: số chưa làm tròn và quy tắc chọn nhãn.
- `diagnostic_early_sd/status.json`: nhãn CE/LOG_RANK/UNRESOLVED, trạng thái E0/E1.
- `diagnostic_early_sd/e0_e1_CE.json`, `e0_e1_LOG_RANK.json`: kết quả chẩn đoán.
- Các CSV utility/features/validation và cache bindings để đối chiếu.

Không publish `.pt`. Chưa có số thì không giả số. Bảng được tính lại trên cùng
hai hệ tham chiếu seed 42/43; cột Val có thể khác bảng cũ dùng một hoặc ba seed.
Đây là chẩn đoán early SD, không tự xác nhận đủ điều kiện chọn backbone/S3.
Không mở đánh giá Test mới. Các báo cáo early/late đã có được giữ nguyên.

Sau khi chạy, commit kết quả nhẹ trước một lượt inference mới nếu clean-tree
guard đang bật. Không sửa code/config giữa chừng; source đổi thì dùng output mới.
