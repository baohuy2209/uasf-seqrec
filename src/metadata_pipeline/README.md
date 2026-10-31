# Metadata Pipeline for TAIMMA Sequential Recommendation

Pipeline chuẩn bị metadata (Category, Brand, Title) cho các tập dữ liệu **Beauty**, **Sports and Outdoors**, và **Toys and Games** (Amazon 2014 benchmark) phục vụ nghiên cứu TAIMMA / UASF.

Pipeline tuân thủ nghiêm ngặt các nguyên tắc:
1. **Khóa Catalog & Chia tập**: Sử dụng chính xác split Train-base (1..n-3), Train-meta (n-2), Val (n-1), Test (n) theo chỉ đạo memo 26/10. Catalog chuẩn `model_item_id` 1..N được suy ra từ Train-base interactions.
2. **True Missingness vs Unknown Token**: Chỉ đánh dấu `has_source=True` khi nguồn dữ liệu thực sự tồn tại, không coi token unknown hay placeholder là metadata thật.
3. **Alignment**: Thứ tự `model_item_id` trong `metadata.csv`, `availability_mask.csv` và `title_embeddings.npy` khớp nhau 100%.
4. **Perturbations (Beauty)**: Sinh synthetic masks (0% đến 80%) và permutations với hạt giống xác định.

---

## 1. Cài đặt môi trường

```powershell
Set-Location uasf-seqrec
# Sử dụng virtualenv đã tạo
.\.venv\Scripts\python.exe -m pip install -e .
```

---

## 2. Các bước thực thi

### Bước 1: Chuẩn bị dữ liệu thô (Raw Data)
Tải reviews 5-core và metadata chính thức từ Stanford SNAP:
```powershell
.\.venv\Scripts\python.exe scripts/download_amazon2014.py --dataset Beauty
.\.venv\Scripts\python.exe scripts/download_amazon2014.py --dataset Sports_and_Outdoors
.\.venv\Scripts\python.exe scripts/download_amazon2014.py --dataset Toys_and_Games
```
*Dữ liệu sẽ được lưu tại `data/raw/amazon_reviews/2014`.*

---

### Bước 2: Chạy trích xuất metadata và sinh Title Embeddings (G01 & G02)
```powershell
# Chạy cho Beauty
.\.venv\Scripts\python.exe -m metadata_pipeline.prepare --config configs/metadata.yaml --dataset beauty

# Chạy cho Sports và Toys
.\.venv\Scripts\python.exe -m metadata_pipeline.prepare --config configs/metadata.yaml --dataset sports
.\.venv\Scripts\python.exe -m metadata_pipeline.prepare --config configs/metadata.yaml --dataset toys

# Hoặc chạy cho tất cả 3 bộ dữ liệu:
.\.venv\Scripts\python.exe -m metadata_pipeline.prepare --config configs/metadata.yaml --dataset all
```

---

### Bước 3: Sinh Availability Mask & Perturbations cho Beauty (G03)
Sinh các tệp mô phỏng thiếu metadata (từ 0.0 đến 0.8) và các phép hoán vị (permutations) theo các seed ngẫu nhiên cố định:
```powershell
.\.venv\Scripts\python.exe -m metadata_pipeline.build_perturbations --config configs/metadata.yaml --dataset beauty
```

---

### Bước 4: Tạo Bảng 1 tóm tắt độ phủ (G04)
Xuất thống kê số lượng và độ phủ Category/Brand/Title cho cả 3 bộ dữ liệu (cả trên Train-base catalog và toàn bộ cohort):
```powershell
.\.venv\Scripts\python.exe -m metadata_pipeline.report --output reports/table_1_metadata.csv
```
*Tệp kết quả sẽ có tại: `reports/table_1_metadata.csv` và `reports/table_1_metadata.md`.*

---

### Bước 5: Kiểm thử và xác thực (G05 & Acceptance Criteria)
Chạy kiểm tra tự động toàn bộ ràng buộc kỹ thuật và kiểm thử tích hợp loader:
```powershell
.\.venv\Scripts\python.exe -m metadata_pipeline.validate --dataset all
```

---

## 3. Cấu trúc thư mục artifacts bàn giao

```text
artifacts/metadata/<dataset>/
├── item_map.csv               # raw_item_id <-> model_item_id (cùng cờ in_train_base)
├── metadata.csv               # model_item_id (1..N), raw_item_id, category, brand, title, has_*
├── title_embeddings.npy       # Shape [N, dim], row order = model_item_id (1..N)
├── availability_mask.csv      # model_item_id, has_category, has_brand, has_title
├── coverage.json              # Độ phủ chi tiết theo Train-base catalog & Toàn bộ cohort
├── manifest.json              # Hash nguồn dữ liệu, thông số encoder, shape, seeds
└── perturbations/             # Riêng cho Beauty (G03)
    ├── masks/                 # Các mask tỷ lệ 0.0 - 0.8 theo seed
    ├── permutations/          # Các mapping hoán vị 1-1 theo seed
    └── perturbations_manifest.json
```

