# Metadata pipeline

Prepare Category (all non-root labels), Brand, and real SentenceTransformer Title
vectors for Amazon 2014 Beauty, Sports and Toys. Catalog IDs come only from
Train-base and stay 1..N; array row i-1 represents model item i. Missing values
have explicit availability flags. Vocabularies are fitted only on Train-base.

## Install and obtain sources

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe scripts/download_amazon2014.py --dataset Beauty
.\.venv\Scripts\python.exe scripts/download_amazon2014.py --dataset Sports_and_Outdoors
.\.venv\Scripts\python.exe scripts/download_amazon2014.py --dataset Toys_and_Games
```

The downloader defaults to `data/raw`. Existing uncompressed reviews may live
in category subdirectories. Both reviews and metadata must exist. Encoder errors,
missing metadata, and mismatched expected statistics stop preparation. No random
Title replacement is permitted.

## Build and validate

```powershell
.\.venv\Scripts\python.exe -m metadata_pipeline.prepare --dataset all
.\.venv\Scripts\python.exe -m metadata_pipeline.build_perturbations --dataset all
.\.venv\Scripts\python.exe -m metadata_pipeline.validate --dataset all --verify-raw-hash
.\.venv\Scripts\python.exe -m metadata_pipeline.report
```

Choose a fresh `--output_dir` for preparation. `--overwrite` explicitly replaces
an existing dataset only after all new metadata and embeddings have been built;
old perturbations are discarded and must be regenerated. Never overwrite
artifacts associated with completed experiment runs.

For a relocated checkout, pass `--raw_root data/raw` to validation. Raw input
checksums are preserved and compared to the selected files. Compressed and
uncompressed review dumps have different byte hashes: use the same source format
as the manifest, or regenerate into a new output directory rather than editing
checksums. Default validation checks raw file availability; `--verify-raw-hash`
also checks SHA256 and reconstructs the catalog and metadata from raw inputs.

If only the handoff artifacts are available, explicitly use `--artifacts-only`.
Its result says `NOT_CHECKED_ARTIFACTS_ONLY` for source provenance. It still
checks artifact hashes, full coverage, vocabularies, all masks/permutations and
the actual sequence loader. This mode is not a raw-source certification.

## Schema version 2

- `metadata.csv`: ordered Train-base catalog; `categories` and `category_indices`
  are JSON lists. `brand_idx` is an integer. Index zero means padding/missing.
- `cohort_metadata.csv`: metadata and availability for all unique cohort items,
  including OOV items, so full-cohort coverage can be independently counted.
- `item_map.csv`, `availability_mask.csv`, `title_embeddings.npy`: aligned IDs,
  explicit source flags and float32 vectors. Missing Title rows are exactly zero.
- `category_vocab.json`, `brand_vocab.json`: sorted Train-base vocabularies.
- `coverage.json`, `statistics.json`: measured coverage and full Table 1 inputs.
- `manifest.json`: schema, preprocessing/category policy, source and output hashes.
- `perturbations`: per-source masks for 0–80% of originally available metadata,
  three seeds, and permutations among available items with natural missing rows
  fixed. The ratio is additional missingness, not total natural plus synthetic
  missingness. Training perturbations require a separate experiment invocation;
  these mappings do not modify interactions or targets.

Legacy single-category artifacts must be regenerated from raw metadata. Labels
that were discarded cannot be recovered from the old CSV. Regenerate residual
models using Category after this schema change; do not reuse checkpoints whose
category vocabulary was fitted to the former single-label representation.

## Load for inference

```python
from metadata_pipeline.loader import load_catalog
attrs, masks = load_catalog("artifacts/metadata/beauty")
# Pass the same attrs/masks to MultimodalSequenceDataset and the model's catalog
# arguments so the history and candidate sides use identical perturbations.
attrs, masks = load_catalog(
    "artifacts/metadata/beauty",
    mask_files={"brand": "artifacts/metadata/beauty/perturbations/masks/mask_brand_ratio_0.80_seed_42.csv"},
    permutation_files={"title": "artifacts/metadata/beauty/perturbations/permutations/perm_title_seed_42.csv"},
)
```

Category tensors have shape `[N,L]` and dtype long; Brand `[N]` long; Title
`[N,D]` float32. Padding is added by MultimodalSequenceDataset, not by shifting
catalog arrays. Availability masks must also be passed to the residual model
so masked candidate scores are disabled. ID-only checkpoint/vocabulary IDs
remain unchanged.

Regression checks:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_metadata_pipeline.py
```
