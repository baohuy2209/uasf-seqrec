"""
validate.py - Comprehensive acceptance criteria verification for metadata artifacts.

Verifications performed:
1. Item Mapping & Row Alignment (item_map.csv vs metadata.csv vs availability_mask.csv):
   - model_item_id is strictly 1..N continuous.
   - All train-base items in item_map.csv match metadata.csv model_item_id and raw_item_id exactly.
   - OOV items have model_item_id == 0 and in_train_base == False.
2. Title Embeddings Integrity (title_embeddings.npy):
   - Dimension, dtype (float32), finite values (no NaN or Inf).
   - Missing titles (has_title == False) strictly map to zero vectors.
   - Available titles (has_title == True) have non-zero norm.
3. Coverage Consistency & Ground-Truth Cross-Check (coverage.json vs metadata.csv):
   - Available + Missing == Total.
   - Coverage percentage formula: CoveragePct == round(Available / Total * 100, 2).
   - Ground-truth counts in coverage.json match exact boolean sums in metadata.csv.
   - Full cohort coverage consistency against item_map.csv.
4. Manifest Deep Integrity & Provenance (manifest.json):
   - num_catalog_items, num_total_cohort_items, num_oov_items match metadata and item_map.
   - title_embedding shape, dimension, and relative path match actual array.
   - Raw data files exist and have valid SHA-256 signatures; optional re-computation via --verify-raw-hash.
5. G05 Acceptance Test (Mask Dominance in MultimodalSequenceDataset):
   - Evaluates both Dense (2D) and Categorical (1D) modalities.
   - Uses synthetic non-zero attribute tensors with positive and negative controls.
   - Mathematically proves that availability_mask == 0 completely zeroes features without false passes.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Enable UTF-8 console output if available
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd
import torch

# Ensure src is on sys.path
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from taimma.paths import resolve_project_path
from taimma.data.split import SequenceSample, compute_file_sha256
from taimma.training.multimodal_trainer import MultimodalSequenceDataset


def is_valid_sha256(hash_str: Optional[str]) -> bool:
    """Validate 64-character hexadecimal SHA256 string."""
    if not hash_str or not isinstance(hash_str, str):
        return False
    return bool(re.fullmatch(r"^[a-fA-F0-9]{64}$", hash_str.strip()))


def validate_dataset_artifacts(
    dataset_dir: Path,
    verify_raw_hash: bool = False,
) -> Dict[str, Any]:
    """
    Validate all metadata artifacts for a given dataset directory.
    Raises AssertionError or FileNotFoundError if any check fails.
    """
    dataset_name = dataset_dir.name
    print(f"\n=======================================================")
    print(f"Validating Metadata Artifacts: {dataset_name}")
    print(f"Directory: {dataset_dir}")
    print(f"=======================================================")

    item_map_path = dataset_dir / "item_map.csv"
    meta_path = dataset_dir / "metadata.csv"
    mask_path = dataset_dir / "availability_mask.csv"
    embed_path = dataset_dir / "title_embeddings.npy"
    coverage_path = dataset_dir / "coverage.json"
    manifest_path = dataset_dir / "manifest.json"

    # 0. Check Existence of All Required Artifacts
    required_files = [
        (item_map_path, "item_map.csv"),
        (meta_path, "metadata.csv"),
        (mask_path, "availability_mask.csv"),
        (embed_path, "title_embeddings.npy"),
        (coverage_path, "coverage.json"),
        (manifest_path, "manifest.json"),
    ]
    for p, name in required_files:
        if not p.exists():
            raise FileNotFoundError(f"Missing required artifact: {p}")

    df_map = pd.read_csv(item_map_path)
    df_meta = pd.read_csv(meta_path)
    df_mask = pd.read_csv(mask_path)
    embeddings = np.load(embed_path)

    with open(coverage_path, "r", encoding="utf-8") as f:
        cov_data = json.load(f)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    n_train_base = len(df_meta)
    assert n_train_base >= 2, f"Catalog must contain at least 2 items for verification, found {n_train_base}"

    # -------------------------------------------------------------
    # 1. Item Mapping & Row Alignment (Addresses Issue 1)
    # -------------------------------------------------------------
    print(f"1. Checking Item Mapping, Row Alignment & 1..N Ordering: {n_train_base} items")

    # model_item_id in metadata must be strictly 1..N continuous
    expected_ids = np.arange(1, n_train_base + 1)
    meta_model_ids = df_meta["model_item_id"].to_numpy()
    assert (meta_model_ids == expected_ids).all(), (
        f"metadata.csv model_item_id must be strictly 1..{n_train_base} continuous"
    )

    # availability_mask.csv row alignment
    assert len(df_mask) == n_train_base, (
        f"Row count mismatch between mask ({len(df_mask)}) and metadata ({n_train_base})"
    )
    assert (df_mask["model_item_id"].to_numpy() == expected_ids).all(), (
        "availability_mask.csv model_item_id order must match metadata exactly"
    )

    # item_map.csv validation against metadata.csv
    df_train_map = df_map[df_map["in_train_base"].astype(bool)].sort_values("model_item_id")
    assert len(df_train_map) == n_train_base, (
        f"item_map.csv in_train_base item count ({len(df_train_map)}) does not match metadata ({n_train_base})"
    )
    assert (df_train_map["model_item_id"].to_numpy() == expected_ids).all(), (
        "item_map.csv train-base model_item_id must match 1..N continuous"
    )
    assert (df_train_map["raw_item_id"].astype(str).to_numpy() == df_meta["raw_item_id"].astype(str).to_numpy()).all(), (
        "CRITICAL: Raw item ID mapping mismatch between item_map.csv and metadata.csv!"
    )
    assert df_train_map["raw_item_id"].nunique() == n_train_base, (
        "raw_item_id contains duplicate values in train-base catalog!"
    )

    # Validate OOV rows in item_map.csv
    df_oov_map = df_map[~df_map["in_train_base"].astype(bool)]
    if len(df_oov_map) > 0:
        assert (df_oov_map["model_item_id"].to_numpy() == 0).all(), (
            "All OOV items in item_map.csv must have model_item_id == 0"
        )
        assert df_oov_map["raw_item_id"].notna().all(), (
            "OOV items in item_map.csv must have non-null raw_item_id"
        )

    print(f"  -> [PASS] item_map.csv, metadata.csv, and availability_mask.csv are 100% aligned ({n_train_base} train-base, {len(df_oov_map)} OOV).")

    # -------------------------------------------------------------
    # 2. Title Embeddings Integrity & Zero Placeholders (Addresses Issue 2 & 3)
    # -------------------------------------------------------------
    embed_shape = embeddings.shape
    dim = embed_shape[1]
    print(f"2. Checking Title Embeddings Integrity: shape={embed_shape}, dtype={embeddings.dtype}")
    assert embed_shape[0] == n_train_base, (
        f"Embedding rows ({embed_shape[0]}) do not match catalog items ({n_train_base})"
    )
    assert embeddings.dtype == np.float32, (
        f"Embedding dtype must be float32, got {embeddings.dtype}"
    )
    assert np.isfinite(embeddings).all(), "Title embeddings contain NaN or Inf values!"

    # Check zero placeholders for missing titles
    has_title_flags = df_mask["has_title"].astype(bool).to_numpy()
    missing_title_indices = np.where(~has_title_flags)[0]
    available_title_indices = np.where(has_title_flags)[0]

    if len(missing_title_indices) > 0:
        missing_vecs = embeddings[missing_title_indices]
        assert np.allclose(missing_vecs, 0.0), (
            "Items with has_title=False MUST have strictly zero embedding vectors!"
        )

    if len(available_title_indices) > 0:
        avail_norms = np.linalg.norm(embeddings[available_title_indices], axis=1)
        assert (avail_norms > 1e-5).all(), (
            "Items with has_title=True must have non-zero embedding vectors!"
        )

    print(f"  -> [PASS] Embeddings finite: {len(available_title_indices)} non-zero vectors, {len(missing_title_indices)} zero placeholders verified.")

    # -------------------------------------------------------------
    # 3. Coverage Arithmetic, Percentage & Ground Truth (Addresses Issue 4)
    # -------------------------------------------------------------
    print("3. Checking Coverage Arithmetic, Formulas, and Ground-Truth Cross-Check")
    assert "train_base_catalog" in cov_data, "coverage.json missing 'train_base_catalog' section"
    tb_cov = cov_data["train_base_catalog"]
    assert tb_cov["total_items"] == n_train_base, (
        f"coverage.json train_base total_items ({tb_cov['total_items']}) != metadata items ({n_train_base})"
    )

    for attr in ["category", "brand", "title"]:
        assert attr in tb_cov, f"Attribute '{attr}' missing in train_base_catalog coverage"
        c_info = tb_cov[attr]
        avail = c_info["available"]
        miss = c_info["missing"]
        cov_pct = c_info["coverage_pct"]

        # A) Arithmetic: Available + Missing == Total
        assert avail + miss == n_train_base, (
            f"Coverage sum mismatch for '{attr}': {avail} + {miss} != {n_train_base}"
        )

        # B) Percentage Formula Verification: CoveragePct == round(Available / Total * 100, 2)
        expected_pct = round(avail / max(1, n_train_base) * 100.0, 2)
        assert np.isclose(expected_pct, cov_pct, atol=1e-2), (
            f"Coverage percentage formula mismatch for '{attr}': expected {expected_pct}%, got {cov_pct}%"
        )

        # C) Ground-Truth cross-check with metadata.csv boolean flags
        gt_avail = int(df_meta[f"has_{attr}"].astype(bool).sum())
        gt_miss = n_train_base - gt_avail
        assert avail == gt_avail, (
            f"Ground truth mismatch for '{attr}': coverage.json available ({avail}) != metadata.csv true count ({gt_avail})"
        )
        assert miss == gt_miss, (
            f"Ground truth mismatch for '{attr}': coverage.json missing ({miss}) != metadata.csv false count ({gt_miss})"
        )

    # Validate all_unique_items_cohort in coverage.json
    if "all_unique_items_cohort" in cov_data:
        cohort_cov = cov_data["all_unique_items_cohort"]
        total_cohort = cohort_cov["total_items"]
        assert total_cohort == len(df_map), (
            f"Cohort total items ({total_cohort}) != item_map.csv total items ({len(df_map)})"
        )
        for attr in ["category", "brand", "title"]:
            c_info = cohort_cov[attr]
            assert c_info["available"] + c_info["missing"] == total_cohort
            exp_cohort_pct = round(c_info["available"] / max(1, total_cohort) * 100.0, 2)
            assert np.isclose(exp_cohort_pct, c_info["coverage_pct"], atol=1e-2)

    assert cov_data.get("oov_items_count", 0) == len(df_oov_map), (
        f"OOV count mismatch in coverage.json: {cov_data.get('oov_items_count')} != {len(df_oov_map)}"
    )
    print("  -> [PASS] Coverage counts, mathematical percentages, and ground-truth metadata cross-checks verified.")

    # -------------------------------------------------------------
    # 4. Manifest Deep Integrity & Provenance (Addresses Issue 5)
    # -------------------------------------------------------------
    print("4. Checking Manifest Deep Integrity & Data Provenance")
    assert manifest["dataset"].strip().lower().startswith(dataset_name.lower()) or dataset_name.lower() in manifest["dataset"].lower(), (
        f"Manifest dataset name '{manifest.get('dataset')}' does not match directory '{dataset_name}'"
    )
    assert manifest["num_catalog_items"] == n_train_base, (
        f"Manifest num_catalog_items ({manifest['num_catalog_items']}) != metadata items ({n_train_base})"
    )
    assert manifest["num_total_cohort_items"] == len(df_map), (
        f"Manifest num_total_cohort_items ({manifest['num_total_cohort_items']}) != item_map items ({len(df_map)})"
    )
    assert manifest["num_oov_items"] == len(df_oov_map), (
        f"Manifest num_oov_items ({manifest['num_oov_items']}) != actual OOV items ({len(df_oov_map)})"
    )

    # Embedding metadata in manifest
    manifest_embed = manifest["title_embedding"]
    assert manifest_embed["shape"] == list(embed_shape), (
        f"Manifest title_embedding shape {manifest_embed['shape']} != actual {list(embed_shape)}"
    )
    assert manifest_embed["dim"] == dim, (
        f"Manifest title_embedding dim ({manifest_embed['dim']}) != actual {dim}"
    )
    assert manifest_embed["path"] == embed_path.name, (
        f"Manifest embedding path '{manifest_embed['path']}' != '{embed_path.name}'"
    )

    # Raw files provenance check
    reviews_file = Path(manifest["reviews_file"])
    assert reviews_file.exists(), f"Raw reviews file in manifest does not exist: {reviews_file}"
    assert is_valid_sha256(manifest["reviews_sha256"]), (
        f"reviews_sha256 in manifest is not a valid 64-hex hash: {manifest.get('reviews_sha256')}"
    )

    if manifest.get("metadata_file"):
        meta_file = Path(manifest["metadata_file"])
        assert meta_file.exists(), f"Raw metadata file in manifest does not exist: {meta_file}"
        assert is_valid_sha256(manifest["metadata_sha256"]), (
            f"metadata_sha256 in manifest is not a valid 64-hex hash: {manifest.get('metadata_sha256')}"
        )

    # Optional deep cryptographic audit of raw files
    if verify_raw_hash:
        print("  -> Computing & verifying raw data SHA-256 checksums (deep verification enabled)...")
        act_rev_hash = compute_file_sha256(reviews_file)
        assert act_rev_hash == manifest["reviews_sha256"], (
            f"Raw reviews SHA-256 mismatch! Manifest: {manifest['reviews_sha256']}, Computed: {act_rev_hash}"
        )
        if manifest.get("metadata_file"):
            act_meta_hash = compute_file_sha256(meta_file)
            assert act_meta_hash == manifest["metadata_sha256"], (
                f"Raw metadata SHA-256 mismatch! Manifest: {manifest['metadata_sha256']}, Computed: {act_meta_hash}"
            )
        print("  -> [PASS] Cryptographic SHA-256 checksums strictly verified for raw files.")
    else:
        print("  -> [PASS] Raw file existence and 64-hex SHA-256 signature formats verified.")

    # -------------------------------------------------------------
    # 5. G05 Acceptance Test: Dual Modality Mask Dominance (Addresses Issues 2 & 3)
    # -------------------------------------------------------------
    print("5. G05 Acceptance Test: Dual Modality Mask Dominance in Model Loader")
    # Dynamically select test items without hardcoding or risking out-of-bounds slicing
    k_test = min(n_train_base, 5)

    # Synthetic non-zero tensors guarantee test cannot pass by accidental zeroes:
    # 1. 2D Dense Embedding (Title): distinct non-zero vectors for all test items
    dummy_title_attrs = torch.empty((k_test, dim), dtype=torch.float32)
    for i in range(k_test):
        dummy_title_attrs[i] = torch.full((dim,), fill_value=float(i + 10.0))

    # 2. 1D Categorical Attribute (Category): distinct non-zero integer IDs for all test items
    dummy_cat_attrs = torch.tensor([idx + 100 for idx in range(k_test)], dtype=torch.long)

    # Masks for test items:
    # For Title: Item 1 has mask=0 (disabled), Item 2 has mask=1 (enabled)
    title_mask = torch.ones(k_test, dtype=torch.float32)
    title_mask[0] = 0.0  # Item 1 disabled

    # For Category: Item 1 has mask=1 (enabled), Item 2 has mask=0 (disabled)
    cat_mask = torch.ones(k_test, dtype=torch.float32)
    cat_mask[1] = 0.0  # Item 2 disabled

    # History: [0, 1, 2] -> index 0 is padding (0), index 1 is item 1, index 2 is item 2
    sample = SequenceSample(
        user_idx=0,
        history=[0, 1, 2],
        target=2,
        history_len=2,
    )

    ds = MultimodalSequenceDataset(
        samples=[sample],
        catalog_source_attrs={"title": dummy_title_attrs, "category": dummy_cat_attrs},
        catalog_availability_masks={"title": title_mask, "category": cat_mask},
    )

    batch_item = ds[0]
    source_seqs = batch_item["source_seqs"]
    title_seq = source_seqs["title"]      # Shape: (3, dim)
    cat_seq = source_seqs["category"]     # Shape: (3,)

    # Verification A: Padding position (item 0) MUST produce zeros
    assert torch.allclose(title_seq[0], torch.zeros(dim)), "Padding position did not produce zero title vector"
    assert cat_seq[0].item() == 0, "Padding position did not produce zero category"

    # Verification B: Title Modality (Dense 2D)
    # Item 1 (mask=0): Negative control -> MUST be strictly zero vector
    assert torch.allclose(title_seq[1], torch.zeros(dim)), (
        "CRITICAL: Title mask=0 did NOT block features in MultimodalSequenceDataset!"
    )
    # Item 2 (mask=1): Positive control -> MUST retain the non-zero attribute vector exactly
    assert torch.allclose(title_seq[2], dummy_title_attrs[1]), (
        "CRITICAL: Title mask=1 did NOT preserve the original attribute vector!"
    )
    assert not torch.allclose(title_seq[2], torch.zeros(dim)), (
        "Positive control failed: Item 2 title is unexpectedly zero"
    )

    # Verification C: Category Modality (Categorical 1D)
    # Item 1 (mask=1): Positive control -> MUST retain the category index exactly (100)
    assert cat_seq[1].item() == dummy_cat_attrs[0].item(), (
        "CRITICAL: Category mask=1 did NOT preserve the original category index!"
    )
    # Item 2 (mask=0): Negative control -> MUST be strictly zero
    assert cat_seq[2].item() == 0, (
        "CRITICAL: Category mask=0 did NOT block categorical feature in MultimodalSequenceDataset!"
    )

    print("  -> [PASS] G05 Certified: Dual-modality positive & negative controls prove mask dominance 100%.")

    print(f"\nAll verification checks PASSED for {dataset_name}!\n")
    return {
        "status": "PASS",
        "dataset": dataset_name,
        "items": n_train_base,
        "oov_items": len(df_oov_map),
        "embedding_shape": list(embed_shape),
        "coverage": tb_cov,
    }


def resolve_metadata_base(path_str: str) -> Path:
    """Resolve directory path intelligently from either current dir or repo root."""
    candidate = Path(path_str).expanduser()
    if candidate.exists():
        return candidate.resolve()
    resolved = resolve_project_path(path_str)
    return resolved


def main():
    parser = argparse.ArgumentParser(description="Validate metadata artifacts against acceptance criteria.")
    parser.add_argument(
        "--metadata_dir",
        default="artifacts/metadata",
        help="Base metadata artifacts directory (e.g. artifacts/metadata)",
    )
    parser.add_argument(
        "--dataset",
        choices=["beauty", "sports", "toys", "all"],
        default="all",
        help="Target dataset to validate (default: all)",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Permit missing datasets without failing (default: strict mode fails if any expected dataset is missing)",
    )
    parser.add_argument(
        "--verify-raw-hash",
        action="store_true",
        help="Compute and verify cryptographic SHA-256 checksums of raw input files against manifest",
    )
    args = parser.parse_args()

    meta_base = resolve_metadata_base(args.metadata_dir)
    target_keys = ["beauty", "sports", "toys"] if args.dataset == "all" else [args.dataset]

    print(f"=== Metadata Artifacts Acceptance Suite ===")
    print(f"Base Directory: {meta_base}")
    print(f"Target Datasets: {target_keys}")
    print(f"Mode: {'Permissive (--allow-missing)' if args.allow_missing else 'Strict (All Required)'}")
    print(f"Verify Raw SHA-256: {args.verify_raw_hash}\n")

    summary_results: Dict[str, Dict[str, Any]] = {}
    any_failed = False

    for key in target_keys:
        ds_path = meta_base / key.lower()
        if not ds_path.exists():
            if args.allow_missing:
                print(f"[WARN] Dataset directory not found: {ds_path} (Skipped due to --allow-missing)")
                summary_results[key] = {"status": "SKIPPED", "reason": "Directory not found"}
            else:
                print(f"[FAIL] Required dataset directory not found: {ds_path}")
                summary_results[key] = {"status": "FAIL", "reason": f"Directory missing: {ds_path}"}
                any_failed = True
            continue

        try:
            res = validate_dataset_artifacts(ds_path, verify_raw_hash=args.verify_raw_hash)
            summary_results[key] = res
        except Exception as exc:
            print(f"\n[FAIL] Verification error for {key}: {exc}")
            summary_results[key] = {"status": "FAIL", "error": str(exc)}
            any_failed = True

    # Print Final Acceptance Report Table
    print("\n" + "=" * 65)
    print("FINAL ACCEPTANCE VERIFICATION SUMMARY")
    print("=" * 65)
    print(f"{'Dataset':<15} | {'Status':<10} | {'Details'}")
    print("-" * 65)

    passed_count = 0
    for key in target_keys:
        info = summary_results.get(key, {"status": "UNKNOWN"})
        status = info.get("status", "FAIL")
        if status == "PASS":
            passed_count += 1
            details = f"Items: {info.get('items')}, Dim: {info.get('embedding_shape', [0, 0])[1]}, OOV: {info.get('oov_items')}"
        elif status == "SKIPPED":
            details = info.get("reason", "Skipped")
        else:
            details = info.get("error", info.get("reason", "Unknown failure"))
        print(f"{key:<15} | {status:<10} | {details}")

    print("=" * 65)

    if any_failed or passed_count == 0:
        print(f"\n[RESULT] [FAILED] ACCEPTANCE SUITE FAILED ({passed_count}/{len(target_keys)} passed).")
        sys.exit(1)
    else:
        print(f"\n[RESULT] [SUCCESS] ALL ACCEPTANCE CHECKS PASSED ({passed_count}/{len(target_keys)} passed)!")
        sys.exit(0)


if __name__ == "__main__":
    main()
