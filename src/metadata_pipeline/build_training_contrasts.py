"""
build_training_contrasts.py - Generate training-time contrast artifacts:
1. Shuffled attributes (Option B: Fixed-mask permutation per source x seed).
2. Random attributes (Marginal distribution sampling for Category/Brand, Gaussian unit-norm for Title).

Strict Invariants:
- Zero Leakage: Derived strictly from Train-base catalog (1..N).
- Fixed-mask missingness: Items with has_source=False remain strictly missing (0/placeholder).
- Deterministic reproducibility: Guaranteed by RNG seeds (42, 43, 44).
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from metadata_pipeline.contracts import SOURCES, check_availability, flags, read_json, read_table, require, write_json
from taimma.paths import resolve_project_path


def compute_file_sha256(filepath: Path) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


def generate_shuffled_contrasts(
    meta_df: pd.DataFrame,
    avail_df: pd.DataFrame,
    seeds: List[int],
    output_dir: Path,
) -> Dict[str, Any]:
    shuffled_dir = output_dir / "shuffled"
    shuffled_dir.mkdir(parents=True, exist_ok=True)

    manifest_records = {}
    num_items = len(meta_df)
    model_ids = meta_df["model_item_id"].to_numpy()

    for seed in seeds:
        for source in SOURCES:
            col_name = f"has_{source}"
            avail_mask = flags(avail_df[col_name])
            avail_indices = np.flatnonzero(avail_mask)
            num_avail = len(avail_indices)

            permuted_indices = np.arange(num_items)
            if num_avail > 1:
                rng = np.random.RandomState(seed)
                shuffled_avail_pos = rng.permutation(num_avail)
                permuted_indices[avail_indices] = avail_indices[shuffled_avail_pos]

            donor_ids = model_ids[permuted_indices]

            # Invariant verifications
            require(np.array_equal(donor_ids[~avail_mask], model_ids[~avail_mask]), "Missing items must remain fixed")
            require(len(np.unique(donor_ids)) == num_items, "Permutation must be bijective")

            filename = f"perm_{source}_seed_{seed}.csv"
            out_path = shuffled_dir / filename

            df_out = pd.DataFrame({
                "model_item_id": model_ids,
                "permuted_source_item_id": donor_ids,
                "source": source,
                "seed": seed,
                "is_available": avail_mask,
            })
            df_out.to_csv(out_path, index=False)

            file_key = f"{source}_seed_{seed}"
            manifest_records[file_key] = {
                "file": str(out_path.name),
                "path": str(out_path.as_posix()),
                "source": source,
                "seed": seed,
                "total_items": num_items,
                "permuted_available_items": int(num_avail),
                "fixed_missing_items": int(num_items - num_avail),
                "sha256": compute_file_sha256(out_path),
            }

    return manifest_records


def generate_random_contrasts(
    meta_df: pd.DataFrame,
    avail_df: pd.DataFrame,
    seeds: List[int],
    output_dir: Path,
    title_dim: int = 384,
) -> Dict[str, Any]:
    random_dir = output_dir / "random"
    random_dir.mkdir(parents=True, exist_ok=True)

    manifest_records = {}
    num_items = len(meta_df)
    model_ids = meta_df["model_item_id"].to_numpy()

    # 1. Category Marginal Pool
    cat_avail_mask = flags(avail_df["has_category"])
    cat_raw_labels = [json.loads(val) for val in meta_df["category_indices"]]
    # Pool of all valid single tokens assigned to available items
    cat_pool = []
    for idx in np.flatnonzero(cat_avail_mask):
        cat_pool.extend([tok for tok in cat_raw_labels[idx] if tok > 0])
    require(len(cat_pool) > 0, "Category pool must not be empty")

    # 2. Brand Marginal Pool
    brand_avail_mask = flags(avail_df["has_brand"])
    brand_raw = meta_df["brand_idx"].to_numpy(dtype=np.int64)
    brand_pool = brand_raw[brand_avail_mask & (brand_raw > 0)].tolist()
    require(len(brand_pool) > 0, "Brand pool must not be empty")

    # 3. Title Mask
    title_avail_mask = flags(avail_df["has_title"])

    for seed in seeds:
        rng = np.random.RandomState(seed)

        # A. Random Category (Marginal Sampling)
        random_cat_indices = []
        for i in range(num_items):
            if cat_avail_mask[i]:
                orig_len = max(1, len(cat_raw_labels[i]))
                sampled = rng.choice(cat_pool, size=orig_len, replace=True).tolist()
                random_cat_indices.append(json.dumps(sampled))
            else:
                random_cat_indices.append("[]")

        cat_out_path = random_dir / f"random_category_seed_{seed}.csv"
        df_cat = pd.DataFrame({
            "model_item_id": model_ids,
            "random_category_indices": random_cat_indices,
            "has_category": cat_avail_mask,
        })
        df_cat.to_csv(cat_out_path, index=False)

        manifest_records[f"category_seed_{seed}"] = {
            "file": cat_out_path.name,
            "source": "category",
            "seed": seed,
            "sampling": "marginal_with_replacement",
            "total_items": num_items,
            "available_items": int(cat_avail_mask.sum()),
            "missing_items": int((~cat_avail_mask).sum()),
            "sha256": compute_file_sha256(cat_out_path),
        }

        # B. Random Brand (Marginal Sampling)
        random_brand_arr = np.zeros(num_items, dtype=np.int64)
        num_brand_avail = int(brand_avail_mask.sum())
        if num_brand_avail > 0:
            random_brand_arr[brand_avail_mask] = rng.choice(brand_pool, size=num_brand_avail, replace=True)

        brand_out_path = random_dir / f"random_brand_seed_{seed}.csv"
        df_brand = pd.DataFrame({
            "model_item_id": model_ids,
            "random_brand_idx": random_brand_arr,
            "has_brand": brand_avail_mask,
        })
        df_brand.to_csv(brand_out_path, index=False)

        manifest_records[f"brand_seed_{seed}"] = {
            "file": brand_out_path.name,
            "source": "brand",
            "seed": seed,
            "sampling": "marginal_with_replacement",
            "total_items": num_items,
            "available_items": num_brand_avail,
            "missing_items": int((~brand_avail_mask).sum()),
            "sha256": compute_file_sha256(brand_out_path),
        }

        # C. Random Title (Gaussian Unit-normalized)
        random_title_arr = np.zeros((num_items, title_dim), dtype=np.float32)
        num_title_avail = int(title_avail_mask.sum())
        if num_title_avail > 0:
            # Gaussian N(0, 1) vectors
            raw_vecs = rng.randn(num_title_avail, title_dim).astype(np.float32)
            norms = np.linalg.norm(raw_vecs, axis=1, keepdims=True) + 1e-12
            random_title_arr[title_avail_mask] = raw_vecs / norms

        title_out_path = random_dir / f"random_title_seed_{seed}.npy"
        np.save(title_out_path, random_title_arr)

        manifest_records[f"title_seed_{seed}"] = {
            "file": title_out_path.name,
            "source": "title",
            "seed": seed,
            "sampling": "gaussian_unit_normalized",
            "dim": title_dim,
            "total_items": num_items,
            "available_items": num_title_avail,
            "missing_items": int((~title_avail_mask).sum()),
            "sha256": compute_file_sha256(title_out_path),
        }

    return manifest_records


def build_training_contrasts_for_dataset(
    dataset_name: str = "beauty",
    base_metadata_dir: str = "artifacts/metadata",
    seeds: List[int] = [42, 43, 44],
    output_dir=None,
) -> Dict[str, Any]:
    dataset_dir = resolve_project_path(base_metadata_dir) / dataset_name.lower()
    meta_path = dataset_dir / "metadata.csv"
    avail_path = dataset_dir / "availability_mask.csv"

    require(meta_path.exists(), f"Missing metadata at {meta_path}")
    require(avail_path.exists(), f"Missing availability mask at {avail_path}")

    meta_df = read_table(meta_path)
    avail_df = read_table(avail_path)
    check_availability(meta_df)
    check_availability(avail_df)
    require(len(meta_df) == len(avail_df), "Row count mismatch between metadata and availability mask")

    output_dir = Path(output_dir) if output_dir is not None else dataset_dir / "training_contrasts"
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / 'manifest.json').exists():
        raise FileExistsError('Preserve existing contrasts; use a new output directory')
    title_shape = np.load(dataset_dir/'title_embeddings.npy', mmap_mode='r', allow_pickle=False).shape
    require(len(title_shape)==2 and title_shape[0]==len(meta_df), 'Title/catalog alignment mismatch')

    print(f"\n=======================================================")
    print(f"Building Training Contrasts for: {dataset_name} ({len(meta_df)} items)")
    print(f"=======================================================")

    print("- Generating shuffled attribute contrasts (Option B)...")
    shuffled_manifest = generate_shuffled_contrasts(meta_df, avail_df, seeds, output_dir)

    print("- Generating random attribute contrasts (Marginal + Gaussian)...")
    random_manifest = generate_random_contrasts(meta_df, avail_df, seeds, output_dir, title_dim=title_shape[1])

    manifest_data = {
        "dataset": dataset_name,
        "num_items": len(meta_df),
        "seeds": seeds,
        "sources": list(SOURCES),
        "policy": {
            "shuffled": "fixed_mask_permutation_option_b",
            "random_categorical": "marginal_sampling_with_replacement",
            "random_dense": f"gaussian_unit_normalized_d{title_shape[1]}",
        },
        "shuffled": shuffled_manifest,
        "random": random_manifest,
        "input_sha256": {p.name: compute_file_sha256(p) for p in
                         [meta_path, avail_path, dataset_dir/'title_embeddings.npy']},
    }

    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, manifest_data)
    print(f"Training contrasts successfully written to: {output_dir}")
    print(f"Manifest written to: {manifest_path}")

    return manifest_data


def main():
    parser = argparse.ArgumentParser(description="Generate training contrasts (shuffled and random) for SISR models.")
    parser.add_argument("--dataset", default="beauty", choices=["beauty", "sports", "toys", "all"])
    parser.add_argument("--metadata_dir", default="artifacts/metadata")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--output_dir", default=None, help="New artifact directory; existing manifests are never overwritten")
    args = parser.parse_args()

    targets = ["beauty", "sports", "toys"] if args.dataset == "all" else [args.dataset]
    for ds in targets:
        destination = Path(args.output_dir)/ds if args.output_dir and args.dataset=='all' else args.output_dir
        build_training_contrasts_for_dataset(ds, base_metadata_dir=args.metadata_dir, seeds=args.seeds,
                                            output_dir=destination)


if __name__ == "__main__":
    main()

