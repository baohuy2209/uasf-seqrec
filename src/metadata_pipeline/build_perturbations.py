"""
build_perturbations.py - Generate synthetic missing metadata masks (0% to 80%)
and deterministic metadata permutations for robustness evaluation (Beauty & others).
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import yaml

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from taimma.paths import resolve_project_path
from .contracts import SOURCES, check_availability, flags, read_table, require


def validate_options(table, ratios, seeds, sources):
    check_availability(table)
    require(bool(sources) and set(sources).issubset(SOURCES) and len(set(sources)) == len(sources), "Invalid sources")
    require(bool(seeds) and len(set(seeds)) == len(seeds) and all(isinstance(s, int) and s >= 0 for s in seeds), "Invalid seeds")
    require(bool(ratios) and len(set(ratios)) == len(ratios) and all(0 <= r <= 1 for r in ratios), "Invalid ratios")
    require(len({f"{r:.2f}" for r in ratios}) == len(ratios), "Ratios collide in output filenames")


def generate_synthetic_masks(
    availability_df: pd.DataFrame,
    ratios: List[float],
    seeds: List[int],
    sources: List[str],
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Simulate missing metadata by randomly masking available sources at specified ratios.
    CRITICAL INVARIANT:
      - Only turns 1 (available) -> 0 (missing).
      - NEVER turns 0 (originally missing) -> 1.
      - Fully reproducible by (ratio, seed, source).
    """
    validate_options(availability_df, ratios, seeds, sources)
    masks_dir = output_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    manifest_records = []
    num_items = len(availability_df)

    for ratio in ratios:
        for seed in seeds:
            # We can generate joint mask or single-source masks
            for source in sources:
                col_name = f"has_{source}"
                if col_name not in availability_df.columns:
                    continue

                rng = np.random.RandomState(seed + int(ratio * 1000))
                orig_status = flags(availability_df[col_name])
                avail_indices = np.where(orig_status)[0]

                # Number of items to drop out
                num_to_drop = int(round(len(avail_indices) * ratio))
                drop_indices = rng.choice(avail_indices, size=num_to_drop, replace=False) if num_to_drop > 0 else np.array([], dtype=int)

                simulated_mask = orig_status.copy()
                simulated_mask[drop_indices] = False

                filename = f"mask_{source}_ratio_{ratio:.2f}_seed_{seed}.csv"
                file_path = masks_dir / filename

                df_out = pd.DataFrame({
                    "model_item_id": availability_df["model_item_id"],
                    "original_available": orig_status,
                    "simulated_available": simulated_mask,
                    "is_dropped": False,
                })
                df_out.loc[drop_indices, "is_dropped"] = True
                df_out.to_csv(file_path, index=False, lineterminator="\n")

                actual_dropped_ratio = len(drop_indices) / max(1, len(avail_indices))
                manifest_records.append({
                    "file": filename,
                    "source": source,
                    "target_ratio": ratio,
                    "actual_dropped_ratio": round(actual_dropped_ratio, 4),
                    "seed": seed,
                    "originally_available": int(len(avail_indices)),
                    "simulated_available": int(simulated_mask.sum()),
                })

    return {
        "num_masks_generated": len(manifest_records),
        "ratios": ratios,
        "seeds": seeds,
        "sources": sources,
        "manifest": manifest_records,
    }


def generate_permutations(
    metadata_df: pd.DataFrame,
    seeds: List[int],
    sources: List[str],
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Generate deterministic 1-1 item metadata permutations for inference robustness tests.
    CRITICAL INVARIANT (Option B - Fixed-mask Permutation):
      - Permutes metadata attributes ONLY among items that ACTUALLY possess that attribute.
      - Items originally missing (has_source=False) remain strictly missing at the same position.
      - Preserves both marginal distribution and exact missingness structure.
      - Does NOT alter user interaction sequences, target items, or model catalog IDs.
    """
    validate_options(metadata_df, [0.0], seeds, sources)
    perm_dir = output_dir / "permutations"
    perm_dir.mkdir(parents=True, exist_ok=True)

    manifest_records = []
    num_items = len(metadata_df)
    model_ids = metadata_df["model_item_id"].values

    for seed in seeds:
        for source in sources:
            flag_col = f"has_{source}"
            if flag_col in metadata_df.columns:
                avail_mask = flags(metadata_df[flag_col])
            else:
                avail_mask = np.ones(num_items, dtype=bool)

            avail_indices = np.where(avail_mask)[0]
            num_avail = len(avail_indices)

            # Permuted indices initialized to identity
            permuted_indices = np.arange(num_items)

            if num_avail > 1:
                rng = np.random.RandomState(seed)
                shuffled_avail_pos = rng.permutation(num_avail)
                # Map available items strictly among themselves
                permuted_indices[avail_indices] = avail_indices[shuffled_avail_pos]

            filename = f"perm_{source}_seed_{seed}.csv"
            file_path = perm_dir / filename

            # Map model_item_id -> permuted_source_item_id
            df_perm = pd.DataFrame({
                "model_item_id": model_ids,
                "permuted_source_item_id": model_ids[permuted_indices],
                "source": source,
                "seed": seed,
                "is_available": avail_mask,
            })
            df_perm.to_csv(file_path, index=False, lineterminator="\n")

            manifest_records.append({
                "file": filename,
                "source": source,
                "seed": seed,
                "total_items": num_items,
                "permuted_available_items": int(num_avail),
                "fixed_missing_items": int(num_items - num_avail),
                "is_bijective": bool(len(np.unique(permuted_indices)) == num_items),
                "policy": "fixed_missingness_structure",
            })

    return {
        "num_permutations_generated": len(manifest_records),
        "seeds": seeds,
        "sources": sources,
        "policy": "fixed_missingness_structure",
        "manifest": manifest_records,
    }


def process_dataset_perturbations(
    dataset_name: str,
    metadata_dir: Path,
    ratios: List[float],
    seeds: List[int],
    sources: List[str],
) -> Dict[str, Any]:
    ds_dir = metadata_dir / dataset_name.lower()
    avail_mask_path = ds_dir / "availability_mask.csv"
    meta_path = ds_dir / "metadata.csv"

    if not avail_mask_path.exists():
        raise FileNotFoundError(f"Missing availability mask at: {avail_mask_path}. Run prepare.py first.")

    avail_df = read_table(avail_mask_path)
    meta_df = read_table(meta_path)
    require(np.array_equal(avail_df.model_item_id, meta_df.model_item_id), "Metadata/mask ID mismatch")
    for source in sources:
        require(np.array_equal(flags(avail_df[f"has_{source}"]), flags(meta_df[f"has_{source}"])),
                f"{source}: metadata/mask mismatch")

    pert_out_dir = ds_dir / "perturbations"
    pert_out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Generating perturbations for {dataset_name} ---")
    print(f"Generating synthetic missing metadata masks for {dataset_name}...")
    mask_res = generate_synthetic_masks(avail_df, ratios, seeds, sources, pert_out_dir)

    print(f"Generating metadata permutations for {dataset_name} (Option B: Fixed-mask)...")
    perm_res = generate_permutations(meta_df, seeds, sources, pert_out_dir)

    manifest_path = pert_out_dir / "perturbations_manifest.json"
    manifest_data = {
        "dataset": dataset_name,
        "masks": mask_res,
        "permutations": perm_res,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    print(f"[{dataset_name}] Successfully generated {mask_res['num_masks_generated']} masks and {perm_res['num_permutations_generated']} permutations.")
    return manifest_data


def main():
    parser = argparse.ArgumentParser(description="Build synthetic perturbation masks and permutations.")
    parser.add_argument("--config", default="configs/metadata.yaml", help="Metadata config path")
    parser.add_argument("--dataset", default="all", choices=["beauty", "sports", "toys", "all"], help="Target dataset (default: all)")
    parser.add_argument("--metadata_dir", default="artifacts/metadata", help="Metadata artifacts base directory")
    args = parser.parse_args()

    with open(resolve_project_path(args.config), "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    pert_cfg = config.get("perturbations", {})
    ratios = pert_cfg.get("missing_ratios", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    seeds = pert_cfg.get("seeds", [42, 43, 44])
    sources = pert_cfg.get("sources", ["category", "brand", "title"])

    meta_base = resolve_project_path(args.metadata_dir)
    target_datasets = ["beauty", "sports", "toys"] if args.dataset == "all" else [args.dataset]

    for ds in target_datasets:
        process_dataset_perturbations(ds, meta_base, ratios, seeds, sources)

    print("\nPerturbations generation completed successfully.")


if __name__ == "__main__":
    main()

