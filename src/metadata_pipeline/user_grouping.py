"""
user_grouping.py - History-based Zero-Leakage User Grouping for SISR Analysis (Beauty & others).

Strict Methodology (Supervisor Memo 03/11/2026, Section 4.1):
1. Catalog Head/Tail Partition:
   - Derived strictly from interaction counts in Train-base events (positions 0 .. n-4).
   - Head items = Top 20% most frequent catalog items.
   - Tail items = Remaining 80% catalog items.
2. User History Statistics (NO TARGET LEAKAGE):
   - Validation context: history strictly before validation target (positions 0 .. n-3).
   - Test context: history strictly before test target (positions 0 .. n-2).
   - history_len = unpadded length of user interaction history.
   - tail_ratio = (number of items in history belonging to Tail) / history_len.
3. 10 Balanced Groups (5 length quantiles x 2 tail bins):
   - Length quintiles Q1..Q5.
   - Tail binary split (Median split: low_tail vs high_tail).
   - group_id in 0..9.
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from metadata_pipeline.contracts import read_json, require, write_json
from taimma.paths import resolve_project_path
from taimma.data.ingestion import Amazon2014Ingestor
from taimma.data.split import SequenceSplitter


def compute_head_tail_partition(train_base_events: List[int], num_catalog_items: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Partition catalog items (1..N) into Head (top 20%) and Tail (bottom 80%)
    based strictly on empirical interaction frequencies in Train-base events.
    """
    counts = Counter(train_base_events)
    # Item frequency array for items 1..num_catalog_items (0-indexed array where index i is item i+1)
    freqs = np.zeros(num_catalog_items, dtype=np.int64)
    for itm_id, cnt in counts.items():
        if 1 <= itm_id <= num_catalog_items:
            freqs[itm_id - 1] = cnt

    # Sort descending by frequency
    sorted_item_indices = np.argsort(-freqs)  # 0-based indices
    num_head = int(round(num_catalog_items * 0.20))
    head_indices = set(sorted_item_indices[:num_head] + 1)  # 1-based item IDs

    is_tail = np.ones(num_catalog_items + 1, dtype=bool)
    is_tail[0] = False  # padding is neither
    for h_id in head_indices:
        is_tail[h_id] = False

    head_events = sum(freqs[h - 1] for h in head_indices)
    total_events = len(train_base_events)
    stats = {
        "num_catalog_items": num_catalog_items,
        "num_head_items": num_head,
        "num_tail_items": num_catalog_items - num_head,
        "head_item_ratio": round(num_head / num_catalog_items, 4),
        "total_train_base_events": total_events,
        "head_events_count": int(head_events),
        "head_events_pct": round(head_events / max(1, total_events) * 100.0, 2),
        "tail_events_count": int(total_events - head_events),
        "tail_events_pct": round((total_events - head_events) / max(1, total_events) * 100.0, 2),
    }
    return is_tail, stats


def assign_quantile_and_median_bins(
    lengths: np.ndarray,
    tail_ratios: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Computes:
    - length_quantile: 1..5 (quintiles)
    - tail_bin: 0 (low_tail), 1 (high_tail) using median split
    - group_id: 0..9 (length_quantile - 1) * 2 + tail_bin
    """
    n_users = len(lengths)

    # 1. Length Quantiles (1..5)
    # Using rank-based quintiles to handle duplicate ties gracefully
    length_ranks = pd.Series(lengths).rank(method="first")
    length_quintiles = pd.qcut(length_ranks, q=5, labels=[1, 2, 3, 4, 5]).to_numpy(dtype=np.int64)

    # 2. Tail Ratio Median Split (0: low_tail, 1: high_tail)
    # Perform median split conditionally within each length quintile for optimal balance
    tail_bins = np.zeros(n_users, dtype=np.int64)
    cutoffs = {}

    for q in range(1, 6):
        mask = (length_quintiles == q)
        q_ratios = tail_ratios[mask]
        median_val = float(np.median(q_ratios))
        cutoffs[f"Q{q}"] = {
            "num_users": int(mask.sum()),
            "median_tail_ratio": round(median_val, 4),
            "mean_tail_ratio": round(float(np.mean(q_ratios)), 4),
            "min_len": int(np.min(lengths[mask])),
            "max_len": int(np.max(lengths[mask])),
        }
        # Items strictly <= median are low_tail (0), > median are high_tail (1)
        tail_bins[mask] = (q_ratios > median_val).astype(np.int64)

    # 3. Overall group_id in 0..9
    group_ids = (length_quintiles - 1) * 2 + tail_bins

    binning_summary = {
        "num_groups": 10,
        "quintile_cutoffs": cutoffs,
        "group_counts": {int(g): int((group_ids == g).sum()) for g in range(10)},
    }
    return length_quintiles, tail_bins, group_ids, binning_summary


def generate_user_groups_for_dataset(
    dataset_name: str = "beauty",
    raw_root: str = "data/raw",
    metadata_dir: str = "artifacts/metadata",
) -> Dict[str, Any]:
    print(f"\n=======================================================")
    print(f"Generating History-based User Groups for: {dataset_name}")
    print(f"=======================================================")

    raw_path = resolve_project_path(raw_root)
    meta_path = resolve_project_path(metadata_dir) / dataset_name.lower()
    require(meta_path.exists(), f"Metadata directory missing: {meta_path}")

    # Build sequence splits using locked project ingestor
    ingestor = Amazon2014Ingestor(str(raw_path), dataset_key=dataset_name.capitalize())
    sequences, _ = ingestor.build_interaction_sequences(min_interactions=5)
    splitter = SequenceSplitter(max_seq_len=50, min_seq_len=5)
    split_data = splitter.split(sequences)

    num_users = split_data.num_users
    num_catalog = split_data.num_items
    print(f"Loaded {num_users} users and {num_catalog} catalog items.")

    # 1. Derive Head/Tail partition from Train-base events
    is_tail, catalog_stats = compute_head_tail_partition(split_data.train_base_events, num_catalog)
    print(f"Catalog Head (top 20%): {catalog_stats['num_head_items']} items ({catalog_stats['head_events_pct']}% train-base events)")
    print(f"Catalog Tail (bottom 80%): {catalog_stats['num_tail_items']} items ({catalog_stats['tail_events_pct']}% train-base events)")

    # 2. Extract History Features for each user (0..num_users-1)
    val_samples = split_data.val_samples
    test_samples = split_data.test_samples
    require(len(val_samples) == num_users, f"val_samples mismatch: {len(val_samples)} vs {num_users}")
    require(len(test_samples) == num_users, f"test_samples mismatch: {len(test_samples)} vs {num_users}")

    user_rows = []
    val_lens = np.zeros(num_users, dtype=np.int64)
    val_tail_ratios = np.zeros(num_users, dtype=np.float64)
    test_lens = np.zeros(num_users, dtype=np.int64)
    test_tail_ratios = np.zeros(num_users, dtype=np.float64)

    for u_idx in range(num_users):
        v_samp = val_samples[u_idx]
        t_samp = test_samples[u_idx]
        require(v_samp.user_idx == u_idx and t_samp.user_idx == u_idx, "User index ordering mismatch")

        # Validation history (unpadded)
        v_hist = [i for i in v_samp.history if i > 0]
        v_len = len(v_hist)
        v_tail_cnt = sum(1 for i in v_hist if is_tail[i])
        v_ratio = v_tail_cnt / max(1, v_len)
        val_lens[u_idx] = v_len
        val_tail_ratios[u_idx] = v_ratio

        # Test history (unpadded)
        t_hist = [i for i in t_samp.history if i > 0]
        t_len = len(t_hist)
        t_tail_cnt = sum(1 for i in t_hist if is_tail[i])
        t_ratio = t_tail_cnt / max(1, t_len)
        test_lens[u_idx] = t_len
        test_tail_ratios[u_idx] = t_ratio

    # 3. Binning for Validation Context
    val_q, val_tb, val_gid, val_summary = assign_quantile_and_median_bins(val_lens, val_tail_ratios)

    # 4. Binning for Test Context
    test_q, test_tb, test_gid, test_summary = assign_quantile_and_median_bins(test_lens, test_tail_ratios)

    # Build output table
    records = []
    for u_idx in range(num_users):
        records.append({
            "user_idx": u_idx,
            "raw_user_id": val_samples[u_idx].raw_user_id,
            "val_history_len": int(val_lens[u_idx]),
            "val_tail_items_count": int(round(val_lens[u_idx] * val_tail_ratios[u_idx])),
            "val_tail_ratio": round(float(val_tail_ratios[u_idx]), 4),
            "val_length_quantile": int(val_q[u_idx]),
            "val_tail_group": "high_tail" if val_tb[u_idx] == 1 else "low_tail",
            "val_group_id": int(val_gid[u_idx]),
            "val_group_label": f"Q{val_q[u_idx]}_{'high' if val_tb[u_idx] == 1 else 'low'}",
            "test_history_len": int(test_lens[u_idx]),
            "test_tail_items_count": int(round(test_lens[u_idx] * test_tail_ratios[u_idx])),
            "test_tail_ratio": round(float(test_tail_ratios[u_idx]), 4),
            "test_length_quantile": int(test_q[u_idx]),
            "test_tail_group": "high_tail" if test_tb[u_idx] == 1 else "low_tail",
            "test_group_id": int(test_gid[u_idx]),
            "test_group_label": f"Q{test_q[u_idx]}_{'high' if test_tb[u_idx] == 1 else 'low'}",
        })

    df_groups = pd.DataFrame(records)
    out_csv = meta_path / "user_groups.csv"
    df_groups.to_csv(out_csv, index=False)

    summary_payload = {
        "dataset": dataset_name,
        "num_users": num_users,
        "catalog_partition": catalog_stats,
        "validation_grouping": val_summary,
        "test_grouping": test_summary,
    }
    out_summary = meta_path / "user_groups_summary.json"
    write_json(out_summary, summary_payload)

    print(f"User groups table written to: {out_csv}")
    print(f"Summary written to: {out_summary}")
    return summary_payload


def main():
    parser = argparse.ArgumentParser(description="Generate zero-leakage history-based user grouping.")
    parser.add_argument("--dataset", default="beauty", choices=["beauty", "sports", "toys"])
    parser.add_argument("--raw_root", default="data/raw")
    parser.add_argument("--metadata_dir", default="artifacts/metadata")
    args = parser.parse_args()

    generate_user_groups_for_dataset(
        dataset_name=args.dataset,
        raw_root=args.raw_root,
        metadata_dir=args.metadata_dir,
    )


if __name__ == "__main__":
    main()

