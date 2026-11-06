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
3. 10 Context Groups (5 length quantiles x 2 tail bins):
   - Length quintiles Q1..Q5.
   - Tail binary split (Median split: low_tail vs high_tail).
   - group_id in 0..9.
   - Fit boundaries on validation histories once and reuse on test.
   - Equal context features share a group; ties may yield empty groups.
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
    sorted_item_indices = np.lexsort((np.arange(num_catalog_items), -freqs))
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


def fit_group_rules(lengths: np.ndarray, tail_ratios: np.ndarray) -> Dict[str, Any]:
    """Fit numeric context-only boundaries ONCE, never using targets or test data.

    Equal histories always have equal groups. Tied lengths may produce empty
    groups; do not break ties using user order to manufacture balanced groups.
    """
    lengths=np.asarray(lengths,dtype=float);tail_ratios=np.asarray(tail_ratios,dtype=float)
    require(lengths.ndim==1 and len(lengths)>0 and lengths.shape==tail_ratios.shape,
            'Grouping requires nonempty aligned context features')
    require(np.isfinite(lengths).all() and np.isfinite(tail_ratios).all(), 'Nonfinite group features')
    require(np.all(lengths>=0) and np.all((tail_ratios>=0)&(tail_ratios<=1)), 'Invalid history features')
    boundaries=np.quantile(lengths,[.2,.4,.6,.8]).tolist()
    q=np.searchsorted(boundaries,lengths,side='left')+1
    global_median=float(np.median(tail_ratios))
    medians={str(k):float(np.median(tail_ratios[q==k])) if np.any(q==k) else global_median
             for k in range(1,6)}
    return dict(schema_version=2,length_boundaries=boundaries,tail_medians=medians,
                fit_partition='validation_contexts',tie_policy='identical_features_same_group',num_groups=10)


def apply_group_rules(lengths, tail_ratios, rules):
    lengths=np.asarray(lengths,dtype=float);ratios=np.asarray(tail_ratios,dtype=float)
    require(lengths.ndim==1 and lengths.shape==ratios.shape and np.isfinite(lengths).all()
            and np.isfinite(ratios).all(), 'Grouping features must be finite and aligned')
    require(np.all(lengths>=0) and np.all((ratios>=0)&(ratios<=1)), 'Invalid context features')
    q=np.searchsorted(rules['length_boundaries'],lengths,side='left')+1
    tb=np.asarray([float(r)>rules['tail_medians'][str(int(k))] for k,r in zip(q,ratios)],dtype=np.int64)
    gids=(q-1)*2+tb
    return q.astype(np.int64),tb,gids.astype(np.int64)


def assign_quantile_and_median_bins(
    lengths: np.ndarray,
    tail_ratios: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Convenience fit/apply on one reference partition; test must reuse rules."""
    rules=fit_group_rules(lengths,tail_ratios)
    q,tb,gids=apply_group_rules(lengths,tail_ratios,rules)
    summary=dict(rules=rules,num_groups=10,group_counts={int(g):int((gids==g).sum()) for g in range(10)})
    return q,tb,gids,summary


def generate_user_groups_for_dataset(
    dataset_name: str = "beauty",
    raw_root: str = "data/raw",
    metadata_dir: str = "artifacts/metadata",
    output_dir=None,
) -> Dict[str, Any]:
    print(f"\n=======================================================")
    print(f"Generating History-based User Groups for: {dataset_name}")
    print(f"=======================================================")

    raw_path = resolve_project_path(raw_root)
    meta_path = resolve_project_path(metadata_dir) / dataset_name.lower()
    require(meta_path.exists(), f"Metadata directory missing: {meta_path}")

    # Build sequence splits using locked project ingestor
    dataset_key={'beauty':'Beauty','sports':'Sports_and_Outdoors','toys':'Toys_and_Games'}[dataset_name.lower()]
    ingestor = Amazon2014Ingestor(str(raw_path), dataset_key=dataset_key)
    sequences, _ = ingestor.build_interaction_sequences(min_interactions=5,include_metadata=False)
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

    # Fit on validation histories; use exactly the same inference rule on test.
    rules=fit_group_rules(val_lens,val_tail_ratios)
    val_q,val_tb,val_gid=apply_group_rules(val_lens,val_tail_ratios,rules)
    test_q,test_tb,test_gid=apply_group_rules(test_lens,test_tail_ratios,rules)
    val_summary=dict(rules=rules,group_counts={int(g):int((val_gid==g).sum()) for g in range(10)})
    test_summary=dict(rules_reused_from='validation',group_counts={int(g):int((test_gid==g).sum()) for g in range(10)})

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
    destination=Path(output_dir) if output_dir is not None else meta_path
    destination.mkdir(parents=True,exist_ok=True)
    out_csv = destination / "user_groups.csv"
    if out_csv.exists() or (destination/'user_groups_summary.json').exists():
        raise FileExistsError('Preserve existing grouping; specify a new --output_dir')
    df_groups.to_csv(out_csv, index=False)

    summary_payload = {
        "dataset": dataset_name,
        "num_users": num_users,
        "catalog_partition": catalog_stats,
        "validation_grouping": val_summary,
        "test_grouping": test_summary,
    }
    out_summary = destination / "user_groups_summary.json"
    write_json(out_summary, summary_payload)

    print(f"User groups table written to: {out_csv}")
    print(f"Summary written to: {out_summary}")
    return summary_payload


def main():
    parser = argparse.ArgumentParser(description="Generate zero-leakage history-based user grouping.")
    parser.add_argument("--dataset", default="beauty", choices=["beauty", "sports", "toys"])
    parser.add_argument("--raw_root", default="data/raw")
    parser.add_argument("--metadata_dir", default="artifacts/metadata")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    generate_user_groups_for_dataset(
        dataset_name=args.dataset,
        raw_root=args.raw_root,
        metadata_dir=args.metadata_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

