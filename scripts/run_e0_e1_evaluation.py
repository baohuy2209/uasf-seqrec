"""
run_e0_e1_evaluation.py - Evaluation Runner for E0 (Heterogeneity), E1 (Predictability), and Label Stability.

Adheres strictly to Supervisor Decision Memo (26/10/2026) - Section 5.3 (Minh):
  1. Label stability and noise threshold epsilon_m between 2 reference seeds (Spearman rho).
  2. E0 Utility Heterogeneity (Condition A & Condition B).
  3. E1 Utility Predictability via 5-fold user-grouped OOF CV with standard scaling on train fold only.
  4. 1000-sample bootstrap 95% CI for AUC and Spearman.
  5. Tested on structured synthetic delta (dependent on sequence length + noise) or real delta files.
"""

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TAIMMA_ROOT = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = TAIMMA_ROOT
SRC_DIR = os.path.join(TAIMMA_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from taimma.paths import resolve_project_path
from taimma.evaluation.e0_e1_protocol import (
    compute_noise_threshold_and_stability,
    check_e0_heterogeneity,
    evaluate_e1_predictability,
    evaluate_supervisor_decision,
    LabelStabilityResult,
    E0HeterogeneityResult,
    E1PredictabilityResult,
)


def generate_structured_synthetic_data(
    num_users: int = 1000,
    samples_per_user: int = 5,
    seed: int = 42,
) -> Tuple[np.ndarray, Dict[str, Tuple[np.ndarray, np.ndarray]], List[int], np.ndarray]:
    """
    Generate structured synthetic delta labels:
      - Delta strongly correlated with sequence length: delta = alpha * (seq_len - 15) + noise
      - Two reference seeds to test label stability
      - Features psi_m: [log(seq_len), src_coverage, attr_diversity, recency]
    """
    rng = np.random.default_rng(seed)
    total_samples = num_users * samples_per_user

    user_ids = []
    seq_lens = []
    for u in range(num_users):
        for _ in range(samples_per_user):
            user_ids.append(u)
            # Varying sequence lengths from 5 to 50
            seq_lens.append(rng.integers(5, 51))

    seq_lens = np.array(seq_lens, dtype=np.float32)

    # 1. Permitted context features psi_m(q)
    log_lens = np.log(seq_lens)
    src_cov = rng.uniform(0.2, 1.0, size=total_samples).astype(np.float32)
    attr_div = rng.uniform(0.1, 0.9, size=total_samples).astype(np.float32)
    recency = (src_cov + rng.normal(0, 0.05, size=total_samples)).clip(0, 1).astype(np.float32)

    context_features = np.column_stack([log_lens, src_cov, attr_div, recency])

    # 2. Structured deltas for Category (high signal), Brand (medium), Title (low)
    deltas = {}
    
    # Category: Positive for short seqs (cold), negative for long seqs + Gaussian noise
    base_sig_cat = -0.05 * (seq_lens - 20.0) + 0.3 * src_cov
    noise_s1 = rng.normal(0, 0.15, size=total_samples)
    noise_s2 = rng.normal(0, 0.15, size=total_samples)
    deltas["category"] = (base_sig_cat + noise_s1, base_sig_cat + noise_s2)

    # Brand: Mild correlation
    base_sig_brd = -0.02 * (seq_lens - 20.0) + 0.2 * attr_div
    deltas["brand"] = (
        base_sig_brd + rng.normal(0, 0.2, size=total_samples),
        base_sig_brd + rng.normal(0, 0.2, size=total_samples),
    )

    # Title: Pure noise (negative control)
    deltas["title"] = (
        rng.normal(0, 0.2, size=total_samples),
        rng.normal(0, 0.2, size=total_samples),
    )

    return context_features, deltas, user_ids, seq_lens


import hashlib
import subprocess

def compute_file_sha256(filepath: str) -> str:
    """Compute sha256 checksum of a file in streaming chunks."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL).decode("ascii").strip()
    except Exception:
        return "unknown"


def load_delta_dataset(filepath: str, return_metadata: bool = False):
    """Load aligned real contexts; never impute missing scientific inputs.

    CSV/parquet require user_idx, seq_len, feat_* and paired deltas. Repeated
    users require context_id. Long tables additionally require source_name,
    reference_seed and delta. group_idx, if provided, is an inference-available
    length-quintile x history-popularity group (0..9), never a target feature.
    The first feature returned is always log of the supplied sequence length.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Delta file not found: {filepath}; use --synthetic explicitly for fixtures")
    metadata = {"source_features": {}, "group_indices": None, "reference_seeds": None}
    if filepath.lower().endswith('.npz'):
        with np.load(filepath, allow_pickle=False) as data:
            required = {'context_features', 'user_ids', 'seq_lens'}
            if not required.issubset(data.files):
                raise ValueError(f"NPZ requires {sorted(required)}")
            features = np.asarray(data['context_features'], dtype=float)
            users = data['user_ids'].tolist()
            lengths = np.asarray(data['seq_lens'], dtype=float)
            deltas = {}
            for src in ('category', 'brand', 'title'):
                keys = [f'delta_{src}_seed{i}' for i in (1, 2)]
                if any(k in data.files for k in keys):
                    if not all(k in data.files for k in keys):
                        raise ValueError(f"Missing paired reference seed for {src}")
                    deltas[src] = tuple(np.asarray(data[k], dtype=float) for k in keys)
            if 'group_indices' in data.files:
                metadata['group_indices'] = np.asarray(data['group_indices'])
            metadata['feature_names'] = data['feature_names'].tolist() if 'feature_names' in data.files else None
    else:
        if filepath.lower().endswith('.parquet'):
            df = pd.read_parquet(filepath)
        elif filepath.lower().endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            raise ValueError('Delta input must be .npz, .csv or .parquet')
        if 'user_idx' not in df and 'user_id' in df:
            df = df.rename(columns={'user_id': 'user_idx'})
        feature_cols = sorted(c for c in df if c.startswith('feat_'))
        if not {'user_idx', 'seq_len'}.issubset(df) or not feature_cols:
            raise ValueError('Real delta tables require user_idx, seq_len and explicit feat_* context features')
        if df.empty or df[['user_idx', 'seq_len'] + feature_cols].isna().any().any():
            raise ValueError('Empty or missing context/feature fields in delta input')
        if 'context_id' not in df:
            # The approved Train-meta split has one context per user. Long form
            # repeats that context over sources/seeds; duplicates are checked below.
            df = df.assign(context_id=df['user_idx'])
        keys = ['user_idx', 'context_id']
        if df[keys].isna().any().any():
            raise ValueError('Context/user identifiers cannot be missing')
        extra = ['group_idx'] if 'group_idx' in df else []
        meta_cols = ['seq_len'] + extra
        long_format = 'source_name' in df
        source_frames = {}
        if long_format:
            if not {'reference_seed', 'delta'}.issubset(df):
                raise ValueError('Long format requires reference_seed and delta')
            if df[['source_name', 'reference_seed', 'delta']].isna().any().any():
                raise ValueError('Missing source, seed or delta')
            if df.duplicated(keys + ['source_name', 'reference_seed']).any():
                raise ValueError('Duplicate context/user/source/seed records')
            seeds = sorted(df.reference_seed.unique().tolist())
            if len(seeds) != 2:
                raise ValueError('Exactly two explicitly aligned reference seeds are required')
            metadata['reference_seeds'] = seeds
            if (df.groupby(keys)[meta_cols].nunique(dropna=False) > 1).any().any():
                raise ValueError('Context length/group differs across sources or seeds')
            context_meta = df.groupby(keys, sort=True)[meta_cols].first()
            deltas = {}
            for src, rows in df.groupby('source_name', sort=True):
                if src not in ('category', 'brand', 'title'):
                    raise ValueError(f'Unknown metadata source: {src}')
                pivot = rows.pivot(index=keys, columns='reference_seed', values='delta').reindex(context_meta.index)
                if set(pivot.columns) != set(seeds) or pivot.isna().any().any():
                    raise ValueError(f'Incomplete source/seed cohort for {src}')
                if (rows.groupby(keys)[feature_cols].nunique(dropna=False) > 1).any().any():
                    raise ValueError(f'Context features differ between reference seeds for {src}')
                source_frames[src] = rows.groupby(keys, sort=True)[feature_cols].first().reindex(context_meta.index)
                deltas[src] = tuple(pivot[seed].to_numpy(dtype=float) for seed in seeds)
        else:
            if df.duplicated(keys).any():
                raise ValueError('Repeated users require unique context_id values')
            indexed = df.set_index(keys).sort_index()
            context_meta = indexed[meta_cols]
            deltas = {}
            for src in ('category', 'brand', 'title'):
                cols = [f'delta_{src}_seed{i}' for i in (1, 2)]
                if any(c in df for c in cols):
                    if not all(c in df for c in cols):
                        raise ValueError(f'Missing paired reference seed for {src}')
                    deltas[src] = tuple(indexed[c].to_numpy(dtype=float) for c in cols)
                    source_frames[src] = indexed[feature_cols]
        users = context_meta.index.get_level_values('user_idx').tolist()
        lengths = context_meta.seq_len.to_numpy(dtype=float)
        if not np.isfinite(lengths).all() or np.any(lengths <= 0):
            raise ValueError('seq_len must be positive and finite')
        log_len = np.log(lengths)
        names = ['feat_log_len'] + [c for c in feature_cols if c != 'feat_log_len']
        for src, frame in source_frames.items():
            if 'feat_log_len' in frame and not np.allclose(frame.feat_log_len, log_len):
                raise ValueError('feat_log_len must equal log(seq_len)')
            metadata['source_features'][src] = np.column_stack([log_len] + [frame[c].to_numpy(dtype=float) for c in names[1:]])
        features = next(iter(metadata['source_features'].values()), np.empty((len(users), 0)))
        metadata['feature_names'] = names
        if extra:
            metadata['group_indices'] = context_meta.group_idx.to_numpy()

    n = len(users)
    if pd.isna(np.asarray(users)).any():
        raise ValueError('User IDs cannot be missing')
    if n == 0 or not deltas or lengths.shape != (n,) or features.ndim != 2 or features.shape[0] != n or features.shape[1] == 0:
        raise ValueError('Empty or misaligned delta/context arrays')
    if not np.isfinite(lengths).all() or np.any(lengths <= 0):
        raise ValueError('Sequence lengths must be positive and finite')
    if not np.isfinite(features).all() or not np.allclose(features[:, 0], np.log(lengths)):
        raise ValueError('Features must be finite; column zero must be log(seq_len)')
    for src, pair in deltas.items():
        if any(v.shape != (n,) or not np.isfinite(v).all() for v in pair):
            raise ValueError(f'Nonfinite or misaligned reference deltas for {src}')
    for values in metadata['source_features'].values():
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite source features')
    groups = metadata['group_indices']
    if groups is not None:
        if groups.shape != (n,) or not np.isfinite(groups.astype(float)).all() or not np.isin(groups, np.arange(10)).all():
            raise ValueError('group_idx must be aligned integer IDs 0..9 from inference-available context')
        metadata['group_indices'] = groups.astype(int)
    result = (features, deltas, users, lengths)
    return (*result, metadata) if return_metadata else result


def main():
    parser = argparse.ArgumentParser(description="TAIMMA E0/E1 & Label Stability Evaluation Runner (Minh)")
    parser.add_argument("--synthetic", action="store_true", help="Explicitly run on calibrated structured synthetic deltas")
    parser.add_argument("--delta_file", type=str, default=None, help="Path to real deltas (.npz, .csv, or .parquet)")
    parser.add_argument("--num_users", type=int, default=1000, help="Number of synthetic users")
    parser.add_argument("--n_bootstrap", type=int, default=1000, help="Bootstrap iterations (default 1000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_json", type=str, default=None, help="Optional output JSON path for reports")
    args = parser.parse_args()
    if args.delta_file:
        args.delta_file = str(resolve_project_path(args.delta_file))
    if args.output_json:
        args.output_json = str(resolve_project_path(args.output_json))
    if args.n_bootstrap < 1 or args.num_users < 1:
        raise ValueError('Bootstrap count and user count must be positive')
    if args.output_json and os.path.exists(args.output_json):
        raise FileExistsError(f'Refusing to overwrite report: {args.output_json}')

    print("\n" + "=" * 75)
    print("TAIMMA â€” 5.3: Independent Evaluation & E0/E1 Predictability Protocol")
    print("=" * 75)

    # Mode validation: require explicit mode, prevent silent fallback
    if args.delta_file is None and not args.synthetic:
        raise ValueError(
            "Mode must be explicitly specified. Provide --delta_file <path> for real evaluation "
            "or --synthetic for calibrated structured synthetic testing. Silent fallbacks are strictly prohibited."
        )
    if args.delta_file is not None and args.synthetic:
        raise ValueError("Conflicting modes specified: cannot specify both --synthetic and --delta_file.")

    file_sha256 = None
    if args.delta_file is not None:
        print(f"\n[Real Mode] Loading delta file from: '{args.delta_file}'")
        file_sha256 = compute_file_sha256(args.delta_file)
        context_features, deltas, user_ids, seq_lens, dataset_meta = load_delta_dataset(args.delta_file, return_metadata=True)
        data_mode = "real"
    else:
        print("\n[Synthetic Mode] Generating structured synthetic deltas (seq_len dependent + noise)...")
        context_features, deltas, user_ids, seq_lens = generate_structured_synthetic_data(
            num_users=args.num_users, seed=args.seed
        )
        data_mode = "synthetic"
        dataset_meta = {"source_features": {}, "group_indices": None, "feature_names": ["feat_log_len", "feat_coverage", "feat_diversity", "feat_recency"]}

    print(f"Data mode: [{data_mode.upper()}]")
    print(f"Total evaluation samples: {len(user_ids)} across {len(set(user_ids))} users.")
    print(f"Bootstrap iterations: {args.n_bootstrap}")

    stability_results: Dict[str, LabelStabilityResult] = {}
    e0_results: Dict[str, E0HeterogeneityResult] = {}
    e1_results: Dict[str, E1PredictabilityResult] = {}

    for src_name, (deltas_s1, deltas_s2) in deltas.items():
        print(f"\n" + "-" * 60)
        print(f"Evaluating Metadata Source: [{src_name.upper()}]")
        print("-" * 60)

        # 1. Noise threshold & Label stability (10.1 & 10.2)
        stab = compute_noise_threshold_and_stability(deltas_s1, deltas_s2, source_name=src_name, user_indices=user_ids, n_bootstrap=args.n_bootstrap, seed=args.seed)
        stability_results[src_name] = stab
        print(f"1. Label Stability & Noise Threshold:")
        print(f"   - Epsilon (noise threshold) : {stab.epsilon:.6f}")
        print(f"   - Spearman rho (seed1 vs 2) : {stab.spearman_rho:.4f} (p={stab.p_value:.2e})")
        print(f"   - Stability Regime          : {stab.stability_regime.upper()}")

        regime = {'contextual': 'contextual', 'soft_supervision': 'averaged', 'group_level': 'group'}[stab.stability_regime]
        eval_deltas = deltas_s1 if regime == 'contextual' else (deltas_s1 + deltas_s2) / 2
        # 2. E0 Heterogeneity (10.3)
        e0 = check_e0_heterogeneity(
            deltas=eval_deltas,
            epsilon=stab.epsilon,
            sequence_lengths=seq_lens,
            source_name=src_name,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
            user_indices=user_ids,
        )
        e0_results[src_name] = e0
        print(f"2. E0 Utility Heterogeneity:")
        print(f"   - Condition A (>=15% above +eps, >=15% below -eps): {e0.condition_a_met} (+{e0.frac_positive*100:.1f}%, -{e0.frac_negative*100:.1f}%)")
        print(f"   - Condition B (Q1 vs Q5 opposite signs): {e0.condition_b_met}")
        print(f"       Q1 mean/CI: {e0.q1_mean}, {e0.q1_ci}; Q5 mean/CI: {e0.q5_mean}, {e0.q5_ci}")
        e1 = evaluate_e1_predictability(
            context_features=dataset_meta['source_features'].get(src_name, context_features),
            utility_deltas=eval_deltas, user_indices=user_ids, epsilon=stab.epsilon,
            source_name=src_name, n_splits=5, n_bootstrap=args.n_bootstrap, seed=args.seed,
            evaluated_regime=regime, group_indices=dataset_meta['group_indices'],
        )
        e1_results[src_name] = e1
        print(f"E1 {src_name}: status={e1.status}; regime={e1.evaluated_regime}; AUC={e1.oof_auc_lr}; CI={e1.bootstrap_95_ci}; Spearman={e1.oof_spearman_rho}; CI={e1.oof_spearman_ci}; pass={e1.e1_passed}")
        print(e1.evidence_summary)

    # 4. Supervisor Decision Engine (10.5)
    decision_report = evaluate_supervisor_decision(
        stability_results=stability_results,
        e0_results=e0_results,
        e1_results=e1_results,
    )

    print(f"\n" + "=" * 75)
    print("SUPERVISOR DECISION REPORT (October 26 Policy)")
    print("=" * 75)
    print(f"DATA MODE      : [{data_mode.upper()}]")
    print(f"FINAL DECISION : [{decision_report.decision}]")
    print(f"RATIONALE      : {decision_report.rationale}")
    print("=" * 75)

    if args.output_json:
        report_data = {
            "data_mode": data_mode,
            "delta_file": args.delta_file,
            "delta_file_sha256": file_sha256,
            "git_commit": get_git_commit(),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "decision": decision_report.decision,
            "rationale": decision_report.rationale,
            "qualifying_sources": decision_report.qualifying_sources,
            "gating_regime": decision_report.gating_regime,
            "feature_names": dataset_meta.get('feature_names'),
            "reference_seeds": dataset_meta.get('reference_seeds'),
            "n_bootstrap": args.n_bootstrap,
            "seed": args.seed,
            "sources": {src: {"stability": asdict(stability_results[src]),
                              "e0": asdict(e0_results[src]), "e1": asdict(e1_results[src])}
                        for src in stability_results},
        }
        resolve_project_path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "x", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, allow_nan=False)
        print(f"\nReport serialized to: '{args.output_json}'")


if __name__ == "__main__":
    main()
