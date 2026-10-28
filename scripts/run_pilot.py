"""
run_pilot.py - Pilot runner for TAIMMA Beauty SASRec ID-only pipeline.

Executes the 5 supervisor-mandated sanity checks:
  - Check 1: Dataset statistics and coverage verification.
  - Check 2: Baseline hierarchy ordering (Random < Popularity < SASRec).
  - Check 3: Plausibility against sequential benchmark reference.
  - Check 4: Leakage detection via target-shuffling degradation.
  - Check 5: Deterministic reproducibility with repeat seed.

Strict Data-First:
  - If raw files are missing, fails fast and explicitly reports missing assets.
  - Smoke tests on synthetic fixtures are labeled distinctly as SMOKE_TEST.
"""

import argparse
import copy
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import yaml
import torch
import numpy as np
from pathlib import Path
from dataclasses import asdict

# Add src to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TAIMMA_ROOT = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = TAIMMA_ROOT
SRC_DIR = os.path.join(TAIMMA_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from taimma.paths import resolve_project_path
from taimma.data.ingestion import Amazon2014Ingestor, InteractionSequence
from taimma.data.split import SequenceSplitter, SplitData, save_processed_data
from taimma.models.sasrec import SASRecModel
from taimma.evaluation.evaluator import IndependentEvaluator, EvaluationResult
from taimma.evaluation.baselines import PopularityBaseline, RandomBaseline
from taimma.training.trainer import SASRecTrainer, TrainingHistory, set_seed
from taimma.experiments.logger import ExperimentLogger
from taimma.experiments.reporting import (
    dataset_statistics, context_table, runtime_snapshot, save_training_history,
    save_pilot_report, validate_rank_format,
)



def run_sanity_checks(
    split_data: SplitData,
    model: SASRecModel,
    trainer: SASRecTrainer,
    evaluator: IndependentEvaluator,
    metadata_dict: Dict[str, Dict[str, Optional[str]]],
    seed: int,
    batch_size: int,
    history: Optional[TrainingHistory] = None,
    repeat_history: Optional[TrainingHistory] = None,
    repeat_test_res: Optional[EvaluationResult] = None,
    repeat_val_res: Optional[EvaluationResult] = None,
    original_val_res: Optional[EvaluationResult] = None,
    repro_tolerance: float = 1e-6,
    is_real_data: bool = True,
    raw_unique_items: Optional[int] = None,
    expected_stats: Optional[Dict[str, int]] = None,
    report_results: Optional[Dict] = None,
) -> Dict[str, Dict[str, Any]]:
    """Run the 5 supervisor sanity checks and return structured verdicts."""
    verdicts = {}

    # -------------------------------------------------------------
    # Check 1: Dataset Statistics
    # -------------------------------------------------------------
    n_users = split_data.num_users
    n_items = split_data.num_items
    n_train_samples = len(split_data.train_base_samples)
    n_val_samples = len(split_data.val_samples)
    n_test_samples = len(split_data.test_samples)

    # Calculate metadata coverage over candidate catalog
    cat_covered = sum(
        1 for item in split_data.item2idx if metadata_dict.get(item, {}).get("category") is not None
    )
    brand_covered = sum(
        1 for item in split_data.item2idx if metadata_dict.get(item, {}).get("brand") is not None
    )
    title_covered = sum(
        1 for item in split_data.item2idx if metadata_dict.get(item, {}).get("title") is not None
    )

    cat_cov_pct = (cat_covered / n_items * 100) if n_items > 0 else 0.0
    brand_cov_pct = (brand_covered / n_items * 100) if n_items > 0 else 0.0
    title_cov_pct = (title_covered / n_items * 100) if n_items > 0 else 0.0

    meta_cov_str = (
        "Metadata: ABSENT on disk (0.0% coverage; Category, Brand, Title stages BLOCKED)"
        if (cat_covered == 0 and brand_covered == 0 and title_covered == 0)
        else f"Coverage - Category: {cat_cov_pct:.1f}%, Brand: {brand_cov_pct:.1f}%, Title: {title_cov_pct:.1f}%"
    )

    stats_summary = (
        f"Users: {n_users}, Catalog Items: {n_items}, "
        f"Train Pairs: {n_train_samples}, Val Targets: {n_val_samples}, Test Targets: {n_test_samples} | "
        f"{meta_cov_str}"
    )

    if is_real_data:
        # Check against supervisor memo references: 22,363 users, 12,101 items
        users_match = expected_stats is not None and n_users == expected_stats["users"]
        items_match = expected_stats is not None and raw_unique_items == expected_stats["items"] and n_items == expected_stats["catalog_train_base_items"]
        ref_match = users_match and items_match
        raw_info = f" (Raw unique items: {raw_unique_items})" if raw_unique_items else ""
        verdicts["Check 1: Dataset Statistics"] = {
            "status": "PENDING" if not metadata_dict else "PASS" if ref_match else "FAIL",
            "evidence": stats_summary + f"{raw_info} (Expected stats: {expected_stats}; review-stat match: {ref_match}; metadata_ready={bool(metadata_dict)})",
            "validation_method": "Comparison with supervisor memo (26/10/2026)",
        }
    else:
        verdicts["Check 1: Dataset Statistics"] = {
            "status": "PASS (Fixture)",
            "evidence": stats_summary,
            "validation_method": "Synthetic fixture shape and coverage check",
        }

    # -------------------------------------------------------------
    # Check 2: Baseline Hierarchy Ordering (Random < Popularity < SASRec)
    # -------------------------------------------------------------
    def evaluate_baseline(baseline, samples=None):
        samples = split_data.test_samples if samples is None else samples
        records = []
        for start in range(0, len(samples), batch_size):
            batch = samples[start:start + batch_size]
            res = evaluator.evaluate_ranks(baseline.predict_all(len(batch)),
                [x.target for x in batch], [x.user_idx for x in batch], split_data.seen_train_items_per_user)
            records.extend(res.per_user_records)
        return evaluator.result_from_records(records)

    rand_res = evaluate_baseline(RandomBaseline(num_items=n_items, seed=seed))
    pop_base = PopularityBaseline(num_items=n_items)
    pop_base.fit(train_base_events=split_data.train_base_events)
    pop_res = evaluate_baseline(pop_base)

    # Evaluate SASRec ID-only
    sasrec_res = trainer.evaluate(
        split_data.test_samples,
        seen_items=split_data.seen_train_items_per_user,
        batch_size=batch_size,
    )

    # Strict ordering requirement: Random < Popularity < SASRec
    ordering_satisfied = (rand_res.ndcg_10 < pop_res.ndcg_10 < sasrec_res.ndcg_10)
    verdicts["Check 2: Baseline Ordering"] = {
        "status": "PASS" if ordering_satisfied else "FAIL",
        "evidence": (
            f"Random NDCG@10: {rand_res.ndcg_10:.4f} | "
            f"Popularity NDCG@10: {pop_res.ndcg_10:.4f} | "
            f"SASRec ID NDCG@10: {sasrec_res.ndcg_10:.4f}"
        ),
        "validation_method": "Strict metric ordering: Random < Popularity < SASRec",
        "measurements": {"random_ndcg_10": rand_res.ndcg_10, "popularity_ndcg_10": pop_res.ndcg_10, "sasrec_ndcg_10": sasrec_res.ndcg_10},
    }

    # -------------------------------------------------------------
    # Check 3: SASRec Benchmark Plausibility
    # -------------------------------------------------------------
    verdicts["Check 3: Benchmark Plausibility"] = {
        "status": "PENDING" if is_real_data else "PASS (FIXTURE)",
        "evidence": f"Observed NDCG@10={sasrec_res.ndcg_10:.6f}. A matched reference implementation/config/commit is required; no hardcoded literature interval certifies reproduction.",
        "validation_method": "Independent matched-protocol reference comparison pending",
    }

    # -------------------------------------------------------------
    # Check 4: Leakage Detection (Target Shuffling & Personalization Collapse)
    # -------------------------------------------------------------
    targets = [sample.target for sample in split_data.test_samples]
    rng_perm = np.random.default_rng(seed)
    shuffled_targets = np.asarray(targets)[rng_perm.permutation(len(targets))].tolist()
    records = []
    with torch.no_grad():
        for start in range(0, n_test_samples, batch_size):
            batch = split_data.test_samples[start:start + batch_size]
            seqs = torch.tensor([x.history for x in batch], dtype=torch.long, device=trainer.device)
            scores = trainer.model.predict_all(seqs)
            res = evaluator.evaluate_ranks(scores, shuffled_targets[start:start + len(batch)],
                [x.user_idx for x in batch], split_data.seen_train_items_per_user)
            records.extend(res.per_user_records)
    shuffled_res = evaluator.result_from_records(records)
    if report_results is not None:
        report_results.update({
            'Random': {'test': rand_res, 'validation': evaluate_baseline(RandomBaseline(n_items, seed), split_data.val_samples)},
            'Popularity': {'test': pop_res, 'validation': evaluate_baseline(pop_base, split_data.val_samples)},
            'SASRec_ID': {'test': sasrec_res, 'validation': original_val_res if original_val_res is not None else trainer.evaluate(split_data.val_samples, split_data.seen_train_items_per_user, batch_size)},
            'SASRec_shuffled': {'test': shuffled_res},
        })

    verdicts["Check 4: Leakage Detection"] = {
        "status": "REVIEW" if is_real_data else "PASS (FIXTURE)",
        "evidence": f"SASRec NDCG@10={sasrec_res.ndcg_10:.6f}; shuffled={shuffled_res.ndcg_10:.6f}; Random={rand_res.ndcg_10:.6f}; Popularity={pop_res.ndcg_10:.6f}; permutation seed={seed}. Shuffling preserves target popularity and does not alone certify absence of leakage.",
        "validation_method": "Seeded cross-user target permutation; scientific acceptance requires a predeclared null criterion",
        "measurements": {"original_ndcg_10": sasrec_res.ndcg_10, "shuffled_ndcg_10": shuffled_res.ndcg_10, "seed": seed},
    }

    # -------------------------------------------------------------
    # Check 5: Reproducibility
    # -------------------------------------------------------------
    if repeat_test_res is not None and repeat_val_res is not None and original_val_res is not None and repeat_history is not None and history is not None:
        def matching_results(a, b):
            values = ('ndcg_10', 'recall_10', 'ndcg_20', 'recall_20')
            return (all(abs(getattr(a, k) - getattr(b, k)) <= repro_tolerance for k in values)
                    and [(v.user_idx, v.target_idx, v.rank) for v in a.per_user_records]
                    == [(v.user_idx, v.target_idx, v.rank) for v in b.per_user_records])
        repro_pass = (history.best_epoch == repeat_history.best_epoch
                      and len(history.train_losses) == len(repeat_history.train_losses)
                      and np.allclose(history.train_losses, repeat_history.train_losses, atol=repro_tolerance, rtol=0)
                      and np.allclose(history.val_ndcg_10, repeat_history.val_ndcg_10, atol=repro_tolerance, rtol=0)
                      and matching_results(sasrec_res, repeat_test_res)
                      and matching_results(original_val_res, repeat_val_res))
        verdicts["Check 5: Reproducibility"] = {
            "status": "PASS" if repro_pass else "FAIL",
            "evidence": f"Compared complete histories, best epoch, validation/test metrics and keyed per-user ranks under seed={seed}, atol={repro_tolerance}, rtol=0.",
            "validation_method": "Full independent repeated training; both runs persisted",
            "measurements": {"epochs_run1": len(history.train_losses), "epochs_run2": len(repeat_history.train_losses),
                             "best_epoch_run1": history.best_epoch, "best_epoch_run2": repeat_history.best_epoch,
                             "test_ndcg_10_abs_diff": abs(sasrec_res.ndcg_10 - repeat_test_res.ndcg_10),
                             "validation_ndcg_10_abs_diff": abs(original_val_res.ndcg_10 - repeat_val_res.ndcg_10),
                             "absolute_tolerance": repro_tolerance},
        }
    else:
        verdicts["Check 5: Reproducibility"] = {
            "status": "PENDING",
            "evidence": (
                f"Single run completed under seed {seed}. Full repeat training comparison pending. "
                f"(Pass --verify_reproducibility to execute second run to convergence)."
            ),
            "validation_method": "Supervisor protocol requires independent repeated execution to verify metric tolerance",
        }

    return verdicts


def main():
    parser = argparse.ArgumentParser(description="TAIMMA Beauty SASRec ID-only Pilot Runner")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--real_data', action='store_true')
    mode.add_argument('--smoke_test', action='store_true')
    parser.add_argument('--config')
    parser.add_argument('--dataset_dir')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--eval_batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--runs_dir')
    parser.add_argument('--results_csv')
    parser.add_argument('--rank_format', choices=['csv', 'parquet', 'both'], default='csv')
    parser.add_argument('--repro_tolerance', type=float, default=1e-6)
    parser.add_argument('--verify_reproducibility', action='store_true')
    args = parser.parse_args()
    cfg = {}
    if args.config:
        args.config = str(resolve_project_path(args.config))
        with open(args.config, encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
            raise ValueError('Experiment configuration must be a YAML mapping')
    explicit = {a.split('=')[0] for a in sys.argv[1:] if a.startswith('--')}
    for key, attr in {'max_epochs': 'epochs', 'learning_rate': 'lr', 'batch_size': 'batch_size',
                      'eval_batch_size': 'eval_batch_size', 'patience': 'patience', 'seed': 'seed',
                      'device': 'device', 'weight_decay': 'weight_decay', 'runs_dir': 'runs_dir',
                      'results_csv': 'results_csv', 'repro_tolerance': 'repro_tolerance', 'rank_format': 'rank_format'}.items():
        if key in cfg and '--' + attr not in explicit:
            setattr(args, attr, cfg[key])
    if min(args.epochs, args.patience, args.batch_size, args.eval_batch_size) < 1 or args.repro_tolerance < 0:
        raise ValueError('Epochs, patience and batch sizes must be positive; tolerance must be nonnegative')

    def local_config(name, default):
        path = Path(cfg.get(name, default))
        if not path.is_absolute():
            path = Path(TAIMMA_ROOT) / path
        with path.open(encoding='utf-8') as f:
            value = yaml.safe_load(f)
        if not isinstance(value, dict):
            raise ValueError(f'Configuration is not a mapping: {path}')
        return value

    model_cfg = local_config('model_config', 'configs/models/sasrec_id.yaml')
    dataset_cfg = local_config('dataset_config', 'configs/datasets/beauty_2014.yaml')
    if dataset_cfg.get('dataset_name') != 'Beauty_2014':
        raise ValueError('This pilot runner validates Beauty_2014; use a separate configured experiment for other datasets')
    model_kwargs = {k: model_cfg[k] for k in ('max_seq_len', 'embedding_dim', 'num_blocks', 'num_heads', 'dropout_rate')}
    model_kwargs['act'] = model_cfg.get('hidden_act', 'gelu')
    if model_kwargs['max_seq_len'] != dataset_cfg['max_seq_len']:
        raise ValueError('Model and dataset max_seq_len must agree')
    mask_seen = cfg.get('mask_seen_items', True)
    if not isinstance(mask_seen, bool):
        raise ValueError('mask_seen_items must be a YAML boolean')
    k_list = cfg.get('k_list', [10, 20])
    if cfg.get('primary_metric', 'NDCG@10') != 'NDCG@10' or cfg.get('ranking_protocol', 'full_catalog') != 'full_catalog':
        raise ValueError('The approved protocol uses NDCG@10 and full_catalog')
    args.dataset_dir = str(resolve_project_path(args.dataset_dir or dataset_cfg['raw_data_dir']))
    args.runs_dir = args.runs_dir or str(Path(TAIMMA_ROOT) / 'runs')
    args.results_csv = args.results_csv or str(Path(TAIMMA_ROOT) / 'results/results.csv')
    for attr in ('runs_dir', 'results_csv'):
        path = Path(getattr(args, attr))
        if not path.is_absolute():
            setattr(args, attr, str(resolve_project_path(path)))

    validate_rank_format(args.rank_format)
    logger = ExperimentLogger(runs_root=args.runs_dir, results_csv=args.results_csv)
    exp_prefix = 'pilot_beauty_real' if args.real_data else 'smoke_test_pilot'
    run_id, run_path = logger.create_run_dir(exp_prefix)
    checkpoint_file = os.path.join(run_path, "checkpoint.pt")
    run_start = time.perf_counter()
    with logger.capture_output(run_path):
        print(f"\n{'='*70}")
        print(f"TAIMMA - Amazon Beauty 2014 -> SASRec ID-only Pilot Pipeline")
        print(f"{'='*70}")

        validate_rank_format(args.rank_format)
        meta_path = None
        raw_unique_items = None
        if args.real_data:
            ingestor = Amazon2014Ingestor(args.dataset_dir, dataset_key="Beauty")
            try:
                reviews_path, meta_path = ingestor.resolve_paths()
                ratings_path = ingestor.ratings_path
            except FileNotFoundError as e:
                print("\n[STRICT DATA-FIRST BLOCKED]")
                print(str(e))
                sys.exit(1)

            print(f"\n[Real Data Mode] Loading actual Amazon Beauty 2014 dataset from:")
            print(f"  - Canonical 5-core reviews: '{reviews_path}'")
            if ratings_path:
                print(f"  - Raw ratings dump:         '{ratings_path}'")
            if meta_path is None:
                print(f"[METADATA NOTICE] Product metadata file is absent in '{args.dataset_dir}'.")
                print("  Category, Brand, and Title auxiliary stages are marked as BLOCKED per research protocol.")
                print("  Proceeding with real-data SASRec ID-only pilot.")
            sequences, meta_dict = ingestor.build_interaction_sequences(min_interactions=5)
            raw_unique_items = len(set(i for s in sequences for i in s.items))
            is_real = True
            exp_prefix = "pilot_beauty_real"

        else:
            # Synthetic smoke test
            print("\n[Smoke Test Mode] Generating calibrated synthetic fixture sequences (n >= 5)...")
            sequences = []
            for u in range(100):
                # Sequence of 10 items
                u_items = [f"B00_ITEM_{((u * 3 + i) % 50) + 1}" for i in range(10)]
                u_ts = [1000 + i * 10 for i in range(10)]
                sequences.append(InteractionSequence(user_id=f"USER_{u}", items=u_items, timestamps=u_ts))
            meta_dict = {
                f"B00_ITEM_{i}": {"brand": f"Brand_{i % 5}", "category": f"SubCat_{i % 3}", "title": f"Title {i}"}
                for i in range(1, 51)
            }
            is_real = False
            exp_prefix = "smoke_test_pilot"

        # Sequence Splitting
        print(f"Splitting sequences according to Train-base protocol (n >= 5)...")
        splitter = SequenceSplitter(max_seq_len=model_kwargs["max_seq_len"], min_seq_len=dataset_cfg["min_interactions"], include_oov_in_eval=True)
        split_data = splitter.split(sequences)

        print(f"  Vocabulary size: {split_data.num_items} items (Train-base catalog)")
        print(f"  Train-base pairs: {len(split_data.train_base_samples)}")
        print(f"  Validation targets: {len(split_data.val_samples)}")
        print(f"  Test targets: {len(split_data.test_samples)}")
        print(f"  OOV Targets: {split_data.oov_counts}")

        if is_real:
            processed_dir = str(resolve_project_path(dataset_cfg['processed_dir']) / run_id)
            save_processed_data(
                processed_dir=processed_dir,
                split_data=split_data,
                raw_reviews_path=reviews_path,
                config={
                    "max_seq_len": model_kwargs["max_seq_len"],
                    "min_seq_len": 5,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "seed": args.seed,
                },
            )
            print(f"  Processed dataset artifacts persisted to:\n    '{processed_dir}'")

        metadata_available = meta_path is not None if is_real else True
        stats = dataset_statistics(sequences, split_data, meta_dict, metadata_available,
                                   dataset_cfg['dataset_name'] if is_real else 'Synthetic_Fixture',
                                   dataset_cfg['expected_stats'] if is_real else None)
        contexts = {'validation': context_table(split_data.val_samples, split_data, meta_dict, metadata_available),
                    'test': context_table(split_data.test_samples, split_data, meta_dict, metadata_available)}

        # Build SASRec Model
        print("\nInitializing SASRec ID-only PyTorch model...")
        set_seed(args.seed)
        model = SASRecModel(
            num_items=split_data.num_items,
            **model_kwargs,
        )

        evaluator = IndependentEvaluator(
            num_items=split_data.num_items,
            k_list=k_list,
            mask_seen_items=mask_seen,
        )

        trainer = SASRecTrainer(
            model=model,
            evaluator=evaluator,
            device=args.device,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            patience=args.patience,
            seed=args.seed,
        )

        print(f"\nTraining SASRec model (max_epochs={args.epochs}, device={trainer.device})...")
        history, best_model = trainer.fit(
            train_samples=split_data.train_base_samples,
            val_samples=split_data.val_samples,
            seen_train_items=split_data.seen_train_items_per_user,
            max_epochs=args.epochs,
            checkpoint_path=checkpoint_file,
            verbose=True,
        )

        # Evaluate on Validation and Test
        val_res = trainer.evaluate(
            split_data.val_samples,
            seen_items=split_data.seen_train_items_per_user,
            batch_size=args.eval_batch_size,
        )
        test_res = trainer.evaluate(
            split_data.test_samples,
            seen_items=split_data.seen_train_items_per_user,
            batch_size=args.eval_batch_size,
        )

        print(f"\nFinal Validation Results: {val_res.to_dict()}")
        print(f"Final Test Results:       {test_res.to_dict()}")

        repeat_history = None
        repeat_test_res = None
        repeat_val_res = None
        if args.verify_reproducibility:
            print(f"\n[Check 5 Verification] Executing complete repeat run with fixed seed {args.seed}...")
            repeat_start = time.perf_counter()
            set_seed(args.seed)
            model_rep = SASRecModel(
                num_items=split_data.num_items,
                **model_kwargs,
            )
            trainer_rep = SASRecTrainer(
                model=model_rep,
                evaluator=evaluator,
                device=args.device,
                learning_rate=args.lr,
                weight_decay=args.weight_decay,
                batch_size=args.batch_size,
                patience=args.patience,
                seed=args.seed,
            )
            repeat_path = os.path.join(run_path, "repeat")
            os.makedirs(repeat_path, exist_ok=False)
            repeat_history, best_model_rep = trainer_rep.fit(
                train_samples=split_data.train_base_samples,
                val_samples=split_data.val_samples,
                seen_train_items=split_data.seen_train_items_per_user,
                max_epochs=args.epochs,
                checkpoint_path=os.path.join(repeat_path, "checkpoint.pt"),
                verbose=False,
            )
            repeat_test_res = trainer_rep.evaluate(
                split_data.test_samples,
                seen_items=split_data.seen_train_items_per_user,
                batch_size=args.eval_batch_size,
            )

            repeat_val_res = trainer_rep.evaluate(split_data.val_samples, split_data.seen_train_items_per_user, args.eval_batch_size)

        baseline_results = {}
        # Execute Sanity Checks
        print("\nRunning the 5 Mandatory Sanity Checks...")
        verdicts = run_sanity_checks(
            split_data=split_data,
            model=best_model,
            trainer=trainer,
            evaluator=evaluator,
            metadata_dict=meta_dict,
            seed=args.seed,
            batch_size=args.batch_size,
            history=history,
            repeat_history=repeat_history,
            repeat_test_res=repeat_test_res,
            repeat_val_res=repeat_val_res,
            original_val_res=val_res,
            repro_tolerance=args.repro_tolerance,
            is_real_data=is_real,
            raw_unique_items=raw_unique_items,
            expected_stats=dataset_cfg["expected_stats"],
            report_results=baseline_results,
        )

        for check_name, info in verdicts.items():
            print(f"  [{info['status']}] {check_name}")
            print(f"      Evidence: {info['evidence']}")

        # Save artifacts
        config_dict = {
            "experiment_name": exp_prefix,
            "is_real_data": is_real,
            "seed": args.seed,
            "num_items": split_data.num_items,
            "num_users": split_data.num_users,
            "max_seq_len": model_kwargs["max_seq_len"],
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "loss_type": "full_catalog_cross_entropy",
            "device": str(trainer.device),
            "mask_seen_items": mask_seen,
            "mask_seen_policy": "train_base" if mask_seen else "none",
            "tie_policy": "deterministic_catalog_index_ascending",
            "sanity_checks": verdicts,
            "model": model_kwargs,
            "dataset": dataset_cfg,
            "dataset_dir": str(Path(args.dataset_dir).resolve()),
            "patience": args.patience,
            "eval_batch_size": args.eval_batch_size,
            "weight_decay": args.weight_decay,
            "repro_tolerance": args.repro_tolerance,
            "k_list": k_list,
            "rank_format": args.rank_format,
            "processed_manifest": os.path.join(processed_dir, "manifest.json") if is_real else None,
            "data_provenance": json.loads(Path(processed_dir, 'manifest.json').read_text(encoding='utf-8')) if is_real else {'data_mode': 'synthetic_fixture'},
            "auxiliary_status": "AVAILABLE" if meta_dict else "BLOCKED_MISSING_METADATA",
            "mechanism": "id_only", "sources": [], "control": "none", "missing_rate": 0.0,
        }
        if is_real:
            stats['reviews_file'] = config_dict['data_provenance']['raw_reviews_file']
            stats['reviews_sha256'] = config_dict['data_provenance']['raw_reviews_sha256']
        if meta_path is not None:
            from taimma.data.split import compute_file_sha256
            stats['metadata_file'] = str(Path(meta_path).resolve())
            stats['metadata_sha256'] = compute_file_sha256(meta_path)
        else:
            stats['metadata_file'] = stats['metadata_sha256'] = None
        statuses = [v['status'] for v in verdicts.values()]
        run_status = ('SMOKE_TEST' if not is_real else 'FAIL' if 'FAIL' in statuses else
                      'BLOCKED' if 'BLOCKED' in statuses else 'PENDING' if any(v in ('PENDING', 'REVIEW') for v in statuses)
                      else 'PASS' if all(v == 'PASS' for v in statuses) else 'INVESTIGATE')
        logger.save_run_artifacts(
            run_path=run_path,
            run_id=run_id,
            config=config_dict,
            val_result=val_res,
            test_result=test_res,
            train_time=history.total_training_time,
            best_epoch=history.best_epoch,
            total_epochs=len(history.train_losses),
            actual_device=trainer.device,
            context_tables=contexts,
            rank_format=args.rank_format,
        )

        save_training_history(run_path, history)
        save_pilot_report(run_path, config_dict, stats, history, baseline_results, contexts,
                          runtime_snapshot(trainer.device), time.perf_counter() - run_start,
                          args.rank_format, run_status)
        if repeat_history is not None:
            repeat_config = dict(config_dict, parent_run_id=run_id)
            logger.save_run_artifacts(repeat_path, run_id + "_repeat", repeat_config,
                repeat_val_res, repeat_test_res, repeat_history.total_training_time,
                repeat_history.best_epoch, len(repeat_history.train_losses), actual_device=str(trainer_rep.device),
                context_tables=contexts, rank_format=args.rank_format)
            save_training_history(repeat_path, repeat_history)
            repeated_baselines = {key: value for key, value in baseline_results.items() if key != 'SASRec_shuffled'}
            repeated_baselines['SASRec_ID'] = {'validation': repeat_val_res, 'test': repeat_test_res}
            save_pilot_report(repeat_path, repeat_config, stats, repeat_history, repeated_baselines, contexts,
                              runtime_snapshot(trainer_rep.device), time.perf_counter() - repeat_start,
                              args.rank_format, 'REPRODUCIBILITY_REPEAT')

        logger.register_result(
            run_id=run_id,
            dataset="Beauty_2014" if is_real else "Synthetic_Fixture",
            model="SASRec_ID",
            seed=args.seed,
            epochs=len(history.train_losses),
            best_epoch=history.best_epoch,
            val_result=val_res,
            test_result=test_res,
            train_time=history.total_training_time,
            checkpoint_path=checkpoint_file,
            status=run_status,
        )

        print(f"\nArtifacts successfully written to: {run_path}")
        print(f"Run registered in: {logger.results_csv}")
        print("\nPilot pipeline execution completed!")


if __name__ == "__main__":
    main()
