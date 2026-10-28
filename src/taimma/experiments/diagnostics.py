"""
diagnostics.py - Experimental diagnostics runner for RQ1, RQ2, and RQ3.

Scientific Invariants:
  - RQ1: Add_m = P(ID + m) - P(ID), Drop_m = P(all) - P(all \\ m).
  - RQ2: Strict separation between train-time permutation (retrained), test-time permutation (frozen),
         and random controls (empirical frequency categorical sampling / Gaussian embedding for text).
  - RQ3: Consistent item-level missingness masks across history and candidate scoring.
         Subgroup analysis on sequence length quintiles and popularity head/tail.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd
import torch

from ..data.split import SequenceSample
from ..evaluation.evaluator import IndependentEvaluator, EvaluationResult


# ---------------------------------------------------------------------------
# RQ1: Marginal Utility and Redundancy
# ---------------------------------------------------------------------------

@dataclass
class MarginalUtilityResult:
    source_name: str
    metric_name: str
    p_id: float
    p_id_plus_m: float
    p_all: float
    p_all_minus_m: float
    add_m: float       # P(ID + m) - P(ID)
    drop_m: float      # P(all) - P(all \ m)
    redundancy: float  # Add_m - Drop_m (positive indicates redundancy)


def compute_marginal_utility_and_redundancy(
    subset_metrics: Dict[Tuple[str, ...], float],
    sources: Tuple[str, ...] = ("category", "brand", "title"),
    metric_name: str = "NDCG@10",
) -> List[MarginalUtilityResult]:
    """
    Compute Add_m, Drop_m, and Redundancy across evaluated source combinations.
    
    Args:
        subset_metrics: Dict mapping source tuple to metric value.
                        ID-only is represented by () or ("id",).
                        All-sources is tuple(sources).
        sources: List of available metadata sources.
        metric_name: Name of target metric.
    """
    id_tuple = () if () in subset_metrics else (("id",) if ("id",) in subset_metrics else None)
    if id_tuple is None:
        raise KeyError("subset_metrics must contain ID-only entry: () or ('id',)")

    p_id = subset_metrics[id_tuple]
    p_all = subset_metrics[tuple(sorted(sources))]

    results = []
    for m in sources:
        add_tuple = (m,)
        p_id_plus_m = subset_metrics[add_tuple]
        add_m = p_id_plus_m - p_id

        drop_subset = tuple(sorted([s for s in sources if s != m]))
        p_all_minus_m = subset_metrics[drop_subset]
        drop_m = p_all - p_all_minus_m

        redundancy = add_m - drop_m

        results.append(
            MarginalUtilityResult(
                source_name=m,
                metric_name=metric_name,
                p_id=round(p_id, 6),
                p_id_plus_m=round(p_id_plus_m, 6),
                p_all=round(p_all, 6),
                p_all_minus_m=round(p_all_minus_m, 6),
                add_m=round(add_m, 6),
                drop_m=round(drop_m, 6),
                redundancy=round(redundancy, 6),
            )
        )
    return results


# ---------------------------------------------------------------------------
# RQ2: Semantics vs Regularization (Permutations & Random Controls)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# RQ2: Semantics vs Regularization (Permutations & Random Controls)
# ---------------------------------------------------------------------------

def permute_metadata_train_time(
    catalog_attrs: torch.Tensor,
    availability_mask: Optional[torch.Tensor] = None,
    seed: int = 42,
) -> torch.Tensor:
    """
    Permute item-to-attribute mapping among items that possess the attribute,
    preserving the original missingness pattern and empirical marginal distribution.
    Used for train-time permutation (requires retraining under this mapping).
    """
    rng = np.random.default_rng(seed)
    permuted_attrs = catalog_attrs.clone()

    if availability_mask is not None:
        has_attr = availability_mask > 0
    else:
        if catalog_attrs.is_floating_point() and catalog_attrs.ndim > 1:
            has_attr = torch.linalg.norm(catalog_attrs, dim=-1) > 1e-6
        else:
            if catalog_attrs.dim() > 1:
                has_attr = (catalog_attrs > 0).any(dim=-1)
            else:
                has_attr = catalog_attrs > 0

    active_indices = torch.where(has_attr)[0].cpu().numpy()
    if len(active_indices) > 1:
        permuted_order = rng.permutation(active_indices)
        permuted_attrs[active_indices] = catalog_attrs[permuted_order]

    return permuted_attrs


def permute_metadata_test_time(
    catalog_attrs: torch.Tensor,
    availability_mask: Optional[torch.Tensor] = None,
    seed: int = 42,
) -> torch.Tensor:
    """
    Permute item-to-attribute mapping at inference/test time only (frozen model).
    Preserves missingness pattern by permuting only among items with valid attributes.
    """
    return permute_metadata_train_time(catalog_attrs, availability_mask=availability_mask, seed=seed)


def generate_categorical_random_control(
    catalog_attrs: torch.Tensor,
    vocab_size: Optional[int] = None,
    seed: int = 42,
) -> torch.Tensor:
    """
    Sample random attribute values uniformly from the approved vocabulary (1..vocab_size).
    Preserves missingness (0/padding) and the exact number of labels per item.
    """
    rng = np.random.default_rng(seed)
    attrs_np = catalog_attrs.cpu().numpy()

    if vocab_size is None:
        vocab_size = max(int(attrs_np.max()), 1)

    random_attrs = np.zeros_like(attrs_np)

    if attrs_np.ndim == 1:
        has_attr = attrs_np > 0
        n_active = int(has_attr.sum())
        if n_active > 0:
            random_attrs[has_attr] = rng.integers(1, vocab_size + 1, size=n_active)
    else:
        # Multilabel: preserve count of non-zero labels per item
        for i in range(attrs_np.shape[0]):
            k_i = int(np.count_nonzero(attrs_np[i]))
            if k_i > 0:
                if k_i > vocab_size:
                    raise ValueError("Vocabulary too small to preserve the item's label count")
                k_sample = k_i
                sampled_labels = rng.choice(np.arange(1, vocab_size + 1), size=k_sample, replace=False)
                random_attrs[i, :k_sample] = sampled_labels

    return torch.tensor(random_attrs, dtype=catalog_attrs.dtype, device=catalog_attrs.device)


def generate_title_gaussian_control(
    catalog_title_embeddings: torch.Tensor,
    seed: int = 42,
) -> torch.Tensor:
    """
    Generate synthetic Gaussian text embeddings matching the empirical mean and full covariance
    matrix of real embeddings.
    Explicit numerical stabilization (1e-6 * I) prevents singular covariance.
    Zeroes out embeddings for missing titles (norm == 0).
    """
    rng = np.random.default_rng(seed)
    emb_np = catalog_title_embeddings.cpu().numpy()
    norms = np.linalg.norm(emb_np, axis=-1)
    valid_mask = norms > 1e-6

    if not np.any(valid_mask):
        return catalog_title_embeddings.clone()

    valid_emb = emb_np[valid_mask]
    mean = np.mean(valid_emb, axis=0)
    dim = emb_np.shape[1]

    # Retain off-diagonal structure even when empirical covariance is rank deficient.
    if len(valid_emb) < 2:
        raise ValueError("Full-covariance Title control requires at least two available titles")
    cov = np.atleast_2d(np.cov(valid_emb, rowvar=False)) + 1e-6 * np.eye(dim)

    synthetic_valid = rng.multivariate_normal(mean=mean, cov=cov, size=len(valid_emb)).astype(np.float32)

    synthetic = np.zeros_like(emb_np)
    synthetic[valid_mask] = synthetic_valid

    return torch.tensor(synthetic, dtype=catalog_title_embeddings.dtype, device=catalog_title_embeddings.device)


# ---------------------------------------------------------------------------
# RQ3: Missing Metadata and Subgroup Behavior
# ---------------------------------------------------------------------------

def generate_item_missingness_mask(
    num_items: int,
    missing_rate: float,
    seed: int = 42,
) -> torch.Tensor:
    """
    Generate deterministic item-level availability mask for inference-time missingness.
    1 indicates source is available, 0 indicates missing.
    """
    if not 0 <= missing_rate <= 1 or num_items < 1:
        raise ValueError("missing_rate must be in [0,1] and num_items must be positive")
    if missing_rate <= 0.0:
        return torch.ones(num_items, dtype=torch.float32)
    if missing_rate >= 1.0:
        return torch.zeros(num_items, dtype=torch.float32)

    rng = np.random.default_rng(seed)
    probs = rng.uniform(0.0, 1.0, size=num_items)
    mask = (probs >= missing_rate).astype(np.float32)
    return torch.tensor(mask, dtype=torch.float32)


def partition_by_sequence_length_quintiles(
    samples: List[SequenceSample],
) -> Dict[str, List[SequenceSample]]:
    """Partition user contexts into 5 sequence length quintiles (Q1 to Q5) using rank-based partition."""
    if len(samples) == 0:
        return {f"Q{i}": [] for i in range(1, 6)}

    lens = np.array([s.history_len for s in samples])
    sorted_order = np.argsort(lens)
    n = len(samples)
    q_size = max(1, n // 5)

    partitions: Dict[str, List[SequenceSample]] = {}
    partitions["Q1"] = [samples[i] for i in sorted_order[:q_size]]
    partitions["Q2"] = [samples[i] for i in sorted_order[q_size: 2 * q_size]]
    partitions["Q3"] = [samples[i] for i in sorted_order[2 * q_size: 3 * q_size]]
    partitions["Q4"] = [samples[i] for i in sorted_order[3 * q_size: 4 * q_size]]
    partitions["Q5"] = [samples[i] for i in sorted_order[4 * q_size:]]

    return partitions


def partition_by_item_popularity(
    samples: List[SequenceSample],
    train_base_samples: Optional[List[SequenceSample]] = None,
    num_items: int = 0,
    head_ratio: float = 0.2,
    train_base_events: Optional[List[int]] = None,
) -> Tuple[List[SequenceSample], List[SequenceSample]]:
    """
    Partition evaluation samples based on target item popularity across all Train-base interaction events.
    Deterministic secondary tie-breaking by catalog item index ascending.
    Returns (head_samples, tail_samples).
    """
    if train_base_events is None:
        raise ValueError("Head/tail partition requires complete train_base_events")
    if num_items < 1 or not 0 < head_ratio <= 1:
        raise ValueError("Invalid catalog size or head_ratio")
    pop_counts = np.zeros(num_items + 1, dtype=np.int64)
    for itm in train_base_events:
        if not 0 < itm <= num_items:
            raise ValueError("Train-base event outside catalog")
        pop_counts[itm] += 1

    valid_items = np.where(pop_counts[1:] > 0)[0] + 1
    # Deterministic tie-breaking: primary -pop_counts, secondary itm index
    sorted_items = sorted(valid_items, key=lambda itm: (-pop_counts[itm], itm))
    num_head = max(int(len(sorted_items) * head_ratio), 1)
    head_set = set(sorted_items[:num_head])

    head_samples = [s for s in samples if s.target in head_set]
    tail_samples = [s for s in samples if s.target not in head_set]
    return head_samples, tail_samples


def compute_context_worsened_rank_rate(
    scores_baseline: np.ndarray,
    scores_multimodal: np.ndarray,
    targets: List[int],
    user_indices: Optional[List[int]] = None,
    seen_items_per_user: Optional[Dict[int, Set[int]]] = None,
) -> Tuple[float, List[int]]:
    """
    Compute fraction of contexts where multimodal model has strictly worse rank than baseline.
    Correctly treats OOV targets (target == 0) as tied worst rank N+1 for both methods (no degradation).
    """
    B, N = scores_baseline.shape
    if scores_multimodal.shape != scores_baseline.shape:
        raise ValueError("Score matrices must have matching shapes")
    evaluator = IndependentEvaluator(N, mask_seen_items=seen_items_per_user is not None)
    users = list(range(B)) if user_indices is None else user_indices
    base = evaluator.evaluate_ranks(scores_baseline, targets, users, seen_items_per_user)
    multi = evaluator.evaluate_ranks(scores_multimodal, targets, users, seen_items_per_user)
    worse = [i for i, (a, b) in enumerate(zip(base.per_user_records, multi.per_user_records)) if b.rank > a.rank]
    return len(worse) / B, worse


def compute_supervisor_negative_transfer_groups(
    group_metrics_baseline: Dict[str, float],
    group_metrics_multimodal: Dict[str, float],
) -> Tuple[float, List[str]]:
    """
    Supervisor metric: Fraction of sequence-length groups where multimodal NDCG@10 < ID-only NDCG@10.
    """
    common_groups = [g for g in group_metrics_baseline if g in group_metrics_multimodal]
    if not common_groups:
        return 0.0, []

    worse_groups = [
        g for g in common_groups
        if group_metrics_multimodal[g] < group_metrics_baseline[g]
    ]
    frac_worse = len(worse_groups) / len(common_groups)
    return frac_worse, worse_groups


# Backwards compatibility alias
compute_negative_transfer_rate = compute_context_worsened_rank_rate


def save_control_artifact(directory, dataset: str, source: str, control: str,
                          rate: float, seed: int, values: torch.Tensor):
    """Persist a shared item control with an explicit identity and checksum.

    This saves only the caller's actual control tensor; it does not generate
    data, choose rates or substitute for a missing source.
    """
    import hashlib
    import json
    from pathlib import Path
    if source not in ('category', 'brand', 'title') or control not in ('missing', 'permutation_train', 'permutation_test', 'random'):
        raise ValueError('Unknown source or control type')
    if not 0 <= rate <= 1 or not dataset or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in dataset):
        raise ValueError('Invalid dataset identifier or rate')
    if not torch.isfinite(values).all():
        raise ValueError('Control tensor contains nonfinite values')
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    stem = f'{dataset}_{source}_{control}_rate{rate:g}_seed{seed}'
    array_path, manifest_path = folder / (stem + '.npy'), folder / (stem + '.json')
    if array_path.exists() or manifest_path.exists():
        raise FileExistsError(f'Refusing to overwrite control: {stem}')
    with array_path.open('xb') as f:
        np.save(f, values.detach().cpu().numpy(), allow_pickle=False)
    metadata = dict(dataset=dataset, source=source, control=control, rate=rate, seed=seed,
                    shape=list(values.shape), dtype=str(values.dtype),
                    sha256=hashlib.sha256(array_path.read_bytes()).hexdigest(), file=array_path.name)
    with manifest_path.open('x', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, allow_nan=False)
    return str(array_path), str(manifest_path)
