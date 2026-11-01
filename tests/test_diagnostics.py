"""
test_diagnostics.py - Unit tests for RQ1, RQ2, and RQ3 diagnostic methods.
"""

import sys
import os
import numpy as np
import torch
import pytest

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
TAIMMA_ROOT = os.path.dirname(TEST_DIR)
SRC_DIR = os.path.join(TAIMMA_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from taimma.data.split import SequenceSample
from taimma.experiments.diagnostics import (
    compute_marginal_utility_and_redundancy,
    permute_metadata_train_time,
    permute_metadata_test_time,
    generate_categorical_random_control,
    generate_title_gaussian_control,
    generate_item_missingness_mask,
    partition_by_sequence_length_quintiles,
    partition_by_item_popularity,
    compute_negative_transfer_rate,
)


def test_rq1_marginal_utility_and_redundancy():
    """Verify calculation of Add_m, Drop_m, and Redundancy across subsets."""
    metrics = {
        (): 0.030,
        ("category",): 0.035,
        ("brand",): 0.033,
        ("title",): 0.034,
        ("brand", "category"): 0.036,
        ("category", "title"): 0.037,
        ("brand", "title"): 0.035,
        ("brand", "category", "title"): 0.038,
    }

    results = compute_marginal_utility_and_redundancy(metrics, sources=("category", "brand", "title"))
    assert len(results) == 3

    cat_res = next(r for r in results if r.source_name == "category")
    # Add_cat = P(ID + cat) - P(ID) = 0.035 - 0.030 = 0.005
    assert abs(cat_res.add_m - 0.005) < 1e-5
    # Drop_cat = P(all) - P(all \ cat) = 0.038 - 0.035 = 0.003
    assert abs(cat_res.drop_m - 0.003) < 1e-5
    # Redundancy = Add - Drop = 0.005 - 0.003 = 0.002
    assert abs(cat_res.redundancy - 0.002) < 1e-5


def test_rq2_permutations_and_random_controls():
    """Verify train-time permutation, categorical control, and Gaussian title control."""
    cat_attrs = torch.tensor([1, 2, 3, 4, 5, 0, 1, 2], dtype=torch.long)
    permuted = permute_metadata_train_time(cat_attrs, seed=42)
    assert permuted.shape == cat_attrs.shape
    assert sorted(permuted.tolist()) == sorted(cat_attrs.tolist())

    cat_control = generate_categorical_random_control(cat_attrs, seed=42)
    assert cat_control.shape == cat_attrs.shape
    # Missing attribute at index 5 must remain 0
    assert cat_control[5] == 0

    title_embs = torch.randn(8, 16)
    title_embs[3] = 0.0  # missing title
    title_control = generate_title_gaussian_control(title_embs, seed=42)
    assert title_control.shape == (8, 16)
    assert torch.all(title_control[3] == 0.0)


def test_rq3_missingness_masks_and_subgroups():
    """Verify item missingness masking and subgroup partitioning."""
    mask_0 = generate_item_missingness_mask(100, 0.0)
    assert mask_0.sum().item() == 100.0

    mask_50 = generate_item_missingness_mask(100, 0.5, seed=42)
    assert 40 <= mask_50.sum().item() <= 60

    mask_100 = generate_item_missingness_mask(100, 1.0)
    assert mask_100.sum().item() == 0.0

    samples = [
        SequenceSample(user_idx=i, history=[1]*i, target=i+1, history_len=i)
        for i in range(1, 21)
    ]
    quintiles = partition_by_sequence_length_quintiles(samples)
    assert len(quintiles) == 5
    total_samples = sum(len(q) for q in quintiles.values())
    assert total_samples == 20

    head_s, tail_s = partition_by_item_popularity(samples, num_items=25, head_ratio=0.2, train_base_events=[item for sample in samples for item in sample.history + [sample.target]])
    assert len(head_s) + len(tail_s) == 20

    # Negative transfer test
    scores_base = np.array([[1.0, 5.0, 2.0], [5.0, 1.0, 2.0]])
    scores_multi = np.array([[1.0, 2.0, 5.0], [5.0, 1.0, 2.0]])
    targets = [2, 1]  # User 0 target is item 2, User 1 target is item 1
    neg_rate, worse_contexts = compute_negative_transfer_rate(scores_base, scores_multi, targets)
    # User 0: target 2 had rank 1 in base (score 5.0), in multi target 2 had rank 2 (score 2.0) -> worse!
    assert neg_rate == 0.5
    assert worse_contexts == [0]
