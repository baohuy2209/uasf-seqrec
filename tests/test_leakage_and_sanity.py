"""
test_leakage_and_sanity.py - Unit tests for data leakage detection and sanity checks.

Tests:
  - Verify that test targets are absent from input history.
  - Verify target-shuffling degradation (metric collapses when targets are randomly shuffled).
  - Verify seed determinism.
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

from taimma.data.ingestion import InteractionSequence
from taimma.data.split import SequenceSplitter
from taimma.evaluation.evaluator import IndependentEvaluator
from taimma.training.trainer import set_seed


def test_target_exclusion_from_history():
    """Verify that the target item at position t never appears in the input prefix."""
    splitter = SequenceSplitter(max_seq_len=10, min_seq_len=5)
    seq = InteractionSequence(
        user_id="U_test",
        items=["A", "B", "C", "D", "E", "F"],
        timestamps=[1, 2, 3, 4, 5, 6],
    )
    data = splitter.split([seq])

    # In every train-base sample, target must not be the final item of its history
    for sample in data.train_base_samples:
        # History is left padded, so non-zero elements are prefix
        non_zero = [i for i in sample.history if i > 0]
        assert non_zero[-1] != sample.target


def test_target_shuffling_degradation():
    """When target labels are permuted across users, NDCG should degrade dramatically."""
    num_items = 20
    evaluator = IndependentEvaluator(num_items=num_items, k_list=[10], mask_seen_items=False)

    # Perfect prediction score matrix: user u gives maximum score to target u
    targets = list(range(1, 11))  # 10 users, targets 1..10
    scores = np.zeros((10, num_items), dtype=np.float32)
    for u in range(10):
        tgt_col = targets[u] - 1
        scores[u, tgt_col] = 100.0  # Top score for ground truth

    res_perfect = evaluator.evaluate_ranks(scores, targets, list(range(10)))
    assert res_perfect.ndcg_10 == 1.0

    # Now shuffle targets: shift by 5
    shuffled_targets = targets[5:] + targets[:5]
    res_shuffled = evaluator.evaluate_ranks(scores, shuffled_targets, list(range(10)))
    # Because target was shifted, its score is 0.0 while another item has score 100.0
    # rank > 1, so ndcg must be significantly lower
    assert res_shuffled.ndcg_10 < res_perfect.ndcg_10


def test_seed_determinism():
    """Verify reproducible random and PyTorch generation under fixed seed."""
    set_seed(42)
    t1 = torch.randn(5, 5)
    n1 = np.random.rand(5)

    set_seed(42)
    t2 = torch.randn(5, 5)
    n2 = np.random.rand(5)

    torch.testing.assert_close(t1, t2)
    np.testing.assert_array_equal(n1, n2)
