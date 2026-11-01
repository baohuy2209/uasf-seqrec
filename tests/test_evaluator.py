"""
test_evaluator.py - Unit tests for independent ranking evaluation and baselines.

Tests:
  - Exact NDCG and Recall calculation against ground-truth analytical formulas.
  - Seen-item masking effect on target rank.
  - PopularityBaseline frequency calculation.
  - RandomBaseline reproducibility.
"""

import math
import sys
import os
import numpy as np
import pytest

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
TAIMMA_ROOT = os.path.dirname(TEST_DIR)
SRC_DIR = os.path.join(TAIMMA_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from taimma.evaluation.evaluator import IndependentEvaluator
from taimma.evaluation.baselines import PopularityBaseline, RandomBaseline
from taimma.data.split import SequenceSample


def test_evaluator_exact_metrics():
    """Verify NDCG@10 and Recall@10 on handcrafted score matrix."""
    num_items = 5
    evaluator = IndependentEvaluator(num_items=num_items, k_list=[10, 20], mask_seen_items=False)

    # User 0: scores for items 1..5: [10.0, 50.0, 30.0, 20.0, 40.0]
    # Sorted order of items: item 2 (50), item 5 (40), item 3 (30), item 4 (20), item 1 (10)
    # Target item 3 is rank 3 (scores higher: item 2, item 5 -> 2 items -> rank 3)
    scores = np.array([[10.0, 50.0, 30.0, 20.0, 40.0]], dtype=np.float32)
    targets = [3]  # item 3
    users = [0]

    res = evaluator.evaluate_ranks(scores, targets, users)
    assert len(res.per_user_records) == 1
    rec = res.per_user_records[0]
    assert rec.rank == 3
    assert rec.recall_10 == 1.0
    expected_ndcg = 1.0 / math.log2(3 + 1)  # 1.0 / 2.0 = 0.5
    assert math.isclose(rec.ndcg_10, expected_ndcg, rel_tol=1e-5)


def test_evaluator_seen_item_masking():
    """Verify that seen-item masking improves the target rank by filtering historical items."""
    num_items = 4
    # User interacted with item 2 and item 4 in train
    # Target is item 3
    # Scores for items 1..4: [1.0, 10.0, 5.0, 8.0]
    # Without masking: item 2 (10.0) and item 4 (8.0) > item 3 (5.0) -> rank = 3
    # With masking: item 2 and 4 are masked to -inf -> item 3 is top ranked -> rank = 1
    scores = np.array([[1.0, 10.0, 5.0, 8.0]], dtype=np.float32)
    targets = [3]
    users = [0]
    seen_train = {0: {2, 4}}

    # 1. Without masking
    eval_no_mask = IndependentEvaluator(num_items=num_items, mask_seen_items=False)
    res_no_mask = eval_no_mask.evaluate_ranks(scores, targets, users, seen_items_per_user=seen_train)
    assert res_no_mask.per_user_records[0].rank == 3

    # 2. With masking
    eval_mask = IndependentEvaluator(num_items=num_items, mask_seen_items=True)
    res_mask = eval_mask.evaluate_ranks(scores, targets, users, seen_items_per_user=seen_train)
    assert res_mask.per_user_records[0].rank == 1
    assert math.isclose(res_mask.per_user_records[0].ndcg_10, 1.0)


def test_popularity_baseline_ranking():
    """Verify that PopularityBaseline correctly orders items by Train-base frequency."""
    num_items = 4
    pop = PopularityBaseline(num_items=num_items)
    # Samples where item 3 appears 3 times, item 1 appears 1 time, items 2 and 4 appear 0 times
    samples = [
        SequenceSample(user_idx=0, history=[0, 1], target=3, history_len=2),
        SequenceSample(user_idx=1, history=[0, 3], target=3, history_len=2),
    ]
    pop.fit(train_base_events=[1, 3, 3, 3])
    # item 3 has count 3, item 1 has count 1, others 0
    scores = pop.predict_all(batch_size=1)
    # item 3 (col 2) should be highest, item 1 (col 0) second
    assert scores[0, 2] > scores[0, 0]
    assert scores[0, 0] > scores[0, 1]


def test_random_baseline_reproducibility():
    """Verify that RandomBaseline produces identical outputs under identical seeds."""
    r1 = RandomBaseline(num_items=10, seed=123)
    r2 = RandomBaseline(num_items=10, seed=123)
    s1 = r1.predict_all(batch_size=5)
    s2 = r2.predict_all(batch_size=5)
    np.testing.assert_array_equal(s1, s2)
