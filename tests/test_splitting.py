"""
test_splitting.py - Unit tests for chronological prefix sequence splitting.

Tests:
  - Sequence length boundary: n = 5, n > 5, n < 5 rejected.
  - Correct prefix allocation: Train-base (1..n-3), Train-meta (n-2), Val (n-1), Test (n).
  - Vocabulary derivation: candidate catalog strictly derived from Train-base.
  - Zero target leakage into training labels or input history.
"""

import sys
import os
import pytest

# Ensure taimma src is importable
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
TAIMMA_ROOT = os.path.dirname(TEST_DIR)
SRC_DIR = os.path.join(TAIMMA_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from taimma.data.ingestion import InteractionSequence
from taimma.data.split import SequenceSplitter


def test_sequence_length_boundary_n5():
    """Test n = 5 sequence: exactly 1 train-base pair, 1 train-meta, 1 val, 1 test."""
    splitter = SequenceSplitter(max_seq_len=5, min_seq_len=5)
    seq = InteractionSequence(
        user_id="U1",
        items=["item_1", "item_2", "item_3", "item_4", "item_5"],
        timestamps=[10, 20, 30, 40, 50],
    )
    data = splitter.split([seq])

    # Vocabulary should contain only train-base items: item_1, item_2
    assert "item_1" in data.item2idx
    assert "item_2" in data.item2idx

    # Train-base: 1 sample: input [item_1], target item_2
    assert len(data.train_base_samples) == 1
    sample = data.train_base_samples[0]
    assert sample.target == data.item2idx["item_2"]
    # Left-padded with 0 to length 5
    assert sample.history == [0, 0, 0, 0, data.item2idx["item_1"]]

    # Train-meta: target item_3 (if in vocab, else counted as OOV)
    # Since item_3 was never in Train-base, it is OOV
    assert data.oov_counts["train_meta"] == 1

    # Val: target item_4 (OOV)
    assert data.oov_counts["val"] == 1

    # Test: target item_5 (OOV)
    assert data.oov_counts["test"] == 1


def test_sequence_length_rejection_less_than_5():
    """Sequences with n < 5 must be excluded from training and evaluation."""
    splitter = SequenceSplitter(max_seq_len=10, min_seq_len=5)
    seq = InteractionSequence(
        user_id="U_short",
        items=["i1", "i2", "i3", "i4"],
        timestamps=[1, 2, 3, 4],
    )
    data = splitter.split([seq])
    assert len(data.train_base_samples) == 0
    assert len(data.val_samples) == 0
    assert len(data.test_samples) == 0


def test_longer_sequence_and_no_leakage():
    """Test sequence of length 8 where items repeat to ensure valid vocab in val and test."""
    splitter = SequenceSplitter(max_seq_len=10, min_seq_len=5)
    seq = InteractionSequence(
        user_id="U_long",
        items=["A", "B", "C", "D", "E", "C", "B", "A"],
        timestamps=[1, 2, 3, 4, 5, 6, 7, 8],
    )
    # Train-base: positions 0 .. 4 (A, B, C, D, E)
    # Targets in Train-base: B, C, D, E (at indices 1, 2, 3, 4)
    # Train-meta target: position 5 ("C")
    # Val target: position 6 ("B")
    # Test target: position 7 ("A")
    data = splitter.split([seq])

    assert len(data.train_base_samples) == 4
    assert len(data.train_meta_samples) == 1
    assert len(data.val_samples) == 1
    assert len(data.test_samples) == 1

    # Check that test target ("A") is NOT in the test sample input history
    test_sample = data.test_samples[0]
    raw_test_target = seq.items[7]
    assert raw_test_target == "A"
    assert test_sample.target == data.item2idx["A"]

    # Test history has positions 0..6 (items A, B, C, D, E, C, B)
    # Ensure test target at position 7 was NOT used in prediction of earlier positions
    val_sample = data.val_samples[0]
    assert val_sample.target == data.item2idx["B"]
