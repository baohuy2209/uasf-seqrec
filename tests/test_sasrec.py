"""
test_sasrec.py - Unit tests for PyTorch SASRec model.

Tests:
  - Output tensor dimensions.
  - Causal masking: changes in future positions do not affect past position representations.
  - Full-catalog Cross-Entropy loss computation.
  - Gradient computation and backward pass.
"""

import sys
import os
import torch
import pytest

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
TAIMMA_ROOT = os.path.dirname(TEST_DIR)
SRC_DIR = os.path.join(TAIMMA_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from taimma.models.sasrec import SASRecModel


def test_sasrec_dimensions_and_forward():
    """Verify output shapes of sequence encoding and prediction."""
    num_items = 20
    max_len = 10
    emb_dim = 32

    model = SASRecModel(
        num_items=num_items,
        max_seq_len=max_len,
        embedding_dim=emb_dim,
        num_blocks=1,
        num_heads=2,
        dropout_rate=0.0,
    )

    batch_seqs = torch.tensor([[0, 0, 1, 2, 3], [0, 4, 5, 6, 7]], dtype=torch.long)
    reps = model(batch_seqs)
    assert reps.shape == (2, 5, emb_dim)

    user_rep = model.get_user_representation(batch_seqs)
    assert user_rep.shape == (2, emb_dim)

    logits = model.predict_all(batch_seqs)
    assert logits.shape == (2, num_items)


def test_sasrec_causal_masking():
    """Verify that future tokens do not alter the representations of preceding tokens."""
    num_items = 10
    max_len = 6
    emb_dim = 16

    model = SASRecModel(
        num_items=num_items,
        max_seq_len=max_len,
        embedding_dim=emb_dim,
        num_blocks=2,
        num_heads=2,
        dropout_rate=0.0,
    )
    model.eval()

    # Sequence A: [1, 2, 3]
    seq_a = torch.tensor([[1, 2, 3, 0, 0, 0]], dtype=torch.long)
    # Sequence B: [1, 2, 4] (token at position 2 is different: 4 vs 3)
    seq_b = torch.tensor([[1, 2, 4, 0, 0, 0]], dtype=torch.long)

    with torch.no_grad():
        rep_a = model(seq_a)
        rep_b = model(seq_b)

    # Position 0 (token 1) and position 1 (token 2) must be IDENTICAL between A and B
    torch.testing.assert_close(rep_a[:, 0, :], rep_b[:, 0, :])
    torch.testing.assert_close(rep_a[:, 1, :], rep_b[:, 1, :])

    # Position 2 should differ because the token at pos 2 differs
    assert not torch.allclose(rep_a[:, 2, :], rep_b[:, 2, :])


def test_full_catalog_ce_loss_and_backward():
    """Verify that compute_loss executes cross entropy and gradients flow."""
    num_items = 15
    max_len = 8
    emb_dim = 16

    model = SASRecModel(
        num_items=num_items,
        max_seq_len=max_len,
        embedding_dim=emb_dim,
        num_blocks=1,
        num_heads=2,
    )

    batch_seqs = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    targets = torch.tensor([4, 7], dtype=torch.long)

    loss = model.compute_loss(batch_seqs, targets)
    assert loss.dim() == 0  # Scalar
    assert loss.item() > 0

    loss.backward()
    assert model.item_embedding.weight.grad is not None
