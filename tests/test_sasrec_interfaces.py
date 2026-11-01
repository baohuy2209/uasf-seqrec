"""
test_sasrec_interfaces.py - Unit tests for SASRec expected standard interfaces and checkpointing.

Tests:
  - encode_item_sequence(item_ids, padding_mask) -> hidden_states
  - encode_context(item_ids, sequence_lengths) -> context_embeddings
  - score_candidates(context_embeddings, candidate_item_ids) -> candidate_scores
  - score_full_catalog(context_embeddings) -> catalog_scores
  - Checkpoint round-trip save and load.
"""

import sys
import os
import tempfile
import torch
import pytest

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
TAIMMA_ROOT = os.path.dirname(TEST_DIR)
SRC_DIR = os.path.join(TAIMMA_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from taimma.models.sasrec import SASRecModel


def test_standard_interfaces():
    num_items = 25
    max_len = 10
    emb_dim = 32

    model = SASRecModel(
        num_items=num_items,
        max_seq_len=max_len,
        embedding_dim=emb_dim,
        num_blocks=1,
        num_heads=2,
    )
    model.eval()

    batch_seqs = torch.tensor([[0, 0, 1, 2, 3], [0, 4, 5, 6, 7]], dtype=torch.long)
    lengths = torch.tensor([3, 4], dtype=torch.long)

    # 1. encode_item_sequence
    hidden_states = model.encode_item_sequence(batch_seqs)
    assert hidden_states.shape == (2, 5, emb_dim)

    # 2. encode_context
    context = model.encode_context(batch_seqs, sequence_lengths=lengths)
    assert context.shape == (2, emb_dim)

    # Default without lengths should return last position
    context_last = model.encode_context(batch_seqs)
    torch.testing.assert_close(context_last, hidden_states[:, -1, :])

    # 3. score_candidates (1D broadcast)
    candidates_1d = torch.tensor([1, 5, 10, 15], dtype=torch.long)
    scores_1d = model.score_candidates(context, candidates_1d)
    assert scores_1d.shape == (2, 4)

    # 3b. score_candidates (2D per-user)
    candidates_2d = torch.tensor([[1, 2], [3, 4]], dtype=torch.long)
    scores_2d = model.score_candidates(context, candidates_2d)
    assert scores_2d.shape == (2, 2)

    # 4. score_full_catalog
    catalog_scores = model.score_full_catalog(context)
    # Must exclude padding index 0, so shape is (B, num_items)
    assert catalog_scores.shape == (2, num_items)


def test_checkpoint_round_trip():
    """Verify that saving and loading checkpoint recovers identical model behavior."""
    num_items = 15
    model = SASRecModel(num_items=num_items, max_seq_len=8, embedding_dim=16)
    model.eval()

    batch_seqs = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    with torch.no_grad():
        orig_scores = model.predict_all(batch_seqs)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        torch.save({"model_state_dict": model.state_dict(), "num_items": num_items}, tmp_path)

        new_model = SASRecModel(num_items=num_items, max_seq_len=8, embedding_dim=16)
        ckpt = torch.load(tmp_path, weights_only=True)
        new_model.load_state_dict(ckpt["model_state_dict"])
        new_model.eval()

        with torch.no_grad():
            loaded_scores = new_model.predict_all(batch_seqs)

        torch.testing.assert_close(orig_scores, loaded_scores)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
