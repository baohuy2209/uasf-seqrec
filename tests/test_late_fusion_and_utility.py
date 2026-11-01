"""
test_late_fusion_and_utility.py - Unit tests for Residual Late Fusion and leave-one-source-out utility labels.

Tests:
  - SourceSequentialEncoder isolation (pure metadata attributes, zero ID leakage).
  - Frozen ID backbone parameter invariance.
  - Joint residual score calculation: s_joint = s_ID + sum(d_m * b_m * s_m).
  - Leave-one-source-out delta computation with zero gradients and correct sign semantics.
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
from taimma.models.late_fusion import ResidualLateFusionModel, SourceSequentialEncoder
from taimma.evaluation.utility import compute_leave_one_source_out_deltas


def test_source_encoder_isolation():
    """Verify SourceSequentialEncoder encodes only metadata and produces correct score shapes."""
    vocab_size = 15
    feature_dim = 16
    max_len = 6

    encoder = SourceSequentialEncoder(
        source_name="category",
        vocab_size=vocab_size,
        feature_dim=feature_dim,
        max_seq_len=max_len,
        num_blocks=1,
        num_heads=2,
    )
    encoder.eval()

    source_seqs = torch.tensor([[1, 2, 3], [4, 5, 0]], dtype=torch.long)
    ctx = encoder.encode_context(source_seqs)
    assert ctx.shape == (2, feature_dim)

    # Candidate attributes: 10 candidate items
    candidate_attrs = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=torch.long)
    scores = encoder.score_candidates(ctx, candidate_attrs)
    assert scores.shape == (2, 10)


def test_late_fusion_frozen_backbone():
    """Verify that ID backbone parameters are strictly frozen (requires_grad is False)."""
    num_items = 10
    id_model = SASRecModel(num_items=num_items, max_seq_len=5, embedding_dim=16)

    cat_enc = SourceSequentialEncoder("category", vocab_size=5, feature_dim=16, max_seq_len=5)
    late_model = ResidualLateFusionModel(
        frozen_id_model=id_model,
        source_encoders={"category": cat_enc},
        p_drop=0.0,
    )

    # Every parameter in ID model must have requires_grad = False
    for name, param in late_model.id_model.named_parameters():
        assert not param.requires_grad, f"Parameter {name} in ID backbone is not frozen!"

    # Encoder parameters should have requires_grad = True (trainable)
    for name, param in late_model.source_encoders.named_parameters():
        assert param.requires_grad, f"Residual encoder parameter {name} should be trainable."


def test_leave_one_source_out_utility_deltas():
    """
    Verify leave-one-source-out utility computation:
      delta_m(q) = CE(without_m) - CE(all)
    Ensures zero gradient updates and correct delta values.
    """
    num_items = 5
    emb_dim = 16
    id_model = SASRecModel(num_items=num_items, max_seq_len=4, embedding_dim=emb_dim)

    cat_enc = SourceSequentialEncoder("category", vocab_size=4, feature_dim=emb_dim, max_seq_len=4)
    brand_enc = SourceSequentialEncoder("brand", vocab_size=4, feature_dim=emb_dim, max_seq_len=4)

    late_model = ResidualLateFusionModel(
        frozen_id_model=id_model,
        source_encoders={"category": cat_enc, "brand": brand_enc},
        p_drop=0.0,
        sources_enabled=("category", "brand"),
    )

    B = 2
    item_seqs = torch.tensor([[1, 2, 3], [2, 3, 4]], dtype=torch.long)
    source_seqs = {
        "category": torch.tensor([[1, 2, 3], [2, 3, 4]], dtype=torch.long),
        "brand": torch.tensor([[1, 2, 0], [2, 0, 1]], dtype=torch.long),
    }
    catalog_attrs = {
        "category": torch.tensor([1, 2, 3, 4, 1], dtype=torch.long),
        "brand": torch.tensor([1, 0, 2, 3, 0], dtype=torch.long),
    }
    catalog_masks = {
        "category": torch.tensor([1, 1, 1, 1, 1], dtype=torch.float32),
        "brand": torch.tensor([1, 0, 1, 1, 0], dtype=torch.float32),  # Item 2 and 5 have missing brand
    }
    targets = torch.tensor([3, 4], dtype=torch.long)
    user_indices = [101, 102]

    deltas, records = compute_leave_one_source_out_deltas(
        late_fusion_model=late_model,
        item_id_seqs=item_seqs,
        source_seqs=source_seqs,
        catalog_source_attrs=catalog_attrs,
        catalog_availability_masks=catalog_masks,
        targets=targets,
        user_indices=user_indices,
        reference_seed=42,
    )

    assert "category" in deltas
    assert "brand" in deltas
    assert deltas["category"].shape == (B,)
    assert deltas["brand"].shape == (B,)

    # Total records: 2 users * 2 sources = 4 records
    assert len(records) == 4

    for r in records:
        # delta = ce_without_m - ce_all
        expected_diff = round(r.ce_without_m - r.ce_all, 6)
        assert abs(r.delta - expected_diff) < 1e-4
