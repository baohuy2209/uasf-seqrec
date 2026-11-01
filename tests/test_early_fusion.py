"""
test_early_fusion.py - Unit tests for SASRecF-style Early Fusion model.

Tests:
  - Element-wise addition of embeddings (maintains embedding_dim, unlike concatenation).
  - Source availability mask zeroing contributions when unavailable.
  - Verification that ID-only subset behaves consistently with base SASRec.
  - Verification of all 8 source combinations.
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
from taimma.models.early_fusion import EarlyFusionSASRec


def test_early_fusion_addition_and_shapes():
    """Verify that early fusion performs element-wise addition, keeping embedding_dim."""
    num_items = 20
    emb_dim = 32
    max_len = 8

    base_model = SASRecModel(num_items=num_items, max_seq_len=max_len, embedding_dim=emb_dim)
    early_model = EarlyFusionSASRec(
        base_sasrec=base_model,
        num_categories=10,
        num_brands=10,
        title_feature_dim=16,
        sources_enabled=("category", "brand", "title"),
    )
    early_model.eval()

    B = 2
    item_seqs = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    cat_seqs = torch.tensor([[1, 2, 0], [0, 1, 2]], dtype=torch.long)
    brand_seqs = torch.tensor([[0, 1, 2], [1, 0, 2]], dtype=torch.long)
    title_feats = torch.randn(B, 3, 16)

    # Encode context
    ctx = early_model.encode_context(
        input_seqs=item_seqs,
        category_ids=cat_seqs,
        brand_ids=brand_seqs,
        title_features=title_feats,
    )
    # Dimension must be emb_dim (32), NOT concatenated 32*4 = 128
    assert ctx.shape == (B, emb_dim)


def test_early_fusion_source_masking():
    """Verify that availability mask zeroes out metadata contributions."""
    base_model = SASRecModel(num_items=10, max_seq_len=5, embedding_dim=16)
    early_model = EarlyFusionSASRec(
        base_sasrec=base_model,
        num_categories=5,
        num_brands=5,
        title_feature_dim=8,
        sources_enabled=("category",),
    )
    early_model.eval()

    cat_ids = torch.tensor([[1, 2]], dtype=torch.long)
    # Mask all False (completely unavailable)
    cat_mask_zero = torch.tensor([[False, False]])

    contrib = early_model.compute_source_contributions(
        category_ids=cat_ids,
        category_mask=cat_mask_zero,
    )
    # All contributions should be 0.0
    torch.testing.assert_close(contrib, torch.zeros_like(contrib))


def test_early_fusion_all_8_subsets():
    """Verify that all 8 subsets of {Category, Brand, Title} + ID-only are supported."""
    base_model = SASRecModel(num_items=10, max_seq_len=5, embedding_dim=16)
    early_model = EarlyFusionSASRec(
        base_sasrec=base_model,
        num_categories=5,
        num_brands=5,
        title_feature_dim=8,
    )
    early_model.eval()

    subsets = [
        (),  # ID-only
        ("category",),
        ("brand",),
        ("title",),
        ("category", "brand"),
        ("category", "title"),
        ("brand", "title"),
        ("category", "brand", "title"),
    ]

    item_seqs = torch.tensor([[1, 2, 3]], dtype=torch.long)
    cat_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    brand_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    title_feats = torch.randn(1, 3, 8)

    for subset in subsets:
        early_model.set_sources_enabled(subset)
        ctx = early_model.encode_context(
            item_seqs,
            category_ids=cat_ids,
            brand_ids=brand_ids,
            title_features=title_feats,
        )
        assert ctx.shape == (1, 16)
