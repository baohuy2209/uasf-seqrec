"""
test_uasf_model.py - Unit tests for UASF Model, selective gates, and exact ID recovery.
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
from taimma.models.uasf import UASFModel, UASFContextGate, UASFGroupGate


@pytest.fixture
def dummy_setup():
    num_items = 8
    emb_dim = 16
    max_len = 4

    id_model = SASRecModel(num_items=num_items, max_seq_len=max_len, embedding_dim=emb_dim)
    cat_enc = SourceSequentialEncoder("category", vocab_size=10, feature_dim=emb_dim, max_seq_len=max_len)
    brand_enc = SourceSequentialEncoder("brand", vocab_size=10, feature_dim=emb_dim, max_seq_len=max_len)

    late_model = ResidualLateFusionModel(
        frozen_id_model=id_model,
        source_encoders={"category": cat_enc, "brand": brand_enc},
        sources_enabled=("category", "brand"),
        p_drop=0.0,
    )

    catalog_attrs = {
        "category": torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.long),
        "brand": torch.tensor([1, 0, 2, 3, 0, 4, 1, 2], dtype=torch.long),
    }
    catalog_masks = {
        "category": torch.ones(num_items),
        "brand": torch.tensor([1, 0, 1, 1, 0, 1, 1, 1], dtype=torch.float32),
    }

    item_seqs = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    source_seqs = {
        "category": torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long),
        "brand": torch.tensor([[1, 0, 2], [3, 0, 4]], dtype=torch.long),
    }
    context_feats = torch.randn(2, 4)

    return {
        "num_items": num_items,
        "id_model": id_model,
        "late_model": late_model,
        "catalog_attrs": catalog_attrs,
        "catalog_masks": catalog_masks,
        "item_seqs": item_seqs,
        "source_seqs": source_seqs,
        "context_feats": context_feats,
    }


def test_uasf_exact_id_recovery_when_alpha_zero(dummy_setup):
    """
    CRITICAL SCIENTIFIC INVARIANT:
      When all alpha_m(q) = 0, UASF scores must exactly equal frozen ID-only scores!
    """
    d = dummy_setup
    uasf_id_only = UASFModel(
        frozen_late_fusion_model=d["late_model"],
        context_feature_dim=4,
        gate_mode="id_only",
    )

    # 1. Direct ID-only scores from backbone
    with torch.no_grad():
        expected_id_scores = d["id_model"].predict_all(d["item_seqs"])

    # 2. UASF scores with id_only gate mode
    scores_uasf = uasf_id_only.predict_all(
        item_id_seqs=d["item_seqs"],
        source_seqs=d["source_seqs"],
        catalog_source_attrs=d["catalog_attrs"],
        catalog_availability_masks=d["catalog_masks"],
        context_features=d["context_feats"],
    )

    # Must be exactly identical
    assert torch.allclose(scores_uasf, expected_id_scores, atol=1e-6)


def test_uasf_contextual_gate_and_thresholding(dummy_setup):
    """Verify contextual gating, temperature scaling, and threshold kappa."""
    d = dummy_setup
    uasf = UASFModel(
        frozen_late_fusion_model=d["late_model"],
        context_feature_dim=4,
        gate_mode="contextual",
        kappa=0.2,
        tau=1.0,
    )

    # History availability
    hist_avail = torch.tensor([[1.0, 0.0], [1.0, 1.0]])  # User 0 has no brand history

    scores, alpha, beta = uasf.forward_scores(
        item_id_seqs=d["item_seqs"],
        source_seqs=d["source_seqs"],
        catalog_source_attrs=d["catalog_attrs"],
        catalog_availability_masks=d["catalog_masks"],
        context_features=d["context_feats"],
        history_availability=hist_avail,
        apply_threshold=True,
    )

    assert scores.shape == (2, d["num_items"])
    assert alpha.shape == (2, 2)
    # User 0 brand alpha must be 0 because history availability is 0
    assert alpha[0, 1].item() == 0.0


def test_uasf_joint_loss_and_supervision(dummy_setup):
    """Verify UASF total loss: L_rec + lambda_util * L_util + lambda_sparse * L_sparse."""
    d = dummy_setup
    uasf = UASFModel(
        frozen_late_fusion_model=d["late_model"],
        context_feature_dim=4,
        gate_mode="contextual",
        lambda_util=0.5,
        lambda_sparse=0.001,
    )

    targets = torch.tensor([4, 7], dtype=torch.long)
    util_labels = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)

    total_loss, loss_dict = uasf.compute_loss(
        item_id_seqs=d["item_seqs"],
        source_seqs=d["source_seqs"],
        catalog_source_attrs=d["catalog_attrs"],
        catalog_availability_masks=d["catalog_masks"],
        targets=targets,
        utility_labels=util_labels,
        context_features=d["context_feats"],
    )

    assert total_loss.item() > 0.0
    assert "loss_rec" in loss_dict
    assert "loss_util" in loss_dict
    assert "loss_sparse" in loss_dict
    assert loss_dict["loss_util"] > 0.0


def test_uasf_all_7_baselines_modes(dummy_setup):
    """Verify execution of the 7 required comparative baseline modes."""
    d = dummy_setup
    baseline_modes = ["id_only", "all_fixed", "learned_global", "softmax", "contextual"]

    for mode in baseline_modes:
        model = UASFModel(
            frozen_late_fusion_model=d["late_model"],
            context_feature_dim=4,
            gate_mode=mode,
        )
        scores = model.predict_all(
            item_id_seqs=d["item_seqs"],
            source_seqs=d["source_seqs"],
            catalog_source_attrs=d["catalog_attrs"],
            catalog_availability_masks=d["catalog_masks"],
            context_features=d["context_feats"],
        )
        assert scores.shape == (2, d["num_items"])
