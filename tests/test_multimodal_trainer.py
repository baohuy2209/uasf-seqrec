"""
test_multimodal_trainer.py - Unit tests for MultimodalSequenceDataset and ResidualLateFusionTrainer.
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

from taimma.data.split import SequenceSample
from taimma.models.sasrec import SASRecModel
from taimma.models.late_fusion import ResidualLateFusionModel, SourceSequentialEncoder
from taimma.evaluation.evaluator import IndependentEvaluator
from taimma.training.multimodal_trainer import (
    MultimodalSequenceDataset,
    ResidualLateFusionTrainer,
    collate_multimodal_batch,
)


def test_multimodal_dataset_and_batch_collation():
    """Verify MultimodalSequenceDataset maps item IDs to source attributes."""
    samples = [
        SequenceSample(user_idx=0, history=[1, 2, 3], target=4, history_len=3),
        SequenceSample(user_idx=1, history=[0, 2, 4], target=1, history_len=2),
    ]

    cat_attrs = torch.tensor([10, 20, 30, 40], dtype=torch.long)
    brand_attrs = torch.tensor([1, 2, 0, 3], dtype=torch.long)
    title_attrs = torch.randn(4, 8)

    catalog_attrs = {"category": cat_attrs, "brand": brand_attrs, "title": title_attrs}
    catalog_masks = {
        "category": torch.ones(4),
        "brand": torch.tensor([1, 1, 0, 1], dtype=torch.float32),
        "title": torch.ones(4),
    }

    dataset = MultimodalSequenceDataset(samples, catalog_attrs, catalog_masks)
    assert len(dataset) == 2

    sample_0 = dataset[0]
    assert sample_0["item_seq"].tolist() == [1, 2, 3]
    assert sample_0["source_seqs"]["category"].tolist() == [10, 20, 30]
    assert sample_0["source_seqs"]["brand"].tolist() == [1, 2, 0]
    assert sample_0["source_seqs"]["title"].shape == (3, 8)

    batch = collate_multimodal_batch([dataset[0], dataset[1]])
    assert batch["item_seqs"].shape == (2, 3)
    assert batch["targets"].shape == (2,)
    assert batch["source_seqs"]["category"].shape == (2, 3)
    assert batch["source_seqs"]["title"].shape == (2, 3, 8)


def test_residual_late_fusion_trainer_frozen_backbone():
    """Verify that trainer only optimizes residual branches and preserves frozen ID backbone."""
    num_items = 6
    emb_dim = 16
    max_len = 4

    id_model = SASRecModel(num_items=num_items, max_seq_len=max_len, embedding_dim=emb_dim)
    cat_enc = SourceSequentialEncoder("category", vocab_size=10, feature_dim=emb_dim, max_seq_len=max_len)
    late_model = ResidualLateFusionModel(id_model, {"category": cat_enc}, p_drop=0.2)

    evaluator = IndependentEvaluator(num_items=num_items)
    catalog_attrs = {"category": torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.long)}
    catalog_masks = {"category": torch.ones(num_items)}

    trainer = ResidualLateFusionTrainer(
        model=late_model,
        evaluator=evaluator,
        catalog_source_attrs=catalog_attrs,
        catalog_availability_masks=catalog_masks,
        device="cpu",
        batch_size=2,
    )

    # Initial ID weights snapshot
    initial_id_weights = id_model.item_embedding.weight.clone()

    train_samples = [
        SequenceSample(user_idx=0, history=[1, 2, 3], target=4, history_len=3),
        SequenceSample(user_idx=1, history=[2, 3, 4], target=5, history_len=3),
    ]
    val_samples = [
        SequenceSample(user_idx=0, history=[1, 2, 3, 4], target=5, history_len=4),
        SequenceSample(user_idx=1, history=[2, 3, 4, 5], target=6, history_len=4),
    ]

    history, trained_model = trainer.fit(
        train_samples=train_samples,
        val_samples=val_samples,
        seen_train_items={0: {1, 2, 3}, 1: {2, 3, 4}},
        max_epochs=2,
        verbose=False,
    )

    # ID backbone weights must be exactly identical (zero gradient updates)
    assert torch.equal(initial_id_weights, trained_model.id_model.item_embedding.weight)
    assert len(history.train_losses) == 2
    assert history.best_val_ndcg >= 0.0
