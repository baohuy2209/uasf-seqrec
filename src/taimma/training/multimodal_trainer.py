"""
multimodal_trainer.py - Multimodal dataset, batch collation, and Residual Late Fusion trainer.

Protocol (Supervisor Decisions October 20 & 26, 2026):
  - Stage S1b: Frozen ID-only SASRec backbone.
  - Jointly train residual metadata encoders (Category, Brand, Title).
  - Source dropout d_m ~ Bernoulli(1 - p_drop) during training.
  - Full-catalog Cross-Entropy loss.
  - Validation NDCG@10 early stopping with model checkpointing.
  - Zero gradients backpropagated into the frozen ID backbone.
"""

import copy
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ..data.split import SequenceSample
from ..evaluation.evaluator import IndependentEvaluator, EvaluationResult
from ..experiments.reporting import runtime_snapshot
from ..models.late_fusion import ResidualLateFusionModel
from .trainer import TrainingHistory, set_seed


class MultimodalSequenceDataset(Dataset):
    """
    Multimodal Sequence Dataset.
    Maps item sequences to source attribute sequences using catalog attribute tables.
    """

    def __init__(
        self,
        samples: List[SequenceSample],
        catalog_source_attrs: Dict[str, torch.Tensor],
        catalog_availability_masks: Dict[str, torch.Tensor],
    ):
        self.samples = samples
        if set(catalog_source_attrs) != set(catalog_availability_masks):
            raise ValueError('Every source requires an explicit availability mask')
        self.catalog_source_attrs = {k: v.detach().cpu() for k, v in catalog_source_attrs.items()}
        self.catalog_availability_masks = {k: v.detach().cpu() for k, v in catalog_availability_masks.items()}
        self.sources = list(catalog_source_attrs.keys())

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, int]]:
        sample = self.samples[idx]
        history_items = sample.history  # List[int], length L, 0 is padding, 1..N items
        target = sample.target
        user_idx = sample.user_idx

        # 1. Item ID sequence tensor
        item_seq = torch.tensor(history_items, dtype=torch.long)

        # 2. Source sequence tensors
        source_seqs: Dict[str, torch.Tensor] = {}
        for src_name in self.sources:
            attrs = self.catalog_source_attrs[src_name]
            avail = self.catalog_availability_masks.get(src_name)
            # If attrs has shape (N,) -> 1D attribute (e.g. categorical index)
            # If attrs has shape (N, D) -> 2D attribute (e.g. dense title embedding)
            if attrs.dim() == 1:
                src_seq = torch.zeros(len(history_items), dtype=attrs.dtype)
                for t, itm in enumerate(history_items):
                    if itm > 0 and itm <= attrs.size(0):
                        if avail is not None and not bool(avail[itm - 1].item() > 0):
                            continue
                        src_seq[t] = attrs[itm - 1]
            else:
                feature_dim = attrs.size(1)
                src_seq = torch.zeros(len(history_items), feature_dim, dtype=attrs.dtype)
                for t, itm in enumerate(history_items):
                    if itm > 0 and itm <= attrs.size(0):
                        if avail is not None and not bool(avail[itm - 1].item() > 0):
                            continue
                        src_seq[t] = attrs[itm - 1]

            source_seqs[src_name] = src_seq

        return {
            "item_seq": item_seq,
            "target": torch.tensor(target, dtype=torch.long),
            "user_idx": user_idx,
            "history_len": sample.history_len,
            "source_seqs": source_seqs,
        }


def collate_multimodal_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate individual sample dicts into a batch tensor dict."""
    item_seqs = torch.stack([b["item_seq"] for b in batch], dim=0)
    targets = torch.stack([b["target"] for b in batch], dim=0)
    user_indices = [b["user_idx"] for b in batch]
    history_lens = [b["history_len"] for b in batch]

    sources = list(batch[0]["source_seqs"].keys())
    source_seqs: Dict[str, torch.Tensor] = {}
    for src in sources:
        source_seqs[src] = torch.stack([b["source_seqs"][src] for b in batch], dim=0)

    return {
        "item_seqs": item_seqs,
        "targets": targets,
        "user_indices": user_indices,
        "history_lens": history_lens,
        "source_seqs": source_seqs,
    }


class ResidualLateFusionTrainer:
    """
    Trainer for Stage S1b: Residual Late Fusion Model over frozen ID backbone.
    Enforces freeze on ID backbone, applies source dropout, and optimizes residual branches.
    """

    def __init__(
        self,
        model: ResidualLateFusionModel,
        evaluator: IndependentEvaluator,
        catalog_source_attrs: Dict[str, torch.Tensor],
        catalog_availability_masks: Dict[str, torch.Tensor],
        device: str = "cpu",
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int = 256,
        patience: int = 10,
        seed: int = 42,
    ):
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; explicitly choose device=cpu")
        self.model = model.to(self.device)
        self.evaluator = evaluator
        self.catalog_source_attrs = {
            k: v.to(self.device) for k, v in catalog_source_attrs.items()
        }
        self.catalog_availability_masks = {
            k: v.to(self.device) for k, v in catalog_availability_masks.items()
        }

        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.patience = patience
        self.seed = seed

        # Ensure ID branch is frozen
        self.model.freeze_id_branch()

        # Optimizer strictly over trainable residual encoder parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(
            trainable_params, lr=self.lr, weight_decay=self.weight_decay
        )

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        self.model.id_model.eval()  # Invariant: ID backbone stays in eval mode
        total_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            item_seqs = batch["item_seqs"].to(self.device)
            targets = batch["targets"].to(self.device)
            source_seqs = {k: v.to(self.device) for k, v in batch["source_seqs"].items()}

            self.optimizer.zero_grad()
            loss = self.model.compute_loss(
                item_id_seqs=item_seqs,
                source_seqs=source_seqs,
                catalog_source_attrs=self.catalog_source_attrs,
                catalog_availability_masks=self.catalog_availability_masks,
                targets=targets,
            )
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def evaluate(
        self,
        samples: List[SequenceSample],
        seen_items: Optional[Dict[int, Set[int]]] = None,
        batch_size: int = 256,
    ) -> EvaluationResult:
        """Full-catalog evaluation across validation or test samples."""
        self.model.eval()
        dataset = MultimodalSequenceDataset(
            samples, self.catalog_source_attrs, self.catalog_availability_masks
        )
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_multimodal_batch
        )

        records = []
        with torch.no_grad():
            for batch in dataloader:
                scores = self.model.predict_all(
                    item_id_seqs=batch['item_seqs'].to(self.device),
                    source_seqs={k: v.to(self.device) for k, v in batch['source_seqs'].items()},
                    catalog_source_attrs=self.catalog_source_attrs,
                    catalog_availability_masks=self.catalog_availability_masks,
                )
                result = self.evaluator.evaluate_ranks(scores, batch['targets'].tolist(), batch['user_indices'], seen_items)
                records.extend(result.per_user_records)
        return self.evaluator.result_from_records(records)

    def fit(
        self,
        train_samples: List[SequenceSample],
        val_samples: List[SequenceSample],
        seen_train_items: Dict[int, Set[int]],
        max_epochs: int = 100,
        checkpoint_path: Optional[str] = None,
        verbose: bool = True,
    ) -> Tuple[TrainingHistory, ResidualLateFusionModel]:
        set_seed(self.seed)

        train_dataset = MultimodalSequenceDataset(
            train_samples, self.catalog_source_attrs, self.catalog_availability_masks
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
            collate_fn=collate_multimodal_batch,
        )

        history = TrainingHistory()
        best_val_ndcg = -1.0
        best_weights = None
        patience_counter = 0
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        start_time = time.perf_counter()

        for epoch in range(1, max_epochs + 1):
            ep_start = time.perf_counter()
            loss = self.train_epoch(train_loader)
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            ep_duration = time.perf_counter() - ep_start

            val_res = self.evaluate(val_samples, seen_items=seen_train_items, batch_size=self.batch_size)
            val_ndcg = val_res.ndcg_10
            val_rec = val_res.recall_10

            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            history.val_metrics_by_epoch.append(val_res.to_dict())
            history.train_losses.append(loss)
            history.val_ndcg_10.append(val_ndcg)
            history.val_recall_10.append(val_rec)
            history.epoch_durations.append(ep_duration)

            if verbose and (epoch % 5 == 0 or epoch == 1 or val_ndcg > best_val_ndcg):
                print(
                    f"Epoch {epoch:03d} | Residual Loss: {loss:.4f} | "
                    f"Val NDCG@10: {val_ndcg:.4f} | Val Recall@10: {val_rec:.4f} | "
                    f"Time: {ep_duration:.2f}s"
                )

            # Early stopping check on Val NDCG@10
            if val_ndcg > best_val_ndcg:
                best_val_ndcg = val_ndcg
                history.best_epoch = epoch
                history.best_val_ndcg = val_ndcg
                best_weights = copy.deepcopy(self.model.state_dict())
                patience_counter = 0

                if checkpoint_path:
                    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": best_weights,
                            "val_ndcg_10": val_ndcg,
                            "val_recall_10": val_rec,
                            "seed": self.seed,
                            "sources_enabled": self.model.sources_enabled,
                        },
                        checkpoint_path,
                    )
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    if verbose:
                        print(f"Early stopping triggered at epoch {epoch}. Best epoch: {history.best_epoch}")
                    history.stop_reason = 'early_stopping'
            history.epoch_wall_durations.append(time.perf_counter() - ep_start)
            if patience_counter >= self.patience:
                break

        history.total_training_time = time.perf_counter() - start_time
        history.resource_usage = runtime_snapshot(self.device)

        if best_weights is not None:
            self.model.load_state_dict(best_weights)

        return history, self.model
