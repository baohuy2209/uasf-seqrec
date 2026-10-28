"""
trainer.py - Training pipeline with deterministic seeding and early stopping on Val NDCG@10.

Protocol:
  - Seed determinism across python, numpy, torch.
  - Full-catalog Cross-Entropy loss on Train-base interactions.
  - Model selection via Validation NDCG@10.
  - Graceful checkpoint saving.
"""

import copy
import os
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from ..data.split import SequenceSample
from ..evaluation.evaluator import IndependentEvaluator, EvaluationResult
from ..experiments.reporting import runtime_snapshot
from ..models.sasrec import SASRecModel


def set_seed(seed: int = 42):
    """Enforce strict seed determinism across libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class SequenceDataset(Dataset):
    """PyTorch Dataset wrapper for sequence samples."""

    def __init__(self, samples: List[SequenceSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        s = self.samples[idx]
        return (
            torch.tensor(s.history, dtype=torch.long),
            torch.tensor(s.target, dtype=torch.long),
            s.user_idx,
            s.history_len,
        )


@dataclass
class TrainingHistory:
    train_losses: List[float] = field(default_factory=list)
    val_ndcg_10: List[float] = field(default_factory=list)
    val_recall_10: List[float] = field(default_factory=list)
    epoch_durations: List[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_ndcg: float = 0.0
    total_training_time: float = 0.0
    epoch_wall_durations: List[float] = field(default_factory=list)
    val_metrics_by_epoch: List[Dict[str, float]] = field(default_factory=list)
    stop_reason: str = 'max_epochs'
    resource_usage: Dict = field(default_factory=dict)


class SASRecTrainer:
    """Trainer for SASRec ID-only backbone."""

    def __init__(
        self,
        model: SASRecModel,
        evaluator: IndependentEvaluator,
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
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.patience = patience
        self.seed = seed

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch_seqs, batch_targets, _, _ in dataloader:
            batch_seqs = batch_seqs.to(self.device)
            batch_targets = batch_targets.to(self.device)

            self.optimizer.zero_grad()
            loss = self.model.compute_loss(batch_seqs, batch_targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def predict_scores(
        self,
        samples: List[SequenceSample],
        batch_size: int = 256,
    ) -> Tuple[np.ndarray, List[int], List[int]]:
        """Compute full-catalog predictions: returns (scores, targets, user_indices)."""
        self.model.eval()
        dataset = SequenceDataset(samples)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        all_scores: List[np.ndarray] = []
        all_targets: List[int] = []
        all_users: List[int] = []

        with torch.no_grad():
            for batch_seqs, batch_targets, batch_u, _ in dataloader:
                batch_seqs = batch_seqs.to(self.device)
                logits = self.model.predict_all(batch_seqs)  # (B, num_items)
                all_scores.append(logits.cpu().numpy())
                all_targets.extend(batch_targets.tolist())
                all_users.extend(batch_u.tolist())

        if not all_scores:
            return np.empty((0, self.model.num_items), dtype=np.float32), [], []

        return np.concatenate(all_scores, axis=0), all_targets, all_users

    def evaluate(
        self,
        samples: List[SequenceSample],
        seen_items: Optional[Dict[int, Set[int]]] = None,
        batch_size: int = 256,
    ) -> EvaluationResult:
        """Full-catalog evaluation across test or validation samples."""
        self.model.eval()
        records = []
        with torch.no_grad():
            for start in range(0, len(samples), batch_size):
                batch = samples[start:start + batch_size]
                seqs = torch.tensor([s.history for s in batch], dtype=torch.long, device=self.device)
                result = self.evaluator.evaluate_ranks(
                    self.model.predict_all(seqs), [s.target for s in batch],
                    [s.user_idx for s in batch], seen_items,
                )
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
    ) -> Tuple[TrainingHistory, SASRecModel]:
        set_seed(self.seed)

        train_dataset = SequenceDataset(train_samples)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
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
                    f"Epoch {epoch:03d} | Train Loss: {loss:.4f} | "
                    f"Val NDCG@10: {val_ndcg:.4f} | Val Recall@10: {val_rec:.4f} | "
                    f"Time: {ep_duration:.2f}s"
                )

            # Early stopping check
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

        # Load best weights into model
        if best_weights is not None:
            self.model.load_state_dict(best_weights)

        return history, self.model
