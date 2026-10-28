"""
evaluator.py - Independent Full-Catalog Ranking Evaluator for TAIMMA.

Requirements:
  - Model-agnostic interface (operates on catalog scores/logits).
  - Configurable seen-item masking policy.
  - Metrics: NDCG@{10, 20}, Recall@{10, 20}. Primary metric: NDCG@10.
  - Per-user rank serialization for downstream statistical analysis.
  - Zero reliance on sampled negatives.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np
import torch


@dataclass
class UserRankRecord:
    user_idx: int
    target_idx: int
    rank: int
    ndcg_10: float
    recall_10: float
    ndcg_20: float
    recall_20: float


@dataclass
class EvaluationResult:
    num_users: int
    ndcg_10: float
    recall_10: float
    ndcg_20: float
    recall_20: float
    mask_seen_policy: bool
    per_user_records: List[UserRankRecord] = field(default_factory=list)

    def to_dict(self) -> Dict[str, float]:
        return {
            "num_evaluated_users": self.num_users,
            "NDCG@10": round(self.ndcg_10, 6),
            "Recall@10": round(self.recall_10, 6),
            "NDCG@20": round(self.ndcg_20, 6),
            "Recall@20": round(self.recall_20, 6),
        }


class IndependentEvaluator:
    """Evaluates sequential recommenders under the full-ranking protocol."""

    def __init__(
        self,
        num_items: int,
        k_list: List[int] = [10, 20],
        mask_seen_items: bool = True,
    ):
        if num_items < 1:
            raise ValueError("num_items must be positive")
        if not k_list or any(k not in (10, 20) for k in k_list):
            raise ValueError("Supported cutoffs are 10 and 20")
        self.num_items = num_items
        self.k_list = sorted(k_list)
        self.mask_seen_items = mask_seen_items

    @staticmethod
    def _compute_ndcg(rank: int, k: int) -> float:
        if rank <= k:
            return 1.0 / math.log2(rank + 1)
        return 0.0

    @staticmethod
    def _compute_recall(rank: int, k: int) -> float:
        return 1.0 if rank <= k else 0.0

    def evaluate_ranks(
        self,
        scores: Union[np.ndarray, torch.Tensor],
        targets: List[int],
        user_indices: List[int],
        seen_items_per_user: Optional[Dict[int, Set[int]]] = None,
    ) -> EvaluationResult:
        """
        Evaluate full ranking from score matrix (B, num_items) where items are 1..num_items.
        
        Args:
            scores: Array/Tensor of shape (B, num_items). Column c corresponds to item index c + 1.
            targets: List of target item indices (1..num_items).
            user_indices: List of user indices for per-user tracking.
            seen_items_per_user: Dict mapping user_idx -> set of item indices already interacted with.
        """
        if isinstance(scores, torch.Tensor):
            scores_np = scores.detach().cpu().numpy().copy()
        else:
            scores_np = scores.copy()

        if scores_np.ndim != 2:
            raise ValueError("scores must be a two-dimensional catalog matrix")
        B, N = scores_np.shape
        if N != self.num_items:
            raise ValueError(f"Expected {self.num_items} catalog columns, got {N}")
        if len(user_indices) != B:
            raise ValueError("user_indices must align with score rows")
        if any(not isinstance(t, (int, np.integer)) for t in targets):
            raise ValueError("targets must be integer item indices")
        if not np.issubdtype(scores_np.dtype, np.floating):
            scores_np = scores_np.astype(np.float64)
        if B != len(targets):
            raise ValueError(f"Batch size {B} does not match targets count {len(targets)}")

        # Strict score validation: reject NaN or positive inf in raw input scores
        if np.isnan(scores_np).any():
            raise ValueError("Invalid score matrix: NaN detected in model output scores prior to masking.")
        if np.isposinf(scores_np).any():
            raise ValueError("Invalid score matrix: +inf detected in model output scores prior to masking.")

        if np.isneginf(scores_np).any():
            raise ValueError("Invalid score matrix: -inf detected before internal masking")

        per_user_records: List[UserRankRecord] = []
        ndcg_10_list: List[float] = []
        recall_10_list: List[float] = []
        ndcg_20_list: List[float] = []
        recall_20_list: List[float] = []

        col_indices = np.arange(N)

        for b in range(B):
            u_idx = user_indices[b]
            tgt = targets[b]
            user_scores = scores_np[b]

            # Seen-item masking: set score to -inf for items already in user history
            if self.mask_seen_items and seen_items_per_user and u_idx in seen_items_per_user:
                seen = seen_items_per_user[u_idx]
                for s_item in seen:
                    # In 0-based column indexing: item i is at column i - 1
                    col = s_item - 1
                    if 0 <= col < N and s_item != tgt:
                        user_scores[col] = -np.inf

            # Column of ground truth target
            tgt_col = tgt - 1
            if tgt_col < 0 or tgt_col >= N:
                # OOV target, assigned worst rank (always a miss across all k <= N)
                rank = N + 1
            else:
                tgt_score = user_scores[tgt_col]
                # Documented tie policy:
                # 1. Primary order: strictly higher score ranks better.
                # 2. Secondary order: ties broken deterministically by catalog column index (ascending).
                # Independent of ground-truth target identity; guarantees strict permutation over items.
                greater_count = np.sum(user_scores > tgt_score)
                tied_before = np.sum((user_scores == tgt_score) & (col_indices < tgt_col))
                rank = int(greater_count + tied_before + 1)

            valid_target = 0 <= tgt_col < N
            ndcg_10 = self._compute_ndcg(rank, 10) if valid_target else 0.0
            recall_10 = self._compute_recall(rank, 10) if valid_target else 0.0
            ndcg_20 = self._compute_ndcg(rank, 20) if valid_target else 0.0
            recall_20 = self._compute_recall(rank, 20) if valid_target else 0.0

            ndcg_10_list.append(ndcg_10)
            recall_10_list.append(recall_10)
            ndcg_20_list.append(ndcg_20)
            recall_20_list.append(recall_20)

            per_user_records.append(
                UserRankRecord(
                    user_idx=u_idx,
                    target_idx=tgt,
                    rank=rank,
                    ndcg_10=ndcg_10,
                    recall_10=recall_10,
                    ndcg_20=ndcg_20,
                    recall_20=recall_20,
                )
            )

        return self.result_from_records(per_user_records)

    def result_from_records(self, records: List[UserRankRecord]) -> EvaluationResult:
        if not records:
            raise ValueError("Cannot evaluate an empty cohort")
        return EvaluationResult(
            num_users=len(records),
            ndcg_10=float(np.mean([r.ndcg_10 for r in records])),
            recall_10=float(np.mean([r.recall_10 for r in records])),
            ndcg_20=float(np.mean([r.ndcg_20 for r in records])),
            recall_20=float(np.mean([r.recall_20 for r in records])),
            mask_seen_policy=self.mask_seen_items,
            per_user_records=records,
        )
