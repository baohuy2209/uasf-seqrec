"""
baselines.py - Reference Random and Popularity baselines for TAIMMA.

Requirements:
  - PopularityBaseline: frequency computed strictly from Train-base interactions.
  - RandomBaseline: reproducible random ranking under fixed seed.
"""

from collections import Counter
from typing import Dict, List, Optional
import numpy as np
import torch
from ..data.split import SequenceSample


class PopularityBaseline:
    """Predicts items sorted by historical popularity in Train-base."""

    def __init__(self, num_items: int):
        self.num_items = num_items
        # Item scores: 1-indexed (array length num_items)
        self.item_scores = np.zeros(num_items, dtype=np.float32)

    def fit(
        self,
        train_base_samples: Optional[List[SequenceSample]] = None,
        train_base_events: Optional[List[int]] = None,
    ):
        """
        Count interaction frequencies in Train-base interactions.
        
        Supervisor Invariant:
          Each interaction event in Train-base is counted exactly once.
          Legitimate repeated interactions by the same user are preserved,
          but prefix-generation duplication across overlapping sub-sequences is eliminated.
        """
        counts = Counter()
        if train_base_events is None:
            raise ValueError("Pass complete train_base_events; truncated overlapping samples cannot recover event frequencies")
        if any(i < 1 or i > self.num_items for i in train_base_events):
            raise ValueError("Train-base event outside catalog")
        counts.update(train_base_events)

        self.item_counts = counts
        for item_idx in range(1, self.num_items + 1):
            self.item_scores[item_idx - 1] = float(counts[item_idx])

    def predict_all(self, batch_size: int) -> np.ndarray:
        """Broadcast popularity scores for a batch of users: (B, num_items)."""
        return np.tile(self.item_scores, (batch_size, 1))


class RandomBaseline:
    """Predicts uniform random scores across catalog items with fixed seed."""

    def __init__(self, num_items: int, seed: int = 42):
        self.num_items = num_items
        self.rng = np.random.RandomState(seed)

    def predict_all(self, batch_size: int) -> np.ndarray:
        """Generate random scores (B, num_items)."""
        return self.rng.rand(batch_size, self.num_items).astype(np.float32)
