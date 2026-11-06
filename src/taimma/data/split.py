"""
split.py - Chronological Train-base / Train-meta / Val / Test splitting for TAIMMA.

Protocol (Supervisor Memo 26/10/2026):
  For an eligible user sequence S_u = [i_1, i_2, ..., i_n] with n >= 5:
    - Train-base: positions 1 .. n-3.
    - Train-meta target: position n-2.
    - Validation target: position n-1.
    - Test target: position n.

Invariants:
  - Train-base catalog derivation: candidate items 1..N derived strictly from Train-base.
  - Zero leakage: Validation and Test targets never appear in Train-base labels or context.
  - Left-padding with index 0.
"""

import datetime
import hashlib
import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from .ingestion import InteractionSequence


@dataclass
class SequenceSample:
    user_idx: int
    history: List[int]       # Left-padded up to max_seq_len with 0
    target: int              # Ground-truth next item idx (1..num_items), 0 if OOV
    history_len: int
    raw_user_id: str = ""
    raw_target_asin: str = ""


@dataclass
class SplitData:
    train_base_samples: List[SequenceSample]
    train_meta_samples: List[SequenceSample]
    val_samples: List[SequenceSample]
    test_samples: List[SequenceSample]
    item2idx: Dict[str, int]
    idx2item: Dict[int, str]
    user2idx: Dict[str, int]
    num_items: int
    num_users: int = 0
    oov_counts: Dict[str, int] = field(default_factory=dict)
    seen_train_items_per_user: Dict[int, Set[int]] = field(default_factory=dict)
    train_base_events: List[int] = field(default_factory=list)


class SequenceSplitter:
    """Performs strict prefix-based chronological sequence splitting."""

    def __init__(self, max_seq_len: int = 50, min_seq_len: int = 5, include_oov_in_eval: bool = True, protocol: str = "four_segment"):
        if protocol not in ('four_segment','loo'):
            raise ValueError('protocol must be four_segment or loo')
        self.protocol = protocol
        self.holdout = 3 if protocol == 'four_segment' else 2
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len
        self.include_oov_in_eval = include_oov_in_eval

    def _pad_sequence(self, seq: List[int]) -> List[int]:
        """Left-pad sequence with 0 up to max_seq_len or truncate to last max_seq_len."""
        if len(seq) >= self.max_seq_len:
            return seq[-self.max_seq_len:]
        return [0] * (self.max_seq_len - len(seq)) + seq

    def split(self, sequences: List[InteractionSequence]) -> SplitData:
        """
        Split interaction sequences into Train-base, Train-meta, Val, Test.
        Derives candidate vocabulary exclusively from Train-base interactions.
        """
        eligible = [s for s in sequences if len(s.items) >= self.min_seq_len]

        # 1. Build Train-base vocabulary
        train_base_items: Set[str] = set()
        user2idx: Dict[str, int] = {}

        for u_idx, seq in enumerate(eligible):
            user2idx[seq.user_id] = u_idx
            n = len(seq.items)
            # Train-base items are positions 0 to n-4 inclusive (1 to n-3 in 1-based indexing)
            train_base_slice = seq.items[: n - self.holdout]
            train_base_items.update(train_base_slice)

        # Stable sorted item vocabulary (index 0 is padding)
        sorted_items = sorted(list(train_base_items))
        item2idx: Dict[str, int] = {item: idx + 1 for idx, item in enumerate(sorted_items)}
        idx2item: Dict[int, str] = {idx: item for item, idx in item2idx.items()}
        num_items = len(item2idx)
        num_users = len(user2idx)

        train_base_samples: List[SequenceSample] = []
        train_meta_samples: List[SequenceSample] = []
        val_samples: List[SequenceSample] = []
        test_samples: List[SequenceSample] = []

        oov_counts = {"train_meta": 0, "val": 0, "test": 0}
        seen_train_items_per_user: Dict[int, Set[int]] = {}
        train_base_events: List[int] = []

        for seq in eligible:
            u_idx = user2idx[seq.user_id]
            n = len(seq.items)
            seen_train: Set[int] = set()

            # Map raw items to integer indices, 0 if OOV
            int_items = [item2idx.get(item, 0) for item in seq.items]

            # Collect each Train-base event exactly once (items at positions 0 .. n-4)
            for itm_idx in int_items[: n - self.holdout]:
                if itm_idx > 0:
                    train_base_events.append(itm_idx)

            # A. Train-base: generate next-item pairs up to position n-3
            # In 0-based: positions 0 .. n-4 -> targets 1 .. n-3
            for pos in range(1, n - self.holdout):
                hist = int_items[:pos]
                tgt = int_items[pos]
                if tgt > 0:  # Valid vocabulary target
                    padded_hist = self._pad_sequence(hist)
                    train_base_samples.append(
                        SequenceSample(
                            user_idx=u_idx,
                            history=padded_hist,
                            target=tgt,
                            history_len=min(len(hist), self.max_seq_len),
                            raw_user_id=seq.user_id,
                            raw_target_asin=seq.items[pos],
                        )
                    )
                # Keep track of seen items in train
                if tgt > 0:
                    seen_train.add(tgt)
            if int_items[0] > 0:
                seen_train.add(int_items[0])
            seen_train_items_per_user[u_idx] = seen_train

            # B. Train-meta target: position n-2 (0-indexed: n-3)
            meta_hist = int_items[: n - 3]
            meta_tgt = int_items[n - 3]
            padded_meta_hist = self._pad_sequence(meta_hist)
            if self.protocol == "four_segment" and meta_tgt > 0:
                train_meta_samples.append(
                    SequenceSample(
                        user_idx=u_idx,
                        history=padded_meta_hist,
                        target=meta_tgt,
                        history_len=min(len(meta_hist), self.max_seq_len),
                        raw_user_id=seq.user_id,
                        raw_target_asin=seq.items[n - 3],
                    )
                )
            elif self.protocol == "four_segment":
                oov_counts["train_meta"] += 1

            # C. Validation target: position n-1 (0-indexed: n-2)
            val_hist = int_items[: n - 2]
            val_tgt = int_items[n - 2]
            padded_val_hist = self._pad_sequence(val_hist)
            if val_tgt > 0:
                val_samples.append(
                    SequenceSample(
                        user_idx=u_idx,
                        history=padded_val_hist,
                        target=val_tgt,
                        history_len=min(len(val_hist), self.max_seq_len),
                        raw_user_id=seq.user_id,
                        raw_target_asin=seq.items[n - 2],
                    )
                )
            else:
                oov_counts["val"] += 1
                if self.include_oov_in_eval:
                    val_samples.append(
                        SequenceSample(
                            user_idx=u_idx,
                            history=padded_val_hist,
                            target=0,
                            history_len=min(len(val_hist), self.max_seq_len),
                            raw_user_id=seq.user_id,
                            raw_target_asin=seq.items[n - 2],
                        )
                    )

            # D. Test target: position n (0-indexed: n-1)
            test_hist = int_items[: n - 1]
            test_tgt = int_items[n - 1]
            padded_test_hist = self._pad_sequence(test_hist)
            if test_tgt > 0:
                test_samples.append(
                    SequenceSample(
                        user_idx=u_idx,
                        history=padded_test_hist,
                        target=test_tgt,
                        history_len=min(len(test_hist), self.max_seq_len),
                        raw_user_id=seq.user_id,
                        raw_target_asin=seq.items[n - 1],
                    )
                )
            else:
                oov_counts["test"] += 1
                if self.include_oov_in_eval:
                    test_samples.append(
                        SequenceSample(
                            user_idx=u_idx,
                            history=padded_test_hist,
                            target=0,
                            history_len=min(len(test_hist), self.max_seq_len),
                            raw_user_id=seq.user_id,
                            raw_target_asin=seq.items[n - 1],
                        )
                    )

        return SplitData(
            train_base_samples=train_base_samples,
            train_meta_samples=train_meta_samples,
            val_samples=val_samples,
            test_samples=test_samples,
            item2idx=item2idx,
            idx2item=idx2item,
            user2idx=user2idx,
            num_items=num_items,
            num_users=num_users,
            oov_counts=oov_counts,
            seen_train_items_per_user=seen_train_items_per_user,
            train_base_events=train_base_events,
        )


def compute_file_sha256(filepath: str) -> str:
    """Compute sha256 checksum of a file in streaming chunks."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def save_processed_data(
    processed_dir: str,
    split_data: SplitData,
    raw_reviews_path: str,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """
    Persist processed dataset artifacts following Strict Data-First:
      - Item and user ID mappings
      - Dataset split summary statistics
      - Provenance manifest with SHA256 checksum
    """
    p_dir = Path(processed_dir)
    if not Path(raw_reviews_path).is_file():
        raise FileNotFoundError(f'Cannot persist splits without source reviews: {raw_reviews_path}')
    if p_dir.exists() and any(p_dir.iterdir()):
        raise FileExistsError(f'Refusing to overwrite processed artifacts: {p_dir}')
    p_dir.mkdir(parents=True, exist_ok=True)
    split_hashes = {}
    for name in ('train_base_samples', 'train_meta_samples', 'val_samples', 'test_samples'):
        dest = p_dir / (name + '.jsonl')
        with dest.open('x', encoding='utf-8') as f:
            for sample in getattr(split_data, name):
                f.write(json.dumps(asdict(sample), ensure_ascii=False) + '\n')
        split_hashes[dest.name] = compute_file_sha256(str(dest))
    with (p_dir / 'train_base_events.json').open('x', encoding='utf-8') as f:
        json.dump(split_data.train_base_events, f)
    split_hashes['train_base_events.json'] = compute_file_sha256(str(p_dir / 'train_base_events.json'))

    with open(p_dir / "item2idx.json", "w", encoding="utf-8") as f:
        json.dump(split_data.item2idx, f, indent=2)

    with open(p_dir / "user2idx.json", "w", encoding="utf-8") as f:
        json.dump(split_data.user2idx, f, indent=2)

    stats = {
        "num_users": split_data.num_users,
        "num_items": split_data.num_items,
        "train_base_samples": len(split_data.train_base_samples),
        "train_meta_samples": len(split_data.train_meta_samples),
        "val_samples": len(split_data.val_samples),
        "test_samples": len(split_data.test_samples),
        "oov_counts": split_data.oov_counts,
    }
    with open(p_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    for name in ('item2idx.json', 'user2idx.json', 'stats.json'):
        split_hashes[name] = compute_file_sha256(str(p_dir / name))
    raw_path_obj = Path(raw_reviews_path)
    manifest = {
        "dataset_name": "Beauty_2014",
        "raw_reviews_file": str(raw_path_obj.resolve()),
        "raw_reviews_sha256": compute_file_sha256(raw_reviews_path) if raw_path_obj.is_file() else "missing",
        "raw_reviews_size_bytes": raw_path_obj.stat().st_size if raw_path_obj.is_file() else 0,
        "created_at": datetime.datetime.now().isoformat(),
        "stats": stats,
        "split_sha256": split_hashes,
        "config": config or {},
    }
    with open(p_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return {
        "item2idx": str((p_dir / "item2idx.json").resolve()),
        "user2idx": str((p_dir / "user2idx.json").resolve()),
        "stats": str((p_dir / "stats.json").resolve()),
        "manifest": str((p_dir / "manifest.json").resolve()),
    }
