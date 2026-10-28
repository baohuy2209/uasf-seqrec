"""
utility.py - Per-context leave-one-source-out utility label computation for TAIMMA.

Formula (Supervisor Memo October 20 & 26, 2026):
  For context q and metadata source m on a frozen Late Fusion reference model:
    s_all(q, ·) = s_ID(q, ·) + SUM_{k} b_(·, k) * s_k(q, ·)
    s_without_m(q, ·) = s_all(q, ·) - b_(·, m) * s_m(q, ·)
    delta_m(q) = CE(s_without_m(q, ·), target) - CE(s_all(q, ·), target)

Invariants:
  - Calculated with CE reduction="none" per context.
  - Frozen model: zero parameter updates or gradients (torch.no_grad).
  - Positive delta_m(q) > 0 indicates that source m improved next-item prediction
    (removing source m increased loss).
"""

from dataclasses import dataclass, field, asdict
import hashlib
import json
from typing import Dict, List, Optional, Tuple
import pandas as pd
import torch
import torch.nn.functional as F
from ..models.late_fusion import ResidualLateFusionModel


@dataclass
class UtilityRecord:
    context_id: str
    user_idx: int
    target_idx: int
    source_name: str
    reference_seed: int
    delta: float
    ce_all: float
    ce_without_m: float
    source_available_on_target: bool
    seq_len: int
    features: Dict[str, float] = field(default_factory=dict)


def compute_leave_one_source_out_deltas(
    late_fusion_model: ResidualLateFusionModel,
    item_id_seqs: torch.Tensor,
    source_seqs: Dict[str, torch.Tensor],
    catalog_source_attrs: Dict[str, torch.Tensor],
    catalog_availability_masks: Dict[str, torch.Tensor],
    targets: torch.Tensor,
    user_indices: List[int],
    reference_seed: int = 42,
    context_ids: Optional[List[str]] = None,
) -> Tuple[Dict[str, torch.Tensor], List[UtilityRecord]]:
    """
    Compute leave-one-source-out utility labels for all active sources in late_fusion_model.

    Args:
        late_fusion_model: Frozen ResidualLateFusionModel
        item_id_seqs: LongTensor (B, S)
        source_seqs: Dict[source -> LongTensor (B, S)]
        catalog_source_attrs: Dict[source -> Tensor (num_items, ...)]
        catalog_availability_masks: Dict[source -> Tensor (num_items,)]
        targets: LongTensor (B,) with 1-based target item indices
        user_indices: List of user IDs for per-context tracking
        reference_seed: Seed ID of this reference model

    Returns:
        deltas_per_source: Dict[source_name -> Tensor (B,)]
        records: List of UtilityRecord objects for serialization
    """
    late_fusion_model.eval()
    B = item_id_seqs.size(0)
    target_labels = targets - 1  # 0-based for F.cross_entropy
    if len(user_indices) != B or targets.shape != (B,) or torch.any(targets <= 0) or torch.any(targets > late_fusion_model.num_items):
        raise ValueError('Utility computation requires aligned in-catalog Train-meta targets')
    from ..data.split import SequenceSample
    from .e0_e1_protocol import extract_permitted_context_features
    histories = item_id_seqs.detach().cpu().tolist()
    lengths = [sum(i > 0 for i in history) for history in histories]
    if any(length == 0 for length in lengths):
        raise ValueError('Utility context must have a nonempty known-item history')
    samples = [SequenceSample(user_indices[b], histories[b], int(targets[b]), lengths[b]) for b in range(B)]
    if context_ids is None:
        context_ids = [hashlib.sha256(json.dumps([int(user_indices[b]), histories[b], int(targets[b])]).encode()).hexdigest() for b in range(B)]
    if len(context_ids) != B or len(set(context_ids)) != B:
        raise ValueError('Context IDs must be unique and aligned within a batch')

    with torch.no_grad():
        # Compute s_all (all active sources included, no dropout)
        scores_all, scores_id, scores_per_source = late_fusion_model.compute_all_scores(
            item_id_seqs=item_id_seqs,
            source_seqs=source_seqs,
            catalog_source_attrs=catalog_source_attrs,
            catalog_availability_masks=catalog_availability_masks,
            apply_dropout=False,
        )

        ce_all = F.cross_entropy(scores_all, target_labels, reduction="none")  # (B,)

        deltas_per_source: Dict[str, torch.Tensor] = {}
        records: List[UtilityRecord] = []

        for src_name, s_m in scores_per_source.items():
            features = extract_permitted_context_features(samples, catalog_source_attrs, catalog_availability_masks, src_name)
            avail_mask = catalog_availability_masks[src_name]  # (num_items,)
            masked_sm = s_m * avail_mask.unsqueeze(0).float()  # (B, num_items)

            # Leave-one-source-out score
            scores_without_m = scores_all - masked_sm
            ce_without_m = F.cross_entropy(scores_without_m, target_labels, reduction="none")  # (B,)

            # delta_m = CE(without_m) - CE(all)
            delta_m = ce_without_m - ce_all  # (B,)
            deltas_per_source[src_name] = delta_m

            # Target attribute availability: check if target item has this source
            for b in range(B):
                tgt_col = target_labels[b].item()
                is_avail = bool(avail_mask[tgt_col].item() > 0)
                records.append(
                    UtilityRecord(
                        context_id=str(context_ids[b]),
                        user_idx=user_indices[b],
                        target_idx=targets[b].item(),
                        source_name=src_name,
                        reference_seed=reference_seed,
                        delta=float(delta_m[b].item()),
                        ce_all=float(ce_all[b].item()),
                        ce_without_m=float(ce_without_m[b].item()),
                        source_available_on_target=is_avail,
                        seq_len=lengths[b],
                        features=dict(zip(('feat_log_len', 'feat_coverage', 'feat_entropy', 'feat_recency'), map(float, features[b]))),
                    )
                )

    return deltas_per_source, records


def export_utility_records_to_df(records: List[UtilityRecord]) -> pd.DataFrame:
    """Convert utility records list into a pandas DataFrame."""
    rows = []
    for record in records:
        row = asdict(record)
        row.update(row.pop('features'))
        rows.append(row)
    return pd.DataFrame(rows)


def save_utility_artifact(filepath, records: List[UtilityRecord], dataset_manifest_path,
                          reference_checkpoints: Dict[int, str]):
    """Persist actual S2 records with verified dataset/checkpoint file lineage."""
    from pathlib import Path
    from ..data.split import compute_file_sha256
    from ..experiments.reporting import validate_rank_format, write_json
    destination = Path(filepath)
    format_name = destination.suffix.lstrip('.').lower()
    validate_rank_format(format_name)
    if format_name == 'both':
        raise ValueError('Supply one .csv or .parquet path')
    sidecar = destination.with_suffix(destination.suffix + '.manifest.json')
    if destination.exists() or sidecar.exists():
        raise FileExistsError(f'Refusing to overwrite utility artifact: {destination}')
    source_manifest = Path(dataset_manifest_path)
    if not source_manifest.is_file():
        raise FileNotFoundError(f'Missing dataset manifest: {source_manifest}')
    frame = export_utility_records_to_df(records)
    keys = ['context_id', 'user_idx', 'source_name', 'reference_seed']
    if frame.empty or frame.duplicated(keys).any():
        raise ValueError('Utility records must be nonempty and unique by context/user/source/seed')
    lineage = {}
    for reference_seed in frame.reference_seed.unique():
        if reference_seed not in reference_checkpoints:
            raise ValueError(f'Missing checkpoint lineage for seed {reference_seed}')
        checkpoint = Path(reference_checkpoints[reference_seed])
        if not checkpoint.is_file():
            raise FileNotFoundError(f'Missing reference checkpoint: {checkpoint}')
        lineage[str(reference_seed)] = {'path': str(checkpoint.resolve()), 'sha256': compute_file_sha256(str(checkpoint))}
    destination.parent.mkdir(parents=True, exist_ok=True)
    if format_name == 'csv':
        with destination.open('x', encoding='utf-8', newline='') as f:
            frame.to_csv(f, index=False)
    else:
        frame.to_parquet(destination, index=False)
    write_json(sidecar, {'schema_version': 1, 'file_sha256': compute_file_sha256(str(destination)),
                        'dataset_manifest': str(source_manifest.resolve()),
                        'dataset_manifest_sha256': compute_file_sha256(str(source_manifest)),
                        'reference_checkpoints': lineage, 'records': len(frame)})
    return str(destination), str(sidecar)
