"""Measured artifacts for supervisor pilot/weekly reports, without invented results."""
from collections import Counter
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import torch


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)


def validate_rank_format(rank_format):
    if rank_format not in ('csv', 'parquet', 'both'):
        raise ValueError('rank_format must be csv, parquet or both')
    if rank_format != 'csv' and not any(importlib.util.find_spec(x) for x in ('pyarrow', 'fastparquet')):
        raise ImportError('Parquet requested but no engine installed. Install pyarrow in the training environment or explicitly select --rank_format csv; no silent format substitution.')


def write_rank_table(folder, stem, records, context=None, rank_format='csv'):
    validate_rank_format(rank_format)
    rows = pd.DataFrame([{'user_idx': r.user_idx, 'target_idx': r.target_idx, 'rank': r.rank,
                          'ndcg@10': r.ndcg_10, 'recall@10': r.recall_10,
                          'ndcg@20': r.ndcg_20, 'recall@20': r.recall_20} for r in records])
    if rows.empty or rows.duplicated(['user_idx', 'target_idx']).any():
        raise ValueError('A ranking artifact requires a nonempty unique evaluation cohort')
    if context is not None:
        if len(context) != len(rows):
            raise ValueError('Context/rank cohorts must match exactly')
        rows = rows.merge(context, on=['user_idx', 'target_idx'], how='left', validate='one_to_one', indicator=True)
        if not rows['_merge'].eq('both').all():
            raise ValueError('Context rows do not match ranking keys')
        rows = rows.drop(columns='_merge')
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    suffixes = ['csv', 'parquet'] if rank_format == 'both' else [rank_format]
    for suffix in suffixes:
        if (folder / f'{stem}.{suffix}').exists():
            raise FileExistsError(f'Refusing to overwrite {stem}.{suffix}')
    for suffix in suffixes:
        path = folder / f'{stem}.{suffix}'
        if suffix == 'csv':
            rows.to_csv(path, index=False)
        else:
            rows.to_parquet(path, index=False)
    return rows


def runtime_snapshot(device):
    """Unknown/not-applicable measurements are null, never zero GPU estimates."""
    device = torch.device(device)
    out = dict(device=str(device), cpu_name=platform.processor() or platform.machine(),
               host_ram_bytes=None, process_rss_bytes=None, process_peak_rss_bytes=None,
               ram_measurement_scope='process_lifetime_peak', ram_status='unavailable',
               gpu_name=None, gpu_total_bytes=None, gpu_peak_allocated_bytes=None,
               gpu_peak_reserved_bytes=None, gpu_status='not_applicable_cpu',
               cuda_version=torch.version.cuda, gpu_measurement_scope='PyTorch allocator since last peak reset; excludes other processes and non-PyTorch allocations')
    try:
        import psutil
        info = psutil.Process().memory_info()
        out.update(host_ram_bytes=int(psutil.virtual_memory().total), process_rss_bytes=int(info.rss), psutil_version=psutil.__version__)
        if hasattr(info, 'peak_wset'):
            out['process_peak_rss_bytes'] = int(info.peak_wset)
        elif os.name == 'posix':
            import resource
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            out['process_peak_rss_bytes'] = int(peak * (1 if platform.system() == 'Darwin' else 1024))
        out['ram_status'] = 'measured' if out['process_peak_rss_bytes'] is not None else 'current_rss_only'
    except (ImportError, OSError) as exc:
        out['ram_status'] = f'unavailable: {type(exc).__name__}'
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('Cannot measure requested CUDA device')
        torch.cuda.synchronize(device)
        props = torch.cuda.get_device_properties(device)
        out.update(gpu_name=props.name, gpu_total_bytes=int(props.total_memory),
                   gpu_peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
                   gpu_peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)),
                   gpu_status='measured')
    return out


def source_presence(record, source):
    if source == 'category':
        return bool(record.get('categories') or record.get('category'))
    return bool(record.get(source))


def dataset_statistics(sequences, split, metadata, metadata_available, dataset_name, expected_stats=None):
    lengths = [len(s.items) for s in sequences]
    items = {item for seq in sequences for item in seq.items}
    pairs = sum(len(set(s.items)) for s in sequences)
    def coverage(catalog):
        return {src: (sum(source_presence(metadata.get(i, {}), src) for i in catalog) / len(catalog)
                      if metadata_available and catalog else None) for src in ('category', 'brand', 'title')}
    category_values, brand_values = set(), set()
    if metadata_available:
        for item in split.item2idx:
            rec = metadata.get(item, {})
            category_values.update(rec.get('categories') or ([rec['category']] if rec.get('category') else []))
            if rec.get('brand'):
                brand_values.add(rec['brand'])
    result = dict(dataset=dataset_name, users=len(sequences), original_items=len(items),
                  catalog_train_base_items=split.num_items, interactions=sum(lengths),
                  unique_user_item_pairs=pairs, sparsity=1 - pairs / (len(sequences) * len(items)),
                  sparsity_definition='1 - unique_user_item_pairs / (users * original_items)',
                  mean_sequence_length=float(np.mean(lengths)), min_sequence_length=min(lengths),
                  max_sequence_length=max(lengths), train_base_events=len(split.train_base_events),
                  train_base_pairs=len(split.train_base_samples), train_meta_contexts=len(split.train_meta_samples),
                  validation_contexts=len(split.val_samples), test_contexts=len(split.test_samples),
                  oov_counts=split.oov_counts, metadata_status='available' if metadata_available else 'missing_file',
                  metadata_coverage_original_items=coverage(items),
                  metadata_coverage_train_base_catalog=coverage(split.item2idx),
                  category_vocab_train_base=len(category_values) if metadata_available else None,
                  brand_vocab_train_base=len(brand_values) if metadata_available else None,
                  title_encoder=None, title_encoder_status='not_encoded_by_this_collection',
                  expected_reference_stats=expected_stats)
    result['reference_users_items_match'] = (result['users'] == expected_stats['users'] and len(items) == expected_stats['items']) if expected_stats else None
    return result


def context_table(samples, split, metadata, metadata_available):
    """Offline analysis groups only; target popularity is never a gate input."""
    counts = Counter(split.train_base_events)
    ordered = sorted(range(1, split.num_items + 1), key=lambda i: (-counts[i], i))
    head = set(ordered[:max(1, int(.2 * len(ordered)))])
    order = sorted(range(len(samples)), key=lambda i: (samples[i].history_len, samples[i].user_idx))
    quintile = {idx: q for q, indices in enumerate(np.array_split(order, 5), 1) for idx in indices}
    rows = []
    for i, sample in enumerate(samples):
        known_history = [itm for itm in sample.history if itm > 0]
        row = dict(user_idx=sample.user_idx, target_idx=sample.target, raw_target_asin=sample.raw_target_asin,
                   history_len=sample.history_len, known_history_items=len(known_history),
                   target_is_oov=sample.target <= 0 or sample.target > split.num_items,
                   target_train_base_frequency=counts[sample.target], analysis_length_quintile=quintile[i],
                   analysis_target_popularity_group=('oov' if sample.target == 0 else 'head' if sample.target in head else 'tail'))
        target_record = metadata.get(sample.raw_target_asin, {})
        for src in ('category', 'brand', 'title'):
            row[f'target_metadata_has_{src}'] = source_presence(target_record, src) if metadata_available else None
            row[f'known_history_{src}_fraction'] = (sum(source_presence(metadata.get(split.idx2item[itm], {}), src) for itm in known_history) / len(known_history)
                                                   if metadata_available and known_history else None)
        rows.append(row)
    return pd.DataFrame(rows)


def save_training_history(folder, history):
    folder = Path(folder)
    write_json(folder / 'training_history.json', asdict(history))
    rows = []
    for i, loss in enumerate(history.train_losses):
        metrics = history.val_metrics_by_epoch[i] if i < len(history.val_metrics_by_epoch) else {}
        rows.append(dict(epoch=i + 1, train_loss=loss, selected_checkpoint=(i + 1 == history.best_epoch),
                         train_sec=history.epoch_durations[i],
                         epoch_wall_sec=history.epoch_wall_durations[i] if i < len(history.epoch_wall_durations) else None,
                         **{k.replace('@', '_at_'): v for k, v in metrics.items()}))
    with (folder / 'epochs.csv').open('x', newline='', encoding='utf-8') as f:
        pd.DataFrame(rows).to_csv(f, index=False)


def save_pilot_report(folder, config, stats, history, baseline_results, contexts,
                      runtime, total_wall_sec, rank_format='csv', scientific_status='PENDING'):
    """Save report inputs; no prose claim of acceptance or invented GPU budget."""
    folder = Path(folder)
    write_json(folder / 'dataset_statistics.json', stats)
    baseline_rows, groups = [], []
    serialized = {}
    for method, splits in baseline_results.items():
        serialized[method] = {}
        for split_name, result in splits.items():
            serialized[method][split_name] = result.to_dict()
            baseline_rows.append(dict(method=method, split=split_name, **result.to_dict()))
            context = None if method == 'SASRec_shuffled' else contexts[split_name]
            frame = write_rank_table(folder / 'baselines' / method, 'per_user_' + split_name,
                                     result.per_user_records, context, rank_format)
            if context is not None:
                for column in ('analysis_length_quintile', 'analysis_target_popularity_group'):
                    for label, values in frame.groupby(column, dropna=False):
                        groups.append(dict(method=method, split=split_name, group_type=column, group=str(label),
                                           num_users=len(values), **{metric: float(values[metric].mean()) for metric in ('ndcg@10', 'recall@10', 'ndcg@20', 'recall@20')}))
    with (folder / 'baseline_metrics.csv').open('x', newline='', encoding='utf-8') as f:
        pd.DataFrame(baseline_rows).to_csv(f, index=False)
    with (folder / 'group_metrics.csv').open('x', newline='', encoding='utf-8') as f:
        pd.DataFrame(groups).to_csv(f, index=False)
    costs = dict(train_wall_sec=history.total_training_time, wall_to_reporting_sec=total_wall_sec,
                 wall_measurement_scope='through evaluation, before report serialization; run_state.json records complete execution wall time',
                 mean_epoch_train_sec=float(np.mean(history.epoch_durations)),
                 mean_epoch_wall_sec=float(np.mean(history.epoch_wall_durations)),
                 epoch_wall_definition='training + validation + checkpoint/epoch bookkeeping',
                 total_epochs=len(history.train_losses), best_epoch=history.best_epoch,
                 stop_reason=history.stop_reason, training_resources=history.resource_usage,
                 observed_runtime=runtime)
    # Only the ID-branch block has a measured matching training time in this pilot.
    budget = []
    for block, runs in [('early_fusion', 72), ('late_fusion_id', 9), ('late_fusion_residual', 63),
                        ('train_time_controls', 108), ('uasf_and_7_baselines', 120), ('ablations', 75), ('inference_only_controls', 0)]:
        measured = (block == 'late_fusion_id' and config.get('mechanism') == 'id_only' and config['is_real_data']
                    and str(config['device']).startswith('cuda')
                    and history.resource_usage.get('gpu_status') == 'measured')
        budget.append(dict(block=block, planned_training_runs=runs,
                           estimated_gpu_hours=runs * history.total_training_time / 3600 if measured else None,
                           status='estimate_from_this_id_pilot_only' if measured else 'pending_matching_gpu_measurement',
                           source='memo_20_10_section_4_3; proposed counts, not completed experiments'))
    with (folder / 'compute_budget.csv').open('x', newline='', encoding='utf-8') as f:
        pd.DataFrame(budget).to_csv(f, index=False)
    report = dict(schema_version=2, data_mode='real' if config['is_real_data'] else 'synthetic_fixture',
                  scientific_status=scientific_status, dataset=stats, baseline_metrics=serialized,
                  sanity_checks=config['sanity_checks'], costs=costs, compute_budget=budget,
                  protocol={k: config[k] for k in ('loss_type', 'mask_seen_policy', 'tie_policy', 'k_list', 'seed')},
                  limitations=['CPU runtime is not a GPU-hour measurement',
                               'ID pilot time does not measure fusion/UASF/ablation costs',
                               'Missing metadata coverage is null, not measured zero',
                               'Analysis target-popularity groups must not become gate inputs'],
                  rank_format=rank_format, missing_metadata=stats['metadata_status'] != 'available',
                  open_issues=[{'check': key, **value} for key, value in config['sanity_checks'].items() if not value['status'].startswith('PASS')],
                  report_field_status='Structured inputs saved; null marks missing or unmeasured evidence, not a measured zero')
    write_json(folder / 'report_data.json', report)
    return report


def write_artifact_index(folder):
    folder = Path(folder)
    rows = []
    for path in sorted(folder.rglob('*')):
        if path.is_file() and path.name != 'artifact_index.json':
            sha = hashlib.sha256()
            with path.open('rb') as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b''):
                    sha.update(chunk)
            rows.append(dict(path=path.relative_to(folder).as_posix(), bytes=path.stat().st_size, sha256=sha.hexdigest()))
    write_json(folder / 'artifact_index.json', {'artifacts': rows})
