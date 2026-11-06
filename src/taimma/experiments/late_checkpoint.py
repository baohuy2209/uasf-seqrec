"""Isolated rerun of the frozen-ID late reference, never early or K2--K4.

The model and training loop are shared unchanged with checkpoint2. This runner
fixes only late orchestration, provenance, resume checks and result selection.
"""
from pathlib import Path
import hashlib
import json
import subprocess

import numpy as np
import pandas as pd
import torch

from .checkpoint2 import Checkpoint2, exclusive, subset_name
from .checkpoint2_protocol import (
    SEEDS, SUBSETS, aligned_rank_matrix, check_training_environment,
    context_signature, digest, git_state, group_contexts,
    grouped_subset_analysis, paired_seed_ci, read_ranks, sha, write_json,
)
from ..training.checkpoint2_trainer import ModelSpec


def normalize_meta_signature(data):
    """Support old splits retaining OOV rows and splits storing eligible rows only."""
    split = data['split']
    eligible = [s for s in split.train_meta_samples if s.target > 0]
    retained_oov = sum(s.target == 0 for s in split.train_meta_samples)
    declared_oov = int(split.oov_counts['train_meta'])
    if retained_oov not in (0, declared_oov) or any(s.target < 0 for s in split.train_meta_samples):
        raise ValueError('Inconsistent Train-meta OOV accounting')
    data['signature'].update(
        train_meta_sha256=context_signature(eligible),
        train_meta_eligible=len(eligible), train_meta_oov=declared_oov,
        train_meta_total=len(eligible) + declared_oov,
    )
    return data


def lock_json(path, value):
    """An existing binding must agree; do not silently relabel prior results."""
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != value:
            raise ValueError('Late checkpoint binding changed; use a new output: ' + str(path))
    else:
        write_json(path, value, overwrite=False)


def paired_difference(lhs, rhs, reps):
    """Positive means lhs is better; the shared helper computes control-real."""
    result = paired_seed_ci(rhs, lhs, reps)
    result['delta_lhs_minus_rhs'] = result.pop('delta_control_minus_real')
    result['lhs_by_seed'] = result.pop('control_by_seed')
    result['rhs_by_seed'] = result.pop('real_by_seed')
    return result


class LateCheckpoint(Checkpoint2):
    def dataset(self, key):
        return normalize_meta_signature(super().dataset(key))

    def check_data(self, key):
        result = super().check_data(key)
        signature = self.dataset(key)['signature']
        result.update({name: signature[name] for name in
                       ('train_meta_eligible', 'train_meta_oov', 'train_meta_total')})
        return result

    def prepare(self, key):
        self.guard(inference=True)
        data = self.dataset(key)
        folder = self.output / key
        sources = self.source_signature()
        tracked = set(subprocess.check_output(
            ['git', 'ls-files', '--cached', '--', 'src', 'scripts', 'configs'],
            cwd=self.repo, text=True).splitlines())
        missing = sorted(set(sources) - tracked)
        if missing:
            raise RuntimeError('Commit all execution sources, including src/taimma/data: ' + ', '.join(missing))
        protocol = dict(late_protocol_version=1, config=self.config,
                        source_sha256=sources, data=data['signature'],
                        environment=check_training_environment(self.config['environment']))
        # HEAD is provenance, not a scientific cache input: a results-only commit
        # may change HEAD without changing executable source/config/data/runtime.
        lock_json(folder / 'protocol.lock.json', protocol)
        if not (folder / 'origin_git.json').exists():
            write_json(folder / 'origin_git.json', git_state(self.repo), overwrite=False)
        groups, summary = group_contexts(data['split'])
        lock_json(folder / 'groups.json', dict(**summary, data_signature=data['signature']))
        group_path = folder / 'groups.csv'
        group_text = groups.to_csv(index=False, lineterminator='\n')
        group_hash = hashlib.sha256(group_text.encode('utf-8')).hexdigest()
        lock_json(folder / 'groups_binding.json', dict(csv_sha256=group_hash))
        if group_path.exists():
            if sha(group_path) != group_hash:
                raise ValueError('Late group assignments changed')
        else:
            group_path.write_text(group_text, encoding='utf-8', newline='\n')
        return data

    def assert_protocol(self, key):
        expected = json.loads((self.output / key / 'protocol.lock.json').read_text(encoding='utf-8'))
        if expected['source_sha256'] != self.source_signature():
            raise ValueError('Source changed during the late run; preserve this run and use a new output')
        if expected['config'] != self.config or expected['data'] != self.dataset(key)['signature']:
            raise ValueError('Late config/data signature changed')

    def run_job(self, key, name, spec, train_samples=None, id_job=None, control=None, max_wall_seconds=None):
        if spec.kind not in ('id', 'late') or control is not None or train_samples is not None:
            raise ValueError('This runner permits only ID and the Train-base late reference')
        if spec.kind == 'late':
            if not spec.freeze_id or spec.source_dropout_mode != 'batch' or spec.all_off_probability != 0:
                raise ValueError('K2--K4 settings are separate experiments, not the late reference')
            if id_job != f'id_{spec.seed}':
                raise ValueError('Late must use the independently trained ID with the same seed')
        self.assert_protocol(key)
        if id_job:
            binding = dict(id_job=id_job, id_checkpoint_sha256=self.model_hash(key, id_job))
            lock_json(self.job_path(key, name) / 'id_dependency.json', binding)
        return super().run_job(key, name, spec, id_job=id_job, max_wall_seconds=max_wall_seconds)

    def verify_frozen_id(self, key, name):
        rec = self.completed(key, name)
        seed = rec['spec']['seed']
        ident = self.completed(key, f'id_{seed}')
        late_path = self.job_path(key, name) / rec['checkpoint']
        id_path = self.job_path(key, f'id_{seed}') / ident['checkpoint']
        late = torch.load(late_path, map_location='cpu', weights_only=True)
        base = torch.load(id_path, map_location='cpu', weights_only=True)
        actual = {k[len('id_model.'):]: v for k, v in late['model_state_dict'].items()
                  if k.startswith('id_model.')}
        expected = base['model_state_dict']
        if actual.keys() != expected.keys() or any(not torch.equal(actual[k], expected[k]) for k in expected):
            raise ValueError('Frozen ID changed in ' + name)
        history = json.loads((late_path.parent / 'history.json').read_text(encoding='utf-8'))
        vals = history['val_ndcg_10']
        best = int(np.argmax(vals)) + 1
        if (history['selection_scope'] != 'trained_epochs_only' or rec['best_epoch'] != best
                or late['epoch'] != best or history['best_epoch'] != best
                or history['best_val_ndcg'] != vals[best - 1]
                or rec['val_ndcg_10'] != vals[best - 1]):
            raise ValueError('Checkpoint does not match full-precision trained-epoch selection')
        initial = history['initial_validation_metrics']['NDCG@10']
        if initial != ident['val_ndcg_10']:
            raise ValueError('Zero-initialized late reference did not start at the ID validation metric')
        evidence = dict(status='VERIFIED', frozen_id_unchanged=True,
                        id_checkpoint_sha256=sha(id_path), late_checkpoint_sha256=sha(late_path),
                        initial_val_ndcg_10=initial, selected_epoch=best,
                        selection_scope='trained_epochs_only')
        lock_json(self.job_path(key, name) / 'late_verification.json', evidence)
        return evidence

    def run_late(self, key):
        with exclusive(self.output / key / 'late_execution.lock'):
            self.prepare(key)
            self.id_grid(key)
            for seed in SEEDS:
                for sources in SUBSETS[1:]:
                    name = f'late_{subset_name(sources)}_{seed}'
                    self.run_job(key, name, ModelSpec('late', seed, sources=sources,
                        source_dropout=.2, source_dropout_mode='batch'), id_job=f'id_{seed}')
                    self.verify_frozen_id(key, name)
            return self.summarize(key)

    def summarize(self, key, early_output=None):
        """Read ranks only; neither training nor model inference is performed."""
        split = self.dataset(key)['split']
        folder = self.output / key
        protocol = json.loads((folder / 'protocol.lock.json').read_text(encoding='utf-8'))
        if protocol['data'] != self.dataset(key)['signature']:
            raise ValueError('Summary dataset differs from the late run')
        groups_meta = json.loads((folder / 'groups.json').read_text(encoding='utf-8'))
        if groups_meta['data_signature'] != protocol['data']:
            raise ValueError('Group dataset binding differs from the late run')
        groups = pd.read_csv(folder / 'groups.csv')
        binding = json.loads((folder / 'groups_binding.json').read_text(encoding='utf-8'))
        if binding['csv_sha256'] != sha(folder / 'groups.csv'):
            raise ValueError('Late group assignments changed')
        frames = {part: {} for part in ('val', 'test')}
        rows, grouped = [], []
        for seed in SEEDS:
            for part in frames:
                per_subset = {}
                for sources in SUBSETS:
                    label = subset_name(sources)
                    job = f'late_{label}_{seed}' if sources else f'id_{seed}'
                    rec = self.completed(key, job)
                    if rec is None:
                        raise FileNotFoundError('Incomplete late grid: ' + job)
                    if sources:
                        proof = json.loads((self.job_path(key, job) / 'late_verification.json').read_text(encoding='utf-8'))
                        if proof['late_checkpoint_sha256'] != self.model_hash(key, job):
                            raise ValueError('Late verification does not match ' + job)
                    ranks = read_ranks((self.job_path(key, job) / rec['checkpoint']).parent / f'per_user_{part}.csv')
                    per_subset[label] = ranks
                    if part == 'test':
                        rows.append(dict(subset=label, seed=seed, ndcg_10=float(ranks['ndcg@10'].mean()),
                                         recall_10=float(ranks['recall@10'].mean()), users=len(ranks)))
                aligned_rank_matrix(per_subset, getattr(split, f'{part}_samples'))
                frames[part][seed] = per_subset
            result, users = grouped_subset_analysis(frames['val'][seed], frames['test'][seed], groups)
            users.to_csv(folder / f'group_late_{seed}.csv', index=False, lineterminator='\n')
            grouped.append(dict(mechanism='late', seed=seed, **result))
        per_seed = pd.DataFrame(rows)
        summary = per_seed.groupby('subset', sort=False).ndcg_10.agg(['mean', 'std']).reset_index()
        comparisons = {}
        for source in ('C', 'B', 'T'):
            without = ''.join(s for s in 'CBT' if s != source)
            for kind, lhs, rhs in [('add', source, 'ID'), ('drop', 'CBT', without)]:
                comparisons[f'{kind}_{source}'] = paired_difference(
                    {s: frames['test'][s][lhs] for s in SEEDS},
                    {s: frames['test'][s][rhs] for s in SEEDS}, self.config['bootstrap_reps'])
        per_seed.to_csv(folder / 'late_per_seed.csv', index=False, lineterminator='\n')
        summary.to_csv(folder / 'late_summary.csv', index=False, lineterminator='\n')
        write_json(folder / 'late_add_drop.json', dict(drop_definition='full_minus_without_source', comparisons=comparisons))
        write_json(folder / 'late_group_analysis.json', grouped)
        result = dict(status='COMPLETE', dataset=key, mechanism='frozen_id_late_reference',
                      seeds=list(SEEDS), jobs=24, groups_use='this_output_only_no_legacy',
                      results=summary.to_dict(orient='records'))
        if early_output is not None:
            root = Path(early_output).resolve() / key
            early_protocol = json.loads((root / 'protocol.lock.json').read_text(encoding='utf-8'))
            for field in ('raw_reviews_sha256', 'metadata_files', 'catalog_sha256',
                          'train_base_sha256', 'val_sha256', 'test_sha256'):
                if early_protocol['data'][field] != protocol['data'][field]:
                    raise ValueError('Early/late data differ: ' + field)
            pairings = {}
            for sources in SUBSETS:
                label = subset_name(sources)
                by_seed = {}
                for seed in SEEDS:
                    name = f'early_{label}_{seed}' if sources else f'id_{seed}'
                    job = root / 'jobs' / name
                    rec = json.loads((job / 'completed.json').read_text(encoding='utf-8'))
                    for rel, h in rec['artifacts'].items():
                        path = (job / rel).resolve()
                        if not path.is_relative_to(job.resolve()) or not path.is_file() or sha(path) != h:
                            raise ValueError('Early artifact changed: ' + str(path))
                    by_seed[seed] = read_ranks((job / rec['checkpoint']).parent / 'per_user_test.csv')
                    aligned_rank_matrix(dict(early=by_seed[seed], late=frames['test'][seed][label]), split.test_samples)
                pairings[label] = paired_difference(by_seed, {s: frames['test'][s][label] for s in SEEDS}, self.config['bootstrap_reps'])
            write_json(folder / 'early_minus_late.json', dict(early_output=str(root), comparisons=pairings))
            result['early_comparison'] = str(folder / 'early_minus_late.json')
        write_json(folder / 'late_report.json', result)
        return result
