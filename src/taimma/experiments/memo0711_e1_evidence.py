"""Replay the locked E1 predictors and persist independently auditable OOF rows.

This module never fits S1 or gates.  The fold/scaler/estimator pattern is copied
from evaluate_e1_predictability in the unchanged e0_e1_protocol.py; the original
reports remain the authority and replay mismatches are errors.
"""
from pathlib import Path
import hashlib
import importlib.util
import importlib.metadata
import json
import sys
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def replay_predictions(X, delta, users, epsilon, regime, group_indices=None):
    """Exactly the five user-held-out folds used by the existing E1 engine."""
    X = np.asarray(X, dtype=float); delta = np.asarray(delta, dtype=float)
    users = np.asarray(users); y = (delta > epsilon).astype(int)
    if X.ndim != 2 or X.shape[0] != len(delta) or users.shape != delta.shape:
        raise ValueError('Misaligned E1 replay inputs')
    if not np.isfinite(X).all() or not np.isfinite(delta).all():
        raise ValueError('Nonfinite E1 replay inputs')
    if regime not in ('contextual', 'averaged', 'group'):
        raise ValueError('Unknown E1 replay regime')
    if len(np.unique(y)) < 2 or len(np.unique(users)) < 5:
        raise ValueError('E1 replay is not evaluable: classes or user folds missing')
    if regime == 'group':
        group_indices = np.asarray(group_indices)
        if group_indices.shape != delta.shape or not np.isin(group_indices, np.arange(10)).all():
            raise ValueError('Group E1 replay requires aligned inference-available group IDs')
    predictions = np.empty(len(delta)); tree_predictions = np.empty(len(delta))
    length_predictions = np.empty(len(delta)); folds = np.empty(len(delta), dtype=int)
    fold_evidence = []
    for fold, (train, test) in enumerate(GroupKFold(n_splits=5).split(X, y, users)):
        if set(users[train]) & set(users[test]):
            raise ValueError('User leakage in E1 OOF folds')
        if len(np.unique(y[train])) < 2:
            raise ValueError('E1 replay training fold has only one class')
        scaler = StandardScaler().fit(X[train])
        train_X, test_X = scaler.transform(X[train]), scaler.transform(X[test])
        if regime == 'group':
            means = {g: float(y[train][group_indices[train] == g].mean()) for g in np.unique(group_indices[train])}
            if any(g not in means for g in np.unique(group_indices[test])):
                raise ValueError('Held-out group absent from E1 training fold')
            predictions[test] = [means[g] for g in group_indices[test]]
            tree_predictions[test] = predictions[test]
        else:
            lr = LogisticRegression(random_state=42, max_iter=200).fit(train_X, y[train])
            tree = DecisionTreeClassifier(max_depth=3, random_state=42).fit(train_X, y[train])
            predictions[test] = lr.predict_proba(test_X)[:, 1]
            tree_predictions[test] = tree.predict_proba(test_X)[:, 1]
        length_lr = LogisticRegression(random_state=42, max_iter=200).fit(train_X[:, :1], y[train])
        length_predictions[test] = length_lr.predict_proba(test_X[:, :1])[:, 1]
        folds[test] = fold
        fold_evidence.append(dict(fold=fold, train_users=sorted(set(map(int, users[train]))),
                                  heldout_users=sorted(set(map(int, users[test])))))
    rho = None if np.ptp(predictions) == 0 or np.ptp(delta) == 0 else float(stats.spearmanr(predictions, delta).statistic)
    metrics = dict(oof_auc_lr=float(roc_auc_score(y, predictions)),
                   oof_auc_tree=float(roc_auc_score(y, tree_predictions)),
                   baseline_len_auc=float(roc_auc_score(y, length_predictions)), oof_spearman_rho=rho)
    frame = pd.DataFrame(dict(user_idx=users, fold=folds, delta=delta, label=y,
                              probability_lr=predictions, probability_tree=tree_predictions,
                              probability_length=length_predictions))
    return frame, metrics, fold_evidence


def _assert_metrics(actual, recorded):
    for name, value in actual.items():
        old = recorded.get(name)
        if old is None and value is None:
            continue
        if old is None or value is None or not np.isclose(old, value, rtol=0, atol=1e-10):
            raise ValueError('OOF replay differs from the bound original E1 report: ' + name)


def export_e1_evidence(repo, output, datasets=('beauty',)):
    """CPU analysis, not report-only: replay LR/tree fits from frozen utility CSVs.

    A verified cache performs no fits. Missing inputs and NOT_EVALUABLE reports
    produce explicit status rows, not invented probabilities. Original utility,
    E1 JSON, S1 checkpoints, and gate checkpoints are never modified.
    """
    repo, output = Path(repo).resolve(), Path(output).resolve()
    script = repo / 'scripts/run_e0_e1_evaluation.py'
    engine = repo / 'src/taimma/evaluation/e0_e1_protocol.py'
    module_spec = importlib.util.spec_from_file_location('_memo_e1_original_loader', script)
    loader = importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(loader)
    outcomes = []; fitted_sources = 0
    for dataset in datasets:
        root = output / dataset / 'selected_s2'
        for metric in ('CE', 'LOG_RANK'):
            delta_path, report_path = root / f'utility_{metric}.csv', root / f'e0_e1_{metric}.json'
            if not delta_path.is_file() or not report_path.is_file():
                outcomes.append(dict(dataset=dataset, metric=metric, status='MISSING_INPUTS'))
                continue
            report = _json(report_path)
            if report.get('seed', 42) != 42:
                raise ValueError('OOF replay requires the originally locked seed 42')
            if report.get('delta_file_sha256') != sha(delta_path):
                raise ValueError('E1 report does not bind the supplied delta CSV')
            old_cache = report_path.with_suffix('.cache.json')
            if not old_cache.is_file():
                raise ValueError('Missing original E1 cache binding: ' + str(old_cache))
            binding = _json(old_cache)
            if (binding.get('sha256') != sha(report_path)
                    or binding.get('signature', {}).get('input_sha256') != sha(delta_path)
                    or binding['signature'].get('source_sha256') != sha(engine)):
                raise ValueError('Original E1 report/input/engine binding changed')
            sources = {str(p): sha(p) for p in [delta_path, report_path, old_cache, script, engine, Path(__file__)]}
            for evidence in [root / 'alignment_cache.json', root / 'features_train_meta.csv',
                             output / dataset / 'protocol.lock.json']:
                if evidence.is_file():
                    sources[str(evidence)] = sha(evidence)
            versions = {name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'scipy', 'scikit-learn']}
            protocol = output / dataset / 'protocol.lock.json'
            if protocol.is_file():
                expected_env = _json(protocol).get('config', {}).get('environment', {})
                if any(str(expected_env.get(k, versions[k])) != str(versions[k]) for k in versions):
                    raise ValueError('CPU replay libraries differ from the locked E1 environment')
            signature = dict(schema_version=1, sources=sources, dataset_root=str(root.parent), repository_root=str(repo), seed=42, folds=5, library_versions=versions,
                             estimator_contract='unchanged_evaluate_e1_predictability', metric=metric)
            folder = output / dataset / 'e1_oof_evidence' / metric
            marker = folder / 'manifest.json'
            if marker.is_file():
                cached = _json(marker)
                if cached['signature'] != signature:
                    raise ValueError('E1 replay cache dependencies changed; preserve output and use a new version')
                for relative, digest in cached['files'].items():
                    p = (folder / relative).resolve()
                    if not p.is_relative_to(folder.resolve()) or not p.is_file() or sha(p) != digest:
                        raise ValueError('E1 OOF evidence cache changed')
                outcomes.append(dict(dataset=dataset, metric=metric, status='VERIFIED_CACHE', manifest=str(marker)))
                continue
            if (output / 'freeze.json').is_file():
                raise RuntimeError('Cannot fit E1 replay predictors after study freeze')
            X, deltas, users, lengths, meta = loader.load_delta_dataset(str(delta_path), return_metadata=True)
            if report.get('feature_names') is not None and report['feature_names'] != meta['feature_names']:
                raise ValueError('OOF replay feature names/order differ from original E1 report')
            if report.get('reference_seeds') is not None and report['reference_seeds'] != meta['reference_seeds']:
                raise ValueError('OOF replay reference seeds differ from original E1 report')
            raw = pd.read_csv(delta_path)
            if 'context_id' not in raw:
                raw['context_id'] = raw['user_idx']
            contexts = raw[['user_idx', 'context_id']].drop_duplicates().sort_values(['user_idx', 'context_id'])
            if contexts.user_idx.tolist() != list(users):
                raise ValueError('OOF context order differs from the original E1 loader')
            rows, summaries, fold_records = [], [], []
            for source, record in report.get('sources', {}).items():
                e1 = record['e1']; stability = record['stability']
                if e1.get('status') != 'EVALUATED':
                    summaries.append(dict(source=source, status='NOT_EVALUABLE', reason=e1.get('evidence_summary')))
                    continue
                regime = e1['evaluated_regime']
                expected_regime = {'contextual':'contextual', 'soft_supervision':'averaged', 'group_level':'group'}[stability['stability_regime']]
                if regime != expected_regime:
                    raise ValueError('E1 stability and evaluated regime disagree')
                first, second = deltas[source]
                delta = first if regime == 'contextual' else (first + second) / 2
                frame, metrics, folds = replay_predictions(meta['source_features'].get(source, X), delta,
                    users, stability['epsilon'], regime, meta['group_indices'])
                fitted_sources += 1
                _assert_metrics(metrics, e1)
                frame['context_id'] = contexts.context_id.to_numpy()
                frame['source_name'] = source; frame['metric'] = metric; frame['evaluated_regime'] = regime
                frame['reference_seeds'] = json.dumps(meta['reference_seeds'])
                frame['delta_reference_1'] = first; frame['delta_reference_2'] = second
                frame['epsilon'] = stability['epsilon']; rows.append(frame)
                summaries.append(dict(source=source, status='MATCHED_ORIGINAL_E1', **metrics))
                fold_records.append(dict(source=source, folds=folds))
            folder.mkdir(parents=True, exist_ok=True)
            files = {}
            if rows:
                p = folder / 'paired_oof_predictions.csv'
                pd.concat(rows, ignore_index=True).to_csv(p, index=False, lineterminator='\n'); files[p.name] = sha(p)
            summary = folder / 'replay_summary.json'
            _write(summary, dict(sources=summaries, folds=fold_records, feature_names=meta['feature_names'],
                                 metric_check_tolerance=1e-10, s1_training=False, gate_training=False))
            files[summary.name] = sha(summary)
            _write(marker, dict(signature=signature, files=files, analysis='CPU E1 predictor replay only',
                                original_report_unchanged=True))
            outcomes.append(dict(dataset=dataset, metric=metric, status='EXPORTED', manifest=str(marker)))
    return dict(s1_training=False, gate_training=False, cpu_predictor_fitting=bool(fitted_sources), fitted_sources=fitted_sources, outcomes=outcomes)


def inspect_e1_evidence(dataset_root):
    """Validate paired evidence using hashes/metrics only; never fit predictors."""
    root = Path(dataset_root).resolve()
    checked = []
    for metric in ('CE', 'LOG_RANK'):
        report_path = root / 'selected_s2' / f'e0_e1_{metric}.json'
        marker = root / 'e1_oof_evidence' / metric / 'manifest.json'
        if not report_path.is_file() or not marker.is_file():
            checked.append(dict(metric=metric, status='MISSING_E1_OOF_EVIDENCE'))
            continue
        record = _json(marker)
        if record.get('signature', {}).get('schema_version') != 1:
            raise ValueError('Unknown E1 OOF evidence schema')
        for absolute, expected in record['signature']['sources'].items():
            p = Path(absolute)
            if not p.is_file():
                # A copied bundle may be audited on Windows without changing its
                # original Linux provenance. Rebase only declared dataset/repo roots.
                normalized=str(absolute).replace('\\','/')
                for old, new in [(record['signature'].get('dataset_root'), root),
                                 (record['signature'].get('repository_root'), root.parents[2])]:
                    prefix=str(old).replace('\\','/').rstrip('/')+'/' if old else ''
                    if prefix and normalized.startswith(prefix):
                        candidate=(new/normalized[len(prefix):]).resolve()
                        if not candidate.is_relative_to(new.resolve()):raise ValueError('E1 evidence path escapes declared root')
                        p=candidate;break
            if not p.is_file() or sha(p) != expected:
                raise ValueError('E1 evidence source changed: ' + str(p))
        for relative, expected in record['files'].items():
            p = (marker.parent / relative).resolve()
            if not p.is_relative_to(marker.parent.resolve()) or not p.is_file() or sha(p) != expected:
                raise ValueError('E1 OOF evidence artifact changed')
        summary = _json(marker.parent / 'replay_summary.json')
        reports = _json(report_path)['sources']
        csv_path = marker.parent / 'paired_oof_predictions.csv'
        frame = pd.read_csv(csv_path) if csv_path.is_file() else pd.DataFrame()
        for item in summary['sources']:
            source = item['source']
            if item['status'] == 'NOT_EVALUABLE':
                if reports[source]['e1'].get('status') == 'EVALUATED':
                    raise ValueError('Evaluable E1 source cannot omit paired predictions')
                continue
            rows = frame.loc[frame.source_name == source]
            if rows.empty or rows.duplicated(['user_idx', 'context_id']).any():
                raise ValueError('Missing or duplicate E1 OOF contexts')
            if rows.groupby('user_idx').fold.nunique().max() != 1 or set(rows.fold) != set(range(5)):
                raise ValueError('OOF users span folds or folds are missing')
            y = rows.label.to_numpy(); probabilities = rows[['probability_lr', 'probability_tree', 'probability_length']].to_numpy()
            if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)) or not np.array_equal(y, (rows.delta > rows.epsilon).astype(int)):
                raise ValueError('Invalid OOF labels or probabilities')
            rho = None if np.ptp(rows.probability_lr) == 0 or np.ptp(rows.delta) == 0 else float(stats.spearmanr(rows.probability_lr, rows.delta).statistic)
            actual = dict(oof_auc_lr=float(roc_auc_score(y, probabilities[:, 0])),
                          oof_auc_tree=float(roc_auc_score(y, probabilities[:, 1])),
                          baseline_len_auc=float(roc_auc_score(y, probabilities[:, 2])), oof_spearman_rho=rho)
            _assert_metrics(actual, reports[source]['e1'])
        checked.append(dict(metric=metric, status='PASS', manifest=str(marker), manifest_sha256=sha(marker)))
    return dict(status='PASS' if all(r['status'] == 'PASS' for r in checked) else 'INCOMPLETE',
                cpu_predictor_fitting=False, checks=checked)
