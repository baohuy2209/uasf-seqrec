"""OOF evidence tests use synthetic CPU classifiers only, never research data."""
import sys
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from taimma.experiments import memo0711_e1_evidence as module
from taimma.evaluation.e0_e1_protocol import evaluate_e1_predictability


@pytest.mark.parametrize('regime', ['contextual', 'averaged', 'group'])
def test_replay_matches_existing_engine_and_user_folds(regime):
    rng = np.random.default_rng(42)
    users = np.repeat(np.arange(40), 2)
    X = rng.normal(size=(len(users), 3)); delta = .7 * X[:, 1] + rng.normal(size=len(users))
    groups = np.zeros(len(users), dtype=int)
    original = evaluate_e1_predictability(X, delta, users, .1, n_bootstrap=20,
        evaluated_regime=regime, group_indices=groups)
    frame, metrics, folds = module.replay_predictions(X, delta, users, .1, regime, groups)
    module._assert_metrics(metrics, asdict(original))
    assert frame.groupby('user_idx').fold.nunique().max() == 1
    assert all(not (set(f['train_users']) & set(f['heldout_users'])) for f in folds)


def make_evidence_input(tmp_path, metric='CE'):
    root = tmp_path / 'runs/memo/beauty/selected_s2'; root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    n = 60; lengths = rng.integers(5, 30, n); signal = rng.normal(size=n)
    delta = signal + rng.normal(size=n); second = delta + rng.normal(0, .05, n)
    rows = []
    for source in ['category', 'brand', 'title']:
        for seed, values in [(42, delta), (43, second)]:
            for i in range(n):
                rows.append(dict(user_idx=i, context_id=f'ctx{i:03}', source_name=source,
                    reference_seed=seed, seq_len=int(lengths[i]), group_idx=0,
                    feat_log_len=float(np.log(lengths[i])), feat_signal=float(signal[i]), delta=float(values[i])))
    path = root / f'utility_{metric}.csv'; pd.DataFrame(rows).to_csv(path, index=False)
    X = np.column_stack([np.log(lengths), signal])
    e1 = asdict(evaluate_e1_predictability(X, delta, np.arange(n), .1, n_bootstrap=20))
    report = dict(seed=42, delta_file_sha256=module.sha(path), feature_names=['feat_log_len','feat_signal'],
        reference_seeds=[42,43], sources={s:dict(stability=dict(epsilon=.1, stability_regime='contextual'),e1=e1)
                                    for s in ['category','brand','title']})
    report_path = root / f'e0_e1_{metric}.json'; report_path.write_text(json.dumps(report), encoding='utf-8')
    cache = dict(signature=dict(input_sha256=module.sha(path), source_sha256=module.sha(REPO/'src/taimma/evaluation/e0_e1_protocol.py')),
                 sha256=module.sha(report_path))
    report_path.with_suffix('.cache.json').write_text(json.dumps(cache), encoding='utf-8')
    return tmp_path/'runs/memo', root


def test_export_cache_and_pure_inspector(tmp_path, monkeypatch):
    output, root = make_evidence_input(tmp_path)
    make_evidence_input(tmp_path, 'LOG_RANK')
    original = {p: p.read_bytes() for p in root.iterdir()}
    result = module.export_e1_evidence(REPO, output)
    assert all(row['status'] == 'EXPORTED' for row in result['outcomes'])
    inspection = module.inspect_e1_evidence(output/'beauty')
    assert inspection['status'] == 'PASS' and not inspection['cpu_predictor_fitting']
    frame = pd.read_csv(output/'beauty/e1_oof_evidence/CE/paired_oof_predictions.csv')
    assert len(frame) == 180 and frame.groupby(['source_name','user_idx']).fold.nunique().max() == 1
    assert np.array_equal(frame.delta, frame.delta_reference_1)
    assert all(p.read_bytes() == b for p,b in original.items())
    monkeypatch.setattr(module, 'replay_predictions', lambda *args: pytest.fail('Valid cache must not refit'))
    cached = module.export_e1_evidence(REPO, output)
    assert all(row['status'] == 'VERIFIED_CACHE' for row in cached['outcomes'])
    p = output/'beauty/e1_oof_evidence/CE/paired_oof_predictions.csv'
    p.write_text(p.read_text(encoding='utf-8') + '\n', encoding='utf-8')
    with pytest.raises(ValueError, match='artifact changed'):
        module.inspect_e1_evidence(output/'beauty')


def test_missing_data_never_fabricates_rows(tmp_path):
    output = tmp_path/'runs/absent'
    result = module.export_e1_evidence(REPO, output)
    assert all(row['status'] == 'MISSING_INPUTS' for row in result['outcomes'])
    assert not output.exists()


def test_mismatched_original_auc_is_rejected(tmp_path):
    output, root = make_evidence_input(tmp_path)
    p = root/'e0_e1_CE.json'; d=json.loads(p.read_text(encoding='utf-8'))
    d['sources']['category']['e1']['oof_auc_lr'] = .01
    p.write_text(json.dumps(d), encoding='utf-8')
    c = p.with_suffix('.cache.json'); binding=json.loads(c.read_text(encoding='utf-8'))
    binding['sha256']=module.sha(p); c.write_text(json.dumps(binding), encoding='utf-8')
    with pytest.raises(ValueError, match='differs from the bound original'):
        module.export_e1_evidence(REPO, output)
    assert not (output/'beauty/e1_oof_evidence/CE/manifest.json').exists()
