"""Regression fixtures for the independently reproduced October 28 audit.

All files and model training here are test fixtures under pytest's temporary
directory. No research datasets or existing experiment artifacts are modified.
"""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from taimma.data.ingestion import Amazon2014Ingestor
from taimma.data.split import SequenceSample
from taimma.evaluation.evaluator import IndependentEvaluator
from taimma.evaluation.baselines import PopularityBaseline
from taimma.evaluation.e0_e1_protocol import (
    evaluate_e1_predictability, compute_noise_threshold_and_stability,
    evaluate_supervisor_decision, LabelStabilityResult, E0HeterogeneityResult,
    E1PredictabilityResult, make_utility_labels,
)
from taimma.evaluation.utility import compute_leave_one_source_out_deltas, export_utility_records_to_df
from taimma.experiments.logger import ExperimentLogger
from taimma.experiments.diagnostics import generate_title_gaussian_control
from taimma.models.late_fusion import SourceSequentialEncoder, ResidualLateFusionModel
from taimma.models.sasrec import SASRecModel


def script(name):
    spec = importlib.util.spec_from_file_location('test_' + name, ROOT / 'scripts' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def delta_script():
    return script('run_e0_e1_evaluation')


def long_rows():
    return [dict(context_id=cid, user_idx=uid, seq_len=length, source_name=src,
                 reference_seed=seed, delta=cid / 100, feat_coverage=coverage)
            for cid, uid, length in [(20, 200, 20), (10, 100, 10)]
            for src, coverage in [('category', .8), ('brand', .3)]
            for seed in (7, 11)]


def test_long_delta_alignment_and_source_features(delta_script, tmp_path):
    path = tmp_path / 'delta.csv'
    pd.DataFrame(long_rows()).to_csv(path, index=False)
    features, deltas, users, lengths, meta = delta_script.load_delta_dataset(str(path), return_metadata=True)
    assert users == [100, 200]
    assert deltas['category'][0].tolist() == [.1, .2]
    np.testing.assert_allclose(features[:, 0], np.log(lengths))
    np.testing.assert_allclose(meta['source_features']['category'][:, 1], .8)
    np.testing.assert_allclose(meta['source_features']['brand'][:, 1], .3)


@pytest.mark.parametrize('mutation', ['missing_seed', 'duplicate', 'feature_changed', 'nan', 'three_seeds'])
def test_incomplete_delta_contract_rejected(delta_script, tmp_path, mutation):
    rows = long_rows()
    if mutation == 'missing_seed':
        rows.pop()
    elif mutation == 'duplicate':
        rows.append(rows[0].copy())
    elif mutation == 'feature_changed':
        rows[0]['feat_coverage'] = .5
    elif mutation == 'nan':
        rows[0]['delta'] = np.nan
    else:
        rows.append(dict(rows[0], reference_seed=13))
    path = tmp_path / 'invalid.csv'
    pd.DataFrame(rows).to_csv(path, index=False)
    with pytest.raises(ValueError):
        delta_script.load_delta_dataset(str(path))


def test_missing_context_is_never_imputed(delta_script, tmp_path):
    path = tmp_path / 'missing.csv'
    pd.DataFrame(dict(user_idx=[1, 2], delta_category_seed1=[.1, -.2], delta_category_seed2=[.2, -.1])).to_csv(path, index=False)
    with pytest.raises(ValueError, match='seq_len'):
        delta_script.load_delta_dataset(str(path))


def test_real_metadata_parser_case_and_multilabel(tmp_path):
    path = tmp_path / 'meta.json'
    path.write_text('\n'.join(repr(row) for row in [
        dict(asin='A', brand=' ACME   Corp ', categories=[['Beauty', 'Skin', 'Face'], ['Beauty', 'Care']]),
        dict(asin='B', brand='acme corp'), dict(asin='C', brand='Unknown'), dict(asin='D', brand='Generic')]), encoding='utf-8')
    parsed = Amazon2014Ingestor().parse_metadata(str(path))
    assert parsed['A']['brand'] == parsed['B']['brand'] == 'acme corp'
    assert parsed['A']['categories'] == ['Skin', 'Face', 'Care']
    assert parsed['C']['brand'] is None
    assert parsed['D']['brand'] == 'generic'


def test_multilabel_late_fusion_and_delta_export(delta_script, tmp_path):
    torch.manual_seed(42)
    torch.set_num_threads(2)
    encoder = SourceSequentialEncoder('category', 8, feature_dim=16, max_seq_len=3)
    model = ResidualLateFusionModel(SASRecModel(4, max_seq_len=3, embedding_dim=16),
                                    {'category': encoder}, sources_enabled=('category',), p_drop=0)
    model.eval()
    attrs = {'category': torch.tensor([[1, 2], [3, 0], [4, 5], [6, 0]])}
    masks = {'category': torch.ones(4)}
    records = []
    # Different batches deliberately have the same batch-local row index zero.
    for user in [100, 200]:
        for seed in [7, 11]:
            _, rec = compute_leave_one_source_out_deltas(model, torch.tensor([[1, 2, 3]]),
                {'category': attrs['category'][torch.tensor([[0, 1, 2]])]}, attrs, masks,
                torch.tensor([4]), [user], reference_seed=seed)
            records.extend(rec)
    assert records[0].context_id == records[1].context_id
    assert records[0].context_id != records[2].context_id
    frame = export_utility_records_to_df(records)
    assert {'seq_len', 'feat_entropy', 'feat_log_len'}.issubset(frame)
    path = tmp_path / 'export.csv'
    frame.to_csv(path, index=False)
    features, deltas, users, lengths = delta_script.load_delta_dataset(str(path))
    assert features.shape == (2, 4)
    assert users == [100, 200]
    assert np.isfinite(deltas['category']).all()


@pytest.mark.parametrize('catalog', [1, 5, 15])
def test_oov_is_miss_for_every_cutoff(catalog):
    r = IndependentEvaluator(catalog).evaluate_ranks(np.zeros((1, catalog)), [0], [0])
    assert r.recall_10 == r.recall_20 == r.ndcg_10 == r.ndcg_20 == 0


@pytest.mark.parametrize('bad', [-np.inf, np.inf, np.nan])
def test_invalid_scores_rejected(bad):
    with pytest.raises(ValueError):
        IndependentEvaluator(5).evaluate_ranks(np.full((1, 5), bad), [1], [0])


def test_alignment_and_catalog_width_rejected():
    ev = IndependentEvaluator(5)
    with pytest.raises(ValueError):
        ev.evaluate_ranks(np.zeros((1, 3)), [1], [0])
    with pytest.raises(ValueError):
        ev.evaluate_ranks(np.zeros((1, 5)), [1], [])


def test_logger_serializes_device_and_protects_artifacts(tmp_path):
    logger = ExperimentLogger(str(tmp_path), str(tmp_path / 'registry.csv'))
    run_id, run_dir = logger.create_run_dir('fixture')
    result = IndependentEvaluator(2).evaluate_ranks(np.array([[.1, .2]]), [2], [0])
    logger.save_run_artifacts(run_dir, run_id, {'device': torch.device('cpu')}, result, result, .1, 1, 1, actual_device=torch.device('cpu'))
    assert json.loads((Path(run_dir) / 'config.json').read_text())['device'] == 'cpu'
    assert json.loads((Path(run_dir) / 'run_manifest.json').read_text())['device'] == 'cpu'
    assert (Path(run_dir) / 'per_user_val.csv').is_file()
    with pytest.raises(FileExistsError):
        logger.save_run_artifacts(run_dir, run_id, {}, result, result, .1, 1, 1)
    assert logger.create_run_dir('fixture')[0] != run_id


def test_rank_deficient_title_control_retains_covariance(monkeypatch):
    seen = {}
    real_rng = np.random.default_rng
    class Sampler:
        def multivariate_normal(self, mean, cov, size):
            seen['cov'] = cov
            return real_rng(42).multivariate_normal(mean, cov, size=size)
    monkeypatch.setattr('taimma.experiments.diagnostics.np.random.default_rng', lambda seed: Sampler())
    generate_title_gaussian_control(torch.tensor([[1., 2., 3., 4.], [2., 4., 6., 8.], [3., 6., 9., 12.]]))
    assert seen['cov'][0, 1] == pytest.approx(2.)


def test_single_class_is_not_a_measured_auc():
    res = evaluate_e1_predictability(np.ones((10, 2)), np.ones(10), list(range(10)), 0, n_bootstrap=10)
    assert res.status == 'NOT_EVALUABLE'
    assert res.oof_auc_lr is None and res.bootstrap_95_ci is None
    with pytest.raises(ValueError):
        compute_noise_threshold_and_stability(np.array([np.nan]), np.array([1.]), n_bootstrap=10)


def test_group_evidence_is_fitted_on_training_users_only():
    # Length is constant; a history-available binary group supplies the signal.
    groups = np.tile([0, 1], 100)
    delta = np.where(groups == 1, 1., -1.)
    X = np.column_stack([np.ones(200), groups])
    res = evaluate_e1_predictability(X, delta, list(range(200)), .1, n_bootstrap=30,
                                    evaluated_regime='group', group_indices=groups)
    assert res.status == 'EVALUATED' and res.e1_passed
    assert res.oof_spearman_ci is not None
    missing = evaluate_e1_predictability(X, delta, list(range(200)), .1, n_bootstrap=10, evaluated_regime='group')
    assert missing.status == 'NOT_EVALUABLE'
    np.testing.assert_allclose(make_utility_labels(np.array([-1., 0., 1.]), 0, temperature=1), [1/(1+np.e), .5, 1/(1+np.exp(-1))])


def test_missing_fallback_cannot_be_conditional():
    stab = {'category': LabelStabilityResult('category', .1, .05, .3, 'group_level')}
    e0 = {'category': E0HeterogeneityResult('category', .1, .3, .3, True, 1, (.5, 1.5), -1, (-1.5, -.5), True, True)}
    e1 = {'category': E1PredictabilityResult('category', .8, .8, .5, (.7, .9), True, 'context fixture')}
    assert evaluate_supervisor_decision(stab, e0, e1).decision == 'PENDING'
    e1['category'] = replace(e1['category'], evaluated_regime='group')
    assert evaluate_supervisor_decision(stab, e0, e1).decision == 'CONDITIONAL'
    assert evaluate_supervisor_decision({}, {}, {}).decision == 'PENDING'


def test_pilot_entrypoint_complete_repeat(tmp_path):
    env = dict(os.environ, PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    command = [sys.executable, '-B', str(ROOT / 'scripts/run_pilot.py'), '--smoke_test',
               '--epochs', '1', '--batch_size', '100', '--verify_reproducibility',
               '--runs_dir', str(tmp_path / 'runs'), '--results_csv', str(tmp_path / 'registry.csv')]
    proc = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', env=env, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    row = pd.read_csv(tmp_path / 'registry.csv').iloc[0]
    assert row.status == 'SMOKE_TEST'
    folder = tmp_path / 'runs' / row.run_id
    for prefix in [folder, folder / 'repeat']:
        for name in ('config.json', 'metrics.json', 'run_manifest.json', 'checkpoint.pt', 'training_history.json', 'per_user_val.csv', 'per_user_test.csv'):
            assert (prefix / name).is_file(), (prefix, name)
    cfg = json.loads((folder / 'config.json').read_text())
    assert cfg['sanity_checks']['Check 5: Reproducibility']['status'] == 'PASS'
    report = json.loads((folder / 'report_data.json').read_text())
    assert report['data_mode'] == 'synthetic_fixture'
    assert report['costs']['training_resources']['gpu_peak_allocated_bytes'] is None
    assert report['costs']['stop_reason'] == 'max_epochs'
    for name in ('Random', 'Popularity', 'SASRec_ID'):
        assert set(report['baseline_metrics'][name]) == {'validation', 'test'}
        assert set(report['baseline_metrics'][name]['validation']) >= {'NDCG@10', 'NDCG@20', 'Recall@10', 'Recall@20'}
    frame = pd.read_csv(folder / 'per_user_test.csv')
    assert {'history_len', 'target_train_base_frequency', 'analysis_length_quintile', 'target_is_oov'}.issubset(frame)
    assert len(frame) == 100
    assert len(pd.read_csv(folder / 'epochs.csv')) == 1
    assert pd.read_csv(folder / 'compute_budget.csv')['estimated_gpu_hours'].isna().all()
    assert json.loads((folder / 'run_state.json').read_text())['execution_status'] == 'COMPLETED'
    assert 'Training SASRec model' in (folder / 'run.log').read_text(encoding='utf-8')
    index = json.loads((folder / 'artifact_index.json').read_text())
    assert any(row['path'] == 'report_data.json' for row in index['artifacts'])
    assert pd.read_csv(tmp_path / 'registry.csv').iloc[0]['report_data_path'].endswith('report_data.json')


def test_e0_cli_writes_explicit_fixture_report(delta_script, monkeypatch, tmp_path):
    out = tmp_path / 'report.json'
    monkeypatch.setattr(sys, 'argv', ['run_e0_e1_evaluation.py', '--synthetic', '--num_users', '40', '--n_bootstrap', '20', '--output_json', str(out)])
    delta_script.main()
    report = json.loads(out.read_text())
    assert report['data_mode'] == 'synthetic'
    assert report['n_bootstrap'] == 20
    assert 'oof_spearman_ci' in report['sources']['category']['e1']
    with pytest.raises(FileExistsError):
        delta_script.main()


def test_shared_control_artifact_is_checksummed_and_immutable(tmp_path):
    import hashlib
    from taimma.experiments.diagnostics import save_control_artifact
    values = torch.tensor([1., 0., 1.])
    array_path, manifest_path = save_control_artifact(tmp_path, 'Beauty_2014', 'category', 'missing', .2, 42, values)
    info = json.loads(Path(manifest_path).read_text())
    assert info['sha256'] == hashlib.sha256(Path(array_path).read_bytes()).hexdigest()
    np.testing.assert_array_equal(np.load(array_path, allow_pickle=False), values.numpy())
    with pytest.raises(FileExistsError):
        save_control_artifact(tmp_path, 'Beauty_2014', 'category', 'missing', .2, 42, values)


@pytest.mark.parametrize('arguments', [[], ['--real_data', '--smoke_test'], ['--smoke_test', '--config', 'absent-config.yaml']])
def test_pilot_rejects_missing_mode_or_config(monkeypatch, arguments):
    module = script('run_pilot')
    monkeypatch.setattr(sys, 'argv', ['run_pilot.py'] + arguments)
    with pytest.raises((SystemExit, FileNotFoundError)):
        module.main()


def test_config_is_effective_and_explicit_cli_takes_precedence(tmp_path):
    import yaml
    model = tmp_path / 'model.yaml'
    model.write_text(yaml.safe_dump(dict(max_seq_len=50, embedding_dim=16, num_blocks=1, num_heads=1, dropout_rate=0., hidden_act='gelu')), encoding='utf-8')
    config = tmp_path / 'experiment.yaml'
    config.write_text(yaml.safe_dump(dict(model_config=str(model), max_epochs=3, seed=7, mask_seen_items=False,
                                        eval_batch_size=33, weight_decay=.01)), encoding='utf-8')
    env = dict(os.environ, PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    proc = subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/run_pilot.py'), '--smoke_test', '--config', str(config),
                           '--epochs', '1', '--runs_dir', str(tmp_path / 'runs'), '--results_csv', str(tmp_path / 'registry.csv')],
                          capture_output=True, text=True, encoding='utf-8', env=env, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    saved = json.loads(next((tmp_path / 'runs').glob('*/config.json')).read_text())
    assert saved['model']['embedding_dim'] == 16
    assert saved['epochs'] == 1 and saved['seed'] == 7
    assert saved['weight_decay'] == .01 and saved['eval_batch_size'] == 33
    assert saved['mask_seen_policy'] == 'none'
