"""Late-only orchestration regressions. Synthetic CPU fixtures, no research fits."""
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from tests.test_metadata_pipeline import inputs, handoff
from taimma.experiments.checkpoint2 import Checkpoint2
from taimma.experiments.late_checkpoint import LateCheckpoint, normalize_meta_signature, lock_json, paired_difference
from taimma.experiments.checkpoint2_protocol import SEEDS, SUBSETS, sha, write_json
from taimma.training.checkpoint2_trainer import ModelSpec, ModelAdapter, build_model
from taimma.training.multimodal_trainer import gather_source_sequences


def test_meta_counts_support_both_historical_split_contracts():
    known = SimpleNamespace(user_idx=0, history=[1], target=1)
    unknown = SimpleNamespace(user_idx=1, history=[1], target=0)
    outputs = []
    for samples in ([known], [known, unknown]):
        data = dict(split=SimpleNamespace(train_meta_samples=samples, oov_counts={'train_meta': 1}), signature={})
        outputs.append(normalize_meta_signature(data)['signature'])
    assert outputs[0] == outputs[1]
    assert outputs[0]['train_meta_eligible'] == outputs[0]['train_meta_oov'] == 1
    assert outputs[0]['train_meta_total'] == 2


def test_paired_ci_reports_lhs_minus_rhs_with_matching_interval_direction():
    def frames(score):
        return {s: pd.DataFrame(dict(user_idx=[0,1], target_idx=[2,3], rank=[1,2],
                    **{'ndcg@10':[score,score]})) for s in SEEDS}
    result = paired_difference(frames(.3), frames(.1), reps=20)
    assert result['delta_lhs_minus_rhs'] == pytest.approx(.2)
    assert result['ci95'] == pytest.approx([.2,.2])
    assert result['lhs_by_seed'][0]['value'] == .3
    assert result['rhs_by_seed'][0]['value'] == .1


def test_late_reference_starts_at_id_and_preserves_frozen_eval_mode():
    attrs = dict(category=torch.tensor([[1], [2], [1]]), brand=torch.tensor([1, 2, 1]), title=torch.randn(3, 4))
    masks = {s: torch.ones(3) for s in attrs}
    spec = ModelSpec('late', 42, embedding_dim=8, max_seq_len=3,
                     source_dropout=.2, source_dropout_mode='batch')
    model = build_model(spec, 3, attrs)
    adapter = ModelAdapter(model, spec, attrs, masks)
    ids = torch.tensor([[0, 1, 2]])
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(adapter.predict(ids), model.id_model.predict_all(ids), rtol=0, atol=0)
        for encoder in model.source_encoders.values():
            encoder.output_projection.weight.fill_(.1)
        off = adapter.predict(ids, weights=torch.zeros(1, 3))
        torch.testing.assert_close(off, model.id_model.predict_all(ids), rtol=0, atol=0)
    model.train()
    assert model.training and not model.id_model.training
    assert all(not p.requires_grad for p in model.id_model.parameters())
    adapter.predict(ids).sum().backward()
    assert all(p.grad is None for p in model.id_model.parameters())
    assert any(p.grad is not None for p in model.source_encoders.parameters())


def test_candidate_missingness_zeroes_only_its_residual():
    attrs = dict(category=torch.tensor([[1], [2], [1]]), brand=torch.tensor([1, 2, 1]), title=torch.randn(3, 4))
    masks = {s: torch.ones(3) for s in attrs}; masks['brand'][1] = 0
    spec = ModelSpec('late', 42, sources=('brand',), embedding_dim=8, max_seq_len=3)
    model = build_model(spec, 3, attrs).eval()
    with torch.no_grad():
        model.source_encoders['brand'].output_projection.weight.fill_(.1)
        ids = torch.tensor([[0, 1, 3]])
        joint, ident, _ = model.compute_all_scores(ids, gather_source_sequences(ids, attrs, masks), attrs, masks)
    assert torch.equal(joint[:, 1], ident[:, 1])


def test_run_only_schedules_3_id_and_21_frozen_late_jobs(tmp_path, monkeypatch):
    run = LateCheckpoint(tmp_path, {}, 'runs/checkpoint2_late_fixture', device='cpu')
    calls = []
    monkeypatch.setattr(run, 'prepare', lambda key: None)
    monkeypatch.setattr(run, 'run_job', lambda key, name, spec, **kw: calls.append((name, spec, kw)))
    monkeypatch.setattr(run, 'verify_frozen_id', lambda *a: None)
    monkeypatch.setattr(run, 'summarize', lambda key: {'status': 'fixture'})
    assert run.run_late('beauty')['status'] == 'fixture'
    assert len(calls) == 24
    assert sum(spec.kind == 'id' for _, spec, _ in calls) == 3
    for name, spec, kwargs in calls[3:]:
        assert spec.kind == 'late' and spec.freeze_id and spec.source_dropout_mode == 'batch'
        assert spec.source_dropout == .2 and spec.all_off_probability == 0
        assert kwargs['id_job'] == f'id_{spec.seed}'


@pytest.mark.parametrize('spec,kwargs', [
    (ModelSpec('early', 42), {}),
    (ModelSpec('late', 42, freeze_id=False), {'id_job': 'id_42'}),
    (ModelSpec('late', 42, source_dropout_mode='sample'), {'id_job': 'id_42'}),
    (ModelSpec('late', 42, source_dropout_mode='batch'), {'id_job': 'id_43'}),
])
def test_refuses_early_k4_and_wrong_seed_before_execution(tmp_path, spec, kwargs):
    run = LateCheckpoint(tmp_path, {}, 'runs/checkpoint2_late_fixture', device='cpu')
    with pytest.raises(ValueError):
        run.run_job('beauty', 'bad', spec, **kwargs)


def test_changed_id_dependency_is_rejected_before_reuse(tmp_path, monkeypatch):
    run = LateCheckpoint(tmp_path, {}, 'runs/checkpoint2_late_fixture', device='cpu')
    monkeypatch.setattr(run, 'assert_protocol', lambda key: None)
    monkeypatch.setattr(run, 'model_hash', lambda *a: 'new-id-hash')
    monkeypatch.setattr(Checkpoint2, 'run_job', lambda *a, **k: pytest.fail('Must reject stale ID first'))
    lock_json(run.job_path('beauty', 'late_C_42') / 'id_dependency.json',
              dict(id_job='id_42', id_checkpoint_sha256='old-id-hash'))
    with pytest.raises(ValueError, match='binding changed'):
        run.run_job('beauty', 'late_C_42', ModelSpec('late', 42, sources=('category',), source_dropout_mode='batch'), id_job='id_42')


def fixture_prepare(run, tmp_path, monkeypatch):
    import taimma.experiments.late_checkpoint as module
    monkeypatch.setattr(run, 'guard', lambda **k: None)
    monkeypatch.setattr(run, 'source_signature', lambda: {'src/file.py': 'fixed'})
    monkeypatch.setattr(module.subprocess, 'check_output', lambda *a, **k: 'src/file.py\n')
    monkeypatch.setattr(module, 'git_state', lambda *a, **k: dict(commit='one', dirty=False))
    monkeypatch.setattr(module, 'check_training_environment', lambda *a, **k: dict(gpu='fixture'))


def test_prepare_never_builds_training_controls_and_allows_results_only_commit(tmp_path, handoff, monkeypatch):
    import taimma.experiments.late_checkpoint as module
    import taimma.experiments.checkpoint2 as shared
    meta, raw = handoff
    run = LateCheckpoint(tmp_path, dict(raw_root=str(raw), metadata_root=str(meta.parent), environment={}),
                         'runs/checkpoint2_late_fixture', device='cpu')
    fixture_prepare(run, tmp_path, monkeypatch)
    monkeypatch.setattr(shared, 'build_training_contrasts_for_dataset', lambda *a, **k: pytest.fail('Late needs no contrasts'))
    run.prepare('beauty')
    assert not (run.output / 'beauty/training_contrasts').exists()
    monkeypatch.setattr(module, 'git_state', lambda *a, **k: dict(commit='results-only-commit', dirty=False))
    run.prepare('beauty')
    assert json.loads((run.output / 'beauty/origin_git.json').read_text())['commit'] == 'one'
    monkeypatch.setattr(run, 'source_signature', lambda: {'src/file.py': 'changed'})
    with pytest.raises(ValueError, match='binding changed'): run.prepare('beauty')


def test_group_csv_tampering_is_rejected(tmp_path, handoff, monkeypatch):
    meta, raw = handoff
    run = LateCheckpoint(tmp_path, dict(raw_root=str(raw), metadata_root=str(meta.parent), environment={}),
                         'runs/checkpoint2_late_fixture', device='cpu')
    fixture_prepare(run, tmp_path, monkeypatch)
    run.prepare('beauty')
    p = run.output / 'beauty/groups.csv'
    p.write_text(p.read_text() + '\n')
    with pytest.raises(ValueError, match='group assignments changed'): run.prepare('beauty')


def test_cli_plan_requires_no_training_dependencies():
    path = Path(__file__).resolve().parents[1] / 'scripts/run_late_checkpoint.py'
    result = subprocess.run([sys.executable, str(path)], capture_output=True, text=True, check=True)
    plan = json.loads(result.stdout)
    assert plan['execution'] == 'DISABLED' and plan['per_dataset']['total_jobs'] == 24


def test_frozen_id_verifier_rejects_modified_id_and_wrong_epoch(tmp_path, monkeypatch):
    run = LateCheckpoint(tmp_path, {}, 'runs/checkpoint2_late_fixture', device='cpu')
    base = run.job_path('beauty', 'id_42'); late = run.job_path('beauty', 'late_C_42')
    for p in (base, late): (p / 'attempt_001').mkdir(parents=True)
    torch.save(dict(model_state_dict={'w': torch.tensor([1.])}), base / 'attempt_001/checkpoint.pt')
    state = dict(epoch=2, model_state_dict={'id_model.w': torch.tensor([1.])})
    torch.save(state, late / 'attempt_001/checkpoint.pt')
    write_json(late / 'attempt_001/history.json', dict(selection_scope='trained_epochs_only',
        val_ndcg_10=[.1, .2], best_epoch=2, best_val_ndcg=.2, initial_validation_metrics={'NDCG@10': .3}))
    entries = {'id_42': dict(checkpoint='attempt_001/checkpoint.pt', val_ndcg_10=.3),
               'late_C_42': dict(checkpoint='attempt_001/checkpoint.pt', spec={'seed':42}, best_epoch=2, val_ndcg_10=.2)}
    monkeypatch.setattr(run, 'completed', lambda key, name: entries[name])
    assert run.verify_frozen_id('beauty', 'late_C_42')['status'] == 'VERIFIED'
    state['model_state_dict']['id_model.w'] = torch.tensor([2.])
    torch.save(state, late / 'attempt_001/checkpoint.pt')
    with pytest.raises(ValueError, match='Frozen ID changed'): run.verify_frozen_id('beauty', 'late_C_42')
    state.update(epoch=1, model_state_dict={'id_model.w': torch.tensor([1.])})
    torch.save(state, late / 'attempt_001/checkpoint.pt')
    with pytest.raises(ValueError, match='trained-epoch selection'): run.verify_frozen_id('beauty', 'late_C_42')


def test_summary_uses_new_grid_only_without_initializing_models(tmp_path, handoff, monkeypatch):
    import taimma.experiments.checkpoint2 as shared
    meta, raw = handoff
    config = dict(raw_root=str(raw), metadata_root=str(meta.parent), environment={}, bootstrap_reps=30)
    run = LateCheckpoint(tmp_path, config, 'runs/checkpoint2_late_fixture', device='cpu')
    fixture_prepare(run, tmp_path, monkeypatch)
    data = run.prepare('beauty')
    monkeypatch.setattr(run, 'legacy_grid', lambda *a: pytest.fail('Never consume legacy ranks'))
    monkeypatch.setattr(shared, 'build_model', lambda *a, **k: pytest.fail('Summary must not initialize a model'))
    from taimma.experiments.checkpoint2 import subset_name
    for seed in SEEDS:
        for sources in SUBSETS:
            label = subset_name(sources)
            name = f'late_{label}_{seed}' if sources else f'id_{seed}'
            root = run.job_path('beauty', name)
            attempt = root / 'attempt_001'; attempt.mkdir(parents=True)
            (attempt / 'checkpoint.pt').write_bytes(b'fixture, not a model')
            for part in ('val', 'test'):
                samples = getattr(data['split'], part + '_samples')
                pd.DataFrame([dict(user_idx=s.user_idx, target_idx=s.target, rank=1 if s.target>0 else 10000,
                    **{'ndcg@10':float(s.target>0), 'recall@10':float(s.target>0)}) for s in samples]).to_csv(
                        attempt / f'per_user_{part}.csv', index=False)
            rec = dict(checkpoint='attempt_001/checkpoint.pt',
                artifacts={p.relative_to(root).as_posix():sha(p) for p in attempt.iterdir()})
            write_json(root / 'completed.json', rec)
            if sources: write_json(root / 'late_verification.json', dict(late_checkpoint_sha256=sha(attempt/'checkpoint.pt')))
    result = run.summarize('beauty')
    assert result['status'] == 'COMPLETE' and len(result['results']) == 8
    assert result['groups_use'] == 'this_output_only_no_legacy'
    stats = pd.read_csv(run.output / 'beauty/late_summary.csv')
    assert (stats['std'] == 0).all()
    assert len(json.loads((run.output / 'beauty/late_group_analysis.json').read_text())) == 3
