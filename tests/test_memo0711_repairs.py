"""Regression evidence for memo repair; fake adapters never fit research models."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from taimma.experiments.memo0711 import Memo0711, SeparateID


class FakeAdapter:
    def __init__(self):
        self.model = torch.nn.Dropout(.9)
        self.device = torch.device('cpu')
        self.attrs = {}
        self.masks = {'category': torch.ones(2)}

    def predict(self, ids, weights=None):
        return self.masks['category'].expand(len(ids), -1)


def test_wrapper_mask_assignment_reaches_scoring_adapter():
    inner = FakeAdapter()
    wrapper = SeparateID(inner, FakeAdapter())
    wrapper.masks = {'category': torch.zeros(2)}
    assert wrapper.masks is inner.masks
    assert torch.equal(wrapper.predict(torch.ones(2, 1)), torch.zeros(2, 2))


def test_raw_add_one_uses_eval_and_cache_does_not_reload_model(tmp_path, monkeypatch):
    from taimma.experiments import memo0711 as module
    run = Memo0711(tmp_path, {'supervisor_memo': '2026-11-07', 'batch_size': 2},
                   'runs/memo0711_fixture', device='cpu')
    run.dataset = lambda key: {'signature': {'fixture': 1}, 'split': SimpleNamespace(train_meta_samples=[1, 2])}
    run.model_hash = lambda *args: 'fixed-model'
    adapter = FakeAdapter()
    adapter.weights = lambda n, enabled: torch.full((n, 3), float(bool(enabled)))
    calls = []

    def predict(ids, weights):
        assert not adapter.model.training, 'no_grad alone does not disable S1 dropout'
        calls.append(1)
        return torch.tensor([[1., 0.], [0., 1.]]) * (1. + weights[:, :1])

    adapter.predict = predict
    run.load_job = lambda *args: adapter
    monkeypatch.setattr(module, 'batches', lambda *args: [dict(
        item_seqs=torch.ones(2, 1, dtype=torch.long), targets=torch.tensor([1, 2]), user_indices=[0, 1])])
    first = run.raw_add_one('beauty', 'early_sd_42', 'CE')
    assert len(first) == 6 and np.isfinite(first.delta).all()
    count = len(calls)
    run.load_job = lambda *args: pytest.fail('Verified utility cache must not load a model')
    pd.testing.assert_frame_equal(first, run.raw_add_one('beauty', 'early_sd_42', 'CE'), check_exact=False)
    assert len(calls) == count


def test_fairness_adoption_uses_meta_and_base_targets(tmp_path):
    from taimma.experiments.memo0711_adoption import training_samples
    base, meta = [SimpleNamespace(target=1)], [SimpleNamespace(target=2)]
    run = SimpleNamespace(dataset=lambda key: {'split': SimpleNamespace(train_base_samples=base, train_meta_samples=meta)})
    assert training_samples(run, 'beauty', 'id_plus_meta_42') == base + meta
    assert training_samples(run, 'beauty', 'early_plus_meta_46') == base + meta
    assert training_samples(run, 'beauty', 'id_42') == base


def test_framing_uses_teacher_nested_family_not_fairness_controls():
    from taimma.experiments.memo0711 import framing_from_ranks
    def frame(value):
        return pd.DataFrame(dict(user_idx=np.arange(30), target_idx=np.arange(30)+1,
                                 rank=np.ones(30), **{'ndcg@10': np.full(30, value)}))
    nested = ['all_fixed', 'global_subset', 'group_subset', 'global_gate', 'unsupervised', 'softmax_null']
    frames = {name: {s: frame(.2) for s in range(42, 47)} for name in nested}
    frames['uasf'] = {s: frame(.3) for s in range(42, 47)}
    frames['id_plus_meta'] = {s: frame(.4) for s in range(42, 47)}
    result = framing_from_ranks(frames, 30)
    assert result['framing'] == 'F1'
    assert result['best_baseline'] in nested


def test_cli_preserves_yaml_roots_until_explicit_override(tmp_path):
    import importlib.util
    path = Path(__file__).resolve().parents[1] / 'scripts/run_memo0711.py'
    spec = importlib.util.spec_from_file_location('memo_cli_fixture', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = {'input_roots': {'early': str(tmp_path/'history/early'),
                              'late': str(tmp_path/'other-checkout/late'),
                              'diagnostics': str(tmp_path/'history/diag')}}
    args = SimpleNamespace(early_root=None, late_root=None, diagnostics_root=None)
    original = module.resolve_input_roots(tmp_path, config, args)
    assert original['late'] == str((tmp_path/'other-checkout/late').resolve())
    args.late_root = 'runs/explicit_late'
    assert module.resolve_input_roots(tmp_path, config, args)['late'] == str((tmp_path/'runs/explicit_late').resolve())


def test_report_marks_absent_measurements_instead_of_zero(tmp_path):
    from taimma.experiments.memo0711_report import write_memo_report
    output = tmp_path/'runtime'
    output.mkdir()
    destination = tmp_path/'tables'
    write_memo_report(output, {'datasets': ['beauty', 'sports', 'toys']}, destination)
    table = pd.read_csv(destination/'B_three_datasets.csv')
    assert len(table) > 0
    metric_cols = [c for c in table if 'mean' in c.lower() or 'std' in c.lower()]
    assert metric_cols, 'Missing measurements need explicit metric columns'
    assert table[metric_cols].isna().all().all()
    assert 'MISSING' in table.to_string()


def adoption_fixture(tmp_path, name='id_plus_meta_42'):
    import json
    from dataclasses import asdict
    from taimma.data.split import SequenceSample
    from taimma.training.checkpoint2_trainer import ModelSpec
    from taimma.experiments.study_completion import COMPATIBLE_MODEL_FILES
    from taimma.experiments.checkpoint2_protocol import sha, digest, context_signature, write_json
    repo = tmp_path/'repo'
    signatures = {}
    for rel in COMPATIBLE_MODEL_FILES:
        p = repo/rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b'# immutable model fixture\n')
        signatures[rel] = sha(p)
    cfg = dict(supervisor_memo='2026-11-07', epochs=1, patience=1, batch_size=2)
    run = Memo0711(repo, cfg, 'runs/new', device='cpu')
    base, meta = [SequenceSample(0,[1],2,1)], [SequenceSample(0,[1,2],3,2)]
    signature = {k: 'fixture' for k in ['raw_reviews_sha256','metadata_files','catalog_sha256',
                                      'train_base_sha256','train_meta_sha256','val_sha256','test_sha256']}
    signature.update(train_meta_eligible=1, train_meta_oov=1, train_meta_total=2)
    run.dataset = lambda key: dict(signature=signature, split=SimpleNamespace(train_base_samples=base, train_meta_samples=meta))
    old = tmp_path/'old'
    source = old/'beauty/jobs'/name
    attempt = source/'attempt_001'
    attempt.mkdir(parents=True)
    spec = asdict(ModelSpec('id',42,sources=()))
    fingerprint = digest(dict(spec=spec,train=context_signature(base+meta),config=digest(cfg),data=signature,control=None,id_job=None))
    write_json(old/'beauty/protocol.lock.json',dict(config=cfg,data=signature))
    write_json(attempt/'config.json',dict(spec=spec,data_signature=signature,fingerprint=fingerprint))
    write_json(attempt/'source_snapshot.json',signatures)
    write_json(attempt/'run_manifest.json',dict(python_version='fixture',torch_version='fixture',library_versions={}))
    write_json(attempt/'history.json',{})
    (attempt/'per_user_val.csv').write_text('user_idx,target_idx,rank,ndcg@10\n0,3,1,1.0\n')
    (attempt/'checkpoint.pt').write_bytes(b'unchanged checkpoint fixture')
    artifacts={p.relative_to(source).as_posix():sha(p) for p in attempt.iterdir()}
    write_json(source/'completed.json',dict(status='COMPLETED',spec=spec,control=None,
               checkpoint='attempt_001/checkpoint.pt',fingerprint=fingerprint,artifacts=artifacts))
    return run,old,source,fingerprint


def test_adopted_fairness_checkpoint_preserves_source_and_fingerprint(tmp_path):
    from taimma.experiments.memo0711_adoption import adopt_s1
    from taimma.experiments.checkpoint2_protocol import sha
    run,old,source,fingerprint = adoption_fixture(tmp_path)
    before={str(p.relative_to(source)):sha(p) for p in source.rglob('*') if p.is_file()}
    adopt_s1(run,'beauty','id_plus_meta_42',old)
    assert run.completed('beauty','id_plus_meta_42')['fingerprint'] == fingerprint
    assert before == {str(p.relative_to(source)):sha(p) for p in source.rglob('*') if p.is_file()}
    adopt_s1(run,'beauty','id_plus_meta_42',old)
    assert run.completed('beauty','id_plus_meta_42')['fingerprint'] == fingerprint


def test_adoption_refuses_missing_weights_without_training(tmp_path):
    from taimma.experiments.memo0711_adoption import adopt_s1
    run,old,source,_ = adoption_fixture(tmp_path)
    (source/'attempt_001/checkpoint.pt').unlink()
    with pytest.raises(FileNotFoundError,match='Missing original artifact'):
        adopt_s1(run,'beauty','id_plus_meta_42',old)
    assert not run.job_path('beauty','id_plus_meta_42').exists()


def test_migrated_run_refuses_implicit_s1_retraining(tmp_path):
    from taimma.training.checkpoint2_trainer import ModelSpec
    run,old,source,_ = adoption_fixture(tmp_path)
    run.reuse_memo_source = old
    run.assert_protocol = lambda key: None
    run.guard = lambda **kwargs: pytest.fail('No real training guard should be entered')
    with pytest.raises(FileNotFoundError,match='refusing implicit retraining'):
        run.run_job('beauty','id_43',ModelSpec('id',43,sources=()))


def test_imported_legacy_config_uses_normalized_memo_meta_binding(tmp_path):
    import json
    from taimma.experiments.memo0711_adoption import adopt_s1
    from taimma.experiments.checkpoint2_protocol import sha,write_json
    run,old,source,_ = adoption_fixture(tmp_path)
    p = source/'attempt_001/config.json'
    cfg = json.loads(p.read_text())
    cfg['data_signature'].update(train_meta_sha256='old_hash_including_oov',
                                 train_meta_eligible=2, train_meta_total=3)
    write_json(p,cfg)
    rec = json.loads((source/'completed.json').read_text())
    rec['artifacts']['attempt_001/config.json'] = sha(p)
    write_json(source/'completed.json',rec)
    write_json(source/'import_provenance.json',dict(source='original_historical_job',scientific_inputs_checked=True))
    before = {str(p.relative_to(source)):sha(p) for p in source.rglob('*') if p.is_file()}
    adopt_s1(run,'beauty','id_plus_meta_42',old)
    assert run.completed('beauty','id_plus_meta_42')
    assert before == {str(p.relative_to(source)):sha(p) for p in source.rglob('*') if p.is_file()}


def test_adoption_manifest_retains_previously_adopted_cohorts(tmp_path,monkeypatch):
    from taimma.experiments import memo0711_adoption as module
    from taimma.experiments.checkpoint2_protocol import write_json
    old = tmp_path/'old'
    for key in ['beauty','sports']:
        write_json(old/key/'jobs/id_42/completed.json',{})
        write_json(old/key/'protocol.lock.json',dict(config={'old':1}))
    run=SimpleNamespace(output=tmp_path/'new', protocol_hash='different-config', prepare=lambda key:None)
    monkeypatch.setattr(module,'adopt_s1',lambda *args:None)
    module.adopt_memo(run,old,['beauty'])
    result=module.adopt_memo(run,old,['sports'])
    assert set(result['jobs']) == {'beauty','sports'}
    assert result['jobs']['beauty'] == ['id_42']


def test_report_removes_owned_stale_tables_but_preserves_user_notes(tmp_path):
    from taimma.experiments.memo0711_report import write_memo_report
    from taimma.experiments.checkpoint2_protocol import write_json
    output=tmp_path/'runtime'; output.mkdir()
    destination=tmp_path/'tables'; destination.mkdir()
    (destination/'obsolete.csv').write_text('obsolete')
    (destination/'personal.md').write_text('keep me')
    write_json(destination/'package_manifest.json',{'files':{'obsolete.csv':'oldhash'}})
    write_memo_report(output,{'datasets':['beauty']},destination)
    assert not (destination/'obsolete.csv').exists()
    write_memo_report(output,{'datasets':['beauty']},destination)
    assert (destination/'personal.md').read_text() == 'keep me'


def test_report_refuses_tampered_gate_rank_csv(tmp_path):
    from taimma.experiments.memo0711_report import write_memo_report
    from taimma.experiments.checkpoint2_protocol import write_json
    output=tmp_path/'runtime'
    path=output/'beauty/gates/uasf_42/per_user_val.csv'
    path.parent.mkdir(parents=True)
    path.write_text('user_idx,target_idx,rank,ndcg@10\n0,1,1,1.0\n')
    write_json(path.with_suffix('.binding.json'),{'sha256':'not-the-content-hash'})
    with pytest.raises(ValueError,match='Changed|changed|hash'):
        write_memo_report(output,{'datasets':['beauty']},tmp_path/'tables')


def test_transfer_no_go_is_explicit_hash_bound_and_never_trains_gate(tmp_path,monkeypatch):
    import datetime
    import json
    from taimma.experiments.memo0711 import SOURCES,SEEDS
    from taimma.experiments.checkpoint2_protocol import write_json,sha
    class TransferDate(datetime.datetime):
        @classmethod
        def now(cls,tz=None): return cls(2026,11,12,tzinfo=tz)
    monkeypatch.setattr(datetime,'datetime',TransferDate)
    run=Memo0711(tmp_path,dict(supervisor_memo='2026-11-07',datasets=['sports'],missing_rates=[0.]),
                 'runs/memo0711_fixture',device='cpu')
    write_json(run.output/'beauty/framing_decision.json',dict(framing='F1'))
    run.foundation=lambda *args:None
    status=dict(s3_eligible=False,decision='NO-GO')
    def utility(*args,**kwargs):
        write_json(run.output/'sports/selected_s2/status.json',status)
        return status
    run.utility=utility
    run.gate_run=lambda *args:pytest.fail('NO-GO cannot train or evaluate transfer gates')
    run.uasf('sports',transfer=True)
    outcome=json.loads((run.output/'sports/framing_decision.json').read_text())
    assert outcome['framing']=='TRANSFER_NO_ELIGIBLE'
    assert outcome['s2_status_sha256']==sha(run.output/'sports/selected_s2/status.json')
    run.completed=lambda *args:{'fixture':'present'}
    run.dataset=lambda key:{'signature':{'fixture':1}}
    rows=[dict(seed=s,model=m,source=src,missing_rate=0.,ndcg_10=.1)
          for s in SEEDS for m in ['early_sd','early_CBT'] for src in SOURCES]
    p=run.output/'sports/robustness.csv'
    pd.DataFrame(rows).to_csv(p,index=False)
    write_json(p.with_name('robustness_binding.json'),dict(evaluation_version=2,sha256=sha(p),data={'fixture':1}))
    write_json(run.output/'sports/group_analysis.json',{})
    for rel in ['beauty/title_only_comparison.json','beauty/late_rank_diagnostics.csv',
                'beauty/selected_s2/features_train_meta.csv','beauty/selected_s2/validation_same_model.csv']:
        path=run.output/rel;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('fixture')
    assert run.validate_ready()=={'sports':'PASS'}
    write_json(run.output/'sports/selected_s2/status.json',dict(s3_eligible=False,decision='different-reason'))
    with pytest.raises(RuntimeError,match='bound explicit outcome'):
        run.validate_ready()
