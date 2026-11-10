"""CPU-only audit/report regressions; no checkpoint loads or research inference."""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from taimma.experiments.memo0711_report import audit_runtime,validate_rank_measurements,write_memo_report


def cli_module():
    spec=importlib.util.spec_from_file_location('memo_cli_v3',ROOT/'scripts/run_memo0711.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_reuse_memo_roots_preserve_old_protocol_until_explicit_override(tmp_path):
    source=tmp_path/'runs/old/beauty';source.mkdir(parents=True)
    roots={key:str(tmp_path/key) for key in ['early','late','diagnostics']}
    (source/'protocol.lock.json').write_text(json.dumps({'config':{'input_roots':roots}}))
    args=SimpleNamespace(reuse_memo='runs/old',early_root=None,late_root=None,diagnostics_root=None)
    result=cli_module().resolve_input_roots(tmp_path,{'input_roots':{'late':'unrelated'}},args)
    assert result=={k:str(Path(v).resolve()) for k,v in roots.items()}
    args.late_root='explicit'
    assert result['late']!=cli_module().resolve_input_roots(tmp_path,{},args)['late']


def test_audit_existing_incomplete_three_cohorts_is_readonly_and_no_crash(tmp_path):
    output=tmp_path/'memo'
    for key in ['beauty','sports','toys']:
        folder=output/key;folder.mkdir(parents=True)
        (folder/'user_note.txt').write_bytes(b'immutable run note')
    before={str(p.relative_to(output)):p.read_bytes() for p in output.rglob('*') if p.is_file()}
    result=audit_runtime(output,{'datasets':['beauty','sports','toys']})
    after={str(p.relative_to(output)):p.read_bytes() for p in output.rglob('*') if p.is_file()}
    assert result['status']=='INCOMPLETE' and result['issues']>0
    assert result['training'] is False and result['inference'] is False
    assert before==after
    assert any(row['status']=='MISSING_JOB' for row in result['checks'])


def test_cli_audit_returns_before_model_import_or_registry_write(tmp_path,monkeypatch,capsys):
    module=cli_module();cfg=tmp_path/'config.yaml'
    cfg.write_text('datasets: [beauty, sports, toys]\n')
    output=tmp_path/'absent'
    monkeypatch.setattr(sys,'argv',['run_memo0711.py','--stage','audit','--config',str(cfg),'--output',str(output)])
    assert module.main()==2
    result=json.loads(capsys.readouterr().out)
    assert result['status']=='INCOMPLETE' and not output.exists()


def test_rank_recomputation_rejects_oov_target_with_nonzero_metric():
    frame=pd.DataFrame({'user_idx':[0,1],'target_idx':[0,2],'rank':[1,2],'ndcg@10':[0.,1/np.log2(3)]})
    validate_rank_measurements(frame,'fixture')
    frame.loc[0,'ndcg@10']=1.
    with pytest.raises(ValueError,match='OOV'):validate_rank_measurements(frame,'fixture')


def test_secondary_comparison_and_historical_f2_are_separate_outputs(tmp_path,monkeypatch):
    from taimma.experiments import memo0711_report as module
    monkeypatch.setattr(module,'measured_figures',lambda *args:([],{'status':'FIXTURE_NO_PLOTS'}))
    output=tmp_path/'repo/runs/memo';root=output/'beauty';root.mkdir(parents=True)
    family=['all_fixed','global_subset','group_subset','global_gate','unsupervised','softmax_null','independent_ID','early_no_sd','id_plus_meta','early_plus_meta']
    decision={'framing':'F2','framing_family':family,'best_baseline':'early_plus_meta','framing_policy':'memo0711_v1_locked_ten_v3',
              'comparison':{'mean_diff':-.01,'ci_lower':-.02,'ci_upper':-.001},
              'secondary_six_selectors':{'framing':'F1','family':family[:6],'best_baseline':'all_fixed','comparison':{'mean_diff':.01}}}
    (root/'framing_decision.json').write_text(json.dumps(decision))
    historic=output/'historic_decisions/beauty';historic.mkdir(parents=True)
    (historic/'framing_decision.json').write_text(json.dumps({'framing':'F2','best_baseline':'early_plus_meta'}))
    dest=tmp_path/'tables'
    write_memo_report(output,{'datasets':['beauty']},dest)
    table=pd.read_csv(dest/'framing_official_and_secondary.csv')
    assert table.loc[table.scope=='official','framing'].iloc[0]=='F2'
    assert table.loc[table.scope=='secondary_six_selectors','framing'].iloc[0]=='F1'
    assert not table.loc[table.scope=='secondary_six_selectors','used_for_primary_decision'].iloc[0]
    assert pd.read_csv(dest/'historical_framing.csv').recorded_framing.tolist()==['F2']
    assert (dest/'framing_official_and_secondary.tex').is_file()


def test_unverified_v2_robustness_is_not_exported_as_v3(tmp_path,monkeypatch):
    from taimma.experiments import memo0711_report as module
    monkeypatch.setattr(module,'measured_figures',lambda *args:([],{'status':'FIXTURE_NO_PLOTS'}))
    output=tmp_path/'memo';root=output/'beauty';root.mkdir(parents=True)
    (root/'robustness.csv').write_text('seed,model,source,missing_rate,ndcg_10\n42,early_sd,title,0.8,0.1\n')
    (root/'robustness_binding.json').write_text(json.dumps({'evaluation_version':2}))
    dest=tmp_path/'tables';write_memo_report(output,{'datasets':['beauty']},dest)
    assert not (dest/'beauty_robustness.csv').exists()
    gaps=pd.read_csv(dest/'completeness_gaps.csv')
    assert 'UNVERIFIED_ROBUSTNESS_MISSING_V3_EVIDENCE' in set(gaps.status)


def test_audit_bound_f2_no_eligible_s2_does_not_require_framing_metrics(tmp_path):
    from taimma.experiments.memo0711_report import file_sha
    root=tmp_path/'memo/beauty';(root/'selected_s2').mkdir(parents=True)
    status=root/'selected_s2/status.json'
    status.write_text(json.dumps({'s3_eligible':False,'decision':'NO-GO'}))
    (root/'framing_decision.json').write_text(json.dumps({'framing':'F2','reason':'NO-GO','s2_status_sha256':file_sha(status),'test_used':False}))
    report=audit_runtime(root.parent,{'datasets':['beauty']})
    check=next(row for row in report['checks'] if row['item']=='framing_family')
    assert check['status']=='NOT_APPLICABLE_NO_ELIGIBLE_S2'


def test_audit_requires_every_declared_gate_when_s2_is_eligible(tmp_path):
    root=tmp_path/'memo/beauty';(root/'selected_s2').mkdir(parents=True)
    (root/'selected_s2/status.json').write_text(json.dumps({'s3_eligible':True,'decision':'GO'}))
    result=audit_runtime(root.parent,{'datasets':['beauty']})
    missing=[r for r in result['checks'] if r['status']=='MISSING_OR_INVALID_GATE']
    assert len(missing)==45
    assert result['status']=='INCOMPLETE'


def test_e1_evidence_cli_uses_cpu_export_before_model_runner(tmp_path,monkeypatch,capsys):
    module=cli_module();cfg=tmp_path/'config.yaml';cfg.write_text('datasets: [beauty]\n')
    calls=[]
    fake=SimpleNamespace(export_e1_evidence=lambda repo,output,datasets: calls.append((repo,output,datasets)) or {'status':'CPU_FIXTURE'})
    monkeypatch.setitem(sys.modules,'taimma.experiments.memo0711_e1_evidence',fake)
    monkeypatch.setattr(sys,'argv',['run_memo0711.py','--stage','e1-evidence','--config',str(cfg),'--output',str(tmp_path/'runtime'),'--datasets','beauty'])
    assert module.main()==0
    assert len(calls)==1 and calls[0][2]==['beauty']
    assert json.loads(capsys.readouterr().out)['status']=='CPU_FIXTURE'
    assert not (tmp_path/'runtime').exists()
