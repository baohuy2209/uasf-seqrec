"""CPU fixtures for the inference-only final release. No research inference."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pandas as pd
import pytest

REPO=Path(__file__).resolve().parents[1]
if not (REPO/'src').exists(): REPO=Path(r'D:\Github\uasf-seqrec')
sys.path.insert(0,str(REPO/'src'))
try:
    from taimma.experiments import memo0711_final as module
except ImportError:
    spec=importlib.util.spec_from_file_location('taimma.experiments.memo0711_final',Path(__file__).with_name('memo0711_final.py'))
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)


def frame(values: list[float], targets: list[int] | None=None) -> pd.DataFrame:
    return pd.DataFrame(dict(user_idx=np.arange(len(values)),target_idx=targets or list(range(1,len(values)+1)),
        rank=np.ones(len(values)),**{name:values for name in ['ndcg@10','ndcg@20','recall@10','recall@20']}))


def test_paired_ci_retains_matched_seed_users() -> None:
    left={s:frame([.2,.4]) for s in (42,43,44)};right={s:frame([.1,.3]) for s in (42,43,44)}
    out=module.paired_difference(left,right,30)
    assert np.isclose(out['difference'],.1) and np.isclose(out['ci_lower'],.1)
    right[43]=frame([.1,.3],[9,2])
    with pytest.raises(ValueError,match='pairing'):module.paired_difference(left,right)


def test_runtime_allowlist_does_not_hide_source() -> None:
    assert module.allowed_runtime_path('finalization/release.lock')
    assert module.allowed_runtime_path('beauty/gates/uasf_42/per_user_test.csv')
    assert not module.allowed_runtime_path('finalization/model.py')
    assert not module.allowed_runtime_path('beauty/gates/uasf_42/completed.json')
    assert not module.allowed_runtime_path('beauty/evaluations/early_sd_42/CBT_val.csv')


def test_git_check_allows_only_untracked_generated_files(tmp_path: Path,monkeypatch: pytest.MonkeyPatch) -> None:
    state=[b'?? runs/v3/finalization/release.lock\0']
    def output(cmd: list[str],**kwargs):
        if cmd[1]=='status':return state[0]
        return 'reviewed-commit\n'
    monkeypatch.setattr(module.subprocess,'check_output',output)
    assert module.reviewed_git(tmp_path,tmp_path/'runs/v3','reviewed-tag')['dirty_code'] is False
    for raw in [b' M src/model.py\0',b'?? runs/v3/finalization/model.py\0',b'?? unrelated.txt\0']:
        state[0]=raw
        with pytest.raises(RuntimeError,match='unexpected'):module.reviewed_git(tmp_path,tmp_path/'runs/v3')


def test_manifest_checkpoint_cannot_escape_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError,match='escapes'):module.inside(tmp_path,'../other.pt')
    result=module.checkpoint_inventory(tmp_path)
    assert result['status']=='INCOMPLETE' and len(result['failures'])==45


def test_early_freeze_requires_reason_without_mutating_config(tmp_path: Path,monkeypatch: pytest.MonkeyPatch) -> None:
    root=tmp_path/'runs/v3';(root/'beauty').mkdir(parents=True)
    module.write(root/'beauty/framing_decision.json',dict(framing='F2',framing_family=list(module.BASELINES)))
    module.write(root/'beauty/selected_s2/status.json',{'label_choice':'CE'})
    run=SimpleNamespace(output=root,repo=tmp_path,config={'test_release_not_before':'2999-01-01','datasets':['beauty']},
        validate_ready=lambda:None,source_signature=lambda:{'code':'sha'})
    before=dict(run.config)
    monkeypatch.setattr(module,'reviewed_git',lambda *a,**k:{'commit':'frozen'})
    monkeypatch.setattr(module,'checkpoint_inventory',lambda *a:{'status':'PASS'})
    monkeypatch.setattr(module,'source_compatibility',lambda *a:{'status':'PASS'})
    with pytest.raises(RuntimeError,match='early-release-reason'):module.freeze(run,'reviewed-tag')
    lock=module.freeze(run,'reviewed-tag','User explicitly requested an early release')
    assert lock['framing']=='F2' and lock['early_release_authorization'] and run.config==before
    assert lock['methods']==list(module.METHODS)


def test_freeze_detects_changed_file(tmp_path: Path,monkeypatch: pytest.MonkeyPatch) -> None:
    root=tmp_path/'runs/v3';root.mkdir(parents=True);p=root/'fixed.csv';p.write_text('original')
    run=SimpleNamespace(output=root,repo=tmp_path,source_signature=lambda:{'code':'x'})
    module.write(root/'freeze.json',dict(finalization_version=1,source_sha256={'code':'x'},git={'commit':'c'},files={'fixed.csv':module.sha(p)}))
    monkeypatch.setattr(module,'reviewed_git',lambda *a,**k:{'commit':'c'})
    module.verify_freeze(run);p.write_text('changed')
    with pytest.raises(ValueError,match='Frozen artifact'):module.verify_freeze(run)


def test_missing_freeze_refuses_before_checkpoint_load(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError,match='--stage freeze'):
        module.verify_freeze(SimpleNamespace(output=tmp_path))


def test_release_all_methods_once_and_resume_without_inference(tmp_path: Path,monkeypatch: pytest.MonkeyPatch) -> None:
    root=tmp_path/'runs/v3';root.mkdir(parents=True);(root/'freeze.json').write_text('{}')
    module.write(root/'beauty/selected_s2/status.json',{'label_choice':'CE'})
    module.write(root/'beauty/selected_s2/e0_e1_CE.json',{})
    monkeypatch.setattr(module,'verify_freeze',lambda *a:{'framing':'F2'})
    monkeypatch.setattr(module,'validation_frames',lambda *a:{m:frame([1.,1.]) for m in module.METHODS})
    from taimma.experiments import checkpoint2_protocol
    monkeypatch.setattr(checkpoint2_protocol,'aligned_rank_matrix',lambda *a:None)
    calls=[]
    monkeypatch.setattr(module,'frozen_gate_test',lambda run,seed,mode,report:(calls.append((seed,mode)) or frame([1.,1.])))
    run=SimpleNamespace(output=root,config={'bootstrap_reps':20},execute_inference=True,
        subset_frames=lambda *a:{m:frame([1.,1.]) for m in ('all_fixed','global_subset','group_subset')},
        inference_ranks=lambda *a:frame([1.,1.]),
        gate_run=lambda key,seed,mode,*a:(calls.append((seed,mode)) or frame([1.,1.])),
        dataset=lambda *a:{'split':SimpleNamespace(val_samples=[],test_samples=[])})
    result=module.release(run)
    assert result['status']=='COMPLETE' and len(calls)==20 and len(pd.read_csv(root/'finalization/table5_validation_test.csv'))==11
    run.gate_run=lambda *a:pytest.fail('Resume must use the completed release')
    assert module.release(run)==result
    (root/'finalization/test_paired_ci.csv').write_text('tampered')
    with pytest.raises(ValueError,match='Released output'):module.release(run)


def test_cli_plan_is_nonexecuting_and_has_no_training_flag() -> None:
    import subprocess
    script=Path(__file__).with_name('finalize_memo0711.py')
    if not script.exists():script=REPO/'scripts/finalize_memo0711.py'
    result=subprocess.run([sys.executable,str(script)],capture_output=True,text=True)
    assert result.returncode==0 and '"training": false' in result.stdout
    result=subprocess.run([sys.executable,str(script),'--execute-training'],capture_output=True,text=True)
    assert result.returncode!=0 and 'unrecognized arguments' in result.stderr
