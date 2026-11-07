"""Synthetic regression tests only: no original experiment is trained/scored."""
from pathlib import Path
from types import SimpleNamespace
import hashlib,json,sys,subprocess
import numpy as np
import pandas as pd
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from taimma.experiments.study_completion import StudyCompletion,choose_backbone,paired_summary,COMPATIBLE_MODEL_FILES
from taimma.experiments.study_controls import build_controls
from taimma.experiments.checkpoint2_protocol import sha,write_json
from taimma.experiments.study_gates import seeds_for,release_test,SoftmaxNullGate


def test_backbone_enforces_three_seed_fallback_and_priority():
    r=choose_backbone([1]*3,[1.2]*3,[1.19]*3,[.94]*3,[.9]*3)
    assert r['backbone'] is None and r['status']=='NO_ELIGIBLE_BACKBONE'
    assert choose_backbone([1]*3,[1.2]*3,[1.19]*3,[.95]*3,[1.3]*3)['backbone']=='early_sd'
    assert choose_backbone([1]*3,[1.2]*3,[1.0]*3,[.9]*3,[1.01]*3)['backbone']=='late_joint'
    with pytest.raises(ValueError):choose_backbone([1]*3,[1]*3,[1]*3,[1]*3,[2])


def test_minimum_main_seeds_and_softmax_null():
    with pytest.raises(ValueError):seeds_for({'main_seeds':{'beauty':[42,43,44]}},'beauty')
    from datetime import datetime,timezone
    reduced={'main_seeds':{'sports':[42,43,44]},'allow_reduced_transfer_seeds':True}
    with pytest.raises(ValueError):seeds_for(reduced,'sports',now=datetime(2026,11,11,tzinfo=timezone.utc))
    assert len(seeds_for(reduced,'sports',now=datetime(2026,11,13,tzinfo=timezone.utc)))==3
    gate=SoftmaxNullGate(4)
    alpha,beta=gate(torch.zeros(2,4),torch.zeros(2,dtype=torch.long),torch.ones(2,3))
    assert (alpha.sum(1)<1).all() and torch.isfinite(alpha).all()


def test_paired_ci_direction_and_at_least_three_seeds():
    def f(v):return pd.DataFrame(dict(user_idx=[0,1],target_idx=[2,3],rank=[1,2],**{'ndcg@10':[v,v]}))
    x=paired_summary({s:f(.3) for s in [42,43,44]},{s:f(.1) for s in [42,43,44]},20)
    assert x['delta_left_minus_right']==pytest.approx(.2) and x['ci95']==pytest.approx([.2,.2])
    with pytest.raises(ValueError):paired_summary({42:f(.3)},{42:f(.1)})


def test_controls_uniform_not_marginal_and_gaussian_covariance(tmp_path):
    n=4000;meta=tmp_path/'metadata';meta.mkdir();rng=np.random.default_rng(8)
    titles=(rng.normal(size=(n,3))@np.diag([.3,1.,2.])+np.array([2.,-1.,.5])).astype(np.float32)
    data=pd.DataFrame(dict(model_item_id=np.arange(1,n+1),category_indices=['[1, 2]']*n,brand_idx=[1]*(n-1)+[2]))
    avail=pd.DataFrame(dict(model_item_id=np.arange(1,n+1),has_category=[True]*n,has_brand=[True]*n,has_title=[True]*n))
    data.to_csv(meta/'metadata.csv',index=False);avail.to_csv(meta/'availability_mask.csv',index=False);np.save(meta/'title_embeddings.npy',titles)
    out=tmp_path/'controls';m=build_controls(meta,out,seeds=[42])
    random_brand=pd.read_csv(out/'random/random_brand_seed_42.csv').random_brand_idx
    assert .45<(random_brand==1).mean()<.55
    random_title=np.load(out/'random/random_title_seed_42.npy')
    assert np.max(np.abs(random_title.mean(0)-titles.mean(0)))<.1
    assert np.linalg.norm(np.cov(random_title,rowvar=False)-np.cov(titles,rowvar=False))/np.linalg.norm(np.cov(titles,rowvar=False))<.12
    assert not np.allclose(np.linalg.norm(random_title,axis=1),1.)
    assert m==build_controls(meta,out,seeds=[42])
    reused=tmp_path/'reused'
    build_controls(meta,reused,seeds=[42],shuffled_source=out)
    assert (reused/'shuffled/perm_brand_seed_42.csv').read_bytes()==(out/'shuffled/perm_brand_seed_42.csv').read_bytes()
    (out/'random/random_brand_seed_42.csv').write_text('modified')
    with pytest.raises(ValueError):build_controls(meta,out,seeds=[42])


def test_adoption_rejects_a_different_historical_runtime(tmp_path):
    run,old,source=import_fixture(tmp_path)
    run.config['environment']=dict(torch='2.13.0+cu130',gpu_contains='RTX 4060')
    write_json(source/'attempt_001/run_manifest.json',dict(python_version='3.12.16',torch_version='1.0',library_versions={}))
    write_json(source/'attempt_001/history.json',dict(resource_usage={'gpu_name':'NVIDIA GeForce RTX 4060'}))
    with pytest.raises(ValueError,match='environment differs'):run.import_job('beauty','id_42',old)


def import_fixture(tmp_path):
    from taimma.data.split import SequenceSample
    repo=tmp_path/'repo';repo.mkdir();source=tmp_path/'old/beauty/jobs/id_42';attempt=source/'attempt_001';attempt.mkdir(parents=True)
    signatures={}
    for rel in COMPATIBLE_MODEL_FILES:
        p=repo/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'# locked model\n');signatures[rel]=sha(p)
    data={k:'fixture' for k in ['raw_reviews_sha256','metadata_files','catalog_sha256','train_base_sha256','val_sha256','test_sha256']}
    sample=SequenceSample(0,[0,1],2,1)
    cfg=dict(spec={'kind':'id','seed':42,'sources':[]},data_signature=data)
    write_json(attempt/'config.json',cfg);write_json(attempt/'source_snapshot.json',signatures)
    (attempt/'checkpoint.pt').write_bytes(b'checkpoint fixture')
    # Text was downloaded through Git: bytes changed only by CRLF.
    (attempt/'run.log').write_bytes(b'fixture\r\n')
    artifacts={p.relative_to(source).as_posix():sha(p) for p in attempt.iterdir()}
    artifacts['attempt_001/run.log']=hashlib.sha256(b'fixture\n').hexdigest()
    write_json(source/'completed.json',dict(spec=cfg['spec'],checkpoint='attempt_001/checkpoint.pt',fingerprint='original',control=None,artifacts=artifacts))
    run=StudyCompletion(repo,{},'runs/completion_fixture',device='cpu')
    run.dataset=lambda key:dict(signature=data,split=SimpleNamespace(train_base_samples=[sample]))
    return run,tmp_path/'old',source


def test_adoption_verifies_checkpoint_and_does_not_change_original(tmp_path):
    run,old,source=import_fixture(tmp_path)
    before={p.relative_to(source).as_posix():sha(p) for p in source.rglob('*') if p.is_file()}
    assert run.import_job('beauty','id_42',old)
    assert before=={p.relative_to(source).as_posix():sha(p) for p in source.rglob('*') if p.is_file()}
    assert (run.job_path('beauty','id_42')/'import_provenance.json').exists()


def test_adoption_refuses_missing_weights_and_changed_model(tmp_path):
    run,old,source=import_fixture(tmp_path)
    p=source/'attempt_001/checkpoint.pt';p.unlink()
    with pytest.raises(FileNotFoundError,match='Original .pt'):run.import_job('beauty','id_42',old)
    p.write_bytes(b'checkpoint fixture')
    (run.repo/COMPATIBLE_MODEL_FILES[0]).write_bytes(b'# altered algorithm\n')
    with pytest.raises(ValueError,match='Model/trainer changed'):run.import_job('beauty','id_42',old)


@pytest.mark.parametrize('label',['CE','UNRESOLVED'])
def test_diagnostic_e1_never_authorizes_s3(tmp_path,monkeypatch,label):
    import taimma.experiments.study_completion as module
    run=StudyCompletion(tmp_path,dict(bootstrap_reps=2),'runs/completion_fixture',device='cpu')
    run.validate_independent_references=lambda *a:None
    p=tmp_path/'src/taimma/evaluation/e0_e1_protocol.py';p.parent.mkdir(parents=True);p.write_text('# fixture')
    write_json(run.output/'beauty/groups.json',dict(rules={}))
    monkeypatch.setattr(module,'apply_group_rules',lambda *a:(None,None,np.zeros(len(a[0]),dtype=int)))
    def align(key,jobs,folder):
        rows=[]
        for seed in [42,43]:
            for source in ['category','brand','title']:
                rows.append(dict(user_idx=0,context_id='x',source_name=source,reference_seed=seed,seq_len=1,
                                 feat_log_len=0.,feat_history_tail_ratio=0.,delta_ce=.1,delta_rank=.2))
        root=run.output/key/folder;root.mkdir(parents=True)
        pd.DataFrame(rows).to_csv(root/'utility_both_labels.csv',index=False)
        return label
    run.alignment=align
    def e1(command,**kwargs):write_json(Path(command[command.index('--output_json')+1]),dict(decision='GO'))
    monkeypatch.setattr(module.subprocess,'run',e1)
    result=run.utility('beauty',['early_sd_42','early_sd_43'],'diagnostic',selected=False)
    assert result['s3_eligible'] is False and result['diagnostic_only'] is True
    assert result['decision']==('GO' if label=='CE' else 'UNRESOLVED')


def test_release_before_freeze_cannot_initialize_model(tmp_path):
    run=StudyCompletion(tmp_path,{},'runs/completion_fixture',device='cpu')
    run.load_job=lambda *a:pytest.fail('No model before freeze')
    with pytest.raises(RuntimeError,match='Freeze'):release_test(run,'beauty')
    run.config['test_release_not_before']='2999-11-13'
    with pytest.raises(RuntimeError,match='before'):run.freeze()


def test_freeze_is_idempotent_and_rejects_tampering(tmp_path,monkeypatch):
    import taimma.experiments.study_completion as module
    run=StudyCompletion(tmp_path,dict(test_release_not_before='2000-01-01',datasets=['beauty']),'runs/completion_fixture',device='cpu')
    run.source_signature=lambda:{'fixture.py':'hash'}
    monkeypatch.setattr(module,'git_state',lambda *a,**kw:dict(commit='fixture',dirty=False))
    monkeypatch.setattr(module.subprocess,'check_output',lambda *a,**kw:'fixture\n')
    p=run.output/'beauty/framing_decision.json';write_json(p,dict(framing='F2',test_used=False))
    (p.parent/'report_tables.csv').write_text('pre-release presentation')
    run.freeze('reviewed-fixture');first=(run.output/'freeze.json').read_bytes();run.freeze('reviewed-fixture')
    assert first==(run.output/'freeze.json').read_bytes()
    (p.parent/'report_tables.csv').write_text('post-release presentation; scientific decisions unchanged')
    run.verify_freeze()
    write_json(p,dict(framing='F1',test_used=True))
    with pytest.raises(ValueError,match='Frozen artifact'):run.verify_freeze()


def test_old_s3_route_cannot_open_test(tmp_path):
    from taimma.experiments.checkpoint2 import Checkpoint2
    run=Checkpoint2(tmp_path,{},'runs/fixture',device='cpu')
    with pytest.raises(RuntimeError,match='retired'):run.s3('beauty')


def test_reference_systems_require_independently_trained_id_weights(tmp_path):
    run=StudyCompletion(tmp_path,{},'runs/completion_fixture',device='cpu')
    for seed in [42,43]:
        folder=run.job_path('beauty',f'early_sd_{seed}');folder.mkdir(parents=True)
        torch.save(dict(model_state_dict={'base_sasrec.weight':torch.tensor([1.])}),folder/'checkpoint.pt')
    run.completed=lambda key,name:dict(spec={'seed':int(name.rsplit('_',1)[1])},checkpoint='checkpoint.pt')
    with pytest.raises(ValueError,match='identical'):run.validate_independent_references('beauty',['early_sd_42','early_sd_43'])
    torch.save(dict(model_state_dict={'base_sasrec.weight':torch.tensor([2.])}),run.job_path('beauty','early_sd_43')/'checkpoint.pt')
    run.validate_independent_references('beauty',['early_sd_42','early_sd_43'])


def test_completion_cli_plan_has_no_execution():
    p=Path(__file__).resolve().parents[1]/'scripts/complete_checkpoint2.py'
    r=subprocess.run([sys.executable,str(p)],capture_output=True,text=True,check=True)
    assert json.loads(r.stdout)['execution']=='DISABLED'


def test_f1_validation_runs_five_seeds_all_ablations_without_test(tmp_path,monkeypatch):
    import taimma.experiments.study_gates as gates
    from taimma.data.split import SequenceSample
    cfg=dict(main_seeds={'beauty':[42,43,44,45,46]},bootstrap_reps=2,gate_tau=1.,gate_kappa=.1,
             soft_label_temperature=.2,epochs=1,patience=1,batch_size=2,lambda_util=.5,lambda_sparse=.001)
    run=StudyCompletion(tmp_path,cfg,'runs/completion_fixture',device='cpu')
    root=run.output/'beauty/selected_s2';root.mkdir(parents=True)
    sources=['category','brand','title']
    report=dict(qualifying_sources=sources,sources={s:dict(stability=dict(stability_regime='contextual',epsilon=.01)) for s in sources})
    write_json(root/'e0_e1_CE.json',report)
    rows=[]
    for seed in [42,43]:
        for u in [0,1]:
            for s in sources:rows.append(dict(user_idx=u,reference_seed=seed,source_name=s,delta=.1+u*.2,delta_ce=.1+u*.2,delta_rank=.2+u*.1))
    frame=pd.DataFrame(rows);frame.to_csv(root/'utility_CE.csv',index=False)
    samples=[SequenceSample(0,[0,1],2,1),SequenceSample(1,[0,2],3,1)]
    class Split:
        train_meta_samples=samples;train_base_samples=samples;val_samples=samples
        @property
        def test_samples(self):pytest.fail('Validation phase must never access gate test samples')
    run.dataset=lambda key:dict(split=Split())
    run.utility=lambda *a,**kw:(run.output/'beauty/ablation_no_sd_labels').mkdir(parents=True,exist_ok=True)
    no=root.parent/'ablation_no_sd_labels';no.mkdir();frame.to_csv(no/'utility_both_labels.csv',index=False)
    run.id_grid_seed=lambda *a:None
    fits=[];run.run_job=lambda key,name,spec,**kw:fits.append(name)
    run.model_hash=lambda *a:'model-hash';run.guard=lambda **kw:None
    run.load_job=lambda *a:SimpleNamespace()
    features=pd.DataFrame(dict(user_idx=[0,1],group_idx=[0,0],feat_log_len=[0.,0.],feat_state_000=[.1,.2]))
    def feat(run,key,job,part):
        p=run.output/key/'gate_features'/job/(part+'.csv');p.parent.mkdir(parents=True,exist_ok=True);features.to_csv(p,index=False)
        return features
    monkeypatch.setattr(gates,'features_for',feat)
    labels=pd.DataFrame(dict(user_idx=[0,1],category=[1.,1.],brand=[1.,1.],title=[1.,1.]))
    monkeypatch.setattr(gates,'add_one_labels',lambda *a,**kw:labels)
    monkeypatch.setattr(gates,'validation_baselines',lambda *a:[dict(mode=b,seed=s,val_ndcg_10=.1) for s in [42,43,44,45,46] for b in gates.BASELINES if b not in ['global_gate','unsupervised']])
    trained=[]
    def train(adapter,gate,train,val,train_f,val_f,labels,columns,attempt,*a,**kw):
        trained.append((gate,columns,kw));torch.save(dict(model_state_dict=gate.state_dict()),attempt/'gate.pt')
        pd.DataFrame([dict(epoch=1,val_ndcg_10=.2)]).to_csv(attempt/'gate_epochs.csv',index=False)
        return gate,np.zeros(len(columns)),np.ones(len(columns)),[dict(val_ndcg_10=.2)]
    monkeypatch.setattr(gates,'train_gate',train)
    monkeypatch.setattr(gates,'gate_evaluate',lambda *a,**kw:SimpleNamespace(ndcg_10=.2))
    monkeypatch.setattr(gates,'rank_frame',lambda r:pd.DataFrame(dict(user_idx=[0,1],target_idx=[2,3],rank=[1,1],**{'ndcg@10':[.2,.2]})))
    gates.fit_validation(run,'beauty','early_sd',dict(s3_eligible=True,label_choice='CE'))
    assert len(trained)==5*(3+5)
    assert not list(run.output.rglob('per_user_test.csv'))
    assert sum(name.startswith('id_plus_meta_') for name in fits)==5
    assert sum(name.startswith('early_plus_meta_') for name in fits)==5
    recs=list((run.output/'beauty/gates').glob('*/completed.json'))
    assert len(recs)==40
    assert any('45' in p.parent.name for p in recs) and any('46' in p.parent.name for p in recs)


def test_report_serializes_exactly_seven_tables_from_registry(tmp_path):
    import zipfile
    from xml.etree import ElementTree as E
    from taimma.experiments.completion_report import write_report
    root=tmp_path/'beauty';root.mkdir()
    pd.DataFrame([dict(job=f'id_{s}',seed=s,val_ndcg_10=.1,test_ndcg_10=.08,best_epoch=3,best_epoch_le5=True) for s in [42,43,44]]).to_csv(root/'results.csv',index=False)
    p=write_report(root,dict(bootstrap_reps=5))
    with zipfile.ZipFile(p) as z:
        xml=E.fromstring(z.read('word/document.xml'))
        assert len(xml.findall('.//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}tbl'))==7
    assert (root/'report_tables.csv').exists() and (root/'utility_summary.csv').exists()
