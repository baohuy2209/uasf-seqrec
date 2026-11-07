import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
import pytest
import torch
from types import SimpleNamespace
from taimma.experiments.memo0711 import SeparateID,NullGate,framing_from_ranks,Memo0711

class Fake(torch.nn.Module):
    def __init__(self,scores):super().__init__();self.register_buffer('scores',torch.tensor(scores,dtype=torch.float32));self.model=self if False else torch.nn.Linear(1,1)
    def predict(self,ids,weights=None):return self.scores.expand(len(ids),-1)+(0 if weights is None else weights.sum(1,keepdim=True))

def test_only_all_off_uses_independent_id_and_retains_alpha_gradient():
    a=Fake([[1.,2.]]);b=Fake([[8.,9.]]);adapter=SeparateID(a,b)
    weights=torch.tensor([[0.,0.,0.],[.1,.2,0.]],requires_grad=True)
    scores=adapter.predict(torch.ones(2,3,dtype=torch.long),weights)
    assert torch.allclose(scores[0],torch.tensor([8.,9.]))
    assert torch.allclose(scores[1],torch.tensor([1.3,2.3]))
    scores.sum().backward();assert weights.grad[1].abs().sum()>0
    adapter.use_id_fallback=False
    assert torch.allclose(adapter.predict(torch.ones(2,3,dtype=torch.long),weights)[0],torch.tensor([1.,2.]))

def test_null_winner_turns_all_sources_off_at_inference():
    gate=NullGate(2)
    with torch.no_grad():
        for p in gate.parameters():p.zero_()
        gate.mlp[-1].bias[0]=8
    alpha,_=gate(torch.zeros(3,2),torch.zeros(3,dtype=torch.long),torch.ones(3,3),threshold=True)
    assert torch.equal(alpha,torch.zeros_like(alpha))

def frame(v):return pd.DataFrame(dict(user_idx=np.arange(len(v)),target_idx=np.arange(len(v))+1,rank=np.ones(len(v)),**{'ndcg@10':v}))

def test_framing_requires_paired_ci_not_only_point_improvement():
    base=np.full(100,.2);small=base+np.r_[np.full(51,.1),np.full(49,-.1)]
    frames={'uasf':{s:frame(small) for s in range(42,47)},'baseline':{s:frame(base) for s in range(42,47)}}
    assert framing_from_ranks(frames,200)['framing']=='F2'
    frames['uasf']={s:frame(base+.05) for s in range(42,47)}
    assert framing_from_ranks(frames,200)['framing']=='F1'
    del frames['uasf'][46]
    with pytest.raises(ValueError):framing_from_ranks(frames)

def test_missing_historical_job_never_trains(tmp_path):
    run=Memo0711(tmp_path,{'supervisor_memo':'2026-11-07'},'runs/memo0711_test',device='cpu')
    with pytest.raises(FileNotFoundError,match='no retraining fallback'):run.import_required('beauty','early_sd_42',tmp_path/'absent')
    with pytest.raises(RuntimeError,match='forbids'):run.run_job('beauty','K4',None)

def test_new_stage_default_is_nonexecuting_plan():
    import subprocess
    p=subprocess.run([sys.executable,str(Path(__file__).resolve().parents[1]/'scripts/run_memo0711.py')],capture_output=True,text=True)
    assert p.returncode==0 and '"training": false' in p.stdout and 'extra K4' in p.stdout

def test_new_job_never_opens_test_and_reuses_verified_checkpoint(tmp_path,monkeypatch):
    from taimma.data.split import SequenceSample
    from taimma.training.checkpoint2_trainer import ModelSpec
    from taimma.experiments import memo0711 as m
    class Split:
        num_items=5
        train_base_samples=[SequenceSample(0,[0,1],2,1),SequenceSample(1,[0,2],3,1)]
        val_samples=[SequenceSample(0,[1,2],3,2),SequenceSample(1,[2,3],4,2)]
        @property
        def test_samples(self):raise AssertionError('Pre-freeze Test was opened')
    cfg=dict(supervisor_memo='2026-11-07',epochs=1,patience=1,batch_size=2)
    run=Memo0711(tmp_path,cfg,'runs/memo0711_test',device='cpu',execute_training=True)
    attrs={'category':torch.ones(5,1,dtype=torch.long),'brand':torch.ones(5,dtype=torch.long),'title':torch.ones(5,2)}
    masks={s:torch.ones(5) for s in attrs}
    run.dataset=lambda key:dict(split=Split(),attrs=attrs,masks=masks,signature={'fixture':1})
    run.catalog=lambda *args:(attrs,masks);run.guard=lambda **kwargs:None;run.assert_protocol=lambda *args:None
    monkeypatch.setattr(m,'git_state',lambda *args,**kwargs:{'commit':'fixture','dirty':False})
    rec=run.run_job('beauty','id_42',ModelSpec('id',42))
    assert rec['test_status']=='NOT_OPENED' and 'test_ndcg_10' not in rec
    monkeypatch.setattr(m,'build_model',lambda *args:pytest.fail('Cached model must not train'))
    cached=run.run_job('beauty','id_42',ModelSpec('id',42))
    assert cached['fingerprint']==rec['fingerprint'] and cached['checkpoint']==rec['checkpoint']

def test_five_seed_gate_orchestration_does_not_open_test(tmp_path,monkeypatch):
    from taimma.experiments import memo0711 as m
    from taimma.data.split import SequenceSample
    from taimma.evaluation.evaluator import EvaluationResult,UserRankRecord
    from taimma.experiments.checkpoint2_protocol import write_json
    cfg=dict(supervisor_memo='2026-11-07',gate_tau=1.,gate_kappa=.1,epochs=1,patience=1,batch_size=2,
             lambda_util=.5,lambda_sparse=.001,bootstrap_reps=20)
    run=Memo0711(tmp_path,cfg,'runs/memo0711_fixture',device='cpu',execute_training=True)
    class Split:
        train_meta_samples=[SequenceSample(0,[1],2,1),SequenceSample(1,[2],3,1)]
        val_samples=train_meta_samples
        @property
        def test_samples(self):raise AssertionError('Gate orchestration touched Test')
    run.dataset=lambda key:{'split':Split()}
    run.foundation=lambda *args:None;run.fairness=lambda *args:None;run.guard=lambda **kwargs:None
    report=dict(qualifying_sources=m.SOURCES,sources={s:{'stability':{'stability_regime':'contextual','epsilon':.1}} for s in m.SOURCES})
    def utility(key,jobs,folder,selected):
        write_json(run.output/key/folder/'e0_e1_CE.json',report)
        return dict(s3_eligible=True,label_choice='CE')
    run.utility=utility
    run.subset_frames=lambda *args:{n:frame([.1,.1]) for n in ['all_fixed','global_subset','group_subset']}
    run.inference_ranks=lambda *args:frame([.1,.1])
    run.model_hash=lambda key,job:job
    run.load_job=lambda *args:SimpleNamespace(device=torch.device('cpu'),model=torch.nn.Linear(1,1))
    features=pd.DataFrame(dict(user_idx=[0,1],group_idx=[0,0],feat_log_len=[0.,1.],feat_state_000=[.1,.2]))
    monkeypatch.setattr(m,'features_for',lambda *args:features)
    seen=[]
    def labels(key,job,*args):
        seen.append(job);return pd.DataFrame(dict(user_idx=[0,1],**{s:[1.,0.] for s in m.SOURCES}))
    run.seed_labels=labels
    def fit_gate(adapter,gate,*args,**kwargs):
        columns=args[5];folder=args[6]
        torch.save(dict(model_state_dict=gate.state_dict(),feature_columns=columns,feature_mean=[0.]*len(columns),feature_scale=[1.]*len(columns)),folder/'gate.pt')
        return gate,None,None,[{'val_ndcg_10':.2}]
    monkeypatch.setattr(m,'train_gate',fit_gate)
    def evaluation(*args,**kwargs):
        return EvaluationResult(2,.2,.2,.2,.2,True,[UserRankRecord(i,i+1,1,.2,.2,.2,.2) for i in [0,1]])
    monkeypatch.setattr(m,'gate_evaluate',evaluation)
    run.uasf('beauty')
    assert len(seen)==5*len(m.MODES)
    assert all(f'early_sd_{s}' in seen for s in range(42,47))
    assert not list(run.output.rglob('per_user_test.csv'))
    assert (run.output/'beauty/framing_decision.json').exists()
    a=torch.load(run.output/'beauty/gates/uasf_42/attempt_001/gate.pt',weights_only=True)['model_state_dict']
    b=torch.load(run.output/'beauty/gates/fallback_s1_42/attempt_001/gate.pt',weights_only=True)['model_state_dict']
    assert all(torch.equal(a[k],b[k]) for k in a), 'Fallback ablation must start from the same gate initialization'
