"""Small CPU fixtures only; never train or infer on the research datasets."""
from pathlib import Path
import json
import subprocess
import sys
import zipfile
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from taimma.models.sasrec import SASRecModel
from taimma.models.early_fusion import EarlyFusionSASRec
from taimma.models.late_fusion import SourceSequentialEncoder,ResidualLateFusionModel
from taimma.models.source_dropout import sample_source_weights
from taimma.experiments.checkpoint2_protocol import (split_meta_users,grouped_subset_analysis,
    choose_utility_label,paired_seed_ci,aligned_rank_matrix)
from metadata_pipeline.user_grouping import fit_group_rules,apply_group_rules
from taimma.data.split import SequenceSample
from taimma.training.multimodal_trainer import gather_source_sequences
from taimma.experiments.mini_checkpoint_report import write_mini_report


@pytest.fixture
def tiny():
    torch.manual_seed(42)
    attrs={'category':torch.tensor([[1,2],[2,0],[1,0],[2,1],[1,0]]),
        'brand':torch.tensor([1,2,1,0,2]),'title':torch.randn(5,8)}
    masks={'category':torch.ones(5),'brand':torch.tensor([1.,1.,1.,0.,1.]),'title':torch.ones(5)}
    ids=torch.tensor([[0,1,2],[0,2,3]])
    return attrs,masks,ids


def early_forward(model,attrs,masks,ids,alpha):
    hist=gather_source_sequences(ids,attrs,masks);idx=(ids-1).clamp(min=0)
    active={s:masks[s][idx]*(ids>0) for s in attrs}
    ctx=model.encode_context(ids,category_ids=hist['category'],brand_ids=hist['brand'],
        title_features=hist['title'],category_mask=active['category'],brand_mask=active['brand'],
        title_mask=active['title'],source_weights=alpha)
    return model.score_full_catalog(ctx,catalog_category_ids=attrs['category'],catalog_brand_ids=attrs['brand'],
        catalog_title_features=attrs['title'],catalog_category_mask=masks['category'],
        catalog_brand_mask=masks['brand'],catalog_title_mask=masks['title'],source_weights=alpha)


def test_early_zero_weights_is_same_s1_id_mode_and_gradients_survive(tiny):
    attrs,masks,ids=tiny
    base=SASRecModel(5,max_seq_len=3,embedding_dim=8,num_blocks=1,num_heads=2,dropout_rate=0)
    model=EarlyFusionSASRec(base,2,2,8,dropout_rate=0).eval()
    with torch.no_grad():
        got=early_forward(model,attrs,masks,ids,torch.zeros(2,3))
        torch.testing.assert_close(got,base.predict_all(ids))
    for p in model.parameters():p.requires_grad=False
    alpha=torch.full((2,3),.5,requires_grad=True)
    # Check derivatives, not optimizer steps or model fitting.
    early_forward(model,attrs,masks,ids,alpha).sum().backward()
    assert alpha.grad is not None and torch.count_nonzero(alpha.grad)>0
    assert all(p.grad is None for p in model.parameters())


def test_early_same_weights_match_manual_both_sides(tiny):
    attrs,masks,ids=tiny
    base=SASRecModel(5,max_seq_len=3,embedding_dim=8,num_blocks=1,num_heads=2,dropout_rate=0)
    model=EarlyFusionSASRec(base,2,2,8,dropout_rate=0).eval()
    # Legacy all-enabled path versus explicit alpha=1.
    torch.testing.assert_close(early_forward(model,attrs,masks,ids,torch.ones(2,3)),
        early_forward(model,attrs,masks,ids,None),atol=2e-5,rtol=2e-5)
    alpha=torch.tensor([[1.,0.,1.],[0.,1.,0.]])
    together=early_forward(model,attrs,masks,ids,alpha)
    separate=torch.cat([early_forward(model,attrs,masks,ids[i:i+1],alpha[i:i+1]) for i in range(2)])
    torch.testing.assert_close(together,separate,atol=2e-5,rtol=2e-5)


@pytest.mark.parametrize('frozen',[True,False])
def test_late_freeze_controls_gradient_and_eval_state(tiny,frozen):
    attrs,masks,ids=tiny
    base=SASRecModel(5,max_seq_len=3,embedding_dim=8,num_blocks=1,num_heads=2,dropout_rate=0)
    enc=SourceSequentialEncoder('brand',2,feature_dim=8,max_seq_len=3,num_heads=2,dropout_rate=0)
    model=ResidualLateFusionModel(base,{'brand':enc},p_drop=0,sources_enabled=('brand',),freeze_id=frozen)
    model.train();assert model.id_model.training is not frozen
    scores,ident,_=model.compute_all_scores(ids,{'brand':gather_source_sequences(ids,attrs,masks)['brand']},
        {'brand':attrs['brand']},{'brand':masks['brand']})
    torch.testing.assert_close(scores,ident)
    torch.nn.functional.cross_entropy(scores,torch.tensor([2,3])).backward()
    assert all(p.grad is None for p in base.parameters()) if frozen else any(p.grad is not None for p in base.parameters())


def test_source_mask_is_per_user_and_all_off_is_training_only():
    torch.manual_seed(42)
    weights=sample_source_weights(1000,('category','brand','title'),'cpu',.2,.2,True)
    assert weights.shape==(1000,3)
    assert len(torch.unique(weights,dim=0))==8
    assert .15<float((weights.sum(1)==0).float().mean())<.27
    torch.testing.assert_close(sample_source_weights(3,('brand',),'cpu',.9,.9,False),
        torch.tensor([[0.,1.,0.]]*3))


def test_groups_are_context_only_and_test_does_not_refit():
    lengths=np.array([2,2,2,3,4,5,6,8,10,20]);ratios=np.array([.5,.5,.5,.2,.4,.6,.3,.8,.9,.1])
    rules=fit_group_rules(lengths,ratios)
    groups=apply_group_rules(lengths,ratios,rules)[2]
    assert len(set(groups[:3]))==1
    order=np.array([9,2,1,0,8,7,6,5,4,3])
    np.testing.assert_array_equal(apply_group_rules(lengths[order],ratios[order],rules)[2],groups[order])
    a=apply_group_rules([3],[.4],rules)[2][0]
    b=apply_group_rules([3,1000],[.4,.99],rules)[2][0]
    assert a==b


def test_meta_split_is_disjoint_by_user():
    samples=[SequenceSample(u,[0,1],2,1) for u in range(10) for _ in range(2)]
    a,b=split_meta_users(samples)
    assert len(a)+len(b)==len(samples)
    assert not ({s.user_idx for s in a}&{s.user_idx for s in b})


def frame(values):
    return pd.DataFrame(dict(user_idx=[0,1,2,3],target_idx=[1,2,3,4],rank=[1,2,3,4],**{'ndcg@10':values}))


def test_group_choice_uses_validation_not_test():
    val={'ID':frame([.8,.8,.1,.1]),'CBT':frame([.1,.1,.8,.8])}
    test={'ID':frame([.2,.2,.9,.9]),'CBT':frame([.9,.9,.2,.2])}
    groups=pd.DataFrame(dict(user_idx=[0,1,2,3],val_group_id=[0,0,1,1],test_group_id=[0,0,1,1]))
    result,rows=grouped_subset_analysis(val,test,groups)
    assert result['choices_from_validation']=={'0':'ID','1':'CBT'}
    assert result['group_val_selected_test']==pytest.approx(.2)
    assert result['group_test_oracle']==pytest.approx(.9)
    bad=test['ID'].copy();bad.loc[0,'target_idx']=99
    with pytest.raises(ValueError):aligned_rank_matrix({'ID':bad,'CBT':test['CBT']})


def test_label_rule_is_locked_and_never_falls_back_to_test():
    ce=dict(category=1.,brand=-1.,title=1.);rank=dict(category=-1.,brand=1.,title=1.)
    assert choose_utility_label(ce,rank,ce)=='CE'
    assert choose_utility_label(ce,rank,rank)=='LOG_RANK'
    assert choose_utility_label(ce,rank,dict(category=1.,brand=1.,title=-1.))=='UNRESOLVED'
    assert choose_utility_label(ce,{},ce)=='PENDING'


def test_paired_seed_ci_keeps_user_alignment():
    real={s:frame([.1,.2,.3,.4]) for s in [42,43,44]}
    control={s:frame([.2,.3,.4,.5]) for s in [42,43,44]}
    got=paired_seed_ci(real,control,reps=20)
    assert got['delta_control_minus_real']==pytest.approx(.1)
    assert got['ci95']==pytest.approx([.1,.1])


def test_report_is_tables_with_one_page_break_and_missing_values(tmp_path):
    state={'dataset':'fixture only','early':{},'complete':False}
    path=write_mini_report(state,tmp_path)
    with zipfile.ZipFile(path) as z:
        root=ET.fromstring(z.read('word/document.xml'))
        ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
        assert len(root.findall('.//w:br[@w:type="page"]',ns))==1
        assert len(root.findall('.//w:tbl',ns))==9
        text=''.join(root.itertext())
        assert 'Chưa có' in text and 'Chưa đủ' in text


def test_cli_default_is_nonexecuting_plan():
    script=Path(__file__).resolve().parents[1]/'scripts/run_checkpoint2.py'
    out=subprocess.check_output([sys.executable,str(script)],text=True,encoding='utf-8')
    plan=json.loads(out)
    assert plan['execution']=='DISABLED'


def test_execution_guard_rejects_without_touching_gpu(tmp_path):
    from taimma.experiments.checkpoint2 import Checkpoint2
    run=Checkpoint2(tmp_path,{},'runs/new')
    with pytest.raises(RuntimeError,match='disabled'):run.guard(training=True)
    assert not (tmp_path/'runs/new').exists()


@pytest.mark.parametrize('kind',['id','early','late'])
def test_job_wiring_resumes_and_rejects_tampering(tmp_path,monkeypatch,tiny,kind):
    """Exercise real artifact/logger wiring; replace fitting with a no-step stub."""
    from types import SimpleNamespace
    import taimma.experiments.checkpoint2 as module
    from taimma.training.checkpoint2_trainer import ModelSpec,evaluate
    from taimma.training.trainer import TrainingHistory
    attrs,masks,ids=tiny
    samples=[SequenceSample(i,ids[i].tolist(),4+i,2) for i in range(2)]
    split=SimpleNamespace(train_base_samples=samples,val_samples=samples,test_samples=samples,num_items=5)
    run=module.Checkpoint2(tmp_path,dict(epochs=1,patience=1,batch_size=2),'runs/fixture',device='cpu')
    monkeypatch.setattr(run,'guard',lambda **kwargs:None)
    monkeypatch.setattr(run,'dataset',lambda key:dict(split=split,attrs=attrs,masks=masks,signature={'fixture':True}))
    monkeypatch.setattr(run,'catalog',lambda key,control:(attrs,masks))
    monkeypatch.setattr(module.ExperimentLogger,'_snapshot_source',staticmethod(lambda folder:None))
    monkeypatch.setattr(module.ExperimentLogger,'_get_git_info',staticmethod(lambda:dict(commit='fixture',dirty=False)))
    calls=[]
    def no_training(adapter,train,val,folder,*args):
        calls.append('no optimizer, fixture only')
        result=evaluate(adapter,val,2)
        torch.save(dict(model_state_dict=adapter.model.state_dict()),folder/'checkpoint.pt')
        return TrainingHistory(train_losses=[1.],val_ndcg_10=[result.ndcg_10],
            val_recall_10=[result.recall_10],best_epoch=1,best_val_ndcg=result.ndcg_10)
    monkeypatch.setattr(module,'fit',no_training)
    spec=ModelSpec(kind,42,embedding_dim=8,max_seq_len=3,freeze_id=False)
    first=run.run_job('beauty','fixture',spec)
    second=run.run_job('beauty','fixture',spec)
    assert first['fingerprint']==second['fingerprint'] and len(calls)==1
    root=run.job_path('beauty','fixture')
    assert (root/'attempt_001/per_user_val.csv').is_file()
    assert (root/'attempt_001/run_state.json').is_file()
    assert not (root/'execution.lock').exists()
    (root/first['checkpoint']).write_bytes(b'changed fixture')
    with pytest.raises(ValueError,match='artifact changed'):run.completed('beauty','fixture')


def test_availability_has_no_target_dependency(tiny):
    from taimma.training.checkpoint2_trainer import ModelSpec,ModelAdapter,build_model
    attrs,masks,ids=tiny
    spec=ModelSpec('early',42,embedding_dim=8,max_seq_len=3)
    adapter=ModelAdapter(build_model(spec,5,attrs),spec,attrs,masks)
    # Brand is absent for item 4 but exists elsewhere in the candidate catalog.
    history=torch.tensor([[0,0,4],[0,4,2]])
    torch.testing.assert_close(adapter.availability(history),torch.tensor([[1.,0.,1.],[1.,1.,1.]]))


def test_same_history_features_ignore_target_identity(tiny):
    from types import SimpleNamespace
    from taimma.training.checkpoint2_trainer import ModelSpec,ModelAdapter,build_model
    from taimma.evaluation.checkpoint2_utility import collect_ablations
    attrs,masks,ids=tiny
    spec=ModelSpec('early',42,embedding_dim=8,max_seq_len=3)
    adapter=ModelAdapter(build_model(spec,5,attrs),spec,attrs,masks)
    split=SimpleNamespace(train_base_events=[1,2,2,3,4,5],num_items=5)
    samples=[SequenceSample(i,ids[0].tolist(),target,2) for i,target in enumerate([4,5])]
    _,features=collect_ablations(adapter,samples,split,batch_size=2,availability_aware=True)
    columns=[c for c in features if c.startswith('feat_')]
    np.testing.assert_allclose(features.loc[0,columns].to_numpy(float),features.loc[1,columns].to_numpy(float),rtol=1e-5,atol=1e-5)
