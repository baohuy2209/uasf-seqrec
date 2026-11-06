"""Regression coverage for data preparation and orchestration, with CPU fixtures.

No real experiment is trained. Model fitting is replaced with a checkpoint-only
stub where orchestration needs to be tested.
"""
from pathlib import Path
from types import SimpleNamespace
import copy
import json
import sys
import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from tests.test_metadata_pipeline import inputs,handoff
from taimma.data.split import SequenceSample
from taimma.experiments.checkpoint2 import Checkpoint2
from taimma.experiments.checkpoint2_protocol import validate_catalog_mapping,sha,write_json
from taimma.training.checkpoint2_trainer import ModelSpec,ModelAdapter,build_model,evaluate
from taimma.training.trainer import TrainingHistory
from taimma.evaluation.checkpoint2_utility import collect_ablations


def fixture_runner(tmp_path,handoff):
    meta,raw=handoff
    return Checkpoint2(tmp_path,dict(raw_root=str(raw),metadata_root=str(meta.parent),environment={}),
        'runs/fixture',device='cpu')


def test_dataset_preflight_retains_oov_mapping_without_writing(tmp_path,handoff,monkeypatch):
    import taimma.experiments.checkpoint2 as module
    monkeypatch.setattr(module,'build_model',lambda *a,**k:pytest.fail('A data check must never create a model'))
    run=fixture_runner(tmp_path,handoff)
    result=run.check_data('beauty')
    assert result['cohort_items']==6 and result['catalog_items']==3 and result['oov_items']==3
    assert result['status']=='PASS' and result['model_inference_executed'] is False
    assert not run.output.exists()


def test_mapping_rejects_real_misalignment_and_bad_oov(handoff):
    meta,_=handoff
    catalog=pd.read_csv(meta/'metadata.csv',keep_default_na=False)
    mapping=pd.read_csv(meta/'item_map.csv',keep_default_na=False)
    split=SimpleNamespace(num_items=3,idx2item={1:'A',2:'B',3:'C'})
    assert validate_catalog_mapping(catalog,mapping,split)['oov_items']==3
    bad=catalog.copy();bad.loc[0,'raw_item_id']='BAD'
    with pytest.raises(ValueError,match='Raw reviews/catalog mismatch'):validate_catalog_mapping(bad,mapping,split)
    bad=mapping.copy();bad.loc[~bad.in_train_base,'model_item_id']=9
    with pytest.raises(ValueError,match='OOV mapping IDs'):validate_catalog_mapping(catalog,bad,split)


def test_prepare_builds_all_controls_and_resumes_without_models(tmp_path,handoff,monkeypatch):
    import taimma.experiments.checkpoint2 as module
    run=fixture_runner(tmp_path,handoff)
    monkeypatch.setattr(run,'guard',lambda **k:None)
    monkeypatch.setattr(module,'git_state',lambda *a,**k:dict(commit='fixture',dirty=False))
    monkeypatch.setattr(module,'check_training_environment',lambda *a,**k:dict(gpu='fixture'))
    run.prepare('beauty')
    root=run.output/'beauty'
    manifest=json.loads((root/'training_contrasts/manifest.json').read_text())
    assert len(manifest['shuffled'])==len(manifest['random'])==9
    for source in ['category','brand','title']:
        for kind in ['shuffled','random']:
            for seed in [42,43,44]:
                attrs,masks=run.catalog('beauty',(source,kind,seed))
                assert attrs[source].shape[0]==3
    before={p.relative_to(root).as_posix():sha(p) for p in root.rglob('*') if p.is_file()}
    run.prepare('beauty')
    after={p.relative_to(root).as_posix():sha(p) for p in root.rglob('*') if p.is_file()}
    assert before==after


def small_adapter():
    attrs=dict(category=torch.tensor([[1],[2],[3],[1],[3]]),brand=torch.tensor([1,2,0,5,4]),title=torch.randn(5,8))
    masks=dict(category=torch.ones(5),brand=torch.tensor([1.,1.,0.,1.,1.]),title=torch.ones(5))
    spec=ModelSpec('early',42,embedding_dim=8,max_seq_len=3,dropout=0.)
    return ModelAdapter(build_model(spec,5,attrs),spec,attrs,masks)


def test_random_control_keeps_canonical_vocabulary_and_loads(tmp_path,monkeypatch):
    import taimma.experiments.checkpoint2 as module
    adapter=small_adapter();attrs=adapter.attrs;masks=adapter.masks
    control={s:v.clone() for s,v in attrs.items()};control['brand']=torch.tensor([1,1,0,1,1])
    samples=[SequenceSample(0,[0,1,2],4,2),SequenceSample(1,[0,2,3],5,2)]
    split=SimpleNamespace(num_items=5,train_base_samples=samples,val_samples=samples,test_samples=samples)
    run=Checkpoint2(tmp_path,dict(epochs=1,patience=1,batch_size=2),'runs/fixture',device='cpu')
    monkeypatch.setattr(run,'guard',lambda **k:None)
    monkeypatch.setattr(run,'dataset',lambda key:dict(split=split,attrs=attrs,masks=masks,signature={'fixture':True}))
    monkeypatch.setattr(run,'catalog',lambda key,contrast:(control,masks) if contrast else (attrs,masks))
    monkeypatch.setattr(module.ExperimentLogger,'_snapshot_source',staticmethod(lambda folder:None))
    monkeypatch.setattr(module.ExperimentLogger,'_get_git_info',staticmethod(lambda:dict(commit='fixture',dirty=False)))
    def no_training(model,train,val,folder,*a):
        assert model.model.brand_embedding.num_embeddings==6
        result=evaluate(model,val,2)
        torch.save(dict(model_state_dict=model.model.state_dict()),folder/'checkpoint.pt')
        return TrainingHistory(train_losses=[1.],best_epoch=1,best_val_ndcg=result.ndcg_10)
    monkeypatch.setattr(module,'fit',no_training)
    run.run_job('beauty','random',adapter.spec,control=('brand','random',42))
    loaded=run.load_job('beauty','random')
    assert loaded.model.brand_embedding.num_embeddings==6
    torch.testing.assert_close(loaded.attrs['brand'],control['brand'])


def test_validation_ablations_keep_oov_in_denominator():
    adapter=small_adapter()
    split=SimpleNamespace(num_items=5,train_base_events=[1,2,3,4,5])
    samples=[SequenceSample(0,[0,1,2],4,2),SequenceSample(1,[0,1,2],0,2)]
    rows,features=collect_ablations(adapter,samples,split,batch_size=2,allow_oov=True)
    oov=rows[rows.target_idx==0]
    assert len(oov)==3 and oov.delta_ce.isna().all()
    assert (oov.ndcg_full==0).all() and (oov.ndcg_without==0).all() and (oov.delta_rank==0).all()
    for source,group in rows.groupby('source_name'):
        full=evaluate(adapter,samples,2)
        without=evaluate(adapter,samples,2,enabled=tuple(s for s in adapter.spec.sources if s!=source))
        assert (group.ndcg_full-group.ndcg_without).mean()==pytest.approx(full.ndcg_10-without.ndcg_10)
    columns=[c for c in features if c.startswith('feat_')]
    np.testing.assert_allclose(features.loc[0,columns].to_numpy(float),features.loc[1,columns].to_numpy(float),rtol=1e-5,atol=1e-5)


def test_alignment_uses_same_references_for_train_and_validation(tmp_path,monkeypatch):
    import taimma.experiments.checkpoint2 as module
    run=Checkpoint2(tmp_path,dict(batch_size=2),'runs/fixture',device='cpu')
    root=run.output/'beauty';write_json(root/'utility_label_rule.json',{'fixture':True})
    sample=SequenceSample(0,[0,1],2,1)
    split=SimpleNamespace(train_meta_samples=[sample],val_samples=[sample])
    monkeypatch.setattr(run,'guard',lambda **k:None)
    monkeypatch.setattr(run,'dataset',lambda key:dict(split=split,signature={'fixture':True}))
    monkeypatch.setattr(run,'model_hash',lambda key,name:name)
    monkeypatch.setattr(run,'load_job',lambda key,name:SimpleNamespace(name=name))
    calls=[]
    def ablation(adapter,samples,split,batch_size,include_features=True,**kwargs):
        calls.append((adapter.name,kwargs.get('allow_oov',False)))
        sign=1. if adapter.name=='ref42' else -3.
        rows=pd.DataFrame([dict(user_idx=0,context_id='context',source_name=s,delta_ce=sign,delta_rank=sign,ndcg_full=0.1 if sign>0 else 0.,ndcg_without=0. if sign>0 else .3) for s in ['category','brand','title']])
        return rows,pd.DataFrame([dict(user_idx=0,context_id='context',feat_log_len=0.)])
    monkeypatch.setattr(module,'collect_ablations',ablation)
    assert run.alignment('beauty',['ref42','ref43'],'alignment')=='CE'
    assert calls==[('ref42',False),('ref42',True),('ref43',False),('ref43',True)]
    record=json.loads((root/'alignment/alignment.json').read_text())
    assert record['validation_ndcg_full_minus_without']['brand']==pytest.approx(-.1)
    calls.clear()
    assert run.alignment('beauty',['ref42','ref43'],'alignment')=='CE'
    assert calls==[]


def test_trained_epoch_selection_excludes_initial_with_no_parameter_updates(tmp_path,monkeypatch):
    """Run the loop plumbing on a fixture; optimizer.step deliberately does nothing."""
    import taimma.training.checkpoint2_trainer as module
    from taimma.evaluation.evaluator import EvaluationResult
    adapter=small_adapter();before=copy.deepcopy(adapter.model.state_dict());steps=[]
    class NoUpdates:
        def __init__(self,*a,**k):pass
        def zero_grad(self,**k):adapter.model.zero_grad(**k)
        def step(self):steps.append('fixture no-op')
    monkeypatch.setattr(module.torch.optim,'Adam',NoUpdates)
    scores=iter([.9,.6,.5])
    monkeypatch.setattr(module,'evaluate',lambda *a,**k:EvaluationResult(2,next(scores),1.,.8,1.,True))
    samples=[SequenceSample(0,[0,1,2],4,2),SequenceSample(1,[0,2,3],5,2)]
    hist=module.fit(adapter,samples,samples,tmp_path/'loop',epochs=3,patience=1,batch_size=2)
    assert hist.best_epoch==1 and hist.best_val_ndcg==.6 and hist.initial_validation_metrics['NDCG@10']==.9
    assert hist.stop_reason=='early_stopping' and len(steps)==2
    checkpoint=torch.load(tmp_path/'loop/checkpoint.pt',weights_only=True)
    assert checkpoint['epoch']==1 and checkpoint['selection_scope']=='trained_epochs_only'
    log=pd.read_csv(tmp_path/'loop/epochs.csv')
    assert log.epoch.tolist()==[0,1,2] and log.selected_checkpoint.tolist()==[False,True,False]
    for key,value in adapter.model.state_dict().items():torch.testing.assert_close(value,before[key])


def test_report_uses_fresh_late_group_results():
    from taimma.experiments.mini_checkpoint_report import report_tables
    rows=[dict(seed=s,global_subset='CBT',id_test=.04,global_test=.05,group_val_selected_test=.06,group_test_oracle=.07) for s in [42,43,44]]
    tables=report_tables(dict(groups={'late':rows},early={}))
    group_table=next(t for t in tables if t[0].startswith('4 '))
    assert '0.050000' in group_table[2][0][2]
    assert 'Chưa có' not in str(group_table[2][0])


def test_gate_all_fixed_matches_s1_even_when_history_lacks_brand():
    from taimma.training.selective_fusion import SelectiveGate,gate_evaluate
    adapter=small_adapter()
    samples=[SequenceSample(0,[0,0,3],4,1)]
    features=pd.DataFrame([dict(user_idx=0,group_idx=0,feat_brand_coverage=0.)])
    gate=SelectiveGate(1,mode='global')
    with torch.no_grad():gate.global_logits.fill_(100.)
    full=evaluate(adapter,samples,1)
    got=gate_evaluate(adapter,gate,samples,features,['feat_brand_coverage'],np.zeros(1),np.ones(1),1)
    assert got.ndcg_10==full.ndcg_10
    seen=[];original=adapter.predict
    def capture(ids,weights=None):
        seen.append(weights.clone());return original(ids,weights)
    adapter.predict=capture
    gate_evaluate(adapter,gate,samples,features,['feat_brand_coverage'],np.zeros(1),np.ones(1),1)
    torch.testing.assert_close(seen[0],torch.ones(1,3))


def test_complete_ablation_cache_does_not_initialize_model(tmp_path,monkeypatch):
    from taimma.experiments.checkpoint2_protocol import SUBSETS,context_signature
    from taimma.experiments.checkpoint2 import subset_name
    samples=[SequenceSample(0,[0,1],2,1)]
    run=Checkpoint2(tmp_path,dict(batch_size=1),'runs/fixture',device='cpu')
    monkeypatch.setattr(run,'guard',lambda **k:None)
    monkeypatch.setattr(run,'dataset',lambda key:dict(split=SimpleNamespace(val_samples=samples,test_samples=samples)))
    monkeypatch.setattr(run,'model_hash',lambda *a:'fixture-checkpoint')
    monkeypatch.setattr(run,'load_job',lambda *a:pytest.fail('Verified complete ablations should not construct a model'))
    folder=run.output/'beauty/ablations/fixture';folder.mkdir(parents=True)
    for sources in SUBSETS:
        for part in ['val','test']:
            path=folder/f'{subset_name(sources)}_{part}.csv';path.write_text('fixture only')
            signature=dict(model_sha256='fixture-checkpoint',context_sha256=context_signature(samples),sources=list(sources),availability_aware=False)
            write_json(path.with_suffix('.json'),dict(signature=signature,sha256=sha(path)))
    run.ablate_subsets('beauty','fixture')
