"""Synthetic score/evidence tests only; no research dataset/checkpoint is used."""
import sys
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from taimma.experiments import memo0711_robustness as r
from taimma.training.checkpoint2_trainer import ModelAdapter,ModelSpec
from taimma.data.split import SequenceSample


def catalog():
    attrs=dict(category=torch.arange(1,6).reshape(5,1),brand=torch.arange(1,6),title=torch.ones(5,2))
    masks={s:torch.ones(5) for s in attrs};masks['category'][2]=0
    return attrs,masks


class TinyScores(torch.nn.Module):
    num_items=5
    def encode_context(self,ids,**kwargs):
        return kwargs['category_mask'].sum(1,keepdim=True)+1
    def score_full_catalog(self,context,**kwargs):
        return torch.arange(5,dtype=torch.float32)[None,:]+context*kwargs['catalog_category_mask'][None,:]


def adapter():
    attrs,masks=catalog()
    return ModelAdapter(TinyScores(),ModelSpec('early',42),attrs,masks)


def test_mask_nested_available_only_and_reproducible():
    _,base=catalog()
    _,a=r.make_mask(base,'category',.4,42)
    _,b=r.make_mask(base,'category',.8,42)
    _,repeat=r.make_mask(base,'category',.8,42)
    pd.testing.assert_frame_equal(b,repeat)
    assert a.selected_for_mask.sum()==2 and b.selected_for_mask.sum()==3
    assert (a.selected_for_mask<=b.selected_for_mask).all()
    assert not b.loc[b.original_available==0,'selected_for_mask'].any()
    assert base['category'][2]==0 and base['category'].sum()==4


def test_actual_history_candidate_probes_and_method_restoration():
    a=adapter();active,table=r.make_mask(catalog()[1],'category',.8,42)
    probe,scores=r.probe_adapter(a,torch.tensor([[0,1,2],[1,2,4]]),active)
    assert probe['calls']==dict(history=1,candidate=1)
    assert probe['effective_mask_verified'] and scores.shape==(2,5)
    assert 'encode_context' not in a.model.__dict__ and 'score_full_catalog' not in a.model.__dict__
    assert np.isfinite(scores).all()


def test_broken_wrapper_reproduces_v1_and_is_rejected():
    class BrokenWrapper:
        def __init__(self,a):self.adapter=a
        def __getattr__(self,name):return getattr(self.adapter,name)
        def predict(self,*args):return self.adapter.predict(*args)
    a=BrokenWrapper(adapter());active,_=r.make_mask(catalog()[1],'category',.8,42)
    with pytest.raises(ValueError,match='did not reach'):
        r.probe_adapter(a,torch.tensor([[0,1,2]]),active)
    assert 'encode_context' not in a.model.__dict__


@pytest.fixture
def run_fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(r,'SEEDS',(42,));monkeypatch.setattr(r,'MODELS',('early_sd',));monkeypatch.setattr(r,'SOURCES',('category',))
    attrs,masks=catalog()
    samples=[SequenceSample(0,[0,1,2],4,2),SequenceSample(1,[0,2,4],5,2)]
    data=dict(attrs=attrs,masks=masks,signature={'fixture':1},split=SimpleNamespace(test_samples=samples))
    return SimpleNamespace(output=tmp_path/'runs/v3',config=dict(batch_size=2,missing_rates=[0.,.8]),
        foundation=lambda *args:None,guard=lambda **kwargs:None,dataset=lambda key:data,
        model_hash=lambda *args:'fixture-model-sha',load_job=lambda *args:adapter())


def test_complete_evidence_is_reproducible_and_cache_never_loads_model(run_fixture):
    run=run_fixture;out=r.run_robustness(run,'beauty')
    assert out['status']=='PASS' and out['rows']==2
    root=run.output/'beauty';binding=json.loads((root/'robustness_binding.json').read_text())
    assert binding['evaluation_version']==3
    summary=pd.read_csv(root/'robustness.csv')
    assert list(summary.removed_items)==[0,3]
    assert summary.iloc[1].candidate_masked_positions==6
    assert summary.iloc[1].masked_history_tokens>0
    paired=root/'robustness_evidence_v3/early_sd_42/category_0.80/paired_logit_probe.csv'
    frame=pd.read_csv(paired);assert len(frame)==10 and (frame.difference!=0).any()
    assert r.validate_robustness(run,'beauty')['status']=='PASS'
    before=binding['evidence_sha256']
    run.load_job=lambda *args:pytest.fail('Completed evidence must not reload S1')
    assert r.run_robustness(run,'beauty')['status']=='PASS'
    assert before==json.loads((root/'robustness_binding.json').read_text())['evidence_sha256']


def test_tampered_mask_rejected_without_gpu(run_fixture):
    run=run_fixture;r.run_robustness(run,'beauty');root=run.output/'beauty'
    path=next((root/'robustness_evidence_v3/masks').glob('*.csv'))
    path.write_text(path.read_text()+'\n')
    with pytest.raises(ValueError,match='changed'):
        r.inspect_evidence(root)


def test_identical_logits_require_explanation_not_fake_degradation(run_fixture):
    class FlatScores(TinyScores):
        def score_full_catalog(self,context,**kwargs):
            return torch.arange(5,dtype=torch.float32)[None,:].expand(len(context),-1)
    attrs,masks=catalog()
    run=run_fixture;run.load_job=lambda *args:ModelAdapter(FlatScores(),ModelSpec('early',42),attrs,masks)
    out=r.run_robustness(run,'beauty')
    assert out['status']=='REVIEW_REQUIRED' and out['pending_cases']==['early_sd_42/category']
    with pytest.raises(ValueError,match='requires recorded review'):
        r.validate_robustness(run,'beauty')
    root=run.output/'beauty'
    review=dict(diagnostics_sha256=r.sha(root/'robustness_diagnostics.csv'),
                reviewer='Người kiểm định fixture',reviewed_at='2026-11-10T15:00:00+07:00',
                case_reasons={'early_sd_42/category':'Mô hình giả lập cố ý giữ điểm ứng viên hằng số, bỏ qua metadata.'})
    (root/'robustness_review.json').write_text(json.dumps(review,ensure_ascii=False),encoding='utf-8')
    assert r.validate_robustness(run,'beauty')['review_accepted']
    review['diagnostics_sha256']='stale';(root/'robustness_review.json').write_text(json.dumps(review,ensure_ascii=False),encoding='utf-8')
    with pytest.raises(ValueError,match='stale'):
        r.inspect_evidence(root)


def test_flat_metric_with_changed_logits_is_not_automatically_a_bug(run_fixture):
    class OffsetScores(TinyScores):
        def score_full_catalog(self,context,**kwargs):
            return torch.arange(5,dtype=torch.float32)[None,:].expand(len(context),-1)+kwargs['catalog_category_mask'].sum()
    attrs,masks=catalog();run=run_fixture
    run.load_job=lambda *args:ModelAdapter(OffsetScores(),ModelSpec('early',42),attrs,masks)
    assert r.run_robustness(run,'beauty')['status']=='PASS'
    d=pd.read_csv(run.output/'beauty/robustness_diagnostics.csv').iloc[0]
    assert d.flat_metric_curve and d.logits_changed_0_vs_80>0
    assert d.status=='VERIFIED_EFFECT_FLAT_METRIC'


def test_no_available_source_is_not_a_verified_robustness_effect(run_fixture):
    run=run_fixture;data=run.dataset('beauty');data['masks']['category'].zero_()
    attrs,masks=catalog();masks['category'].zero_()
    run.load_job=lambda *args:ModelAdapter(TinyScores(),ModelSpec('early',42),attrs,masks)
    assert r.run_robustness(run,'beauty')['status']=='PASS'
    d=pd.read_csv(run.output/'beauty/robustness_diagnostics.csv').iloc[0]
    assert d.status=='NOT_APPLICABLE_NO_EFFECTIVE_REMOVAL'


def test_diagnostic_claims_cannot_bypass_actual_unchanged_logits(run_fixture):
    run=run_fixture;r.run_robustness(run,'beauty');root=run.output/'beauty'
    p=root/'robustness_diagnostics.csv';d=pd.read_csv(p);d['logits_changed_0_vs_80']=0
    d.to_csv(p,index=False,lineterminator='\n')
    b=root/'robustness_binding.json';binding=json.loads(b.read_text());binding['evidence_sha256']['robustness_diagnostics.csv']=r.sha(p)
    b.write_text(json.dumps(binding))
    with pytest.raises(ValueError,match='Diagnostic claims'):
        r.inspect_evidence(root)


def test_oov_semantics_reject_self_consistently_rehashed_rank(run_fixture):
    run=run_fixture;run.dataset('beauty')['split'].test_samples[0].target=0
    r.run_robustness(run,'beauty');root=run.output/'beauty'
    summary_path=root/'robustness.csv';summary=pd.read_csv(summary_path)
    row=summary.iloc[0];marker=root/row.case_manifest
    rec=json.loads(marker.read_text(encoding='utf-8'))
    relative=next(p for p in rec['files'] if p.endswith('/per_user_test.csv'))
    path=root/relative;ranks=pd.read_csv(path)
    assert ranks.iloc[0].target_idx==0 and ranks.iloc[0]['rank']==6
    ranks.loc[0,'rank']=1;ranks.to_csv(path,index=False,lineterminator='\n')
    rec['files'][relative]=r.sha(path);marker.write_text(json.dumps(rec),encoding='utf-8')
    summary.loc[0,'rank_sha256']=r.sha(path);summary.to_csv(summary_path,index=False,lineterminator='\n')
    binding_path=root/'robustness_binding.json';binding=json.loads(binding_path.read_text(encoding='utf-8'))
    binding['evidence_sha256'][relative]=r.sha(path)
    binding['evidence_sha256'][row.case_manifest]=r.sha(marker)
    binding['sha256']=r.sha(summary_path);binding_path.write_text(json.dumps(binding),encoding='utf-8')
    with pytest.raises(ValueError,match='OOV targets must retain'):
        r.inspect_evidence(root)
