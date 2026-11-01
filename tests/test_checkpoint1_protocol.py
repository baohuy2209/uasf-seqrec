import sys
from pathlib import Path
import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from taimma.evaluation.evaluator import IndependentEvaluator
from taimma.data.ingestion import InteractionSequence
from taimma.data.split import SequenceSplitter
from taimma.models.sasrec import SASRecModel
from taimma.models.late_fusion import SourceSequentialEncoder,ResidualLateFusionModel


def test_strict_history_target_is_miss_even_when_catalog_smaller_than_k():
    ev=IndependentEvaluator(3,mask_policy='input_history_strict')
    result=ev.evaluate_ranks(np.array([[9.,8.,7.]]),[1],[0],histories=[[1,2,3]])
    assert result.ndcg_10==0 and result.recall_20==0


def test_histories_are_per_row_not_per_user():
    ev=IndependentEvaluator(4,mask_policy='input_history_strict')
    result=ev.evaluate_ranks(np.array([[10.,9.,8.,7.],[10.,9.,8.,7.]]),[2,2],[0,0],histories=[[1],[1,2]])
    assert result.per_user_records[0].rank==1
    assert result.per_user_records[1].ndcg_10==0
    with pytest.raises(ValueError,match='aligned history'):
        ev.evaluate_ranks(np.ones((1,4)),[1],[0])


def test_legacy_and_exclude_target_policies_remain_explicit():
    scores=np.array([[10.,9.,8.]])
    strict=IndependentEvaluator(3,mask_policy='input_history_strict')
    exclude=IndependentEvaluator(3,mask_policy='input_history_exclude_target')
    assert strict.evaluate_ranks(scores,[1],[0],histories=[[1,2]]).recall_10==0
    assert exclude.evaluate_ranks(scores,[1],[0],histories=[[1,2]]).recall_10==1
    assert IndependentEvaluator(3).evaluate_ranks(scores,[1],[0],{0:{1,2}}).recall_10==1


def test_loo_uses_prefix_through_n_minus_2_and_preserves_oov_denominator():
    seq=[InteractionSequence('U',list('ABCDEFG'),list(range(7)))]
    four=SequenceSplitter(max_seq_len=7).split(seq)
    loo=SequenceSplitter(max_seq_len=7,protocol='loo').split(seq)
    assert len(four.train_base_samples)==3
    assert len(loo.train_base_samples)==4
    assert set(loo.item2idx)==set('ABCDE')
    assert not loo.train_meta_samples
    assert len(loo.val_samples)==len(loo.test_samples)==1
    assert loo.val_samples[0].target==loo.test_samples[0].target==0


def test_residual_starts_at_exact_id_then_projection_can_learn():
    torch.manual_seed(42)
    encoder=SourceSequentialEncoder('brand',3,feature_dim=8,max_seq_len=3,num_heads=1)
    model=ResidualLateFusionModel(SASRecModel(3,max_seq_len=3,embedding_dim=8,num_heads=1),
                                 {'brand':encoder},sources_enabled=('brand',),p_drop=0)
    model.eval()
    ids=torch.tensor([[0,1,2]])
    attrs={'brand':torch.tensor([1,2,3])};masks={'brand':torch.ones(3)}
    seqs={'brand':torch.tensor([[0,1,2]])}
    total,base,_=model.compute_all_scores(ids,seqs,attrs,masks,apply_dropout=False)
    assert torch.equal(total,base)
    model.compute_loss(ids,seqs,attrs,masks,torch.tensor([3])).backward()
    assert torch.count_nonzero(encoder.output_projection.weight.grad)>0
    assert all(p.grad is None for p in model.id_model.parameters())

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_batch_metadata_gather_is_exactly_equal_to_original(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    from taimma.data.split import SequenceSample
    from taimma.training.multimodal_trainer import MultimodalSequenceDataset,collate_multimodal_batch,gather_source_sequences
    samples=[SequenceSample(0,[0,1,2,9],1,3),SequenceSample(1,[3,0,2,1],2,3)]
    attrs={'category':torch.tensor([[1,2],[0,0],[3,0]]),'brand':torch.tensor([1,0,2]),
           'title':torch.arange(12,dtype=torch.float32).reshape(3,4)}
    masks={'category':torch.tensor([1.,0.,1.]),'brand':torch.tensor([1.,0.,1.]),'title':torch.tensor([0.,1.,1.])}
    original=MultimodalSequenceDataset(samples,attrs,masks)
    expected=collate_multimodal_batch([original[i] for i in range(2)])
    deferred=MultimodalSequenceDataset(samples,attrs,masks,defer_features=True)
    batch=collate_multimodal_batch([deferred[i] for i in range(2)])
    gathered=gather_source_sequences(batch['item_seqs'].to(device),
                {s:v.to(device) for s,v in attrs.items()},{s:v.to(device) for s,v in masks.items()})
    for source in attrs:assert torch.equal(gathered[source].cpu(),expected['source_seqs'][source])
