"""Shared trained-epochs-only loop for the 03 November diagnostic experiments.

Importing this module never trains a model. The CLI additionally requires an
explicit execution flag and the locked RTX4060 environment before using fit.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import copy
import json
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

from ..models.sasrec import SASRecModel
from ..models.early_fusion import EarlyFusionSASRec
from ..models.late_fusion import ResidualLateFusionModel, SourceSequentialEncoder
from ..models.source_dropout import SOURCES, sample_source_weights
from ..evaluation.evaluator import IndependentEvaluator
from ..experiments.reporting import exact_validation_metrics, write_epoch_log, save_checkpoint_atomic, runtime_snapshot
from .trainer import TrainingHistory, set_seed
from .multimodal_trainer import MultimodalSequenceDataset, collate_multimodal_batch, gather_source_sequences


@dataclass(frozen=True)
class ModelSpec:
    kind: str
    seed: int
    sources: tuple = SOURCES
    embedding_dim: int = 64
    max_seq_len: int = 50
    dropout: float = .2
    source_dropout: float = 0.
    all_off_probability: float = 0.
    source_dropout_mode: str = 'sample'
    freeze_id: bool = True
    learning_rate: float = .001
    weight_decay: float = 0.


def build_model(spec, num_items, attrs, id_model=None):
    set_seed(spec.seed)
    base = id_model if id_model is not None else SASRecModel(
        num_items,max_seq_len=spec.max_seq_len,embedding_dim=spec.embedding_dim,
        num_blocks=2,num_heads=2,dropout_rate=.2)
    if spec.kind=='id':
        return base
    if spec.kind=='early':
        return EarlyFusionSASRec(base,num_categories=int(attrs['category'].max()),
            num_brands=int(attrs['brand'].max()),title_feature_dim=attrs['title'].shape[1],
            sources_enabled=tuple(spec.sources),dropout_rate=spec.dropout)
    if spec.kind!='late':
        raise ValueError('Unknown model kind')
    encoders={s:SourceSequentialEncoder(s,attrs[s].shape[1] if s=='title' else int(attrs[s].max()),
        feature_dim=spec.embedding_dim,max_seq_len=spec.max_seq_len,num_blocks=1,num_heads=2,
        dropout_rate=spec.dropout,is_dense_feature=s=='title') for s in spec.sources}
    return ResidualLateFusionModel(base,encoders,p_drop=spec.source_dropout,
        sources_enabled=tuple(spec.sources),freeze_id=spec.freeze_id,
        source_dropout_mode=spec.source_dropout_mode,p_all_off=spec.all_off_probability)


class ModelAdapter:
    def __init__(self,model,spec,attrs,masks,device='cpu'):
        self.device=torch.device(device)
        self.model=model.to(self.device);self.spec=spec
        self.attrs={s:v.to(self.device) for s,v in attrs.items()}
        self.masks={s:v.to(self.device) for s,v in masks.items()}

    def weights(self,batch_size,enabled=None):
        return sample_source_weights(batch_size,self.spec.sources if enabled is None else enabled,
            self.device,training=False)

    def availability(self,ids):
        ids=ids.to(self.device);indices=(ids-1).clamp(min=0)
        return torch.stack([((self.masks[s][indices]>0)&(ids>0)).any(dim=1)
            for s in SOURCES],dim=1).to(torch.float32)

    def predict(self,ids,weights=None):
        ids=ids.to(self.device)
        if self.spec.kind=='id':
            return self.model.predict_all(ids)
        hist=gather_source_sequences(ids,self.attrs,self.masks)
        if self.spec.kind=='late':
            return self.model.compute_all_scores(ids,hist,self.attrs,self.masks,
                apply_dropout=self.model.training,source_weights=weights)[0]
        if weights is None:
            weights=sample_source_weights(ids.size(0),self.spec.sources,self.device,
                self.spec.source_dropout,self.spec.all_off_probability,self.model.training)
        indices=(ids-1).clamp(min=0)
        active={s:self.masks[s][indices]*(ids>0) for s in SOURCES}
        context=self.model.encode_context(ids,category_ids=hist['category'],brand_ids=hist['brand'],
            title_features=hist['title'],category_mask=active['category'],brand_mask=active['brand'],
            title_mask=active['title'],source_weights=weights)
        return self.model.score_full_catalog(context,catalog_category_ids=self.attrs['category'],
            catalog_brand_ids=self.attrs['brand'],catalog_title_features=self.attrs['title'],
            catalog_category_mask=self.masks['category'],catalog_brand_mask=self.masks['brand'],
            catalog_title_mask=self.masks['title'],source_weights=weights)

    def context(self,ids,weights=None):
        """Ungated frozen reference state for inference-available psi only."""
        ids=ids.to(self.device)
        if self.spec.kind=='id':return self.model.encode_context(ids)
        if self.spec.kind=='late':return self.model.id_model.encode_context(ids)
        hist=gather_source_sequences(ids,self.attrs,self.masks);idx=(ids-1).clamp(min=0)
        active={s:self.masks[s][idx]*(ids>0) for s in SOURCES}
        return self.model.encode_context(ids,category_ids=hist['category'],brand_ids=hist['brand'],
            title_features=hist['title'],category_mask=active['category'],brand_mask=active['brand'],
            title_mask=active['title'],source_weights=self.weights(ids.size(0)) if weights is None else weights)


def batches(samples,attrs,masks,batch_size,shuffle=False,seed=42):
    dataset=MultimodalSequenceDataset(samples,attrs,masks,defer_features=True)
    return DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),collate_fn=collate_multimodal_batch)


def evaluate(adapter,samples,batch_size=256,enabled=None,availability_aware=False):
    adapter.model.eval();ev=IndependentEvaluator(adapter.model.num_items,mask_policy='input_history_strict')
    records=[]
    with torch.no_grad():
        for b in batches(samples,adapter.attrs,adapter.masks,batch_size):
            weights=None if enabled is None else adapter.weights(b['item_seqs'].size(0),enabled)
            if availability_aware:
                weights=(adapter.weights(b['item_seqs'].size(0)) if weights is None else weights)*adapter.availability(b['item_seqs'])
            result=ev.evaluate_ranks(adapter.predict(b['item_seqs'],weights),b['targets'].tolist(),
                b['user_indices'],histories=b['item_seqs'].tolist())
            records.extend(result.per_user_records)
    return ev.result_from_records(records)


def fit(adapter,train_samples,val_samples,folder,epochs=100,patience=20,batch_size=256,
        max_wall_seconds=None):
    if not train_samples or any(s.target<=0 for s in train_samples):
        raise ValueError('Training requires nonempty in-catalog labels')
    if epochs<1 or patience<1:raise ValueError('Positive epoch/patience required')
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    if (folder/'checkpoint.pt').exists():
        raise FileExistsError('Refuse to overwrite a model checkpoint; resume verified completed jobs instead')
    set_seed(adapter.spec.seed)
    optimizer=torch.optim.Adam([p for p in adapter.model.parameters() if p.requires_grad],
        lr=adapter.spec.learning_rate,weight_decay=adapter.spec.weight_decay)
    history=TrainingHistory();history.stop_reason='running'
    t=time.perf_counter();initial=evaluate(adapter,val_samples,batch_size)
    history.initial_validation_metrics=exact_validation_metrics(initial)
    history.initial_validation_duration=time.perf_counter()-t
    write_epoch_log(folder,history)
    best=-1.;best_state=None;stale=0;begin=time.perf_counter()
    # A single persistent loader: deterministic, fresh shuffling each epoch.
    loader=batches(train_samples,adapter.attrs,adapter.masks,batch_size,True,adapter.spec.seed)
    for epoch in range(1,epochs+1):
        if max_wall_seconds is not None and time.perf_counter()-begin>=max_wall_seconds:
            history.stop_reason='diagnostic_time_limit';break
        start=time.perf_counter();adapter.model.train();loss_sum=0.;count=0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            y=batch['targets'].to(adapter.device)-1
            loss=torch.nn.functional.cross_entropy(adapter.predict(batch['item_seqs']),y)
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            loss.backward();optimizer.step();loss_sum+=float(loss.detach())*len(y);count+=len(y)
        if adapter.device.type=='cuda':torch.cuda.synchronize(adapter.device)
        train_duration=time.perf_counter()-start
        result=evaluate(adapter,val_samples,batch_size)
        history.train_losses.append(loss_sum/count);history.val_ndcg_10.append(result.ndcg_10)
        history.val_recall_10.append(result.recall_10);history.epoch_durations.append(train_duration)
        history.val_metrics_by_epoch.append(exact_validation_metrics(result))
        if result.ndcg_10>best:
            best=result.ndcg_10;history.best_epoch=epoch;history.best_val_ndcg=best;stale=0
            best_state=copy.deepcopy(adapter.model.state_dict())
            save_checkpoint_atomic(folder/'checkpoint.pt',dict(epoch=epoch,model_state_dict=best_state,
                val_ndcg_10=best,seed=adapter.spec.seed,selection_scope='trained_epochs_only',spec=asdict(adapter.spec)))
        else:stale+=1
        history.epoch_wall_durations.append(time.perf_counter()-start)
        print(f'Epoch {epoch:03d} | loss={loss_sum/count:.6f} | Val NDCG@10={result.ndcg_10:.10f}',flush=True)
        write_epoch_log(folder,history)
        if stale>=patience:history.stop_reason='early_stopping';break
    if best_state is None:raise RuntimeError('No trained epoch completed within diagnostic time budget')
    if history.stop_reason=='running':history.stop_reason='max_epochs'
    history.total_training_time=time.perf_counter()-begin;history.resource_usage=runtime_snapshot(adapter.device)
    adapter.model.load_state_dict(best_state);adapter.model.eval();write_epoch_log(folder,history)
    return history
