"""First real CBT early-fusion run for measured checkpoint resource budgeting."""
import copy
import time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from metadata_pipeline.contracts import write_json
from taimma.models.early_fusion import EarlyFusionSASRec
from taimma.training.trainer import TrainingHistory,set_seed
from taimma.training.multimodal_trainer import MultimodalSequenceDataset,collate_multimodal_batch,gather_source_sequences
from taimma.experiments.reporting import runtime_snapshot
from taimma.evaluation.evaluator import IndependentEvaluator


def run_early(runner):
    key='early_CBT_seed42'
    if key in runner.progress:return runner.progress[key]
    set_seed(42)
    model=EarlyFusionSASRec(runner.model(runner.split,42,runner.backbone_dim),
          num_categories=int(runner.attrs['category'].max()),num_brands=int(runner.attrs['brand'].max()),
          title_feature_dim=runner.attrs['title'].shape[1]).to(runner.device)
    attrs={s:v.to(runner.device) for s,v in runner.attrs.items()}
    masks={s:v.to(runner.device) for s,v in runner.masks.items()}
    def loader(samples,shuffle=False):
        return DataLoader(MultimodalSequenceDataset(samples,runner.attrs,runner.masks,defer_features=True),batch_size=runner.args.batch_size,
                 shuffle=shuffle,collate_fn=collate_multimodal_batch,generator=torch.Generator().manual_seed(42))
    def predict(batch):
        ids=batch['item_seqs'].to(runner.device)
        history=gather_source_sequences(ids,attrs,masks)
        indices=(ids-1).clamp(min=0)
        active={s:masks[s][indices]*(ids>0) for s in masks}
        ctx=model.encode_context(ids,category_ids=history['category'],brand_ids=history['brand'],
              title_features=history['title'],category_mask=active['category'],brand_mask=active['brand'],title_mask=active['title'])
        return model.score_full_catalog(ctx,catalog_category_ids=attrs['category'],catalog_brand_ids=attrs['brand'],
              catalog_title_features=attrs['title'],catalog_category_mask=masks['category'],
              catalog_brand_mask=masks['brand'],catalog_title_mask=masks['title'])
    ev=IndependentEvaluator(runner.split.num_items,mask_policy='input_history_strict')
    def evaluate(samples):
        model.eval()
        records=[]
        with torch.no_grad():
            for batch in loader(samples):
                result=ev.evaluate_ranks(predict(batch),batch['targets'].tolist(),batch['user_indices'],
                                         histories=batch['item_seqs'].tolist())
                records.extend(result.per_user_records)
        return ev.result_from_records(records)
    run_id,folder=runner.logger.create_run_dir(key)
    folder=Path(folder)
    history=TrainingHistory()
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    best=-1;weights=None;stale=0
    config=dict(dataset='Beauty',mechanism='early_fusion',sources=['category','brand','title'],seed=42,
                embedding_dim=runner.backbone_dim,mask_policy='input_history_strict',device=str(runner.device))
    with runner.logger.capture_output(str(folder)):
        if runner.device.type=='cuda':torch.cuda.reset_peak_memory_stats()
        begin=time.perf_counter()
        for epoch in range(1,runner.args.epochs+1):
            then=time.perf_counter();model.train();losses=[]
            for batch in loader(runner.split.train_base_samples,True):
                optimizer.zero_grad()
                loss=torch.nn.functional.cross_entropy(predict(batch),batch['targets'].to(runner.device)-1)
                loss.backward();optimizer.step();losses.append(loss.item())
            if runner.device.type=='cuda':torch.cuda.synchronize()
            train_sec=time.perf_counter()-then
            val=evaluate(runner.split.val_samples)
            history.train_losses.append(float(np.mean(losses)));history.val_ndcg_10.append(val.ndcg_10)
            history.val_recall_10.append(val.recall_10);history.epoch_durations.append(train_sec)
            history.epoch_wall_durations.append(time.perf_counter()-then)
            print(f'Early CBT epoch {epoch}: loss={np.mean(losses):.5f}; val NDCG@10={val.ndcg_10:.6f}; train_sec={train_sec:.2f}',flush=True)
            if val.ndcg_10>best:
                best=val.ndcg_10;weights=copy.deepcopy(model.state_dict());stale=0
                history.best_epoch=epoch;history.best_val_ndcg=best
                torch.save(dict(model_state_dict=weights,epoch=epoch,seed=42),folder/'checkpoint.pt')
            else:stale+=1
            if stale>=runner.args.patience:
                history.stop_reason='early_stopping';break
        history.total_training_time=time.perf_counter()-begin
        history.resource_usage=runtime_snapshot(runner.device)
        model.load_state_dict(weights)
        val=evaluate(runner.split.val_samples);test=evaluate(runner.split.test_samples)
        runner.logger.save_run_artifacts(str(folder),run_id,config,val,test,history.total_training_time,
                        history.best_epoch,len(history.train_losses),actual_device=str(runner.device))
        write_json(folder/'history.json',asdict(history))
        write_json(folder/'report_data.json',dict(data_mode='real',costs=dict(
                    mean_epoch_train_sec=float(np.mean(history.epoch_durations)),mean_epoch_wall_sec=float(np.mean(history.epoch_wall_durations)),
                    stop_reason=history.stop_reason,training_resources=history.resource_usage)))
        pd.DataFrame(dict(epoch=range(1,len(history.train_losses)+1),loss=history.train_losses,
                   train_sec=history.epoch_durations,wall_sec=history.epoch_wall_durations,
                   val_ndcg_10=history.val_ndcg_10)).to_csv(folder/'epochs.csv',index=False)
        runner.logger.register_result(run_id,'Beauty','EarlyFusion_CBT',42,len(history.train_losses),history.best_epoch,
                  val,test,history.total_training_time,str(folder/'checkpoint.pt'))
    entry=dict(folder=str(folder),checkpoint=str(folder/'checkpoint.pt'),training_sec=history.total_training_time,
               validation=val.to_dict(),test=test.to_dict())
    runner.progress[key]=entry;runner.save_progress()
    return entry
