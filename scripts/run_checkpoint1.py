"""Resumable, real-data execution for the supervisor's 29 October checkpoint."""
import argparse
import copy
import itertools
import json
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from metadata_pipeline.contracts import read_json, write_json, require
from metadata_pipeline.loader import load_catalog
from taimma.data.ingestion import Amazon2014Ingestor
from taimma.data.split import SequenceSplitter, compute_file_sha256
from taimma.evaluation.evaluator import IndependentEvaluator
from taimma.evaluation.baselines import RandomBaseline, PopularityBaseline
from taimma.evaluation.utility import compute_leave_one_source_out_deltas, save_utility_artifact
from taimma.experiments.diagnostics import compute_marginal_utility_and_redundancy
from taimma.experiments.logger import ExperimentLogger
from taimma.experiments.reporting import runtime_snapshot
from taimma.models.sasrec import SASRecModel
from taimma.models.late_fusion import SourceSequentialEncoder, ResidualLateFusionModel
from taimma.training.trainer import SASRecTrainer, set_seed
from taimma.training.multimodal_trainer import ResidualLateFusionTrainer, MultimodalSequenceDataset, collate_multimodal_batch, gather_source_sequences
from torch.utils.data import DataLoader

SOURCES = ('category','brand','title')
POLICIES = ('none','train_base_exclude_target','input_history_exclude_target','input_history_strict')


def evaluator(n,policy='input_history_strict'):
    return IndependentEvaluator(n,mask_policy=policy)


def atomic_json(path,value):
    temp = path.with_suffix('.tmp.json')
    write_json(temp,value)
    temp.replace(path)


def metrics(result):
    return result.to_dict()


def evaluate_predictor(predict,samples,n,seen,policy='input_history_strict',batch_size=256,targets=None):
    ev = evaluator(n,policy)
    records=[]
    with torch.no_grad():
        for start in range(0,len(samples),batch_size):
            batch=samples[start:start+batch_size]
            result=ev.evaluate_ranks(predict(batch),[s.target for s in batch] if targets is None else targets[start:start+len(batch)],
                    [s.user_idx for s in batch],seen,histories=[s.history for s in batch])
            records.extend(result.per_user_records)
    return ev.result_from_records(records)


class Checkpoint:
    def __init__(self,args):
        self.args=args
        self.data_mode="real"
        self.root=(ROOT/args.output).resolve()
        self.root.mkdir(parents=True,exist_ok=True)
        self.device=torch.device(args.device)
        if self.device.type == 'cuda' and not args.prepare_only and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; choose an explicitly supported execution environment')
        torch.set_num_threads(args.threads)
        self.logger=ExperimentLogger(str(self.root/'runs'),str(self.root/(f'results_{args.train_job}.csv' if args.train_job else 'results.csv')))
        self.progress=read_json(self.root/'progress.json') if (self.root/'progress.json').exists() else {}
        for p in (self.root/'completed').glob('*.json'):
            self.progress[p.stem]=read_json(p)
        signature={str(p.relative_to(ROOT)):compute_file_sha256(str(p)) for folder in ['src','scripts','configs']
                   for p in sorted((ROOT/folder).rglob('*')) if p.suffix in ('.py','.yaml') and '__pycache__' not in p.parts}
        lock=dict(code_sha256=signature,seeds=args.seeds,max_epochs=args.epochs,patience=args.patience,
                  learning_rate=0.001,batch_size=args.batch_size,main_mask_policy='input_history_strict',
                  loo_reference_mask_policy='none',source_dropout=0.2,device=args.device,
                  metadata_manifest_sha256={key:compute_file_sha256(str(ROOT/f'artifacts/metadata/{key}/manifest.json')) for key in ['beauty','sports','toys']})
        lock_path=self.root/'protocol.lock.json'
        if lock_path.exists():
            require(read_json(lock_path)==lock,'Code/config/data changed; use a new --output to preserve previous runs')
        else:
            write_json(lock_path,lock)
            ExperimentLogger._snapshot_source(self.root)
        if args.reuse_completed_from and not (self.root/'reused_stages.json').exists():
            require(not self.progress, 'Reuse is only allowed into a fresh output directory')
            from taimma.experiments.checkpoint_state import reuse_completed_id_stages
            self.progress.update(reuse_completed_id_stages((ROOT/args.reuse_completed_from).resolve(),self.root,
                    lock,Path(__file__).read_text(encoding='utf-8')))
            self.save_progress()
        self.ingestor=Amazon2014Ingestor(str(ROOT/'data/raw'),'Beauty')
        self.reviews,_=self.ingestor.resolve_paths()
        self.sequences,_=self.ingestor.build_interaction_sequences(include_metadata=False)
        require(len(self.sequences)==22363 and sum(len(s.items) for s in self.sequences)==198502,'Beauty cohort mismatch')
        self.split=SequenceSplitter().split(self.sequences)
        require(len(self.split.train_base_samples)==109050 and self.split.num_items==11998,'Four-segment split mismatch')
        self.attrs,self.masks=load_catalog(ROOT/'artifacts/metadata/beauty')
        self.backbone_dim=self.progress.get('backbone_dim',64)

    def save_progress(self):
        atomic_json(self.root/'progress.json',self.progress)

    def model(self,split,seed,dim):
        set_seed(seed)
        return SASRecModel(split.num_items,max_seq_len=50,embedding_dim=dim,num_blocks=2,num_heads=2,dropout_rate=.2)

    def residual_model(self,id_model,seed,sources):
        set_seed(seed)
        encoders={s:SourceSequentialEncoder(s,int(self.attrs[s].shape[1]) if s=='title' else int(self.attrs[s].max()),
                           feature_dim=self.backbone_dim,max_seq_len=50,num_blocks=1,num_heads=2,dropout_rate=.2,
                           is_dense_feature=s=='title') for s in sources}
        return ResidualLateFusionModel(id_model,encoders,p_drop=.2,sources_enabled=sources)

    def train(self,key,split,seed,dim=64,sources=(),id_model=None,policy='input_history_strict'):
        if key in self.progress:
            entry=self.progress[key]
            require(Path(entry['checkpoint']).is_file(),'Completed checkpoint disappeared')
            if entry.get('checkpoint_sha256'):
                require(compute_file_sha256(entry['checkpoint'])==entry['checkpoint_sha256'],'Completed checkpoint checksum changed')
            return entry
        model=self.model(split,seed,dim) if not sources else self.residual_model(copy.deepcopy(id_model),seed,sources)
        run_id,folder=self.logger.create_run_dir(key)
        folder=Path(folder)
        config=dict(dataset='Beauty' if self.data_mode=='real' else 'SyntheticFixture',data_mode=self.data_mode,protocol='loo' if split is not self.split else 'four_segment',seed=seed,
                    embedding_dim=dim,sources=list(sources),mask_policy=policy,epochs=self.args.epochs,
                    patience=self.args.patience,batch_size=self.args.batch_size,learning_rate=.001,device=str(self.device))
        with self.logger.capture_output(str(folder)):
            print(f'[{key}] training {run_id}',flush=True)
            ev=evaluator(split.num_items,policy)
            if sources:
                trainer=ResidualLateFusionTrainer(model,ev,{s:self.attrs[s] for s in sources},{s:self.masks[s] for s in sources},
                         device=str(self.device),batch_size=self.args.batch_size,patience=self.args.patience,seed=seed)
                # The source projections start at zero, so this must be exactly the ID system.
                sample=torch.tensor([s.history for s in split.val_samples[:2]],device=self.device)
                dataset=MultimodalSequenceDataset(split.val_samples[:2],self.attrs,self.masks)
                batch=collate_multimodal_batch([dataset[0],dataset[1]])
                model.eval()
                with torch.no_grad():
                    joint=model.predict_all(sample,{s:batch['source_seqs'][s].to(self.device) for s in sources},
                                trainer.catalog_source_attrs,trainer.catalog_availability_masks)
                    require(torch.equal(joint,model.id_model.predict_all(sample)),'Residual initialization != ID scores')
            else:
                trainer=SASRecTrainer(model,ev,device=str(self.device),batch_size=self.args.batch_size,
                                    patience=self.args.patience,seed=seed)
            history,_=trainer.fit(split.train_base_samples,split.val_samples,split.seen_train_items_per_user,
                         max_epochs=self.args.epochs,checkpoint_path=str(folder/'checkpoint.pt'))
            val=trainer.evaluate(split.val_samples,split.seen_train_items_per_user)
            test=trainer.evaluate(split.test_samples,split.seen_train_items_per_user)
            self.logger.save_run_artifacts(str(folder),run_id,config,val,test,history.total_training_time,
                  history.best_epoch,len(history.train_losses),actual_device=str(self.device))
            pd.DataFrame([dict(epoch=i+1,train_loss=loss,val_ndcg_10=history.val_ndcg_10[i],
                         train_sec=history.epoch_durations[i],wall_sec=history.epoch_wall_durations[i])
                         for i,loss in enumerate(history.train_losses)]).to_csv(folder/'epochs.csv',index=False)
            write_json(folder/'history.json',asdict(history))
            write_json(folder/'report_data.json',dict(data_mode=self.data_mode,costs=dict(
                  mean_epoch_train_sec=float(np.mean(history.epoch_durations)),
                  mean_epoch_wall_sec=float(np.mean(history.epoch_wall_durations)),
                  stop_reason=history.stop_reason,training_resources=history.resource_usage)))
            self.logger.register_result(run_id,'Beauty' ,'ResidualLateFusion' if sources else 'SASRec',seed,
                    len(history.train_losses),history.best_epoch,val,test,history.total_training_time,str(folder/'checkpoint.pt'))
        entry=dict(folder=str(folder),checkpoint=str(folder/'checkpoint.pt'),seed=seed,dim=dim,sources=list(sources),
                   validation=metrics(val),test=metrics(test),test_metrics_exact=dict(recall_10=test.recall_10,ndcg_10=test.ndcg_10),
                   checkpoint_sha256=compute_file_sha256(str(folder/'checkpoint.pt')),training_sec=history.total_training_time)
        self.progress[key]=entry
        completed=self.root/'completed'
        completed.mkdir(exist_ok=True)
        atomic_json(completed/f'{key}.json',entry)
        if not self.args.train_job:
            self.save_progress()
        return entry

    def load_id(self,entry,split=None):
        model=self.model(split or self.split,entry['seed'],entry['dim'])
        state=torch.load(entry['checkpoint'],map_location='cpu',weights_only=True)
        model.load_state_dict(state['model_state_dict'])
        return model.to(self.device).eval()

    def load_residual(self,entry):
        if entry.get('checkpoint_sha256'):
            require(compute_file_sha256(entry['checkpoint'])==entry['checkpoint_sha256'],'Residual checkpoint checksum changed')
        id_entry=self.progress[f'id_seed{entry["seed"]}']
        model=self.residual_model(self.load_id(id_entry),entry['seed'],tuple(entry['sources']))
        state=torch.load(entry['checkpoint'],map_location='cpu',weights_only=True)
        model.load_state_dict(state['model_state_dict'])
        return model.to(self.device).eval()

    def predict_id(self,model,batch):
        return model.predict_all(torch.tensor([s.history for s in batch],device=self.device))

    def predict_residual(self,model,batch,attrs=None,masks=None):
        attrs=attrs or self.attrs
        masks=masks or self.masks
        previous=getattr(self,'_catalog_cache_sources',None)
        if previous is None or previous[0] is not attrs or previous[1] is not masks:
            self._catalog_cache_sources=(attrs,masks)
            self._catalog_cache=({s:v.to(self.device) for s,v in attrs.items()},
                                 {s:v.to(self.device) for s,v in masks.items()})
        attrs_gpu,masks_gpu=self._catalog_cache
        selected_attrs={s:attrs_gpu[s] for s in model.source_encoders}
        selected_masks={s:masks_gpu[s] for s in model.source_encoders}
        ids=torch.tensor([sample.history for sample in batch],device=self.device)
        return model.predict_all(ids,gather_source_sequences(ids,selected_attrs,selected_masks),selected_attrs,selected_masks)

    def audit(self):
        summary={}
        for dataset in ['beauty','sports','toys']:
            directory=ROOT/'artifacts/metadata'/dataset
            manifest=read_json(directory/'manifest.json')
            for name,sha in manifest['artifact_sha256'].items():
                require(compute_file_sha256(str(directory/name))==sha,f'{dataset}: artifact changed: {name}')
            for kind in ['reviews','metadata']:
                path=Path(manifest[f'{kind}_file'])
                if not path.is_absolute():path=ROOT/path
                require(compute_file_sha256(str(path))==manifest[f'{kind}_sha256'],f'{dataset}: raw {kind} checksum changed')
        mapping=pd.read_csv(ROOT/'artifacts/metadata/beauty/metadata.csv',keep_default_na=False)
        require(mapping.raw_item_id.tolist()==[self.split.idx2item[i] for i in range(1,self.split.num_items+1)],'Metadata/ID catalog mismatch')
        summary['metadata_all_three']='VERIFIED_ARTIFACT_AND_RAW_SHA256'
        if 'audit' in self.progress:
            self.progress['audit']['metadata_all_three']=summary['metadata_all_three']
            self.save_progress()
            return
        for name,samples in [('validation',self.split.val_samples),('test',self.split.test_samples)]:
            frame=pd.DataFrame([dict(user_idx=s.user_idx,target=s.target,raw_target_asin=s.raw_target_asin,
                   target_is_oov=s.target<=0,target_in_input_history=s.target>0 and s.target in s.history) for s in samples])
            frame.to_csv(self.root/f'{name}_target_audit.csv',index=False)
            summary[name]=dict(total=len(frame),oov=int(frame.target_is_oov.sum()),
                               overlap=int(frame.target_in_input_history.sum()),denominator='all users, OOV retained as misses')
        checkpoint=ROOT/'runs/pilot_beauty_real_20261028_130530_787799_1f45262e/checkpoint.pt'
        if checkpoint.exists():
            model=self.model(self.split,42,64)
            model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['model_state_dict'])
            model.to(self.device).eval()
            sensitivity=[]
            for name,samples in [('validation',self.split.val_samples),('test',self.split.test_samples)]:
                for policy in POLICIES:
                    result=evaluate_predictor(lambda batch:self.predict_id(model,batch),samples,self.split.num_items,
                                              self.split.seen_train_items_per_user,policy)
                    sensitivity.append(dict(split=name,policy=policy,**metrics(result)))
            pd.DataFrame(sensitivity).to_csv(self.root/'same_checkpoint_mask_sensitivity.csv',index=False)
            summary['historical_checkpoint_sha256']=compute_file_sha256(str(checkpoint))
        self.progress['audit']=summary
        self.save_progress()

    def loo(self):
        split=SequenceSplitter(protocol='loo').split(self.sequences)
        require(len(split.train_base_samples)==131413,'LOO pair count must equal 198502 - 3*22363')
        for dim in [64,256]:
            entry=self.train(f'loo_d{dim}',split,42,dim,policy='none')
            exact=entry.get('test_metrics_exact',dict(recall_10=entry['test']['Recall@10'],ndcg_10=entry['test']['NDCG@10']))
            passed=exact['recall_10']>=.075 and exact['ndcg_10']>=.033
            entry['supervisor_condition_3']='PASS' if passed else 'FAIL'
            self.save_progress()
            if passed:
                self.backbone_dim=dim
                self.progress['backbone_dim']=dim
                self.progress['condition_3']='PASS'
                self.save_progress()
                return True
        self.progress['condition_3']='FAIL_STOP_GRID'
        self.save_progress()
        return False

    def id_scores(self,entry):
        model=None
        evidence={}
        checkpoint_sha=compute_file_sha256(entry['checkpoint'])
        for name,samples in [('train_meta',self.split.train_meta_samples),('validation',self.split.val_samples),('test',self.split.test_samples)]:
            destination=Path(entry['folder'])/f's_ID_{name}.npy'
            sidecar=destination.with_suffix('.rows.json')
            expected_rows=dict(users=[sample.user_idx for sample in samples],targets=[sample.target for sample in samples],
                               checkpoint_sha256=checkpoint_sha)
            if destination.exists():
                require(sidecar.is_file(),f'Incomplete ID cache (missing row evidence): {destination}')
                saved=read_json(sidecar)
                require(all(saved.get(k)==v for k,v in expected_rows.items()),f'ID cache row/checkpoint mismatch: {destination}')
                matrix=np.load(destination,mmap_mode='r',allow_pickle=False)
                require(matrix.shape==(len(samples),self.split.num_items) and matrix.dtype==np.float32,
                        f'ID cache shape/dtype mismatch: {destination}')
                for offset in range(0,len(samples),1024):
                    require(np.isfinite(matrix[offset:offset+1024]).all(),f'Non-finite ID cache: {destination}')
                model=model or self.load_id(entry)
                checks={0,(len(samples)//2//self.args.batch_size)*self.args.batch_size,
                        ((len(samples)-1)//self.args.batch_size)*self.args.batch_size}
                with torch.no_grad():
                    for offset in sorted(checks):
                        actual=self.predict_id(model,samples[offset:offset+self.args.batch_size]).cpu().numpy()
                        require(np.allclose(actual,matrix[offset:offset+len(actual)],atol=1e-5,rtol=1e-5),
                                f'ID cache scores do not match checkpoint: {destination}')
                del matrix
            else:
                import shutil
                require(shutil.disk_usage(destination.parent).free > len(samples)*self.split.num_items*4+512*1024**2,
                        'Insufficient disk space for lossless float32 ID-score cache')
                model=model or self.load_id(entry)
                temp=destination.with_suffix('.partial.npy')
                matrix=np.lib.format.open_memmap(temp,mode='w+',dtype='float32',shape=(len(samples),self.split.num_items))
                with torch.no_grad():
                    for offset in range(0,len(samples),self.args.batch_size):
                        matrix[offset:offset+self.args.batch_size]=self.predict_id(model,samples[offset:offset+self.args.batch_size]).cpu().numpy()
                matrix.flush();del matrix
                temp.rename(destination)
                write_json(sidecar,expected_rows)
            sha=compute_file_sha256(str(destination))
            evidence[name]=dict(path=str(destination),sha256=sha,checkpoint_sha256=checkpoint_sha,
                                rows=len(samples),dtype='float32',data_mode=self.data_mode)
        directory=self.root/'score_cache_integrity'
        directory.mkdir(exist_ok=True)
        evidence_path=directory/f'seed{entry["seed"]}.json'
        if evidence_path.exists():
            require(read_json(evidence_path)==evidence,'Previously verified ID cache bytes changed')
        else:
            write_json(evidence_path,evidence)

    def grid(self):
        for seed in self.args.seeds:
            id_entry=self.train(f'id_seed{seed}',self.split,seed,self.backbone_dim)
            self.id_scores(id_entry)
            # Measure the first full-source residual fit alone before concurrent jobs.
            first_key=f'late_CBT_seed{seed}'
            self.train(first_key,self.split,seed,self.backbone_dim,SOURCES,self.load_id(id_entry))
            jobs=[]
            for count in [3,1,2]:
                for sources in itertools.combinations(SOURCES,count):
                    key=f'late_{"".join(s[0].upper() for s in sources)}_seed{seed}'
                    if key not in self.progress:
                        jobs.append((key,sources))
            if self.args.workers == 1:
                for key,sources in jobs:
                    self.train(key,self.split,seed,self.backbone_dim,sources,self.load_id(id_entry))
            else:
                for offset in range(0,len(jobs),self.args.workers):
                    processes=[]
                    for key,sources in jobs[offset:offset+self.args.workers]:
                        cmd=[sys.executable,str(Path(__file__).resolve()),'--output',self.args.output,'--device',str(self.device),
                             '--epochs',str(self.args.epochs),'--patience',str(self.args.patience),'--batch_size',str(self.args.batch_size),
                             '--threads',str(self.args.threads),'--train_job',key,'--job_seed',str(seed),'--job_sources',*sources]
                        processes.append((key,subprocess.Popen(cmd,cwd=ROOT)))
                    for key,process in processes:
                        require(process.wait()==0,f'Residual job failed: {key}; logs remain in run directory')
                        self.progress[key]=read_json(self.root/'completed'/f'{key}.json')
                    self.save_progress()
            torch.cuda.empty_cache() if self.device.type=='cuda' else None
        rows=[]
        for seed in self.args.seeds:
            subsets={():self.progress[f'id_seed{seed}']['test']['NDCG@10']}
            rows.append(dict(seed=seed,sources='ID',NDCG_at_10=subsets[()]))
            for count in [1,2,3]:
                for sources in itertools.combinations(SOURCES,count):
                    entry=self.progress[f'late_{"".join(s[0].upper() for s in sources)}_seed{seed}']
                    subsets[tuple(sorted(sources))]=entry['test']['NDCG@10']
                    rows.append(dict(seed=seed,sources='+'.join(sources),NDCG_at_10=entry['test']['NDCG@10']))
            pd.DataFrame([asdict(x) for x in compute_marginal_utility_and_redundancy(subsets)]).to_csv(self.root/f'marginal_utility_seed{seed}.csv',index=False)
        frame=pd.DataFrame(rows)
        frame.to_csv(self.root/'diagnostic_B_per_seed.csv',index=False)
        frame.groupby('sources').NDCG_at_10.agg(['mean','std','count']).to_csv(self.root/'diagnostic_B_summary.csv')

    def diagnostics(self):
        destination=self.root/'robustness.csv'
        rows=pd.read_csv(destination).to_dict('records') if destination.exists() else []
        def key(row):
            return (int(row['seed']),row['source'],row['control'],None if row['control']=='permutation' else round(float(row['ratio']),2))
        done={key(row) for row in rows}
        require(len(done)==len(rows),'Duplicate saved robustness measurements')
        def persist():
            temporary=destination.with_suffix('.partial.csv')
            pd.DataFrame(rows).to_csv(temporary,index=False)
            temporary.replace(destination)
        for seed in self.args.seeds:
            entry=self.progress[f'late_CBT_seed{seed}']
            model=self.load_residual(entry)
            for source in SOURCES:
                perm=ROOT/f'artifacts/metadata/beauty/perturbations/permutations/perm_{source}_seed_{seed}.csv'
                for ratio in np.arange(0,0.81,.1):
                    measurement=(seed,source,'missing',round(float(ratio),2))
                    if measurement in done:continue
                    mask=ROOT/f'artifacts/metadata/beauty/perturbations/masks/mask_{source}_ratio_{ratio:.2f}_seed_{seed}.csv'
                    attrs,masks=load_catalog(ROOT/'artifacts/metadata/beauty',mask_files={source:mask})
                    result=evaluate_predictor(lambda batch:self.predict_residual(model,batch,attrs,masks),self.split.test_samples,
                             self.split.num_items,self.split.seen_train_items_per_user)
                    rows.append(dict(seed=seed,source=source,control='missing',ratio=round(float(ratio),2),**metrics(result)))
                    done.add(measurement);persist()
                if (seed,source,'permutation',None) in done:continue
                attrs,masks=load_catalog(ROOT/'artifacts/metadata/beauty',permutation_files={source:perm})
                result=evaluate_predictor(lambda batch:self.predict_residual(model,batch,attrs,masks),self.split.test_samples,
                             self.split.num_items,self.split.seen_train_items_per_user)
                rows.append(dict(seed=seed,source=source,control='permutation',ratio=None,**metrics(result)))
                done.add((seed,source,'permutation',None));persist()
        self.progress['diagnostics']='COMPLETED'
        self.save_progress()

    def utility(self):
        destination=self.root/'utility.csv'
        sidecar=destination.with_suffix('.csv.manifest.json')
        staging=destination.with_name('utility.partial.csv')
        staging_sidecar=staging.with_suffix('.csv.manifest.json')
        if destination.exists() and not sidecar.exists() and staging_sidecar.exists():
            require(read_json(staging_sidecar)['file_sha256']==compute_file_sha256(str(destination)),
                    'Interrupted utility promotion checksum mismatch')
            staging_sidecar.rename(sidecar)
        if destination.exists() and sidecar.exists():
            lineage=read_json(sidecar)
            require(lineage['file_sha256']==compute_file_sha256(str(destination)),'Utility artifact checksum changed')
            require(lineage['dataset_manifest_sha256']==compute_file_sha256(str(ROOT/'artifacts/metadata/beauty/manifest.json')),
                    'Utility metadata provenance changed')
            for seed in self.args.seeds[:2]:
                expected=self.progress[f'late_CBT_seed{seed}']['checkpoint']
                require(lineage['reference_checkpoints'][str(seed)]['sha256']==compute_file_sha256(expected),
                        'Utility reference provenance changed')
            frame=pd.read_csv(destination)
            frame.groupby('source_name').delta.describe(percentiles=[.05,.25,.5,.75,.95]).to_csv(self.root/'delta_distribution.csv')
            return destination
        # Preserve incomplete writes for inspection; do not overwrite an existing scientific result.
        existing=[path for path in [destination,sidecar,staging,staging_sidecar] if path.exists()]
        if existing:
            import uuid
            backup=self.root/'incomplete_utility'/uuid.uuid4().hex
            backup.mkdir(parents=True)
            for path in existing:path.rename(backup/path.name)
        records=[]
        checkpoints={}
        for seed in self.args.seeds[:2]:
            entry=self.progress[f'late_CBT_seed{seed}']
            checkpoints[seed]=entry['checkpoint']
            model=self.load_residual(entry)
            loader=DataLoader(MultimodalSequenceDataset(self.split.train_meta_samples,self.attrs,self.masks,defer_features=True),
                             batch_size=self.args.batch_size,shuffle=False,collate_fn=collate_multimodal_batch)
            attrs={s:v.to(self.device) for s,v in self.attrs.items()}
            masks={s:v.to(self.device) for s,v in self.masks.items()}
            for batch in loader:
                _,items=compute_leave_one_source_out_deltas(model,batch['item_seqs'].to(self.device),
                    gather_source_sequences(batch['item_seqs'].to(self.device),attrs,masks),attrs,masks,
                    batch['targets'].to(self.device),batch['user_indices'],reference_seed=seed)
                records.extend(items)
        save_utility_artifact(staging,records,ROOT/'artifacts/metadata/beauty/manifest.json',checkpoints)
        staging.rename(destination)
        staging_sidecar.rename(sidecar)
        frame=pd.read_csv(destination)
        frame.groupby('source_name').delta.describe(percentiles=[.05,.25,.5,.75,.95]).to_csv(self.root/'delta_distribution.csv')
        return destination

    def baselines(self):
        entry=self.progress[f'id_seed{self.args.seeds[0]}']
        model=self.load_id(entry)
        pop=PopularityBaseline(self.split.num_items)
        pop.fit(train_base_events=self.split.train_base_events)
        random=RandomBaseline(self.split.num_items,42)
        rows=[]
        samples=self.split.test_samples
        targets=np.asarray([s.target for s in samples])
        shuffled=targets[np.random.default_rng(42).permutation(len(targets))].tolist()
        for name,predict in [('Random',lambda batch:random.predict_all(len(batch))),
                             ('Popularity',lambda batch:pop.predict_all(len(batch))),
                             ('SASRec',lambda batch:self.predict_id(model,batch))]:
            for control,labels in [('original',None),('shuffled',shuffled)]:
                result=evaluate_predictor(predict,samples,self.split.num_items,self.split.seen_train_items_per_user,targets=labels)
                rows.append(dict(model=name,control=control,**metrics(result)))
        pd.DataFrame(rows).to_csv(self.root/'baseline_and_shuffle.csv',index=False)

    def run(self):
        if not self.args.prepare_only:
            for key,entry in self.progress.items():
                if key.startswith('id_seed'):
                    saved=read_json(Path(entry['folder'])/'run_manifest.json')
                    require(saved['torch_version']==torch.__version__,
                            'Reused ID checkpoint uses a different PyTorch build; use its recorded environment')
        if self.args.prepare_only:
            self.save_progress()
            plan=dict(data_mode='real',execution='PREPARED_ONLY_NO_TRAINING_OR_EVALUATION',
                device_target=str(self.device),completed_stages=list(self.progress),
                next_steps=['audit/LOO if not reused','throughput benchmark','independent ID and 7 late subsets x3 seeds',
                            'early CBT timing','reproducibility','robustness','two-reference utility and E0/E1'],
                resume_command=f'python scripts/run_checkpoint1.py --output {self.args.output} --device {self.device} --stage all --workers {self.args.workers}')
            write_json(self.root/'execution_plan.json',plan)
            print(json.dumps(plan,indent=2),flush=True)
            return 0
        if self.args.train_job:
            seed=self.args.job_seed
            sources=tuple(self.args.job_sources)
            self.train(self.args.train_job,self.split,seed,self.backbone_dim,sources,
                       self.load_id(self.progress[f'id_seed{seed}']))
            return 0
        self.audit()
        if not self.loo():
            print('Both LOO configurations failed condition (3). Diagnostic grid STOPPED as required.',flush=True)
            return 2
        if self.args.stage=='loo':return 0
        throughput=self.root/'throughput.json'
        if self.device.type != 'cuda':
            self.args.workers=1
            if not throughput.exists():
                write_json(throughput,dict(scope='CUDA concurrency not measured in CPU mode',measurements=[],best_workers=1))
        elif not throughput.exists() or {row['processes'] for row in read_json(throughput).get('measurements',[])} != {1,2,3}:
            subprocess.run([sys.executable,str(ROOT/'scripts/benchmark_checkpoint1.py'),'--dim',str(self.backbone_dim),
                            '--output',str(throughput),'--batch_size',str(self.args.batch_size),'--threads',str(self.args.threads)],check=True,cwd=ROOT)
        self.args.workers=min(self.args.workers,read_json(throughput)['best_workers'])
        self.grid()
        from checkpoint1_early import run_early
        run_early(self)
        self.baselines()
        repeated=self.train('id_seed42_repeat',self.split,42,self.backbone_dim)
        first=self.progress['id_seed42']
        import pandas.testing
        for name in ['per_user_val.csv','per_user_test.csv']:
            pandas.testing.assert_frame_equal(pd.read_csv(Path(first['folder'])/name),pd.read_csv(Path(repeated['folder'])/name))
        write_json(self.root/'reproducibility.json',dict(status='PASS',seed=42,first=first['checkpoint'],repeat=repeated['checkpoint']))
        if self.args.stage=='grid':return 0
        self.diagnostics()
        destination=self.utility()
        report=self.root/'e0_e1.json'
        if not report.exists():
            subprocess.run([sys.executable,str(ROOT/'scripts/run_e0_e1_evaluation.py'),'--delta_file',str(destination),
                 '--n_bootstrap','1000','--seed','42','--output_json',str(report)],check=True,cwd=ROOT)
        self.progress['checkpoint_experiments']='COMPLETED'
        self.save_progress()
        return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='runs/checkpoint1_20261101_batch')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--stage',choices=['loo','grid','all'],default='all')
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--patience',type=int,default=10)
    parser.add_argument('--batch_size',type=int,default=256)
    parser.add_argument('--seeds',type=int,nargs='+',default=[42,43,44])
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--train_job')
    parser.add_argument('--job_seed',type=int)
    parser.add_argument('--job_sources',nargs='+')
    parser.add_argument('--workers',type=int,choices=[1,2,3],default=2)
    parser.add_argument('--prepare-only',action='store_true',dest='prepare_only',help='Validate setup and prepare state only; never initialize CUDA or train')
    parser.add_argument('--reuse-completed-from',dest='reuse_completed_from',help='Explicitly reuse verified completed ID/LOO stages into a fresh output')
    args=parser.parse_args()
    require(all(v>0 for v in [args.epochs,args.patience,args.batch_size,args.threads]),'Epochs/patience/batch/threads must be positive')
    require(args.seeds==[42,43,44],'Checkpoint requires three seeds and independent references 42/43')
    raise SystemExit(Checkpoint(args).run())


if __name__=='__main__':main()
