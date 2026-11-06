"""Resumable stages for the supervisor feedback dated 03 November 2026.

Nothing trains or evaluates real models on import. CLI plan is the default;
training and inference require explicit flags plus clean code/environment.
"""
from dataclasses import asdict, replace
from pathlib import Path
from contextlib import contextmanager
import copy
import itertools
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
import numpy as np
import pandas as pd
import torch

from metadata_pipeline.loader import load_catalog
from metadata_pipeline.build_training_contrasts import build_training_contrasts_for_dataset
from metadata_pipeline.user_grouping import apply_group_rules
from ..data.ingestion import Amazon2014Ingestor
from ..data.split import SequenceSplitter
from ..training.checkpoint2_trainer import ModelSpec, ModelAdapter, build_model, fit, evaluate
from ..training.selective_fusion import SelectiveGate, train_gate, gate_evaluate
from ..evaluation.checkpoint2_utility import collect_ablations, score_contexts, paired_ce_evidence
from ..evaluation.e0_e1_protocol import make_utility_labels
from ..experiments.logger import ExperimentLogger
from .checkpoint2_protocol import (SEEDS,SOURCE_ORDER,SUBSETS,sha,digest,write_json,git_state,
    check_training_environment,context_signature,split_meta_users,read_ranks,group_contexts,
    grouped_subset_analysis,paired_seed_ci,choose_utility_label)

DATASETS={'beauty':'Beauty','sports':'Sports_and_Outdoors','toys':'Toys_and_Games'}


def subset_name(sources):
    return ''.join(s[0].upper() for s in sources) or 'ID'


def rank_frame(result):
    return pd.DataFrame([dict(user_idx=int(r.user_idx),target_idx=int(r.target_idx),rank=int(r.rank),
        **{'ndcg@10':float(r.ndcg_10),'recall@10':float(r.recall_10),
           'ndcg@20':float(r.ndcg_20),'recall@20':float(r.recall_20)}) for r in result.per_user_records])


@contextmanager
def exclusive(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    try:f=path.open('x',encoding='utf-8')
    except FileExistsError as exc:raise RuntimeError(f'Another job or an interrupted process owns {path}; inspect before removing its lock') from exc
    f.write(str(os.getpid()));f.close()
    try:yield
    finally:path.unlink()


class Checkpoint2:
    def __init__(self,repo,config,output,device='cuda',execute_training=False,execute_inference=False):
        self.repo=Path(repo).resolve();self.config=config;self.output=(self.repo/output).resolve()
        runs_root=(self.repo/'runs').resolve()
        if not self.output.is_relative_to(runs_root) or self.output==runs_root:
            raise ValueError('Use a dedicated output subdirectory under repo/runs; never use the repo or dataset root')
        self.device=device;self.execute_training=execute_training
        self.execute_inference=execute_inference or execute_training;self.data={}
        self.protocol_hash=digest(config)

    def guard(self,training=False,inference=False):
        if training and not self.execute_training:
            raise RuntimeError('Real training is disabled. Supply --execute-training on the RTX4060 host.')
        if inference and not self.execute_inference:
            raise RuntimeError('Model inference is disabled. Supply --execute-inference on the locked host.')
        if training or inference:
            git_state(self.repo,require_clean=True)
            check_training_environment(self.config['environment'])

    def dataset(self,key):
        if key in self.data:return self.data[key]
        ing=Amazon2014Ingestor(str(self.repo/self.config['raw_root']),DATASETS[key])
        sequences,_=ing.build_interaction_sequences(include_metadata=False)
        split=SequenceSplitter(max_seq_len=50).split(sequences)
        meta=self.repo/self.config['metadata_root']/key
        attrs,masks=load_catalog(meta)
        manifest=json.loads((meta/'manifest.json').read_text(encoding='utf-8'))
        metadata_hashes={}
        for rel,expected in manifest.get('artifact_sha256',{}).items():
            p=(meta/rel).resolve()
            if not p.is_relative_to(meta.resolve()) or not p.is_file() or sha(p)!=expected:
                raise ValueError(f'{key}: missing or modified metadata artifact {rel}')
            metadata_hashes[rel]=expected
        raw_hash=sha(ing.reviews_path)
        if manifest.get('reviews_sha256')!=raw_hash:raise ValueError('Raw reviews differ from metadata provenance')
        mapping=pd.read_csv(meta/'item_map.csv',keep_default_na=False)
        # Prepared metadata must retain exactly the split's raw-ID catalog.
        raw_col='raw_item_id' if 'raw_item_id' in mapping else 'asin'
        if raw_col not in mapping or mapping[raw_col].astype(str).tolist()!=[split.idx2item[i] for i in range(1,split.num_items+1)]:
            raise ValueError(f'{key}: catalog IDs do not match prepared metadata')
        d=dict(split=split,attrs=attrs,masks=masks,metadata_dir=meta)
        d['signature']=dict(raw_reviews_sha256=raw_hash,metadata_manifest_sha256=sha(meta/'manifest.json'),metadata_files=metadata_hashes,
            catalog_sha256=digest(split.idx2item),train_base_sha256=context_signature(split.train_base_samples),
            train_meta_sha256=context_signature(split.train_meta_samples),val_sha256=context_signature(split.val_samples),
            test_sha256=context_signature(split.test_samples),train_meta_eligible=len(split.train_meta_samples),
            train_meta_oov=split.oov_counts['train_meta'],train_meta_total=len(split.train_meta_samples)+split.oov_counts['train_meta'])
        self.data[key]=d;return d

    def source_signature(self):
        return {p.relative_to(self.repo).as_posix():sha(p) for root in ['src','scripts','configs']
            for p in sorted((self.repo/root).rglob('*')) if p.is_file() and p.suffix in ['.py','.yaml'] and '__pycache__' not in p.parts}

    def prepare(self,key):
        self.guard(inference=True)
        d=self.dataset(key);folder=self.output/key;folder.mkdir(parents=True,exist_ok=True)
        lock=dict(config=self.config,source_sha256=self.source_signature(),data=d['signature'],
            git=git_state(self.repo),environment=check_training_environment(self.config['environment']))
        path=folder/'protocol.lock.json'
        if path.exists():
            if json.loads(path.read_text(encoding='utf-8'))!=lock:
                raise RuntimeError('Code/config/data/runtime changed. Preserve this run and use a new --output.')
        else:write_json(path,lock)
        contrasts=folder/'training_contrasts'
        if not (contrasts/'manifest.json').exists():
            build_training_contrasts_for_dataset(key,str((self.repo/self.config['metadata_root']).resolve()),
                list(SEEDS),output_dir=contrasts)
        manifest=json.loads((contrasts/'manifest.json').read_text(encoding='utf-8'))
        for category in ['shuffled','random']:
            for rec in manifest[category].values():
                path=contrasts/category/rec['file']
                if not path.is_file() or sha(path)!=rec['sha256']:
                    raise ValueError('Missing or modified contrast artifact: '+str(path))
        for name,expected in manifest.get('input_sha256',{}).items():
            if sha(d['metadata_dir']/name)!=expected:raise ValueError('Contrast source changed: '+name)
        groups,summary=group_contexts(d['split'])
        groups.to_csv(folder/'groups.csv',index=False,lineterminator='\n')
        write_json(folder/'groups.json',dict(**summary,data_signature=d['signature']))
        rule=dict(rule='CE if all three nonzero signs match validation; else LOG_RANK if all match; else UNRESOLVED',
            locked_before_computation=True,test_used=False,zero_sign='inconclusive',feature_reference_seed=42)
        rule_path=folder/'utility_label_rule.json'
        if rule_path.exists():
            if json.loads(rule_path.read_text(encoding='utf-8'))!=rule:raise ValueError('Preregistered utility rule changed')
        else:write_json(rule_path,rule)
        return d

    def job_path(self,key,name):return self.output/key/'jobs'/name

    def completed(self,key,name):
        path=self.job_path(key,name)/'completed.json'
        if not path.is_file():return None
        rec=json.loads(path.read_text(encoding='utf-8'))
        for rel,h in rec['artifacts'].items():
            p=(path.parent/rel).resolve()
            if not p.is_relative_to(path.parent.resolve()):raise ValueError('Artifact path escapes job directory')
            if not p.is_file() or sha(p)!=h:raise ValueError('Completed job artifact changed: '+str(p))
        return rec

    def load_job(self,key,name):
        d=self.dataset(key);rec=self.completed(key,name)
        if rec is None:raise FileNotFoundError('Required completed model: '+name)
        spec=ModelSpec(**{**rec['spec'],'sources':tuple(rec['spec']['sources'])})
        model=build_model(spec,d['split'].num_items,d['attrs'])
        state=torch.load(self.job_path(key,name)/rec['checkpoint'],map_location='cpu',weights_only=True)
        model.load_state_dict(state['model_state_dict'])
        attrs,masks=self.catalog(key,rec.get('control'))
        return ModelAdapter(model,spec,attrs,masks,self.device)

    def model_hash(self,key,name):
        rec=self.completed(key,name)
        if rec is None:raise FileNotFoundError('Required completed model: '+name)
        return sha(self.job_path(key,name)/rec['checkpoint'])

    def catalog(self,key,control=None):
        d=self.dataset(key)
        if not control:return d['attrs'],d['masks']
        src,kind,seed=control;root=self.output/key/'training_contrasts'
        if kind=='shuffled':return load_catalog(d['metadata_dir'],permutation_files={src:root/'shuffled'/f'perm_{src}_seed_{seed}.csv'})
        suffix='npy' if src=='title' else 'csv'
        return load_catalog(d['metadata_dir'],random_files={src:root/'random'/f'random_{src}_seed_{seed}.{suffix}'})

    def run_job(self,key,name,spec,train_samples=None,id_job=None,control=None,max_wall_seconds=None):
        self.guard(training=True);d=self.dataset(key);split=d['split']
        train_samples=split.train_base_samples if train_samples is None else train_samples
        fingerprint=digest(dict(spec=asdict(spec),train=context_signature(train_samples),
            config=self.protocol_hash,data=d['signature'],control=control,id_job=id_job))
        existing=self.completed(key,name)
        if existing:
            if existing['fingerprint']!=fingerprint:raise ValueError('Completed job configuration mismatch: '+name)
            return existing
        root=self.job_path(key,name);root.mkdir(parents=True,exist_ok=True)
        with exclusive(root/'execution.lock'):
            existing=self.completed(key,name)
            if existing:return existing
            # Preserve failed/interrupted attempts rather than overwrite checkpoints.
            attempt=root/f'attempt_{len(list(root.glob("attempt_*")))+1:03d}'
            attempt.mkdir()
            attrs,masks=self.catalog(key,control)
            id_model=copy.deepcopy(self.load_job(key,id_job).model) if id_job else None
            model=build_model(spec,split.num_items,attrs,id_model)
            adapter=ModelAdapter(model,spec,attrs,masks,self.device)
            logger=ExperimentLogger(str(attempt.parent),str(attempt.parent/'unused_registry.csv'))
            with logger.capture_output(attempt):
                history=fit(adapter,train_samples,split.val_samples,attempt,
                    self.config['epochs'],self.config['patience'],self.config['batch_size'],max_wall_seconds)
                val=evaluate(adapter,split.val_samples,self.config['batch_size'])
                test=evaluate(adapter,split.test_samples,self.config['batch_size'])
                logger.save_run_artifacts(str(attempt),name,dict(spec=asdict(spec),dataset=key,control=control,
                    fingerprint=fingerprint,train_contexts=len(train_samples),train_users=len({s.user_idx for s in train_samples}),
                    data_signature=d['signature'],selection='trained_epochs_only'),val,test,
                    history.total_training_time,history.best_epoch,len(history.train_losses),actual_device=self.device,
                    history=history)
            rec=dict(spec=asdict(spec),control=control,fingerprint=fingerprint,
                checkpoint=f'{attempt.name}/checkpoint.pt',val_ndcg_10=val.ndcg_10,test_ndcg_10=test.ndcg_10,
                best_epoch=history.best_epoch,epochs=len(history.train_losses),stop_reason=history.stop_reason,
                train_contexts=len(train_samples),artifacts={p.relative_to(root).as_posix():sha(p)
                    for p in attempt.iterdir() if p.is_file()},status='COMPLETED')
            write_json(root/'completed.json',rec)
            del adapter,model
            if torch.cuda.is_available():torch.cuda.empty_cache()
            return rec

    def id_grid(self,key):
        for seed in SEEDS:self.run_job(key,f'id_{seed}',ModelSpec('id',seed,sources=()))

    def early_grid(self,key):
        self.id_grid(key)
        for seed,sources in itertools.product(SEEDS,SUBSETS[1:]):
            self.run_job(key,f'early_{subset_name(sources)}_{seed}',ModelSpec('early',seed,sources=sources))

    def controls(self,key):
        for seed,source,kind in itertools.product(SEEDS,SOURCE_ORDER,['shuffled','random']):
            self.run_job(key,f'control_{kind}_{source}_{seed}',ModelSpec('early',seed),control=(source,kind,seed))

    def source_dropout(self,key):
        for seed in SEEDS:
            self.run_job(key,f'early_sd_{seed}',ModelSpec('early',seed,source_dropout=.2,all_off_probability=.2))
            self.ablate_subsets(key,f'early_sd_{seed}')

    def ablate_subsets(self,key,name,availability_aware=False):
        self.guard(inference=True);split=self.dataset(key)['split'];adapter=self.load_job(key,name)
        target=self.output/key/('gate_ablations' if availability_aware else 'ablations')/name;target.mkdir(parents=True,exist_ok=True)
        for sources,part in itertools.product(SUBSETS,['val','test']):
            path=target/f'{subset_name(sources)}_{part}.csv'
            marker=path.with_suffix('.json')
            signature=dict(model_sha256=sha(self.job_path(key,name)/self.completed(key,name)['checkpoint']),
                context_sha256=context_signature(split.val_samples if part=='val' else split.test_samples),sources=list(sources),availability_aware=availability_aware)
            if path.exists() and marker.exists():
                rec=json.loads(marker.read_text(encoding='utf-8'))
                if rec['signature']!=signature or rec['sha256']!=sha(path):raise ValueError('Ablation cache mismatch')
                continue
            result=evaluate(adapter,split.val_samples if part=='val' else split.test_samples,
                self.config['batch_size'],enabled=sources,availability_aware=availability_aware)
            rank_frame(result).to_csv(path,index=False,lineterminator='\n')
            write_json(marker,dict(signature=signature,sha256=sha(path)))

    def diagnostics(self,key):
        self.guard(training=True);d=self.dataset(key);split=d['split'];root=self.output/key/'diagnostics'
        root.mkdir(parents=True,exist_ok=True);clock=root/'clock.json'
        if not clock.exists():write_json(clock,dict(started=time.time(),budget_hours=self.config['diagnostics_hours']))
        timer=json.loads(clock.read_text(encoding='utf-8'))
        def remaining():return max(0.,timer['budget_hours']*3600-(time.time()-timer['started']))
        if remaining()<=0:return
        base=ModelSpec('late',42,source_dropout=.2,source_dropout_mode='batch')
        self.run_job(key,'late_reference_42',base,id_job='id_42',max_wall_seconds=remaining())
        if remaining()>0:
            marker=root/'K1.cache.json'
            signature=dict(id=self.model_hash(key,'id_42'),late=self.model_hash(key,'late_reference_42'),data=d['signature'])
            if marker.exists():
                cached=json.loads(marker.read_text(encoding='utf-8'))
                if cached['signature']!=signature or any(not (root/n).is_file() or sha(root/n)!=h for n,h in cached['files'].items()):
                    raise ValueError('K1 scalar cache changed')
            else:
                ident=self.load_job(key,'id_42');late=self.load_job(key,'late_reference_42');evidence={}
                for part,samples in [('train_base',split.train_base_samples),('train_meta',split.train_meta_samples)]:
                    a=score_contexts(ident,samples,self.config['batch_size']);b=score_contexts(late,samples,self.config['batch_size'])
                    summary,paired=paired_ce_evidence(a,b);evidence[part]=summary
                    paired.to_csv(root/f'K1_{part}.csv',index=False,lineterminator='\n')
                write_json(root/'K1.json',dict(status='COMPLETED',**evidence))
                write_json(marker,dict(signature=signature,files={n:sha(root/n) for n in ['K1.json','K1_train_base.csv','K1_train_meta.csv']}))
                del ident,late
        a,b=split_meta_users(split.train_meta_samples)
        write_json(root/'K2_split.json',dict(fit_users=sorted({s.user_idx for s in a}),heldout_users=sorted({s.user_idx for s in b}),
            fit_context_sha256=context_signature(a),heldout_context_sha256=context_signature(b),
            restriction='Never use the fit half as out-of-sample S2 labels'))
        jobs=[('K2',base,a,'id_42'),('K3',replace(base,learning_rate=.0001,weight_decay=.0001,dropout=.5,
            source_dropout_mode='sample'),None,'id_42'),('K4',replace(base,freeze_id=False,
            source_dropout_mode='sample',all_off_probability=.2),None,None)]
        for name,spec,samples,id_job in jobs:
            if remaining()<=0:break
            rec=self.run_job(key,name,spec,samples,id_job,max_wall_seconds=remaining())
            write_json(root/f'{name}.json',dict(status='COMPLETED',job=name,val_ndcg_10=rec['val_ndcg_10'],
                test_ndcg_10=rec['test_ndcg_10'],stop_reason=rec['stop_reason'],spec=rec['spec']))

    def legacy_grid(self,key):
        if key!='beauty':return None
        root=(self.repo/self.config['legacy_run_root']).resolve()
        if not (root/'progress.json').is_file():return None
        grid={}
        for seed in SEEDS:
            entries={}
            for sources in SUBSETS:
                name=f'id_seed{seed}' if not sources else f'late_{subset_name(sources)}_seed{seed}'
                candidates=list((root/'runs').glob(name+'_20*'))
                if len(candidates)!=1:return None
                entries[subset_name(sources)]={p:read_ranks(candidates[0]/f'per_user_{p}.csv') for p in ['val','test']}
            grid[seed]=entries
        return grid

    def group_analysis(self,key):
        d=self.dataset(key);root=self.output/key
        if not (root/'groups.csv').exists():raise FileNotFoundError('Run prepare to lock group boundaries first')
        if json.loads((root/'groups.json').read_text(encoding='utf-8')).get('data_signature')!=d['signature']:
            raise ValueError('Group definitions do not match this dataset/cohort')
        groups=pd.read_csv(root/'groups.csv');results=[]
        legacy=self.legacy_grid(key)
        mechanisms=['early']+(['late_legacy'] if legacy else ['late'])
        for mechanism,seed in itertools.product(mechanisms,SEEDS):
            vals={};tests={}
            for sources in SUBSETS:
                subset=subset_name(sources)
                if mechanism=='late_legacy':
                    vals[subset]=legacy[seed][subset]['val'];tests[subset]=legacy[seed][subset]['test']
                else:
                    job=f'{mechanism}_{subset}_{seed}' if sources else f'id_{seed}'
                    rec=self.completed(key,job)
                    if rec is None:break
                    folder=(self.job_path(key,job)/rec['checkpoint']).parent
                    vals[subset]=read_ranks(folder/'per_user_val.csv');tests[subset]=read_ranks(folder/'per_user_test.csv')
            if len(vals)!=8:continue
            from .checkpoint2_protocol import aligned_rank_matrix
            aligned_rank_matrix(vals,d['split'].val_samples);aligned_rank_matrix(tests,d['split'].test_samples)
            summary,peruser=grouped_subset_analysis(vals,tests,groups)
            peruser.to_csv(root/f'group_{mechanism}_{seed}.csv',index=False,lineterminator='\n')
            results.append(dict(mechanism=mechanism,seed=seed,**summary))
        write_json(root/'group_analysis.json',results)

    def alignment(self,key,model_names,output_name,availability_aware=False):
        self.guard(inference=True);d=self.dataset(key);root=self.output/key/output_name;root.mkdir(parents=True,exist_ok=True)
        if not (self.output/key/'utility_label_rule.json').exists():raise RuntimeError('Preregister utility rule with prepare before computation')
        signature=dict(models={n:self.model_hash(key,n) for n in model_names},
            data=d['signature'],rule=sha(self.output/key/'utility_label_rule.json'),availability_aware=availability_aware)
        cache=root/'alignment_cache.json'
        if cache.exists():
            rec=json.loads(cache.read_text(encoding='utf-8'))
            if rec['signature']!=signature:raise ValueError('Alignment model/cohort changed')
            if any(not (root/n).is_file() or sha(root/n)!=h for n,h in rec['files'].items()):raise ValueError('Alignment cache changed')
            return json.loads((root/'alignment.json').read_text(encoding='utf-8'))['label_choice']
        records=[];fixed_features=None
        for name in model_names:
            adapter=self.load_job(key,name)
            frame,features=collect_ablations(adapter,d['split'].train_meta_samples,d['split'],self.config['batch_size'],include_features=fixed_features is None,availability_aware=availability_aware)
            if fixed_features is None:fixed_features=features
            frame=frame.merge(fixed_features,on=['context_id','user_idx'],validate='many_to_one')
            records.append(frame)
        utility=pd.concat(records,ignore_index=True)
        ce=utility.groupby('source_name').delta_ce.mean().to_dict();rank=utility.groupby('source_name').delta_rank.mean().to_dict()
        adapter=self.load_job(key,model_names[0]);val,_=collect_ablations(adapter,
            [s for s in d['split'].val_samples if s.target>0],d['split'],self.config['batch_size'],include_features=False,availability_aware=availability_aware)
        val['ndcg_difference']=val.ndcg_full-val.ndcg_without
        ndcg=val.groupby('source_name').ndcg_difference.mean().to_dict()
        choice=choose_utility_label(ce,rank,ndcg)
        utility.to_csv(root/'utility_both_labels.csv',index=False,lineterminator='\n')
        val.to_csv(root/'validation_same_model.csv',index=False,lineterminator='\n')
        fixed_features.to_csv(root/'features_train_meta.csv',index=False,lineterminator='\n')
        write_json(root/'alignment.json',dict(mean_delta_ce=ce,mean_delta_rank=rank,
            validation_ndcg_full_minus_without=ndcg,label_choice=choice,rule_locked_before_computation=True,
            references=model_names,feature_reference=model_names[0],test_used=False,availability_aware=availability_aware))
        write_json(cache,dict(signature=signature,files={p.name:sha(p) for p in root.iterdir()
            if p.name in ['utility_both_labels.csv','validation_same_model.csv','features_train_meta.csv','alignment.json']}))
        return choice

    def lock_backbone(self,key,backbone,reason):
        if backbone not in ['early_sd','late_joint'] or not reason:raise ValueError('Explicit backbone and reason required')
        path=self.output/key/'backbone_decision.json'
        jobs=['early_sd_42','early_sd_43','early_sd_44'] if backbone=='early_sd' else ['K4']
        if any(self.completed(key,n) is None for n in jobs):raise RuntimeError('Missing validation evidence for proposed backbone')
        evidence={n:self.completed(key,n)['val_ndcg_10'] for n in jobs}
        write_json(path,dict(backbone=backbone,reason=reason,validation_only_evidence=evidence,
            git=git_state(self.repo),timestamp=time.time(),test_used=False),overwrite=False)

    def reference_jobs(self,key):
        path=self.output/key/'backbone_decision.json'
        if not path.exists():raise RuntimeError('AWAITING_BACKBONE_SELECTION: review mini report, then run choose with --backbone and --reason')
        choice=json.loads(path.read_text(encoding='utf-8'))['backbone']
        if choice=='early_sd':return ['early_sd_42','early_sd_43']
        if not self.completed(key,'late_joint_43'):
            self.run_job(key,'late_joint_43',ModelSpec('late',43,freeze_id=False,
                source_dropout=.2,all_off_probability=.2))
        return ['K4','late_joint_43']

    def s2_e1(self,key):
        jobs=self.reference_jobs(key);label=self.alignment(key,jobs,'selected_s2',availability_aware=True)
        if label not in ['CE','LOG_RANK']:return
        root=self.output/key/'selected_s2';frame=pd.read_csv(root/'utility_both_labels.csv')
        groups,summary=group_contexts(self.dataset(key)['split'])
        features=frame.drop_duplicates('user_idx')
        q,t,g=apply_group_rules(np.rint(np.exp(features.feat_log_len.to_numpy())),features.feat_history_tail_ratio.to_numpy(),summary['rules'])
        mapping=dict(zip(features.user_idx,g))
        frame['group_idx']=frame.user_idx.map(mapping)
        frame['delta']=frame.delta_ce if label=='CE' else frame.delta_rank
        frame.to_csv(root/'utility.csv',index=False,lineterminator='\n')
        command=[sys.executable,str(self.repo/'scripts/run_e0_e1_evaluation.py'),
            '--delta_file',str(root/'utility.csv'),'--n_bootstrap',str(self.config['bootstrap_reps']),
            '--seed','42','--output_json',str(root/'e0_e1.json')]
        e1_signature=dict(utility_sha256=sha(root/'utility.csv'),config=self.protocol_hash,
            protocol_sha256=sha(self.repo/'src/taimma/evaluation/e0_e1_protocol.py'))
        marker=root/'e0_e1.cache.json'
        if marker.exists():
            cached=json.loads(marker.read_text(encoding='utf-8'))
            if cached['signature']!=e1_signature or not (root/'e0_e1.json').is_file() or sha(root/'e0_e1.json')!=cached['sha256']:
                raise ValueError('E0/E1 cache inputs or outputs changed')
        else:
            if (root/'e0_e1.json').exists():raise ValueError('Unverified E0/E1 output exists without cache provenance')
            subprocess.run(command,cwd=self.repo,check=True)
            write_json(marker,dict(signature=e1_signature,sha256=sha(root/'e0_e1.json')))
        self.ablate_subsets(key,jobs[0])
        self.nested_baselines(key,jobs[0])

    def s3(self,key):
        self.guard(training=True);root=self.output/key/'selected_s2';report_path=root/'e0_e1.json'
        if not report_path.exists():raise RuntimeError('S2/E1 and explicit backbone selection are required before S3')
        marker=root/'e0_e1.cache.json'
        if not marker.exists():raise RuntimeError('Run S2/E1 to verify eligibility provenance before S3')
        cached=json.loads(marker.read_text(encoding='utf-8'))
        if cached['sha256']!=sha(report_path) or cached['signature']['utility_sha256']!=sha(root/'utility.csv'):
            raise ValueError('S3 eligibility or utility changed after S2/E1')
        report=json.loads(report_path.read_text(encoding='utf-8'))
        if report['decision'] not in ['GO','CONDITIONAL']:
            write_json(root/'s3_status.json',dict(status='NOT_AUTHORIZED_BY_PROTOCOL',decision=report['decision']));return
        if not (root/'nested_subset_summary.json').is_file():
            raise RuntimeError('Review same-S1 global/group/oracle evidence from S2/E1 before requesting S3')
        d=self.dataset(key);jobs=self.reference_jobs(key);adapter=self.load_job(key,jobs[0]);split=d['split']
        utility=pd.read_csv(root/'utility.csv');features=pd.read_csv(root/'features_train_meta.csv')
        rules=json.loads((self.output/key/'groups.json').read_text(encoding='utf-8'))['rules']
        def attach_groups(f):
            f=f.copy();f['group_idx']=apply_group_rules(np.rint(np.exp(f.feat_log_len)),f.feat_history_tail_ratio,rules)[2];return f
        features=attach_groups(features)
        feature_parts={}
        for part,samples in [('val',split.val_samples),('test',split.test_samples)]:
            path=root/f'features_{part}.csv';marker=path.with_suffix('.cache.json')
            signature=dict(model=self.model_hash(key,jobs[0]),contexts=context_signature(samples),
                data=d['signature'],rules=rules,availability_aware=True,config=self.protocol_hash)
            if marker.exists():
                cached=json.loads(marker.read_text(encoding='utf-8'))
                if cached['signature']!=signature or not path.is_file() or sha(path)!=cached['sha256']:
                    raise ValueError('Gate feature cache changed: '+part)
                feature_parts[part]=pd.read_csv(path);continue
            valid=[s for s in samples if s.target>0]
            _,f=collect_ablations(adapter,valid,split,self.config['batch_size'],include_features=True,availability_aware=True)
            # OOV targets remain in evaluation denominators. Features must not
            # depend on the target: use an in-catalog placeholder only to obtain
            # the exact same history-only feature calculations for those rows.
            if len(valid)!=len(samples):
                from dataclasses import replace as dc_replace
                missing=[dc_replace(s,target=1) for s in samples if s.target==0]
                _,extra=collect_ablations(adapter,missing,split,self.config['batch_size'],include_features=True,availability_aware=True)
                f=pd.concat([f,extra],ignore_index=True)
            feature_parts[part]=attach_groups(f)
            feature_parts[part].to_csv(path,index=False,lineterminator='\n')
            write_json(marker,dict(signature=signature,sha256=sha(path)))
        labels=utility.groupby(['user_idx','source_name']).delta.mean().unstack()
        qualified=report['qualifying_sources'];regimes={}
        for source in qualified:
            record=report['sources'][source];regime=record['stability']['stability_regime'];regimes[source]=regime
            temperature=self.config['soft_label_temperature'] if regime=='soft_supervision' else None
            labels[source]=make_utility_labels(labels[source].to_numpy(),record['stability']['epsilon'],temperature)
        for s in SOURCE_ORDER:
            if s not in qualified:labels[s]=0.
        labels=labels.reset_index();columns=sorted(c for c in features if c.startswith('feat_'))
        for mode in ['global','unsupervised','contextual']:
            target=root/f'gate_{mode}';done=target/'completed.json'
            gate_inputs=digest(dict(utility=sha(root/'utility.csv'),model=sha(self.job_path(key,jobs[0])/self.completed(key,jobs[0])['checkpoint']),
                mode=mode,config=self.protocol_hash,eligibility=sha(report_path),
                features={p:sha(root/f'features_{p}.csv') for p in ['train_meta','val','test']}))
            if done.exists():
                rec=json.loads(done.read_text(encoding='utf-8'))
                if rec.get('input_signature')!=gate_inputs or any(not (target/n).is_file() or sha(target/n)!=h for n,h in rec['artifacts'].items()):
                    raise ValueError('Gate cache inputs or outputs changed')
                continue
            from ..training.trainer import set_seed
            set_seed(adapter.spec.seed)
            gate=SelectiveGate(len(columns),mode,regimes if mode=='contextual' else None,
                tau=self.config['gate_tau'],kappa=self.config['gate_kappa'])
            with exclusive(target/'execution.lock'):
                attempt=target/f'attempt_{len(list(target.glob("attempt_*")))+1:03d}'
                attempt.mkdir()
                result,mean,scale,hist=train_gate(adapter,gate,split.train_meta_samples,split.val_samples,
                    features,feature_parts['val'],labels,columns,attempt,self.config['epochs'],self.config['patience'],
                    self.config['batch_size'],lambda_util=self.config['lambda_util'] if mode=='contextual' else 0.,
                    lambda_sparse=self.config['lambda_sparse'],qualified_sources=qualified if mode=='contextual' else ())
                val=gate_evaluate(adapter,result,split.val_samples,feature_parts['val'],columns,mean,scale,self.config['batch_size'])
                test=gate_evaluate(adapter,result,split.test_samples,feature_parts['test'],columns,mean,scale,self.config['batch_size'])
                rank_frame(val).to_csv(attempt/'per_user_val.csv',index=False,lineterminator='\n')
                rank_frame(test).to_csv(attempt/'per_user_test.csv',index=False,lineterminator='\n')
                write_json(done,dict(mode=mode,val_ndcg_10=val.ndcg_10,test_ndcg_10=test.ndcg_10,
                    gate_sha256=sha(attempt/'gate.pt'),regimes=regimes,feature_columns=columns,epochs=len(hist),
                    attempt=attempt.name,input_signature=gate_inputs,
                    artifacts={f'{attempt.name}/{n}':sha(attempt/n) for n in ['gate.pt','per_user_val.csv','per_user_test.csv','gate_epochs.csv']}))
        self.nested_baselines(key,jobs[0])
        self.compare_gate_baselines(key,jobs[0])

    def compare_gate_baselines(self,key,name):
        """Seven declared comparisons, paired users; this S3 analysis is seed42."""
        from ..evaluation.statistics import paired_bootstrap_test,holm_bonferroni_correction,tost_equivalence_test
        from .checkpoint2_protocol import aligned_rank_matrix
        root=self.output/key/'selected_s2'
        def gate_ranks(mode):
            target=root/f'gate_{mode}'
            rec=json.loads((target/'completed.json').read_text(encoding='utf-8'))
            return read_ranks(target/rec['attempt']/'per_user_test.csv')
        uasf=gate_ranks('contextual')
        subsets=pd.read_csv(root/'nested_subset_per_user.csv').set_index('user_idx')
        full=read_ranks(self.output/key/'gate_ablations'/name/'CBT_test.csv')
        global_subset=full.copy();group_subset=full.copy()
        global_subset['ndcg@10']=global_subset.user_idx.map(subsets.ndcg_global)
        group_subset['ndcg@10']=group_subset.user_idx.map(subsets.ndcg_group)
        baselines={'all_fixed':full,'best_global_subset':global_subset,'best_group_subset':group_subset,
            'learned_global':gate_ranks('global'),
            'unsupervised_gate':gate_ranks('unsupervised')}
        for label,job in [('independent_ID','id_42'),('early_without_source_dropout','early_CBT_42')]:
            rec=self.completed(key,job)
            if rec is None:raise RuntimeError('Missing declared baseline '+job)
            baselines[label]=read_ranks((self.job_path(key,job)/rec['checkpoint']).parent/'per_user_test.csv')
        results=[];equivalence={}
        for label,frame in baselines.items():
            _,_,m=aligned_rank_matrix({'UASF':uasf,label:frame})
            results.append(paired_bootstrap_test(m[:,0],m[:,1],method_b_name=label,n_bootstrap=self.config['bootstrap_reps']))
            equivalent,low,high,p=tost_equivalence_test(m[:,0],m[:,1],relative_margin=.01)
            equivalence[label]=dict(equivalent=bool(equivalent),lower_margin=float(low),upper_margin=float(high),p=float(p))
        corrected=holm_bonferroni_correction(results)
        write_json(root/'baseline_comparisons.json',dict(comparisons=asdict(corrected),tost=equivalence,
            seeds=[42],limitation='Paired-user uncertainty conditional on seed42; no multi-seed S3 robustness claim',
            confirmatory_family_size=7))

    def nested_baselines(self,key,name):
        """All fixed / global subset / group subset are on the SAME frozen S1."""
        from .checkpoint2_protocol import aligned_rank_matrix
        self.ablate_subsets(key,name,availability_aware=True);root=self.output/key/'gate_ablations'/name
        val={subset_name(s):read_ranks(root/f'{subset_name(s)}_val.csv') for s in SUBSETS}
        test={subset_name(s):read_ranks(root/f'{subset_name(s)}_test.csv') for s in SUBSETS}
        summary,peruser=grouped_subset_analysis(val,test,pd.read_csv(self.output/key/'groups.csv'))
        peruser.to_csv(self.output/key/'selected_s2/nested_subset_per_user.csv',index=False,lineterminator='\n')
        write_json(self.output/key/'selected_s2/nested_subset_summary.json',summary)

    def transfer(self,key):
        self.id_grid(key)
        for seed,sources in itertools.product(SEEDS,SUBSETS[1:]):
            self.run_job(key,f'late_{subset_name(sources)}_{seed}',ModelSpec('late',seed,sources=sources,
                source_dropout=.2,source_dropout_mode='batch'),id_job=f'id_{seed}')
        self.early_grid(key)

    def mini_state(self,key='beauty'):
        state=dict(dataset=key,early={},controls={},diagnostics={},groups={},source_dropout={},utility={},
            proposal='Chưa chốt kiến trúc nền; không tự chạy S3 trên late fusion cũ.')
        def values(names):
            out=[]
            for n in names:
                rec=self.completed(key,n)
                if rec:out.append(dict(seed=rec['spec']['seed'],value=rec['test_ndcg_10']))
            return out
        for sources in SUBSETS:
            label=subset_name(sources)
            state['early'][label]=values([f'early_{label}_{s}' if sources else f'id_{s}' for s in SEEDS])
        for source,kind in itertools.product(SOURCE_ORDER,['shuffled','random']):
            real={};control={}
            for seed in SEEDS:
                a=self.completed(key,f'early_CBT_{seed}');b=self.completed(key,f'control_{kind}_{source}_{seed}')
                if not a or not b:break
                real[seed]=read_ranks((self.job_path(key,f'early_CBT_{seed}')/a['checkpoint']).parent/'per_user_test.csv')
                control[seed]=read_ranks((self.job_path(key,f'control_{kind}_{source}_{seed}')/b['checkpoint']).parent/'per_user_test.csv')
            if len(real)==3:state['controls'][f'{source}_{kind}']=paired_seed_ci(real,control,self.config['bootstrap_reps'])
        for code in ['K1','K2','K3','K4']:
            p=self.output/key/'diagnostics'/f'{code}.json'
            if p.exists():state['diagnostics'][code]=json.loads(p.read_text(encoding='utf-8'))
        p=self.output/key/'group_analysis.json'
        if p.exists():
            for record in json.loads(p.read_text(encoding='utf-8')):
                state['groups'].setdefault(record['mechanism'],[]).append(record)
        p=self.output/key/'late_alignment/alignment.json'
        if p.exists():state['utility']=json.loads(p.read_text(encoding='utf-8'))
        for sources in SUBSETS:
            if len(sources) not in [0,2,3]:continue
            label=subset_name(sources);rows=[]
            for seed in SEEDS:
                p=self.output/key/'ablations'/f'early_sd_{seed}'/f'{label}_test.csv'
                if p.exists():rows.append(dict(seed=seed,value=float(read_ranks(p)['ndcg@10'].mean())))
            state['source_dropout'][label]=rows
        p=self.output/key/'backbone_decision.json'
        if p.exists():state['proposal']=json.loads(p.read_text(encoding='utf-8'))
        state['complete']=all(len(v)==3 for v in state['early'].values()) and len(state['controls'])==6 and len(state['diagnostics'])==4 and all(len(state['groups'].get(k,[]))==3 for k in ['early','late_legacy']) and state['utility'].get('label_choice') not in [None,'PENDING'] and all(len(v)==3 for v in state['source_dropout'].values()) and isinstance(state['proposal'],dict)
        return state

    def archive(self,destination):
        destination=Path(destination).resolve()
        if not self.output.is_dir():raise FileNotFoundError('No run output to archive')
        if any(self.output.rglob('*.lock')):raise RuntimeError('Inspect active/interrupted jobs before archiving; a run lock exists')
        if destination==self.output or self.output in destination.parents:raise ValueError('Archive must be outside the run directory')
        destination.mkdir(parents=True,exist_ok=True)
        archive=destination/(self.output.name+'.zip')
        if archive.exists():raise FileExistsError('Never overwrite an existing shared archive')
        manifest={}
        with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
            for p in sorted(self.output.rglob('*')):
                if p.is_file() and not p.name.endswith('.lock'):
                    if not p.resolve().is_relative_to(self.output):raise ValueError('Archive cannot follow paths outside the run')
                    rel=p.relative_to(self.output).as_posix();manifest[rel]=sha(p);z.write(p,rel)
            z.writestr('SHA256_MANIFEST.json',json.dumps(manifest,indent=2))
        write_json(archive.with_suffix('.sha256.json'),dict(archive=archive.name,sha256=sha(archive),files=len(manifest)))
        return archive
