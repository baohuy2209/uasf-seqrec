"""Supervisor 07 November protocol. Historical models/trainers stay unchanged."""
from pathlib import Path
from dataclasses import asdict
import copy, json, shutil
import numpy as np
import pandas as pd
import torch
from torch import nn
from .study_completion import StudyCompletion,paired_summary
from .study_gates import features_for, attach_groups, SoftmaxNullGate
from .checkpoint2 import exclusive, rank_frame, subset_name
from .checkpoint2_protocol import digest, sha, write_json, context_signature, aligned_rank_matrix, git_state
from .late_checkpoint import lock_json
from ..training.checkpoint2_trainer import ModelSpec, build_model, ModelAdapter, fit, evaluate, batches
from ..training.selective_fusion import SelectiveGate, train_gate, gate_evaluate
from ..training.trainer import set_seed
from ..evaluation.checkpoint2_utility import collect_ablations
from ..evaluation.e0_e1_protocol import make_utility_labels
from ..evaluation.statistics import paired_bootstrap_test, holm_bonferroni_correction,tost_equivalence_test
from ..experiments.logger import ExperimentLogger

SEEDS=[42,43,44]
LABELS=['ID','C','B','T','CB','CT','BT','CBT']
SOURCES=['category','brand','title']
BASELINES=['all_fixed','global_subset','group_subset','global_gate','unsupervised','softmax_null',
           'independent_ID','early_no_sd','id_plus_meta','early_plus_meta']
MODES=['uasf','global_gate','unsupervised','softmax_null','wo_utility','state_only','add_one','fallback_s1','wo_source_dropout']


class SeparateID:
    """Route ONLY exact all-off rows to the frozen independent ID model."""
    def __init__(self, adapter, fallback):
        self.adapter=adapter;self.fallback=fallback;self.use_id_fallback=True
        fallback.model.eval()
        for p in fallback.model.parameters():p.requires_grad=False
    def __getattr__(self,name):return getattr(self.adapter,name)
    def predict(self,ids,weights=None):
        scores=self.adapter.predict(ids,weights)
        if weights is not None and self.use_id_fallback:
            off=(weights==0).all(1)
            if off.any():
                with torch.no_grad():fallback=self.fallback.predict(ids[off.to(ids.device)])
                scores=scores.clone();scores[off]=fallback
        return scores


class NullGate(SoftmaxNullGate):
    """Predeclared inference rule: winning null OR all beta below kappa routes ID."""
    def forward(self,features,groups,availability,threshold=False):
        probs=torch.softmax(self.mlp(features)/self.tau,dim=-1);beta=probs[:,1:]
        alpha=beta*availability
        if threshold:alpha=alpha*(beta>=self.kappa)*(probs.argmax(1)!=0).unsqueeze(1)
        return alpha,beta


def framing_from_ranks(frames,reps=1000):
    seeds=sorted(frames['uasf'])
    if seeds!=[42,43,44,45,46]:raise ValueError('Beauty framing requires five seeds 42..46')
    arrays={name:[] for name in frames};cohort=None
    for seed in seeds:
        current={name:by_seed[seed] for name,by_seed in frames.items()}
        names,users,m=aligned_rank_matrix(current)
        keys=current[names[0]].sort_values('user_idx')[['user_idx','target_idx']].to_numpy()
        if cohort is not None and not np.array_equal(cohort,keys):raise ValueError('Seed cohorts differ')
        cohort=keys
        for i,name in enumerate(names):arrays[name].append(m[:,i])
    values={name:np.mean(v,axis=0) for name,v in arrays.items()}
    best=max(sorted(set(values)-{'uasf'}),key=lambda n:values[n].mean())
    comparison=paired_bootstrap_test(values['uasf'],values[best],method_b_name=best,n_bootstrap=reps)
    delta=values['uasf']-values[best];rng=np.random.default_rng(42)
    interval=np.percentile([delta[rng.integers(len(delta),size=len(delta))].mean() for _ in range(reps)],[2.5,97.5])
    comparison.mean_diff=float(delta.mean());comparison.ci_lower=float(interval[0]);comparison.ci_upper=float(interval[1])
    return dict(framing='F1' if comparison.mean_diff>0 and comparison.ci_lower>0 else 'F2',
        best_baseline=best,comparison=asdict(comparison),validation_means={n:float(v.mean()) for n,v in values.items()},
        seeds=seeds,test_used=False,rule='mean UASF exceeds best locked baseline and paired-user CI lower > 0')


def rank_diagnostics(roots,output):
    rows=[]
    for seed in SEEDS:
        pair={}
        for mode,root,name in [('ID',roots['early'],f'id_{seed}'),('late_CBT',roots['late'],f'late_CBT_{seed}')]:
            job=Path(root)/'beauty/jobs'/name;rec=json.loads((job/'completed.json').read_text())
            pair[mode]=pd.read_csv((job/rec['checkpoint']).parent/'per_user_test.csv').sort_values('user_idx').reset_index(drop=True)
        aligned_rank_matrix(pair)
        for mode,f in pair.items():
            valid=(f.target_idx>0)
            rows.append(dict(seed=seed,model=mode,n=len(f),oov=int((~valid).sum()),median_rank=float(f['rank'].median()),
                **{f'top{k}':float(((f['rank']<=k)&valid).mean()) for k in [10,100,1000]}))
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out/'late_rank_diagnostics.csv',index=False,lineterminator='\n')
    pd.DataFrame(rows).groupby('model')[['median_rank','top10','top100','top1000']].agg(['mean','std']).to_csv(out/'late_rank_summary.csv',lineterminator='\n')
    write_json(out/'rank_scope.json',dict(training=False,inference=False,top10_brand_composition='Requires top-K item IDs; not inferable from target ranks'))
    return rows


class Memo0711(StudyCompletion):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.config.get('supervisor_memo')!='2026-11-07':raise ValueError('Use the 07/11 configuration')
    def load_job(self,key,name):
        adapter=super().load_job(key,name)
        if adapter.spec.kind=='early':
            return SeparateID(adapter,super().load_job(key,f'id_{adapter.spec.seed}'))
        return adapter
    def model_hash(self,key,name):
        original=super().model_hash(key,name)
        rec=self.completed(key,name)
        if rec['spec']['kind']=='early':
            return digest(dict(early=original,fallback=super().model_hash(key,f"id_{rec['spec']['seed']}"),policy='separate_ID_0711'))
        return original
    def run_job(self,key,name,spec,train_samples=None,id_job=None,control=None,max_wall_seconds=None):
        """New fitting is validation-only. No fabricated or pre-freeze Test metrics."""
        if name.startswith(('K','late_joint','control_random')):raise RuntimeError('07/11 forbids additional K4/random V2 fitting')
        self.assert_protocol(key)
        if (self.output/'freeze.json').exists():raise RuntimeError('Frozen study; no new fitting')
        d=self.dataset(key);samples=d['split'].train_base_samples if train_samples is None else train_samples
        fingerprint=digest(dict(spec=asdict(spec),train=context_signature(samples),config=self.protocol_hash,
                               data=d['signature'],control=control,id_job=id_job))
        existing=self.completed(key,name)
        if existing:
            if existing['fingerprint']!=fingerprint:raise ValueError('Job configuration changed: '+name)
            return existing
        self.guard(training=True)
        root=self.job_path(key,name)
        with exclusive(root/'execution.lock'):
            attempt=root/f'attempt_{len(list(root.glob("attempt_*")))+1:03d}';attempt.mkdir()
            attrs,masks=self.catalog(key,control)
            base=copy.deepcopy(self.load_job(key,id_job).model) if id_job else None
            adapter=ModelAdapter(build_model(spec,d['split'].num_items,d['attrs'],base),spec,attrs,masks,self.device)
            logger=ExperimentLogger(str(root),str(root/'unused.csv'))
            with logger.capture_output(attempt):
                h=fit(adapter,samples,d['split'].val_samples,attempt,self.config['epochs'],self.config['patience'],self.config['batch_size'])
                val=evaluate(adapter,d['split'].val_samples,self.config['batch_size'])
            rank_frame(val).to_csv(attempt/'per_user_val.csv',index=False,lineterminator='\n')
            write_json(attempt/'history.json',asdict(h))
            import platform,importlib.metadata
            write_json(attempt/'run_manifest.json',dict(python_version=platform.python_version(),torch_version=torch.__version__,
                library_versions={n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','scikit-learn']},git=git_state(self.repo)))
            write_json(attempt/'config.json',dict(spec=asdict(spec),data_signature=d['signature'],fingerprint=fingerprint))
            write_json(attempt/'source_snapshot.json',self.source_signature())
            rec=dict(spec=asdict(spec),control=control,fingerprint=fingerprint,checkpoint=f'{attempt.name}/checkpoint.pt',
                val_ndcg_10=val.ndcg_10,best_epoch=h.best_epoch,epochs=len(h.train_losses),stop_reason=h.stop_reason,
                test_status='NOT_OPENED',status='COMPLETED',artifacts={p.relative_to(root).as_posix():sha(p) for p in attempt.iterdir() if p.is_file()})
            write_json(root/'completed.json',rec)
        return rec
    def import_required(self,key,name,root):
        source=Path(root)/key/'jobs'/name
        if not (source/'completed.json').is_file():raise FileNotFoundError('Expected historical job missing; no retraining fallback: '+str(source))
        if not self.import_job(key,name,root):raise RuntimeError('Import failed: '+name)
    def foundation(self,key,seeds):
        self.prepare(key)
        for seed in seeds:
            for name,spec in [(f'id_{seed}',ModelSpec('id',seed,sources=())),
                (f'early_CBT_{seed}',ModelSpec('early',seed)),
                (f'early_sd_{seed}',ModelSpec('early',seed,source_dropout=.2,all_off_probability=.2))]:
                if key=='beauty' and seed in SEEDS:self.import_required(key,name,self.config['input_roots']['early'])
                else:self.run_job(key,name,spec)
        lock_json(self.output/key/'backbone_decision.json',dict(backbone='early_sd',supervisor_memo='2026-11-07',
            fallback='independent_ID_Train-base_same_seed',all_off_95pct_rule='SUPERSEDED_BY_SEPARATE_ID',test_used=False))
    def grid_b(self,key):
        self.prepare(key)
        for seed in SEEDS:
            jobs=[(f'id_{seed}',ModelSpec('id',seed,sources=()),None,'early')]
            for lab in LABELS[1:]:
                sources=tuple(s for s in SOURCES if s[0].upper() in lab)
                jobs += [(f'early_{lab}_{seed}',ModelSpec('early',seed,sources=sources),None,'early'),
                         (f'late_{lab}_{seed}',ModelSpec('late',seed,sources=sources,source_dropout=.2,source_dropout_mode='batch'),f'id_{seed}','late')]
            for name,spec,ident,kind in jobs:
                root=self.config['input_roots'][kind]
                if key=='beauty':self.import_required(key,name,root)
                elif (Path(root)/key/'jobs'/name/'completed.json').exists():self.import_job(key,name,root)
                else:self.run_job(key,name,spec,id_job=ident)
        # Diagnostic B may evaluate Test; UASF gates remain Val-only until freeze.
        self.group_analysis(key)
    def raw_add_one(self,key,job,label):
        path=self.output/key/f'per_seed_s2/{job}/raw_add_one_{label}.csv';marker=path.with_suffix('.json')
        signature=dict(model=self.model_hash(key,job),data=self.dataset(key)['signature'],label=label)
        if marker.exists():
            rec=json.loads(marker.read_text())
            if rec['signature']!=signature or sha(path)!=rec['sha256']:raise ValueError('Add-one cache changed')
            return pd.read_csv(path)
        rows=[];adapter=self.load_job(key,job);samples=self.dataset(key)['split'].train_meta_samples
        for batch in batches(samples,adapter.attrs,adapter.masks,self.config['batch_size']):
            with torch.no_grad():
                ids=batch['item_seqs'];n=len(ids);y=batch['targets'].to(adapter.device)-1
                empty=adapter.predict(ids,adapter.weights(n,()))
                for source in SOURCES:
                    single=adapter.predict(ids,adapter.weights(n,(source,)))
                    if label=='CE':delta=(nn.functional.cross_entropy(empty,y,reduction='none')-nn.functional.cross_entropy(single,y,reduction='none')).cpu().numpy()
                    else:
                        from ..evaluation.evaluator import IndependentEvaluator
                        ev=IndependentEvaluator(adapter.model.num_items,mask_policy='input_history_strict')
                        a=ev.evaluate_ranks(empty,(y+1).cpu().tolist(),batch['user_indices'],histories=ids.tolist())
                        b=ev.evaluate_ranks(single,(y+1).cpu().tolist(),batch['user_indices'],histories=ids.tolist())
                        delta=np.array([np.log2(1+x.rank)-np.log2(1+z.rank) for x,z in zip(a.per_user_records,b.per_user_records)])
                    rows.extend(dict(user_idx=int(u),source_name=source,delta=float(v)) for u,v in zip(batch['user_indices'],delta))
        raw=pd.DataFrame(rows)
        path.parent.mkdir(parents=True,exist_ok=True);raw.to_csv(path,index=False,lineterminator='\n')
        write_json(marker,dict(signature=signature,sha256=sha(path)));return raw
    def seed_labels(self,key,job,report,label,add_one=False):
        self.alignment(key,[job],f'per_seed_s2/{job}')
        raw=pd.read_csv(self.output/key/f'per_seed_s2/{job}/utility_both_labels.csv')
        if add_one:raw=self.raw_add_one(key,job,label)
        else:raw['delta']=raw['delta_ce' if label=='CE' else 'delta_rank']
        values=raw.pivot(index='user_idx',columns='source_name',values='delta')
        for source in SOURCES:
            r=report['sources'][source]
            epsilon=r['stability']['epsilon']
            if add_one:
                family=job.rsplit('_',1)[0];pair=[self.raw_add_one(key,f'{family}_{seed}',label) for seed in [42,43]]
                a,b=[f[f.source_name==source].sort_values('user_idx').delta.to_numpy() for f in pair]
                epsilon=float(np.median(np.abs(a-b)))
            values[source]=make_utility_labels(values[source].to_numpy(),epsilon,
                self.config['soft_label_temperature'] if r['stability']['stability_regime']=='soft_supervision' else None) if source in report['qualifying_sources'] else 0.
        path=self.output/key/f'per_seed_s2/{job}'
        values.reset_index().to_csv(path/('add_one_labels.csv' if add_one else 'labels.csv'),index=False,lineterminator='\n')
        return values.reset_index()
    def fairness(self,key,seeds):
        self.foundation(key,seeds)
        samples=self.dataset(key)['split'].train_base_samples+self.dataset(key)['split'].train_meta_samples
        for seed in seeds:
            self.run_job(key,f'id_plus_meta_{seed}',ModelSpec('id',seed,sources=()),train_samples=samples)
            self.run_job(key,f'early_plus_meta_{seed}',ModelSpec('early',seed),train_samples=samples)
    def subset_frames(self,key,seed,part):
        job=f'early_sd_{seed}';frames={lab:self.inference_ranks(key,job,lab,part) for lab in LABELS}
        folder=self.output/key/'gate_baselines';path=folder/f'{seed}.json'
        if part=='val':
            names,users,m=aligned_rank_matrix(frames);g=pd.read_csv(self.output/key/'groups.csv').set_index('user_idx').loc[users].val_group_id.to_numpy()
            global_lab=names[int(np.argmax(m.mean(0)))];choices={str(int(i)):names[int(np.argmax(m[g==i].mean(0)))] for i in np.unique(g)}
            write_json(path,dict(global_subset=global_lab,groups=choices,test_used=False))
        selection=json.loads(path.read_text());groups=pd.read_csv(self.output/key/'groups.csv').set_index('user_idx')
        pieces=[]
        for gid,rows in groups.groupby('val_group_id' if part=='val' else 'test_group_id'):
            lab=selection['groups'].get(str(int(gid)),selection['global_subset']);pieces.append(frames[lab][frames[lab].user_idx.isin(rows.index)])
        return dict(all_fixed=frames['CBT'],global_subset=frames[selection['global_subset']],
                    group_subset=pd.concat(pieces).sort_values('user_idx').reset_index(drop=True))
    def gate_run(self,key,seed,mode,report,label,part='val'):
        no_sd=mode=='wo_source_dropout';job=f'early_CBT_{seed}' if no_sd else f'early_sd_{seed}'
        adapter=self.load_job(key,job);adapter.use_id_fallback=mode!='fallback_s1'
        features=features_for(self,key,job,'train_meta');columns=sorted(c for c in features if c.startswith('feat_state_' if mode=='state_only' else 'feat_'))
        regimes={s:report['sources'][s]['stability']['stability_regime'] for s in report['qualifying_sources']}
        set_seed(seed)
        gate=NullGate(len(columns),tau=self.config['gate_tau'],kappa=self.config['gate_kappa']) if mode=='softmax_null' else SelectiveGate(len(columns),
            'global' if mode=='global_gate' else 'unsupervised' if mode=='unsupervised' else 'contextual',regimes,tau=self.config['gate_tau'],kappa=self.config['gate_kappa'])
        root=self.output/key/'gates'/f'{mode}_{seed}';marker=root/'completed.json'
        signature=digest(dict(model=self.model_hash(key,job),config=self.protocol_hash,mode=mode,label=label,report=report))
        if marker.exists():
            rec=json.loads(marker.read_text())
            if rec['signature']!=signature or any(sha(root/n)!=h for n,h in rec['artifacts'].items()):raise ValueError('Gate cache changed')
        else:
            if part=='test':raise FileNotFoundError('No trained gate: '+str(root))
            self.guard(training=True)
            labels=self.seed_labels(key,job,report,label,mode=='add_one')
            with exclusive(root/'execution.lock'):
                attempt=root/f'attempt_{len(list(root.glob("attempt_*")))+1:03d}';attempt.mkdir()
                _,_,_,h=train_gate(adapter,gate,self.dataset(key)['split'].train_meta_samples,self.dataset(key)['split'].val_samples,
                    features,features_for(self,key,job,'val'),labels,columns,attempt,self.config['epochs'],self.config['patience'],self.config['batch_size'],
                    lambda_util=0. if mode in ['global_gate','unsupervised','wo_utility'] else self.config['lambda_util'],
                    lambda_sparse=self.config['lambda_sparse'],qualified_sources=report['qualifying_sources'])
                rec=dict(signature=signature,attempt=attempt.name,mode=mode,seed=seed,best_epoch=int(np.argmax([r['val_ndcg_10'] for r in h]))+1,
                         artifacts={p.relative_to(root).as_posix():sha(p) for p in attempt.iterdir() if p.is_file()})
                write_json(marker,rec)
        state=torch.load(root/rec['attempt']/'gate.pt',map_location=adapter.device,weights_only=True)
        gate.load_state_dict(state['model_state_dict']);gate.to(adapter.device)
        if part=='test':self.verify_freeze()
        out=root/f'per_user_{part}.csv';binding=out.with_suffix('.binding.json');sig=dict(gate=signature,part=part)
        if binding.exists():
            m=json.loads(binding.read_text())
            if m['signature']!=sig or sha(out)!=m['sha256']:raise ValueError('Gate rank cache changed')
            return pd.read_csv(out)
        result=gate_evaluate(adapter,gate,getattr(self.dataset(key)['split'],part+'_samples'),features_for(self,key,job,part),
            state['feature_columns'],np.array(state['feature_mean']),np.array(state['feature_scale']),self.config['batch_size'])
        f=rank_frame(result);f.to_csv(out,index=False,lineterminator='\n');write_json(binding,dict(signature=sig,sha256=sha(out)));return f
    def uasf(self,key,transfer=False):
        if key!='beauty':
            decision=json.loads((self.output/'beauty/framing_decision.json').read_text())
            from datetime import datetime,timezone,timedelta
            if not transfer or decision['framing']!='F1' or datetime.now(timezone(timedelta(hours=7))).date().isoformat()<'2026-11-12':
                raise RuntimeError('Transfer UASF requires F1 Beauty and date >=12/11; use transfer stage')
        seeds=list(range(42,47)) if key=='beauty' else SEEDS
        self.foundation(key,seeds);self.fairness(key,seeds)
        status=self.utility(key,['early_sd_42','early_sd_43'],'selected_s2',selected=True)
        if not status['s3_eligible']:
            write_json(self.output/key/'framing_decision.json',dict(framing='F2',reason=status['decision'],test_used=False))
            return
        label=status['label_choice'];report=json.loads((self.output/key/f'selected_s2/e0_e1_{label}.json').read_text())
        frames={name:{} for name in BASELINES+['uasf']};rows=[]
        for seed in seeds:
            current=self.subset_frames(key,seed,'val')
            for mode,job in [('independent_ID',f'id_{seed}'),('early_no_sd',f'early_CBT_{seed}'),('id_plus_meta',f'id_plus_meta_{seed}'),('early_plus_meta',f'early_plus_meta_{seed}')]:
                current[mode]=self.inference_ranks(key,job,'CBT','val')
            for mode in MODES:
                # No-SD labels use own two-reference noise estimate, same chosen label metric.
                r=report
                if mode=='wo_source_dropout':
                    self.utility(key,['early_CBT_42','early_CBT_43'],'no_sd_s2',selected=False)
                    r=json.loads((self.output/key/f'no_sd_s2/e0_e1_{label}.json').read_text())
                f=self.gate_run(key,seed,mode,r,label);rows.append(dict(seed=seed,mode=mode,val_ndcg_10=float(f['ndcg@10'].mean())))
                if mode in frames:current[mode]=f
            for name,f in current.items():frames[name][seed]=f
        if key=='beauty':write_json(self.output/key/'framing_decision.json',framing_from_ranks(frames,self.config['bootstrap_reps']))
        else:write_json(self.output/key/'framing_decision.json',dict(framing='TRANSFER_F1',test_used=False))
        pd.DataFrame(rows).to_csv(self.output/key/'gate_validation.csv',index=False,lineterminator='\n')
    def robustness_current(self,key):
        self.foundation(key,SEEDS);rows=[]
        for seed in SEEDS:
            for family in ['early_sd','early_CBT']:
                for source in SOURCES:
                    mask=self.dataset(key)['masks'];available=np.flatnonzero(mask[source].numpy()>0)
                    order=np.random.default_rng(seed).permutation(available)
                    for rate in self.config['missing_rates']:
                        active={s:v.clone() for s,v in mask.items()};active[source][order[:round(rate*len(order))]]=0
                        f=self.inference_ranks(key,f'{family}_{seed}',f'missing_{source}_{rate:.2f}','test',active)
                        rows.append(dict(seed=seed,model=family,source=source,missing_rate=rate,ndcg_10=float(f['ndcg@10'].mean())))
        pd.DataFrame(rows).to_csv(self.output/key/'robustness.csv',index=False,lineterminator='\n')
    def title_controls(self):
        key='beauty';self.prepare(key)
        from metadata_pipeline.build_training_contrasts import build_training_contrasts_for_dataset
        # Existing uniform/permutation artifacts are NOT used to fit new random models.
        old=Path(self.config['input_roots']['early'])/key/'training_contrasts'
        target=self.output/key/'training_contrasts'
        if not target.exists():shutil.copytree(old,target)
        rows=[]
        for seed in SEEDS:
            self.import_required(key,f'id_{seed}',self.config['input_roots']['early'])
            self.import_required(key,f'early_T_{seed}',self.config['input_roots']['early'])
            rec=self.run_job(key,f'title_shuffled_{seed}',ModelSpec('early',seed,sources=('title',)),control=('title','shuffled',seed))
            for name in [f'early_T_{seed}',f'title_shuffled_{seed}']:
                f=self.inference_ranks(key,name,'T','test');rows.append(dict(seed=seed,job=name,test_ndcg_10=float(f['ndcg@10'].mean())))
        pd.DataFrame(rows).to_csv(self.output/key/'title_only_controls.csv',index=False,lineterminator='\n')
        real={s:self.inference_ranks(key,f'early_T_{s}','T','test') for s in SEEDS}
        shuffled={s:self.inference_ranks(key,f'title_shuffled_{s}','T','test') for s in SEEDS}
        write_json(self.output/key/'title_only_comparison.json',dict(direction='shuffled minus real',
            **paired_summary(shuffled,real,self.config['bootstrap_reps'])))
    def release(self,key):
        self.verify_freeze();label=json.loads((self.output/key/'selected_s2/status.json').read_text())['label_choice']
        report=json.loads((self.output/key/f'selected_s2/e0_e1_{label}.json').read_text());rows=[];frames={n:{} for n in BASELINES+['uasf']}
        seeds=list(range(42,47)) if key=='beauty' else SEEDS
        for seed in seeds:
            current=self.subset_frames(key,seed,'test')
            for name,job in [('independent_ID',f'id_{seed}'),('early_no_sd',f'early_CBT_{seed}'),('id_plus_meta',f'id_plus_meta_{seed}'),('early_plus_meta',f'early_plus_meta_{seed}')]:
                current[name]=self.inference_ranks(key,job,'CBT','test')
            for mode in MODES:
                r=report if mode!='wo_source_dropout' else json.loads((self.output/key/f'no_sd_s2/e0_e1_{label}.json').read_text())
                f=self.gate_run(key,seed,mode,r,label,'test');rows.append(dict(seed=seed,mode=mode,test_ndcg_10=float(f['ndcg@10'].mean())))
                if mode in frames:current[mode]=f
            for name,f in current.items():frames[name][seed]=f
        comparisons=[];equivalence={}
        for mode in BASELINES:
            a=[];b=[]
            for seed in seeds:
                _,_,m=aligned_rank_matrix({'uasf':frames['uasf'][seed],mode:frames[mode][seed]});a.append(m[:,0]);b.append(m[:,1])
            comparisons.append(paired_bootstrap_test(np.mean(a,0),np.mean(b,0),method_b_name=mode,n_bootstrap=self.config['bootstrap_reps']))
            ok,low,high,p=tost_equivalence_test(np.mean(a,0),np.mean(b,0));equivalence[mode]=dict(equivalent=bool(ok),lower=low,upper=high,p=p)
        write_json(self.output/key/'baseline_comparisons.json',dict(family_size=len(BASELINES),holm=asdict(holm_bonferroni_correction(comparisons)),tost=equivalence))
        pd.DataFrame(rows).to_csv(self.output/key/'gate_test.csv',index=False,lineterminator='\n')
    def group_analysis(self,key):
        # New fit jobs are Val-only; diagnostic Test ranks are evaluated here explicitly.
        for seed in SEEDS:
            for mode in ['early','late']:
                for lab in LABELS:
                    name=f'id_{seed}' if lab=='ID' else f'{mode}_{lab}_{seed}'
                    rec=self.completed(key,name);attempt=(self.job_path(key,name)/rec['checkpoint']).parent
                    for part in ['val','test']:
                        path=attempt/f'per_user_{part}.csv'
                        if not path.exists():
                            f=self.inference_ranks(key,name,lab,part);f.to_csv(path,index=False,lineterminator='\n')
                            rec['artifacts'][path.relative_to(self.job_path(key,name)).as_posix()]=sha(path)
                            write_json(self.job_path(key,name)/'completed.json',rec)
                    if 'test_ndcg_10' not in rec:
                        f=pd.read_csv(attempt/'per_user_test.csv')
                        rec.update(test_ndcg_10=float(f['ndcg@10'].mean()),test_status='DIAGNOSTIC_B_EVALUATED')
                        write_json(self.job_path(key,name)/'completed.json',rec)
        return super().group_analysis(key)
