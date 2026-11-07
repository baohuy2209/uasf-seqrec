"""Validation-only UASF, nested baselines and five controlled ablations.

All test evaluation is in release_test, behind the immutable study freeze.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch import nn
from ..training.selective_fusion import SelectiveGate,train_gate,gate_evaluate
from ..training.checkpoint2_trainer import ModelSpec
from ..training.trainer import set_seed
from ..evaluation.checkpoint2_utility import collect_ablations
from ..evaluation.e0_e1_protocol import make_utility_labels,compute_noise_threshold_and_stability
from ..evaluation.evaluator import IndependentEvaluator
from ..training.checkpoint2_trainer import batches
from ..experiments.checkpoint2 import exclusive,rank_frame
from ..experiments.checkpoint2_protocol import sha,digest,write_json,read_ranks,aligned_rank_matrix
from metadata_pipeline.user_grouping import apply_group_rules

BASELINES=['all_fixed','global_subset','group_subset','global_gate','unsupervised','independent_ID','early_no_sd']
ABLATIONS=['wo_utility','state_only','softmax_null','wo_source_dropout','add_one']
MODES=['global_gate','unsupervised','uasf']+ABLATIONS


class SoftmaxNullGate(SelectiveGate):
    def __init__(self,input_dim,**kwargs):
        super().__init__(input_dim,mode='unsupervised',**kwargs)
        self.mlp=nn.Sequential(nn.Linear(input_dim,32),nn.ReLU(),nn.Linear(32,4))
    def forward(self,features,groups,availability,threshold=False):
        beta=torch.softmax(self.mlp(features)/self.tau,dim=-1)[:,1:]
        return beta*availability,beta


def seeds_for(config,key,now=None):
    seeds=list(config['main_seeds'][key])
    if len(seeds)!=len(set(seeds)) or len(seeds)<(5 if key=='beauty' else 3):raise ValueError('Beauty needs 5 seeds; Sports/Toys need at least 3')
    if key!='beauty' and len(seeds)<5:
        from datetime import datetime,timezone,timedelta
        current=now or datetime.now(timezone(timedelta(hours=7)))
        cutoff=datetime(2026,11,12,18,tzinfo=timezone(timedelta(hours=7)))
        if not config.get('allow_reduced_transfer_seeds') or current<cutoff:
            raise ValueError('Reduced Sports/Toys seeds require an explicit decision on/after the evening of 12/11')
    return seeds


def model_job(backbone,seed,no_dropout=False):
    if backbone=='early_sd':return f'early_CBT_{seed}' if no_dropout else f'early_sd_{seed}'
    return f'late_joint_no_sd_{seed}' if no_dropout else 'K4' if seed==42 else f'late_joint_{seed}'


def ensure_foundation(run,key,backbone,seed,no_dropout=False):
    if backbone=='early_sd':spec=ModelSpec('early',seed,source_dropout=0. if no_dropout else .2,all_off_probability=0. if no_dropout else .2)
    else:spec=ModelSpec('late',seed,freeze_id=False,source_dropout=0. if no_dropout else .2,all_off_probability=0. if no_dropout else .2)
    name=model_job(backbone,seed,no_dropout);run.run_job(key,name,spec);return name


def attach_groups(frame,rules):
    frame=frame.copy();frame['group_idx']=apply_group_rules(np.rint(np.exp(frame.feat_log_len)),frame.feat_history_tail_ratio,rules)[2]
    return frame


def features_for(run,key,job,part):
    root=run.output/key/'gate_features'/job;p=root/f'{part}.csv';marker=p.with_suffix('.json')
    d=run.dataset(key);samples=getattr(d['split'],part+'_samples')
    signature=dict(model=run.model_hash(key,job),part=part,data=d['signature'])
    if marker.exists():
        rec=json.loads(marker.read_text())
        if rec['signature']!=signature or sha(p)!=rec['sha256']:raise ValueError('Gate feature cache changed')
        return pd.read_csv(p)
    # Called for test only after the release gate has verified the frozen study.
    adapter=run.load_job(key,job)
    _,f=collect_ablations(adapter,samples,d['split'],run.config['batch_size'],allow_oov=part!='train_meta')
    rules=json.loads((run.output/key/'groups.json').read_text())['rules'];f=attach_groups(f,rules)
    root.mkdir(parents=True,exist_ok=True);f.to_csv(p,index=False,lineterminator='\n')
    write_json(marker,dict(signature=signature,sha256=sha(p)));return f


def add_one_labels(run,key,jobs,qualified,regimes,label='CE'):
    """CE(empty)-CE(single), paired references, with its own noise threshold."""
    root=run.output/key/'selected_s2';p=root/'add_one_labels.csv';marker=p.with_suffix('.json')
    signature=dict(models={j:run.model_hash(key,j) for j in jobs},label='add_one_'+label,qualified=qualified,regimes=regimes)
    if marker.exists():
        m=json.loads(marker.read_text())
        if m['signature']!=signature or sha(p)!=m['sha256']:raise ValueError('Add-one cache changed')
        return pd.read_csv(p)
    all_rows=[];samples=run.dataset(key)['split'].train_meta_samples
    for name in jobs:
        adapter=run.load_job(key,name);adapter.model.eval()
        with torch.no_grad():
            for batch in batches(samples,adapter.attrs,adapter.masks,run.config['batch_size']):
                ids=batch['item_seqs'];targets=batch['targets'].to(adapter.device)-1;n=len(targets)
                empty_scores=adapter.predict(ids,adapter.weights(n,()))
                empty=nn.functional.cross_entropy(empty_scores,targets,reduction='none')
                ev=IndependentEvaluator(adapter.model.num_items,mask_policy='input_history_strict')
                er=ev.evaluate_ranks(empty_scores,(targets+1).cpu().tolist(),batch['user_indices'],histories=ids.tolist())
                values={}
                for source in ['category','brand','title']:
                    scores=adapter.predict(ids,adapter.weights(n,(source,)))
                    single=nn.functional.cross_entropy(scores,targets,reduction='none')
                    sr=ev.evaluate_ranks(scores,(targets+1).cpu().tolist(),batch['user_indices'],histories=ids.tolist())
                    values[source]=(empty-single).cpu().numpy() if label=='CE' else np.array([np.log2(1+a.rank)-np.log2(1+b.rank) for a,b in zip(er.per_user_records,sr.per_user_records)])
                for i,u in enumerate(batch['user_indices']):
                    all_rows.append(dict(user_idx=int(u),seed=adapter.spec.seed,**{s:float(values[s][i]) for s in values}))
    raw=pd.DataFrame(all_rows);out=raw.groupby('user_idx')[['category','brand','title']].mean()
    for source in ['category','brand','title']:
        pair=raw.pivot(index='user_idx',columns='seed',values=source)
        a,b=[pair[c].to_numpy() for c in pair.columns]
        eps=compute_noise_threshold_and_stability(a,b,n_bootstrap=run.config['bootstrap_reps']).epsilon
        out[source]=make_utility_labels(out[source].to_numpy(),eps,run.config['soft_label_temperature'] if regimes.get(source)=='soft_supervision' else None) if source in qualified else 0.
    out=out.reset_index();out.to_csv(p,index=False,lineterminator='\n');write_json(marker,dict(signature=signature,sha256=sha(p)))
    return out


def validation_baselines(run,key,backbone,seeds):
    records=[]
    for seed in seeds:
        job=model_job(backbone,seed)
        frames={label:run.inference_ranks(key,job,label,'val') for label in ['ID','C','B','T','CB','CT','BT','CBT']}
        names,users,m=aligned_rank_matrix(frames);groups=pd.read_csv(run.output/key/'groups.csv').set_index('user_idx').loc[users].val_group_id.to_numpy()
        global_idx=int(np.argmax(m.mean(0)));chosen=np.full(len(users),global_idx)
        for group in np.unique(groups):chosen[groups==group]=int(np.argmax(m[groups==group].mean(0)))
        choices={str(int(g)):names[int(chosen[np.flatnonzero(groups==g)[0]])] for g in np.unique(groups)}
        write_json(run.output/key/'gate_baselines'/f'subsets_{seed}.json',dict(global_subset=names[global_idx],groups=choices,test_used=False))
        for mode,score in [('all_fixed',m[:,names.index('CBT')]),('global_subset',m[:,global_idx]),('group_subset',m[np.arange(len(m)),chosen])]:
            records.append(dict(seed=seed,mode=mode,val_ndcg_10=float(score.mean())))
        for mode,name in [('independent_ID',f'id_{seed}'),('early_no_sd',f'early_CBT_{seed}')]:
            run.run_job(key,name,ModelSpec('id',seed,sources=()) if mode=='independent_ID' else ModelSpec('early',seed))
            records.append(dict(seed=seed,mode=mode,val_ndcg_10=run.completed(key,name)['val_ndcg_10']))
    return records


def fit_validation(run,key,backbone,status):
    if not status['s3_eligible']:
        write_json(run.output/key/'s3_status.json',dict(status='SKIPPED',reason='Selected backbone and eligible E0/E1 required'));return
    if (run.output/'freeze.json').exists():raise RuntimeError('Cannot train gates after freeze')
    root=run.output/key/'selected_s2';label=status['label_choice']
    report=json.loads((root/f'e0_e1_{label}.json').read_text())
    raw=pd.read_csv(root/f'utility_{label}.csv');qualified=report['qualifying_sources']
    regimes={s:report['sources'][s]['stability']['stability_regime'] for s in qualified}
    labels=raw.groupby(['user_idx','source_name']).delta.mean().unstack()
    for s in ['category','brand','title']:
        eps=report['sources'][s]['stability']['epsilon']
        labels[s]=make_utility_labels(labels[s].to_numpy(),eps,run.config['soft_label_temperature'] if regimes.get(s)=='soft_supervision' else None) if s in qualified else 0.
    labels=labels.reset_index();seeds=seeds_for(run.config,key)
    references=[model_job(backbone,s) for s in [42,43]]
    for s in seeds:
        run.id_grid_seed(key,s)
        ensure_foundation(run,key,backbone,s)
        ensure_foundation(run,key,backbone,s,True)
    # The w/o SD ablation regenerates utility on its own no-dropout references.
    no_refs=[model_job(backbone,s,True) for s in [42,43]]
    no_status=run.utility(key,no_refs,'ablation_no_sd_labels',selected=False)
    no_raw=pd.read_csv(run.output/key/'ablation_no_sd_labels/utility_both_labels.csv')
    no_labels=no_raw.groupby(['user_idx','source_name'])[('delta_ce' if label=='CE' else 'delta_rank')].mean().unstack()
    for source in ['category','brand','title']:
        v=no_raw[no_raw.source_name==source];pairs=v.pivot(index='user_idx',columns='reference_seed',values='delta_ce' if label=='CE' else 'delta_rank')
        eps=compute_noise_threshold_and_stability(*[pairs[c].to_numpy() for c in pairs.columns],n_bootstrap=run.config['bootstrap_reps']).epsilon
        no_labels[source]=make_utility_labels(no_labels[source].to_numpy(),eps,run.config['soft_label_temperature'] if regimes.get(source)=='soft_supervision' else None) if source in qualified else 0.
    no_labels=no_labels.reset_index()
    plus_labels=add_one_labels(run,key,references,qualified,regimes,label)
    rows=validation_baselines(run,key,backbone,seeds)
    for seed in seeds:
        for mode in MODES:
            job=model_job(backbone,seed,mode=='wo_source_dropout')
            features=features_for(run,key,job,'train_meta');val_features=features_for(run,key,job,'val')
            columns=sorted(c for c in features if c.startswith('feat_state_') if True) if mode=='state_only' else sorted(c for c in features if c.startswith('feat_'))
            target=run.output/key/'gates'/f'{mode}_{seed}';done=target/'completed.json'
            selected_labels=no_labels if mode=='wo_source_dropout' else plus_labels if mode=='add_one' else labels
            signature=digest(dict(model=run.model_hash(key,job),mode=mode,seed=seed,config=run.protocol_hash,
                                 feature_train_sha256=sha(run.output/key/'gate_features'/job/'train_meta.csv'),
                                 feature_val_sha256=sha(run.output/key/'gate_features'/job/'val.csv'),
                                 labels=selected_labels.to_dict(orient='list'),columns=columns))
            if done.exists():
                rec=json.loads(done.read_text())
                if rec['signature']!=signature or any(sha(target/p)!=h for p,h in rec['artifacts'].items()):raise ValueError('Gate cache changed')
            else:
                run.guard(training=True);set_seed(seed)
                mode_name='global' if mode=='global_gate' else 'unsupervised' if mode=='unsupervised' else 'contextual'
                gate=SoftmaxNullGate(len(columns),tau=run.config['gate_tau'],kappa=run.config['gate_kappa']) if mode=='softmax_null' else SelectiveGate(len(columns),mode_name,regimes if mode_name=='contextual' else None,tau=run.config['gate_tau'],kappa=run.config['gate_kappa'])
                with exclusive(target/'execution.lock'):
                    attempt=target/f'attempt_{len(list(target.glob("attempt_*")))+1:03d}';attempt.mkdir()
                    adapter=run.load_job(key,job)
                    supervised=mode not in ['global_gate','unsupervised','wo_utility']
                    gate,mean,scale,hist=train_gate(adapter,gate,run.dataset(key)['split'].train_meta_samples,run.dataset(key)['split'].val_samples,
                        features,val_features,selected_labels,columns,attempt,run.config['epochs'],run.config['patience'],run.config['batch_size'],
                        lambda_util=run.config['lambda_util'] if supervised else 0.,lambda_sparse=run.config['lambda_sparse'],qualified_sources=qualified if supervised else ())
                    val=gate_evaluate(adapter,gate,run.dataset(key)['split'].val_samples,val_features,columns,mean,scale,run.config['batch_size'])
                    rank_frame(val).to_csv(attempt/'per_user_val.csv',index=False,lineterminator='\n')
                    rec=dict(mode=mode,seed=seed,foundation=job,signature=signature,attempt=attempt.name,columns=columns,
                        val_ndcg_10=val.ndcg_10,best_epoch=int(np.argmax([h['val_ndcg_10'] for h in hist]))+1,
                        regimes=regimes if mode_name=='contextual' else {},gate_mode=mode_name,softmax=mode=='softmax_null',
                        artifacts={p.relative_to(target).as_posix():sha(p) for p in attempt.iterdir() if p.is_file()})
                    write_json(done,rec)
            rows.append(dict(seed=seed,mode=mode,val_ndcg_10=rec['val_ndcg_10']))
    pd.DataFrame(rows).to_csv(run.output/key/'gate_validation.csv',index=False,lineterminator='\n')
    means=pd.DataFrame(rows).groupby('mode').val_ndcg_10.mean()
    f1=all(means.uasf>means[b] for b in BASELINES)
    write_json(run.output/key/'framing_decision.json',dict(framing='F1' if f1 else 'F2',validation_means=means.to_dict(),seeds=seeds,
                rule='E0/E1 eligible and UASF validation strictly beats every nested baseline',test_used=False))
    # Fairness controls add Train-meta only after reference/utility construction.
    train=run.dataset(key)['split'].train_base_samples+run.dataset(key)['split'].train_meta_samples
    for seed in seeds:
        run.run_job(key,f'id_plus_meta_{seed}',ModelSpec('id',seed,sources=()),train_samples=train)
        run.run_job(key,f'early_plus_meta_{seed}',ModelSpec('early',seed),train_samples=train)


def release_test(run,key):
    run.verify_freeze()
    root=run.output/key;rows=[]
    for done in sorted((root/'gates').glob('*/completed.json')):
        rec=json.loads(done.read_text());target=done.parent;out=target/'per_user_test.csv';marker=out.with_suffix('.json')
        signature=dict(freeze=sha(run.output/'freeze.json'),gate=rec['signature'])
        if marker.exists():
            m=json.loads(marker.read_text())
            if m['signature']!=signature or sha(out)!=m['sha256']:raise ValueError('Released test cache changed')
            f=read_ranks(out)
        else:
            run.guard(inference=True);adapter=run.load_job(key,rec['foundation'])
            gate=SoftmaxNullGate(len(rec['columns']),tau=run.config['gate_tau'],kappa=run.config['gate_kappa']) if rec['softmax'] else SelectiveGate(len(rec['columns']),rec['gate_mode'],rec['regimes'],tau=run.config['gate_tau'],kappa=run.config['gate_kappa'])
            state=torch.load(target/rec['attempt']/'gate.pt',map_location=adapter.device,weights_only=True)
            gate.load_state_dict(state['model_state_dict']);gate.to(adapter.device)
            features=features_for(run,key,rec['foundation'],'test')
            result=gate_evaluate(adapter,gate,run.dataset(key)['split'].test_samples,features,rec['columns'],np.array(state['feature_mean']),np.array(state['feature_scale']),run.config['batch_size'])
            rank_frame(result).to_csv(out,index=False,lineterminator='\n');write_json(marker,dict(signature=signature,sha256=sha(out)));f=read_ranks(out)
        rows.append(dict(mode=rec['mode'],seed=rec['seed'],test_ndcg_10=f['ndcg@10'].mean()))
    if not rows:
        write_json(root/'s3_test_status.json',dict(status='NOT_APPLICABLE',reason='No eligible S3 gates were trained'))
        return
    pd.DataFrame(rows).to_csv(root/'gate_test.csv',index=False,lineterminator='\n')
    choice=json.loads((root/'backbone_decision.json').read_text())['backbone']
    seeds=seeds_for(run.config,key);comparison_frames={mode:{} for mode in BASELINES+['uasf']}
    for seed in seeds:
        job=model_job(choice,seed)
        subsets={lab:run.inference_ranks(key,job,lab,'test') for lab in ['ID','C','B','T','CB','CT','BT','CBT']}
        selection=json.loads((root/'gate_baselines'/f'subsets_{seed}.json').read_text())
        group=pd.read_csv(root/'groups.csv').set_index('user_idx')
        global_frame=subsets[selection['global_subset']]
        group_parts=[]
        for gid,users in group.groupby('test_group_id'):
            lab=selection['groups'].get(str(int(gid)),selection['global_subset'])
            group_parts.append(subsets[lab][subsets[lab].user_idx.isin(users.index)])
        frames=dict(all_fixed=subsets['CBT'],global_subset=global_frame,group_subset=pd.concat(group_parts).sort_values('user_idx').reset_index(drop=True))
        for mode in ['global_gate','unsupervised','uasf']:
            frames[mode]=read_ranks(root/'gates'/f'{mode}_{seed}'/'per_user_test.csv')
        for mode,name in [('independent_ID',f'id_{seed}'),('early_no_sd',f'early_CBT_{seed}')]:
            rec=run.completed(key,name);frames[mode]=read_ranks((run.job_path(key,name)/rec['checkpoint']).parent/'per_user_test.csv')
        aligned_rank_matrix(frames,run.dataset(key)['split'].test_samples)
        for mode,f in frames.items():
            p=root/'released_baselines'/str(seed)/(mode+'.csv');p.parent.mkdir(parents=True,exist_ok=True)
            f.to_csv(p,index=False,lineterminator='\n');comparison_frames[mode][seed]=f
    from ..evaluation.statistics import paired_bootstrap_test,holm_bonferroni_correction,tost_equivalence_test
    from dataclasses import asdict
    comparisons=[];equivalence={}
    for mode in BASELINES:
        left=[];right=[]
        for seed in seeds:
            _,_,m=aligned_rank_matrix({'uasf':comparison_frames['uasf'][seed],mode:comparison_frames[mode][seed]})
            left.append(m[:,0]);right.append(m[:,1])
        a=np.mean(left,axis=0);b=np.mean(right,axis=0)
        comparisons.append(paired_bootstrap_test(a,b,method_b_name=mode,n_bootstrap=run.config['bootstrap_reps']))
        ok,low,high,p=tost_equivalence_test(a,b,relative_margin=.01)
        equivalence[mode]=dict(equivalent=bool(ok),lower=low,upper=high,p=p)
    write_json(root/'baseline_comparisons.json',dict(seeds=seeds,paired_user_seed_means=True,
        confirmatory_family_size=7,holm=asdict(holm_bonferroni_correction(comparisons)),tost=equivalence))
