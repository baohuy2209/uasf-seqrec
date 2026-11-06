"""Protocol, provenance and CPU-only analyses for the 03 November feedback."""
from pathlib import Path
from collections import Counter
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import uuid
import numpy as np
import pandas as pd

from metadata_pipeline.user_grouping import fit_group_rules, apply_group_rules, compute_head_tail_partition
from metadata_pipeline.contracts import flags

SEEDS=(42,43,44)
SOURCE_ORDER=('category','brand','title')
SUBSETS=((),('category',),('brand',),('title',),('category','brand'),
         ('category','title'),('brand','title'),SOURCE_ORDER)


def validate_catalog_mapping(catalog,mapping,split):
    """The cohort map includes OOV rows; embeddings contain Train-base rows only."""
    for name,frame in [('metadata.csv',catalog),('item_map.csv',mapping)]:
        if not {'model_item_id','raw_item_id'}.issubset(frame):
            raise ValueError(f'{name}: required model_item_id/raw_item_id columns are missing')
        if frame.raw_item_id.duplicated().any():raise ValueError(f'{name}: duplicate raw item IDs')
    if 'in_train_base' not in mapping:raise ValueError('item_map.csv: missing in_train_base')
    active=flags(mapping.in_train_base)
    if not (mapping.loc[~active,'model_item_id']==0).all():raise ValueError('OOV mapping IDs must be zero')
    train=mapping.loc[active].sort_values('model_item_id')
    expected=[split.idx2item[i] for i in range(1,split.num_items+1)]
    if not np.array_equal(catalog.model_item_id,np.arange(1,split.num_items+1)):
        raise ValueError(f'Catalog model IDs must be 1..{split.num_items}; metadata rows={len(catalog)}')
    if catalog.raw_item_id.astype(str).tolist()!=expected:
        mismatch=next((i+1 for i,(a,b) in enumerate(zip(catalog.raw_item_id.astype(str),expected)) if a!=b),None)
        raise ValueError(f'Raw reviews/catalog mismatch: split={split.num_items}, metadata={len(catalog)}, first differing model ID={mismatch}')
    if not np.array_equal(train.model_item_id,catalog.model_item_id) or train.raw_item_id.astype(str).tolist()!=expected:
        raise ValueError('Train-base rows in item_map.csv differ from metadata.csv')
    return dict(cohort_items=len(mapping),catalog_items=len(catalog),oov_items=int((~active).sum()))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()


def write_json(path,value,overwrite=True):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and not overwrite:raise FileExistsError(path)
    tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.partial')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8',newline='\n')
    tmp.replace(path)


def git_state(repo,require_clean=False):
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    status=subprocess.check_output(['git','status','--porcelain','--untracked-files=all'],cwd=repo,text=True)
    if require_clean and status.strip():
        raise RuntimeError('Commit reviewed code before execution. Working tree is dirty:\n'+status)
    return dict(commit=commit,dirty=bool(status.strip()))


def check_training_environment(expected):
    import torch
    if not torch.cuda.is_available():raise RuntimeError('Locked experiments require the RTX4060 CUDA environment')
    actual=dict(python=platform.python_version(),torch=torch.__version__,gpu=torch.cuda.get_device_name(0))
    actual.update({p:importlib.metadata.version(p) for p in ('numpy','pandas','scipy','scikit-learn')})
    bad={k:(v,actual.get(k)) for k,v in expected.items() if k!='gpu_contains' and str(actual.get(k))!=str(v)}
    if expected.get('gpu_contains','RTX 4060') not in actual['gpu']:
        bad['gpu']=(expected.get('gpu_contains'),actual['gpu'])
    if bad:raise RuntimeError('Environment differs from the locked checkpoint environment; do not auto-upgrade: '+str(bad))
    return actual


def context_signature(samples):
    return digest([[int(s.user_idx),list(map(int,s.history)),int(s.target)] for s in samples])


def split_meta_users(samples,seed=42):
    eligible=[s for s in samples if s.target>0]
    users=np.asarray(sorted({s.user_idx for s in eligible}))
    if len(users)<2:raise ValueError('K2 requires at least two users')
    rng=np.random.default_rng(seed);rng.shuffle(users);fit=set(users[:len(users)//2].tolist())
    return [s for s in eligible if s.user_idx in fit],[s for s in eligible if s.user_idx not in fit]


def read_ranks(path):
    frame=pd.read_csv(path)
    frame=frame.rename(columns={c:c.lower() for c in frame.columns})
    for c in ['user_idx','target_idx','rank','ndcg@10']:
        if c not in frame:raise ValueError(f'Missing rank field: {c}')
    if frame.empty or frame.user_idx.duplicated().any():raise ValueError('One nonempty aligned row per user required')
    if not np.isfinite(frame['ndcg@10']).all():raise ValueError('Nonfinite ranks/metrics')
    return frame.sort_values('user_idx').reset_index(drop=True)


def aligned_rank_matrix(frames,expected_samples=None):
    names=list(frames)
    if not names:raise ValueError('No rank evidence')
    frames={k:f.sort_values('user_idx').reset_index(drop=True) for k,f in frames.items()}
    first=frames[names[0]]
    if first.user_idx.duplicated().any():raise ValueError('Duplicate context user')
    if expected_samples is not None:
        expected=sorted((int(s.user_idx),int(s.target)) for s in expected_samples)
        if list(zip(first.user_idx,first.target_idx))!=expected:raise ValueError('Rank rows do not match split users/targets')
    for name,f in frames.items():
        if list(zip(f.user_idx,f.target_idx))!=list(zip(first.user_idx,first.target_idx)):
            raise ValueError('User/target mismatch between methods: '+name)
    mat=np.column_stack([frames[n]['ndcg@10'].to_numpy(float) for n in names])
    if not np.isfinite(mat).all():raise ValueError('Nonfinite per-user metrics')
    return names,first.user_idx.to_numpy(),mat


def group_contexts(split):
    tail,partition=compute_head_tail_partition(split.train_base_events,split.num_items)
    def features(samples):
        histories=[[i for i in s.history if i>0] for s in samples]
        return np.asarray([len(h) for h in histories]),np.asarray([float(np.mean(tail[h])) if h else 0. for h in histories])
    vl,vt=features(split.val_samples);tl,tt=features(split.test_samples)
    rules=fit_group_rules(vl,vt)
    vg=apply_group_rules(vl,vt,rules)[2];tg=apply_group_rules(tl,tt,rules)[2]
    frame=pd.DataFrame(dict(user_idx=[s.user_idx for s in split.val_samples],val_group_id=vg,
        test_group_id=tg,val_history_len=vl,val_tail_ratio=vt,test_history_len=tl,test_tail_ratio=tt))
    return frame,dict(rules=rules,partition=partition,fit_partition='validation_histories_only')


def grouped_subset_analysis(val_frames,test_frames,groups):
    if set(val_frames)!=set(test_frames):raise ValueError('Subset grid mismatch')
    names,vu,v=aligned_rank_matrix(val_frames);tnames,tu,t=aligned_rank_matrix(test_frames)
    if names!=tnames or not np.array_equal(vu,tu):raise ValueError('Validation/test user grids differ')
    g=groups.set_index('user_idx').reindex(vu)
    if g.isna().any().any():raise ValueError('Missing group assignment')
    vg=g.val_group_id.to_numpy(int);tg=g.test_group_id.to_numpy(int)
    global_idx=int(np.argmax(v.mean(axis=0)));selected=np.full(len(t),global_idx,dtype=int)
    oracle=np.empty(len(t),dtype=int);mapping={};fallback=[]
    for k in sorted(set(tg)|set(vg)):
        if np.any(vg==k):pick=int(np.argmax(v[vg==k].mean(axis=0)))
        else:pick=global_idx;fallback.append(int(k))
        mapping[str(int(k))]=names[pick];selected[tg==k]=pick
        if np.any(tg==k):oracle[tg==k]=int(np.argmax(t[tg==k].mean(axis=0)))
    chosen=t[np.arange(len(t)),selected];upper=t[np.arange(len(t)),oracle]
    return dict(global_subset=names[global_idx],global_test=float(t[:,global_idx].mean()),
        id_test=float(t[:,names.index('ID')].mean()) if 'ID' in names else None,
        group_val_selected_test=float(chosen.mean()),group_test_oracle=float(upper.mean()),
        choices_from_validation=mapping,unseen_validation_groups_fallback_global=fallback,
        oracle_is_descriptive_test_selection=True),pd.DataFrame(dict(user_idx=tu,
            ndcg_global=t[:,global_idx],ndcg_group=chosen,ndcg_oracle=upper))


def paired_seed_ci(real_frames,control_frames,reps=1000,seed=42):
    if set(real_frames)!=set(control_frames) or set(real_frames)!=set(SEEDS):
        raise ValueError('Paired controls require the same three seeds')
    real=[];control=[];key=None
    for s in SEEDS:
        names,users,m=aligned_rank_matrix({'real':real_frames[s],'control':control_frames[s]})
        current=list(zip(real_frames[s].sort_values('user_idx').user_idx,
                         real_frames[s].sort_values('user_idx').target_idx))
        if key is not None and key!=current:raise ValueError('Seed cohorts differ')
        key=current;real.append(m[:,0]);control.append(m[:,1])
    # Pair users, retaining all evaluated seed predictions together for a user.
    delta=np.mean(np.asarray(control)-np.asarray(real),axis=0)
    rng=np.random.default_rng(seed)
    means=[float(delta[rng.integers(0,len(delta),len(delta))].mean()) for _ in range(reps)]
    return dict(delta_control_minus_real=float(delta.mean()),ci95=np.percentile(means,[2.5,97.5]).tolist(),
        real_by_seed=[dict(seed=s,value=float(real[i].mean())) for i,s in enumerate(SEEDS)],
        control_by_seed=[dict(seed=s,value=float(control[i].mean())) for i,s in enumerate(SEEDS)],
        resampling_unit='user_with_seed_predictions_together',replicates=reps,
        limitation='Conditional on these three training seeds; not a training-seed CI')


def choose_utility_label(ce_means,rank_means,val_ndcg_diffs,sources=SOURCE_ORDER):
    if any(s not in ce_means or s not in rank_means or s not in val_ndcg_diffs for s in sources):
        return 'PENDING'
    def matches(values):
        return all(np.isfinite(values[s]) and np.isfinite(val_ndcg_diffs[s])
                   and np.sign(values[s])==np.sign(val_ndcg_diffs[s])
                   and np.sign(val_ndcg_diffs[s])!=0 for s in sources)
    if matches(ce_means):return 'CE'
    if matches(rank_means):return 'LOG_RANK'
    return 'UNRESOLVED'


def history_statistics(samples,split,attrs,masks):
    counts=Counter(split.train_base_events)
    tail,_=compute_head_tail_partition(split.train_base_events,split.num_items)
    a={s:v.detach().cpu().numpy() for s,v in attrs.items()}
    m={s:v.detach().cpu().numpy() for s,v in masks.items()}
    rows=[]
    for sample in samples:
        h=[i for i in sample.history if i>0]
        if not h:raise ValueError('Empty context history')
        row=dict(feat_log_len=float(np.log(len(h))),feat_history_logpop_mean=float(np.mean([np.log1p(counts[i]) for i in h])),
            feat_last_logpop=float(np.log1p(counts[h[-1]])),feat_history_tail_ratio=float(np.mean(tail[h])))
        for s in SOURCE_ORDER:
            available=m[s][np.asarray(h)-1]>0;values=[]
            for i,ok in zip(h,available):
                if not ok:continue
                v=a[s][i-1]
                if s=='title':values.append(v.tobytes())
                else:values.extend(int(x) for x in np.asarray(v).reshape(-1) if x>0)
            c=np.asarray(list(Counter(values).values()),dtype=float);pr=c/c.sum() if len(c) else c
            row.update({f'feat_{s}_coverage':float(available.mean()),
                f'feat_{s}_entropy':float(-np.sum(pr*np.log(pr))) if len(pr) else 0.,
                f'feat_{s}_recency':float(available[-3:].mean())})
        rows.append(row)
    return pd.DataFrame(rows)
