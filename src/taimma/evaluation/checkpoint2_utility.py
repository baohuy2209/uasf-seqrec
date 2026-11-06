"""Same-model CE/log-rank ablations; early fusion is re-encoded for every mask."""
import numpy as np
import pandas as pd
import torch
from .evaluator import IndependentEvaluator
from ..training.checkpoint2_trainer import batches
from ..models.source_dropout import SOURCES
from ..experiments.checkpoint2_protocol import digest, history_statistics


def context_ids(samples):
    return [digest([int(s.user_idx),list(map(int,s.history)),int(s.target)]) for s in samples]


def score_contexts(adapter,samples,batch_size=256):
    """Small scalar evidence cache instead of an unverified full-logit cache."""
    adapter.model.eval();ev=IndependentEvaluator(adapter.model.num_items,mask_policy='input_history_strict')
    rows=[];offset=0;ids=context_ids(samples)
    with torch.no_grad():
        for b in batches(samples,adapter.attrs,adapter.masks,batch_size):
            scores=adapter.predict(b['item_seqs']);targets=b['targets'].to(adapter.device)
            valid=targets>0
            ce=torch.full((len(targets),),float('nan'),device=adapter.device)
            ce[valid]=torch.nn.functional.cross_entropy(scores[valid],targets[valid]-1,reduction='none')
            result=ev.evaluate_ranks(scores,targets.cpu().tolist(),b['user_indices'],histories=b['item_seqs'].tolist())
            for i,rec in enumerate(result.per_user_records):
                rows.append(dict(context_id=ids[offset+i],user_idx=int(rec.user_idx),target_idx=int(rec.target_idx),
                    ce=float(ce[i]) if valid[i] else None,rank=int(rec.rank),ndcg_10=float(rec.ndcg_10)))
            offset+=len(targets)
    return pd.DataFrame(rows)


def paired_ce_evidence(id_rows,late_rows):
    keys=['context_id','user_idx','target_idx']
    if id_rows.duplicated(keys).any() or late_rows.duplicated(keys).any():raise ValueError('Duplicate CE contexts')
    if set(map(tuple,id_rows[keys].to_numpy()))!=set(map(tuple,late_rows[keys].to_numpy())):
        raise ValueError('K1 requires exactly aligned contexts/labels')
    joined=id_rows.merge(late_rows,on=keys,suffixes=('_id','_late'),validate='one_to_one')
    valid=joined.ce_id.notna() & joined.ce_late.notna()
    if not valid.any():raise ValueError('No eligible CE targets')
    joined['ce_late_minus_id']=joined.ce_late-joined.ce_id
    return dict(contexts=len(joined),valid_targets=int(valid.sum()),
        mean_ce_id=float(joined.loc[valid,'ce_id'].mean()),mean_ce_late=float(joined.loc[valid,'ce_late'].mean()),
        mean_ce_late_minus_id=float(joined.loc[valid,'ce_late_minus_id'].mean())),joined


def _masked_for_features(scores,histories):
    arr=scores.detach().cpu().numpy().astype(np.float64,copy=True)
    for i,h in enumerate(histories):
        seen=np.asarray(sorted({int(x)-1 for x in h if 0<int(x)<=arr.shape[1]}),dtype=int)
        arr[i,seen]=-np.inf
    return arr


def collect_ablations(adapter,samples,split,batch_size=256,include_features=True,availability_aware=False,allow_oov=False):
    """No optimizer. All alpha=1 versus alpha_m=0 at BOTH early-fusion sides.

    Features are obtained from ungated reference predictions, never from target
    ranks, losses, labels or a gated encoder. Each reference's utility labels can
    subsequently be paired with the SAME seed42 feature table.
    """
    # S2 labels require known targets. Validation/test can retain OOV misses in
    # ranking denominators; their CE fields are missing, never target -1.
    if any(s.target<0 or s.target>adapter.model.num_items or (s.target==0 and not allow_oov) for s in samples):
        raise ValueError('S2 requires in-catalog targets; pass allow_oov=True only for evaluation/features')
    adapter.model.eval();ev=IndependentEvaluator(adapter.model.num_items,mask_policy='input_history_strict')
    ids=context_ids(samples);rows=[];features=[];offset=0
    history_feats=history_statistics(samples,split,adapter.attrs,adapter.masks) if include_features else None
    with torch.no_grad():
        for b in batches(samples,adapter.attrs,adapter.masks,batch_size):
            n=len(b['targets']);targets=b['targets'].to(adapter.device)
            available=adapter.availability(b['item_seqs']) if availability_aware else torch.ones(n,3,device=adapter.device)
            full_weights=adapter.weights(n)*available
            full=adapter.predict(b['item_seqs'],full_weights)
            valid=targets>0
            def ce_values(scores):
                losses=torch.full((n,),float('nan'),device=adapter.device)
                if valid.any():losses[valid]=torch.nn.functional.cross_entropy(scores[valid],targets[valid]-1,reduction='none')
                return losses.cpu().numpy()
            ce_full=ce_values(full)
            full_rank=ev.evaluate_ranks(full,targets.cpu().tolist(),b['user_indices'],histories=b['item_seqs'].tolist())
            h=adapter.context(b['item_seqs'],full_weights).cpu().numpy() if include_features else None
            base_features=history_feats.iloc[offset:offset+n].to_dict('records') if include_features else None
            if include_features:
                full_masked=_masked_for_features(full,b['item_seqs'].tolist())
                top_n=min(10,adapter.model.num_items)
                # Fixed catalog-index tie rule, never ground-truth identity.
                top_full=np.argsort(-full_masked,axis=1,kind='stable')[:,:top_n]
            for source in SOURCES:
                enabled=tuple(s for s in adapter.spec.sources if s!=source)
                # Recompute the complete forward pass, not scores_full-residual.
                without=adapter.predict(b['item_seqs'],adapter.weights(n,enabled)*available)
                ce_without=ce_values(without)
                ranks=ev.evaluate_ranks(without,targets.cpu().tolist(),b['user_indices'],histories=b['item_seqs'].tolist())
                if include_features:
                    masked=_masked_for_features(without,b['item_seqs'].tolist())
                    top_without=np.argsort(-masked,axis=1,kind='stable')[:,:top_n]
                    x=full.detach().cpu().numpy().astype(float);y=without.detach().cpu().numpy().astype(float)
                    numerator=(x*y).sum(axis=1);denom=np.linalg.norm(x,axis=1)*np.linalg.norm(y,axis=1)
                    cos=numerator/np.maximum(denom,1e-12)
                for i,(r0,r1) in enumerate(zip(full_rank.per_user_records,ranks.per_user_records)):
                    rows.append(dict(context_id=ids[offset+i],user_idx=int(r0.user_idx),target_idx=int(r0.target_idx),
                        source_name=source,reference_seed=adapter.spec.seed,seq_len=int((b['item_seqs'][i]>0).sum()),
                        ce_all=float(ce_full[i]),ce_without_m=float(ce_without[i]),
                        delta_ce=float(ce_without[i]-ce_full[i]),
                        delta_rank=float(np.log2(1+r1.rank)-np.log2(1+r0.rank)),
                        ndcg_full=float(r0.ndcg_10),ndcg_without=float(r1.ndcg_10),
                        rank_full=int(r0.rank),rank_without=int(r1.rank)))
                    if include_features:
                        base_features[i][f'feat_{source}_top10_agreement']=len(set(top_full[i])&set(top_without[i]))/top_n
                        base_features[i][f'feat_{source}_score_cosine']=float(cos[i])
            if include_features:
                for i,row in enumerate(base_features):
                    row.update({f'feat_state_{j:03d}':float(v) for j,v in enumerate(h[i])})
                    row.update(context_id=ids[offset+i],user_idx=int(b['user_indices'][i]))
                    features.append(row)
            offset+=n
    return pd.DataFrame(rows),pd.DataFrame(features)
