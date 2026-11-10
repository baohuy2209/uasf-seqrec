"""Auditable missing-metadata evaluation; import never loads a research model.

The v3 evidence proves the effective item mask reaches BOTH early-fusion calls.
Constant ranking metrics are never proof of immunity; unchanged logits require review.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

from .checkpoint2_protocol import sha, digest, write_json, context_signature

VERSION = 3
SEEDS = (42, 43, 44)
MODELS = ('early_sd', 'early_CBT')
SOURCES = ('category', 'brand', 'title')


def _array(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


def tensor_sha(value):
    a = np.ascontiguousarray(_array(value))
    import hashlib
    h = hashlib.sha256()
    h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def make_mask(original, source, rate, seed):
    """Nested sample from available catalog rows; padding is never a catalog row."""
    if source not in original or not 0 <= rate <= 1:
        raise ValueError('Invalid source or missing rate')
    base = np.asarray(_array(original[source])).reshape(-1)
    if not len(base) or not np.isfinite(base).all() or not np.isin(base, [0, 1]).all():
        raise ValueError('Availability must be a nonempty binary vector')
    eligible = np.flatnonzero(base > 0)
    selected = np.zeros(len(base), dtype=bool)
    selected[np.random.default_rng(seed).permutation(eligible)[:round(float(rate)*len(eligible))]] = True
    active = {s: v.clone() for s, v in original.items()}
    active[source][selected] = 0
    table = pd.DataFrame(dict(model_item_id=np.arange(1, len(base)+1),
        original_available=base.astype(int), selected_for_mask=selected.astype(int),
        effective_available=_array(active[source]).astype(int)))
    return active, table


def _checked(root, relative):
    p = (Path(root)/relative).resolve()
    if not p.is_relative_to(Path(root).resolve()):
        raise ValueError('Evidence path escapes dataset directory')
    return p


def _verify_files(root, files):
    for relative, expected in files.items():
        p = _checked(root, relative)
        if not p.is_file() or sha(p) != expected:
            raise ValueError('Robustness evidence missing or changed: '+str(p))


def _record_file(path, writer):
    """Never overwrite conflicting cached evidence."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True); writer(path)


def _diagnostic(job, seed, family, source, cases):
    a,b=cases[0.0],cases[0.8]
    if not np.array_equal(a[1][['user_idx','target_idx']].to_numpy(),b[1][['user_idx','target_idx']].to_numpy()):
        raise ValueError('Unpaired robustness ranks')
    if a[2]['input_sha256']!=b[2]['input_sha256'] or a[2]['user_indices']!=b[2]['user_indices']:
        raise ValueError('Logit probe contexts differ between zero and 80 percent')
    changed=int(np.count_nonzero(a[1]['rank'].to_numpy()!=b[1]['rank'].to_numpy()))
    flat=max(c[0]['ndcg_10'] for c in cases.values())-min(c[0]['ndcg_10'] for c in cases.values())<=1e-14
    probe=b[2];logits=probe['versus_zero']
    if logits['logits_zero_sha256']!=a[2]['logits_sha256']:
        raise ValueError('Zero-vs-80 logit comparison uses a different baseline')
    if logits['changed_logits']<0 or logits['changed_logits']>np.prod(probe['shape']) or logits['logits_max_abs']<0 or logits['logits_l1']<0:
        raise ValueError('Invalid logit comparison statistics')
    if bool(logits['changed_logits']) != bool(logits['logits_max_abs']) or logits['logits_l1']<logits['logits_max_abs']:
        raise ValueError('Inconsistent logit comparison statistics')
    no_effective_removal=b[0]['counts']['removed_items']==0
    pending=bool(not no_effective_removal and not logits['changed_logits'])
    status=('NOT_APPLICABLE_NO_EFFECTIVE_REMOVAL' if no_effective_removal else
            'REVIEW_REQUIRED' if pending else 'VERIFIED_EFFECT_FLAT_METRIC' if flat else 'VERIFIED_EFFECT')
    reason=('No available items were removed at the configured rounding/count' if no_effective_removal else
            'Identical probe logits despite effective masking require an explanation' if pending else
            'Mask reached both scoring stages; metric flatness is reported without claiming immunity')
    rank_file=lambda c:next(p for p in c[0]['files'] if p.endswith('per_user_test.csv'))
    return dict(case_id=f'{job}/{source}',seed=seed,model=family,source=source,
        changed_rank_users_0_vs_80=changed,identical_rank_files_0_vs_80=a[0]['files'][rank_file(a)]==b[0]['files'][rank_file(b)],
        ndcg_delta_80_minus_0=b[0]['ndcg_10']-a[0]['ndcg_10'],flat_metric_curve=flat,
        history_mask_verified=bool(probe.get('effective_mask_verified')),candidate_mask_verified=bool(probe.get('effective_mask_verified')),
        logits_changed_0_vs_80=int(logits['changed_logits']),logits_max_abs_0_vs_80=float(logits['logits_max_abs']),status=status,reason=reason)


def probe_adapter(adapter, ids, intended):
    """One fixed batch; inspect actual kwargs read by history/candidate scoring.

    ModelAdapter.predict calls encode_context and score_full_catalog directly,
    so nn.Module forward hooks do not observe these calls. Temporary method
    wrappers are restored in finally; weights and model state are not changed.
    """
    import torch
    from ..training.multimodal_trainer import gather_source_sequences
    adapter.model.eval(); ids = ids.to(adapter.device)
    adapter.masks = {s: v.to(adapter.device) for s, v in intended.items()}
    model = adapter.model
    expected_history = gather_source_sequences(ids, adapter.attrs, adapter.masks)
    idx = (ids-1).clamp(min=0)
    calls = dict(history=0, candidate=0)
    observed = {}
    original_encode, original_score = model.encode_context, model.score_full_catalog
    had_encode, had_score = 'encode_context' in model.__dict__, 'score_full_catalog' in model.__dict__

    def same(actual, expected, name):
        if actual is None or actual.shape != expected.shape or not torch.equal(actual, expected):
            raise ValueError('Effective robustness mask did not reach '+name)

    def encode(*args, **kwargs):
        calls['history'] += 1
        for source in intended:
            expected = intended[source].to(adapter.device)[idx] * (ids > 0)
            same(kwargs.get(source+'_mask'), expected, 'history '+source)
            field = {'category':'category_ids', 'brand':'brand_ids', 'title':'title_features'}[source]
            same(kwargs.get(field), expected_history[source], 'history values '+source)
            observed['history_'+source+'_mask_sha256'] = tensor_sha(kwargs[source+'_mask'])
            observed['history_'+source+'_values_sha256'] = tensor_sha(kwargs[field])
        return original_encode(*args, **kwargs)

    def score(*args, **kwargs):
        calls['candidate'] += 1
        for source in intended:
            same(kwargs.get('catalog_'+source+'_mask'), intended[source].to(adapter.device), 'candidate '+source)
            observed['candidate_'+source+'_mask_sha256'] = tensor_sha(kwargs['catalog_'+source+'_mask'])
        return original_score(*args, **kwargs)

    try:
        model.encode_context = encode; model.score_full_catalog = score
        with torch.no_grad():
            scores = adapter.predict(ids, adapter.weights(len(ids)))
        if calls != dict(history=1, candidate=1):
            raise ValueError('Did not observe both early scoring stages')
        if not torch.isfinite(scores).all():raise ValueError('Nonfinite robustness probe logits')
        return dict(calls=calls, effective_mask_verified=True, input_sha256=tensor_sha(ids),
                    logits_sha256=tensor_sha(scores), shape=list(scores.shape), **observed), _array(scores).copy()
    finally:
        if had_encode:model.encode_context = original_encode
        else:delattr(model, 'encode_context')
        if had_score:model.score_full_catalog = original_score
        else:delattr(model, 'score_full_catalog')


def run_robustness(run, key):
    """Re-evaluate frozen S1 only; cache v1/v2 is never used."""
    import torch
    from ..training.checkpoint2_trainer import batches, evaluate
    from .checkpoint2 import rank_frame
    run.foundation(key, SEEDS); run.guard(inference=True)
    data = run.dataset(key); samples = data['split'].test_samples
    root = run.output/key; root.mkdir(parents=True, exist_ok=True)
    original = {s:v.detach().cpu().clone() for s,v in data['masks'].items()}
    contexts = context_signature(samples)
    histories = np.asarray([s.history for s in samples], dtype=np.int64)
    if not len(samples):raise ValueError('No robustness evaluation contexts')
    rates = sorted(map(float, run.config['missing_rates']))
    if len(set(rates)) != len(rates) or 0.0 not in rates or 0.8 not in rates:
        raise ValueError('Unique missing rates must include 0 and 0.8')
    rows, diagnostics, files = [], [], {}
    mask_cache = {}
    for seed in SEEDS:
        for source in SOURCES:
            for rate in rates:
                active, table = make_mask(original, source, rate, seed)
                relative = f'robustness_evidence_v3/masks/{source}_{seed}_{rate:.2f}.csv'
                path = root/relative
                _record_file(path, lambda p, t=table:t.to_csv(p,index=False,lineterminator='\n'))
                saved = pd.read_csv(path)
                if not saved.equals(table):raise ValueError('Saved effective mask differs from deterministic mask')
                files[relative] = sha(path); mask_cache[(seed,source,rate)] = (active,table,relative)
    for seed in SEEDS:
        for family in MODELS:
            job = f'{family}_{seed}'; model_hash = run.model_hash(key, job); adapter = None
            for source in SOURCES:
                cases = {}; zero_scores = None
                for rate in rates:
                    active, table, mask_rel = mask_cache[(seed,source,rate)]
                    mask_signature = digest({s:_array(v).tolist() for s,v in active.items()})
                    relative = f'robustness_evidence_v3/{job}/{source}_{rate:.2f}'
                    folder=root/relative; marker=folder/'manifest.json'
                    signature=dict(evaluation_version=VERSION,model=model_hash,job=job,seed=seed,source=source,
                        missing_rate=rate,contexts_sha256=contexts,data=data['signature'],mask_sha256=mask_signature,
                        mask_file_sha256=files[mask_rel],batch_size=run.config['batch_size'])
                    if marker.exists():
                        rec=json.loads(marker.read_text(encoding='utf-8'))
                        if rec['signature']!=signature:raise ValueError('Robustness case signature changed')
                        _verify_files(root,rec['files'])
                    else:
                        if adapter is None:adapter=run.load_job(key,job)
                        adapter.model.eval()
                        batch=next(iter(batches(samples,adapter.attrs,adapter.masks,run.config['batch_size'])))
                        ids=batch['item_seqs']
                        if rate==0.8 and zero_scores is None:
                            _,zero_scores=probe_adapter(adapter,ids,original)
                        probe,scores=probe_adapter(adapter,ids,active)
                        probe.update(user_indices=list(map(int,batch['user_indices'])),
                                     target_indices=list(map(int,batch['targets'].tolist())))
                        if rate==0:zero_scores=scores
                        if rate==0.8:
                            delta=np.abs(scores.astype(np.float64)-zero_scores.astype(np.float64))
                            probe['versus_zero']=dict(logits_l1=float(delta.sum()),logits_max_abs=float(delta.max()),
                                changed_logits=int(np.count_nonzero(delta)),logits_zero_sha256=tensor_sha(zero_scores))
                        # probe_adapter sets exactly the same mask used for the full evaluation.
                        result=evaluate(adapter,samples,run.config['batch_size'])
                        f=rank_frame(result);folder.mkdir(parents=True,exist_ok=True)
                        ranks=folder/'per_user_test.csv';f.to_csv(ranks,index=False,lineterminator='\n')
                        probe_path=folder/'probe.json';write_json(probe_path,probe)
                        selected=table.selected_for_mask.to_numpy(bool)
                        valid=histories>0; indices=np.maximum(histories-1,0)
                        original_hist=_array(original[source])[indices]*valid
                        effective_hist=_array(active[source])[indices]*valid
                        eligible=int(table.original_available.sum());removed=int(selected.sum());remaining=int(table.effective_available.sum())
                        count=dict(catalog_items=len(table),eligible_items=eligible,removed_items=removed,remaining_items=remaining,
                            original_catalog_coverage=eligible/len(table),effective_catalog_coverage=remaining/len(table),
                            effective_eligible_coverage=remaining/eligible if eligible else 0.,
                            history_tokens=int(valid.sum()),original_available_history_tokens=int(original_hist.sum()),
                            effective_available_history_tokens=int(effective_hist.sum()),masked_history_tokens=int((original_hist-effective_hist).sum()),
                            contexts_with_masked_history=int(((original_hist-effective_hist)>0).any(1).sum()),
                            candidate_masked_items=removed,candidate_masked_positions=removed*len(samples),contexts=len(samples))
                        case_files={mask_rel:files[mask_rel],ranks.relative_to(root).as_posix():sha(ranks),probe_path.relative_to(root).as_posix():sha(probe_path)}
                        if rate==0.8:
                            # Catalog coordinates are fixed without using held-out targets.
                            columns=np.linspace(0,scores.shape[1]-1,min(8,scores.shape[1]),dtype=int)
                            paired=pd.DataFrame([dict(user_idx=int(u),model_item_id=int(c+1),
                                logit_zero=float(zero_scores[i,c]),logit_80=float(scores[i,c]),
                                difference=float(scores[i,c]-zero_scores[i,c]),
                                candidate_source_masked=int(table.selected_for_mask.iloc[c]))
                                for i,u in enumerate(batch['user_indices']) for c in columns])
                            paired_path=folder/'paired_logit_probe.csv';paired.to_csv(paired_path,index=False,lineterminator='\n')
                            case_files[paired_path.relative_to(root).as_posix()]=sha(paired_path)
                        rec=dict(signature=signature,counts=count,ndcg_10=float(f['ndcg@10'].mean()),files=case_files)
                        write_json(marker,rec)
                    files.update(rec['files']);files[marker.relative_to(root).as_posix()]=sha(marker)
                    ranks_path=folder/'per_user_test.csv';f=pd.read_csv(ranks_path)
                    cases[rate]=(rec,f,json.loads((folder/'probe.json').read_text(encoding='utf-8')))
                    rows.append(dict(seed=seed,model=family,source=source,missing_rate=rate,ndcg_10=rec['ndcg_10'],
                        model_sha256=model_hash,mask_sha256=mask_signature,mask_file_sha256=files[mask_rel],rank_sha256=sha(ranks_path),
                        probe_sha256=sha(folder/'probe.json'),case_manifest=marker.relative_to(root).as_posix(),**rec['counts']))
                diagnostics.append(_diagnostic(job,seed,family,source,cases))
    summary=root/'robustness.csv';pd.DataFrame(rows).to_csv(summary,index=False,lineterminator='\n')
    diag=root/'robustness_diagnostics.csv';pd.DataFrame(diagnostics).to_csv(diag,index=False,lineterminator='\n')
    files['robustness_diagnostics.csv']=sha(diag)
    write_json(root/'robustness_binding.json',dict(evaluation_version=VERSION,sha256=sha(summary),rows=len(rows),
        data=data['signature'],contexts_sha256=contexts,evidence_sha256=files,
        context_count=len(samples),context_keys_sha256=digest([[int(s.user_idx),int(s.target)] for s in samples]),
        models={f'{m}_{s}':run.model_hash(key,f'{m}_{s}') for s in SEEDS for m in MODELS},
        source_sha256=sha(Path(__file__)),rates=rates,seeds=list(SEEDS),families=list(MODELS),sources=list(SOURCES)))
    return inspect_evidence(root)


def inspect_evidence(root, expected=None):
    """Pure artifact audit for reporting; no checkpoint loading or inference."""
    root=Path(root);binding=json.loads((root/'robustness_binding.json').read_text(encoding='utf-8'))
    if binding.get('evaluation_version')!=VERSION:raise ValueError('Robustness requires effective-mask evidence v3')
    if binding.get('source_sha256')!=sha(Path(__file__)):raise ValueError('Robustness evidence implementation changed')
    if sha(root/'robustness.csv')!=binding['sha256']:raise ValueError('Robustness summary changed')
    _verify_files(root,binding['evidence_sha256'])
    frame=pd.read_csv(root/'robustness.csv')
    keys={(s,m,src,float(r)) for s in binding['seeds'] for m in binding['families'] for src in binding['sources'] for r in binding['rates']}
    actual=set(zip(frame.seed,frame.model,frame.source,frame.missing_rate.astype(float)))
    if actual!=keys or len(frame)!=len(keys) or len(frame)!=binding['rows'] or not np.isfinite(frame.ndcg_10).all():
        raise ValueError('Incomplete robustness coverage')
    if expected:
        for name,value in expected.items():
            if binding.get(name)!=value:raise ValueError('Robustness binding mismatch: '+name)
    case_groups={}
    for row in frame.to_dict('records'):
        if row['case_manifest'] not in binding['evidence_sha256']:raise ValueError('Unbound case manifest')
        rec=json.loads(_checked(root,row['case_manifest']).read_text(encoding='utf-8'))
        if any(binding['evidence_sha256'].get(p)!=h for p,h in rec['files'].items()):raise ValueError('Unbound case artifact')
        for name in ['evaluation_version','data','contexts_sha256']:
            if rec['signature'][name]!=binding[name]:raise ValueError('Case does not belong to aggregate binding: '+name)
        mask=next(p for p in rec['files'] if '/masks/' in p);table=pd.read_csv(_checked(root,mask))
        if not np.array_equal(table.model_item_id,np.arange(1,len(table)+1)) or not np.isin(table[['original_available','effective_available','selected_for_mask']].to_numpy(),[0,1]).all():
            raise ValueError('Invalid saved item mask')
        original=table.original_available.to_numpy();selected=table.selected_for_mask.to_numpy();effective=table.effective_available.to_numpy()
        if np.any(selected>original) or not np.array_equal(effective,original-selected):raise ValueError('Effective mask resurrected or changed natural missingness')
        count=rec['counts']
        if int(selected.sum())!=round(float(row['missing_rate'])*int(original.sum())) or count['removed_items']!=int(selected.sum()) or count['remaining_items']!=int(effective.sum()):
            raise ValueError('Saved mask counts/rate disagree')
        probe_path=next(p for p in rec['files'] if p.endswith('/probe.json'));probe=json.loads(_checked(root,probe_path).read_text(encoding='utf-8'))
        if not probe.get('effective_mask_verified') or probe.get('calls')!=dict(history=1,candidate=1):raise ValueError('Both scoring masks not verified')
        for name in ['seed','source','missing_rate']:
            if rec['signature'][name]!=row[name]:raise ValueError('Case identity mismatch')
        if rec['signature']['job']!=f"{row['model']}_{row['seed']}" or rec['signature']['model']!=row['model_sha256']:
            raise ValueError('Case model identity mismatch')
        if row['model_sha256']!=binding['models'][rec['signature']['job']]:raise ValueError('Case model differs from checkpoint binding')
        for name,value in count.items():
            if not np.isclose(row[name],value,rtol=0,atol=1e-12):raise ValueError('Summary count differs from case evidence: '+name)
        ranks_path=next(p for p in rec['files'] if p.endswith('/per_user_test.csv'));ranks=pd.read_csv(_checked(root,ranks_path))
        required=['user_idx','target_idx','rank','ndcg@10']
        if not all(c in ranks for c in required) or len(ranks)!=count['contexts'] or len(ranks)!=binding['context_count'] or ranks.user_idx.duplicated().any():
            raise ValueError('Invalid or duplicate robustness users/context count')
        if digest(ranks[['user_idx','target_idx']].astype(int).values.tolist())!=binding['context_keys_sha256']:
            raise ValueError('Rank user/target sequence differs from test contexts')
        integer=ranks[['user_idx','target_idx','rank']].to_numpy(dtype=float)
        if not np.isfinite(integer).all() or not np.equal(integer,np.floor(integer)).all() or np.any(integer[:,:2]<0) or np.any(integer[:,2]<1) or np.any(integer[:,2]>count['catalog_items']+1):
            raise ValueError('Invalid robustness rank or target indices')
        if np.any(ranks.target_idx>count['catalog_items']):raise ValueError('Target index exceeds catalog')
        if np.any((ranks.target_idx==0)&(ranks['rank']!=count['catalog_items']+1)):
            raise ValueError('OOV targets must retain the catalog-plus-one miss rank')
        ndcg=np.where((ranks.target_idx>0)&(ranks['rank']>=1)&(ranks['rank']<=min(10,count['catalog_items'])),1/np.log2(1+np.maximum(ranks['rank'],1)),0.)
        if not np.allclose(ndcg,ranks['ndcg@10'],rtol=0,atol=1e-14) or not np.isclose(float(ndcg.mean()),row['ndcg_10'],rtol=0,atol=1e-14):
            raise ValueError('Robustness metrics disagree with saved ranks')
        if row['rank_sha256']!=rec['files'][ranks_path] or row['probe_sha256']!=rec['files'][probe_path] or row['mask_file_sha256']!=rec['files'][mask]:
            raise ValueError('Summary artifact digest differs from case evidence')
        if float(row['missing_rate'])==.8:
            pair_path=next((p for p in rec['files'] if p.endswith('/paired_logit_probe.csv')),None)
            if pair_path is None:raise ValueError('Missing paired logit values')
            paired=pd.read_csv(_checked(root,pair_path))
            if not np.isfinite(paired[['logit_zero','logit_80','difference']].to_numpy()).all() or not np.allclose(paired.logit_80-paired.logit_zero,paired.difference,rtol=1e-6,atol=1e-7):
                raise ValueError('Invalid paired logit values')
        key=(int(row['seed']),row['model'],row['source'])
        case_groups.setdefault(key,{})[float(row['missing_rate'])]=(rec,ranks,probe)
    diagnostic=pd.read_csv(root/'robustness_diagnostics.csv')
    computed=pd.DataFrame([_diagnostic(f'{m}_{s}',s,m,src,cases) for (s,m,src),cases in case_groups.items()])
    try:
        pd.testing.assert_frame_equal(diagnostic.sort_values('case_id').reset_index(drop=True),
            computed.sort_values('case_id').reset_index(drop=True),check_dtype=False,check_exact=False,rtol=0,atol=1e-14)
    except AssertionError as exc:raise ValueError('Diagnostic claims disagree with actual case evidence') from exc
    pending=diagnostic.loc[diagnostic.status=='REVIEW_REQUIRED','case_id'].tolist()
    accepted=False
    review_path=root/'robustness_review.json'
    if pending and review_path.exists():
        review=json.loads(review_path.read_text(encoding='utf-8'))
        reasons=review.get('case_reasons',{})
        from datetime import datetime
        try:date=datetime.fromisoformat(review.get('reviewed_at','').replace('Z','+00:00'))
        except ValueError:date=None
        accepted=(review.get('diagnostics_sha256')==sha(root/'robustness_diagnostics.csv') and
                  isinstance(review.get('reviewer'),str) and bool(review['reviewer'].strip()) and date is not None and date.tzinfo is not None and
                  set(reasons)==set(pending) and all(isinstance(v,str) and len(v.strip())>=20 for v in reasons.values()))
        if not accepted:raise ValueError('Robustness review is stale or lacks explanations')
    return dict(status='REVIEW_REQUIRED' if pending and not accepted else 'HUMAN_REVIEWED_FLAT' if accepted else 'PASS',pending_cases=pending,
                review_accepted=accepted,review_sha256=sha(review_path) if accepted else None,rows=len(frame),evaluation_version=VERSION)


def validate_robustness(run, key):
    data=run.dataset(key)
    expected=dict(data=data['signature'],contexts_sha256=context_signature(data['split'].test_samples),
                  context_count=len(data['split'].test_samples),context_keys_sha256=digest([[int(s.user_idx),int(s.target)] for s in data['split'].test_samples]),
                  seeds=list(SEEDS),families=list(MODELS),sources=list(SOURCES),rates=sorted(map(float,run.config['missing_rates'])),
                  models={f'{m}_{s}':run.model_hash(key,f'{m}_{s}') for s in SEEDS for m in MODELS})
    result=inspect_evidence(run.output/key,expected)
    if result['status'] not in ['PASS','HUMAN_REVIEWED_FLAT']:
        raise ValueError('Robustness flat/identical evidence requires recorded review: '+', '.join(result['pending_cases']))
    return result
