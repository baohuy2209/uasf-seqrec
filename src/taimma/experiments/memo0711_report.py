"""Read-only measured table generation: no models, inference, or fitted predictors."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def validate_rank_measurements(frame, source):
    """Recompute one-target NDCG, including OOV/strict-history misses as zero."""
    required={'user_idx','target_idx','rank','ndcg@10'}
    if not required.issubset(frame):raise ValueError('Missing rank columns: '+str(source))
    if frame.empty or frame.user_idx.duplicated().any():raise ValueError('Empty/duplicate rank cohort: '+str(source))
    rank=frame['rank'].to_numpy(float);target=frame.target_idx.to_numpy(float)
    metric=frame['ndcg@10'].to_numpy(float)
    if not np.isfinite(rank).all() or not np.isfinite(target).all() or not np.isfinite(metric).all() or np.any(rank<1) or np.any(rank!=np.floor(rank)) or np.any(target<0) or np.any(target!=np.floor(target)):
        raise ValueError('Invalid rank/target/metric: '+str(source))
    expected=np.where((rank<=10)&(target>0),1/np.log2(1+rank),0.)
    if not np.allclose(metric,expected,rtol=0,atol=1e-12):
        raise ValueError('Stored NDCG disagrees with ranks/OOV denominator: '+str(source))


def inspect_group_lineage(root):
    root=Path(root).resolve();path=root/'group_lineage.json'
    value=json.loads(path.read_text(encoding='utf-8'))
    body={k:v for k,v in value.items() if k!='lineage_id'}
    expected=hashlib.sha256(json.dumps(body,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
    if value.get('schema_version')!=3 or value.get('lineage_id')!=expected:raise ValueError('Group lineage declaration changed')
    def check(relative,h):
        p=(root/relative).resolve()
        if not p.is_relative_to(root):raise ValueError('Group artifact escapes cohort')
        if not p.is_file() or file_sha(p)!=h:raise ValueError('Group artifact missing/changed: '+str(p))
    for name,h in value['files'].items():check(name,h)
    expected_analyses={(kind,seed) for kind in ['early','late'] for seed in [42,43,44]}
    if {(r['mechanism'],r['seed']) for r in value['analyses']}!=expected_analyses or len(value['analyses'])!=6:raise ValueError('Incomplete group analysis coverage')
    for analysis in value['analyses']:
        for name,h in analysis['ranks_sha256'].items():check(name,h)
        for job,rec in analysis['models'].items():check('jobs/'+job+'/completed.json',rec['completed_sha256'])
        check(f"group_{analysis['mechanism']}_{analysis['seed']}.csv",analysis['selected_users_sha256'])
    return value


def audit_runtime(output,config):
    """CPU metadata/rank inspection only: never writes or imports a model runner.

    This is a completeness audit, not permission to bypass validate_ready/freeze.
    """
    output=Path(output).resolve();checks=[]
    def result(dataset,item,status,detail=''):
        checks.append(dict(dataset=dataset,item=item,status=status,detail=str(detail)))
    if not output.is_dir():
        return dict(status='INCOMPLETE',training=False,inference=False,source=str(output),
                    checks=[dict(dataset='all',item='runtime',status='MISSING',detail=str(output))])
    for dataset in config.get('datasets',['beauty','sports','toys']):
        root=output/dataset
        for name in ['group_analysis.json','group_lineage.json','selected_s2/status.json']:
            if not (root/name).is_file():
                # Transfer S2 is conditional; absence outside Beauty is explicit N/A under F2.
                if dataset!='beauty' and name.startswith('selected_s2'):
                    b=output/'beauty/framing_decision.json'
                    decision=json.loads(b.read_text(encoding='utf-8')) if b.is_file() else {}
                    if decision.get('framing')=='F2':result(dataset,name,'NOT_APPLICABLE_F2');continue
                result(dataset,name,'MISSING')
            else:result(dataset,name,'PRESENT')
        if (root/'group_lineage.json').is_file():
            try:
                lineage=inspect_group_lineage(root);result(dataset,'group_lineage','PASS',lineage['lineage_id'])
            except (ValueError,KeyError,FileNotFoundError) as exc:result(dataset,'group_lineage','FAILED',exc)
        try:
            from .memo0711_robustness import inspect_evidence
            robust=inspect_evidence(root)
            result(dataset,'robustness',robust['status'],json.dumps(robust,ensure_ascii=False))
        except (FileNotFoundError,ValueError,KeyError,ImportError) as exc:
            result(dataset,'robustness','INCOMPLETE',exc)
        if (root/'selected_s2/status.json').is_file():
            try:
                from .memo0711_e1_evidence import inspect_e1_evidence
                e1=inspect_e1_evidence(root)
                result(dataset,'e1_oof_evidence',e1['status'],json.dumps(e1,ensure_ascii=False))
            except (FileNotFoundError,ValueError,KeyError,ImportError) as exc:result(dataset,'e1_oof_evidence','MISSING_OR_INVALID_E1_OOF',exc)
        expected_jobs={f'id_{s}' for s in [42,43,44]}
        for kind in ['early','late']:
            expected_jobs.update(f'{kind}_{lab}_{s}' for lab in ['C','B','T','CB','CT','BT','CBT'] for s in [42,43,44])
        if dataset=='beauty':
            expected_jobs.update(f'{prefix}_{s}' for prefix in ['id','early_CBT','early_sd','id_plus_meta','early_plus_meta'] for s in range(42,47))
            expected_jobs.update(f'title_shuffled_{s}' for s in [42,43,44])
        for name in sorted(expected_jobs):
            if not (root/'jobs'/name/'completed.json').is_file():result(dataset,name,'MISSING_JOB')
        for marker in sorted((root/'jobs').glob('*/completed.json')):
            try:
                rec=json.loads(marker.read_text(encoding='utf-8'))
                job=marker.parent
                for rel,expected in rec.get('artifacts',{}).items():
                    p=(job/rel).resolve()
                    if not p.is_relative_to(job.resolve()):raise ValueError('Artifact escapes job')
                    if not p.is_file():raise FileNotFoundError(p)
                    if file_sha(p)!=expected:raise ValueError('Artifact hash changed: '+str(p))
                for part in ['val','test']:
                    p=(job/rec['checkpoint']).parent/f'per_user_{part}.csv'
                    if not p.is_file():
                        if part=='test' and rec.get('test_status')=='NOT_OPENED':continue
                        if part=='val':raise FileNotFoundError(p)
                        continue
                    f=pd.read_csv(p);validate_rank_measurements(f,p)
                    value=rec.get(part+'_ndcg_10')
                    if value is not None and not np.isclose(f['ndcg@10'].mean(),value,rtol=0,atol=1e-12):raise ValueError('Summary disagrees with ranks: '+str(p))
                result(dataset,marker.parent.name,'PASS')
            except (ValueError,KeyError,FileNotFoundError) as exc:result(dataset,marker.parent.name,'FAILED',exc)
        status_path=root/'selected_s2/status.json'
        status=json.loads(status_path.read_text(encoding='utf-8')) if status_path.is_file() else {}
        if status.get('s3_eligible'):
            seeds=range(42,47) if dataset=='beauty' else [42,43,44]
            modes=['uasf','global_gate','unsupervised','softmax_null','wo_utility','state_only','add_one','fallback_s1','wo_source_dropout'] if dataset=='beauty' else ['uasf','global_gate','unsupervised','softmax_null']
            for seed in seeds:
                for mode in modes:
                    folder=root/'gates'/f'{mode}_{seed}'
                    try:
                        rec=json.loads((folder/'completed.json').read_text(encoding='utf-8'))
                        for rel,h in rec['artifacts'].items():
                            artifact=(folder/rel).resolve()
                            if not artifact.is_relative_to(folder.resolve()) or not artifact.is_file() or file_sha(artifact)!=h:raise ValueError('Gate checkpoint artifact missing/changed')
                        rank=folder/'per_user_val.csv';binding=json.loads(rank.with_suffix('.binding.json').read_text(encoding='utf-8'))
                        if binding.get('signature')!={'gate':rec['signature'],'part':'val'} or binding.get('sha256')!=file_sha(rank):raise ValueError('Gate validation binding changed')
                        validate_rank_measurements(pd.read_csv(rank),rank)
                        result(dataset,'gate:'+folder.name,'PASS')
                    except (ValueError,KeyError,FileNotFoundError) as exc:result(dataset,'gate:'+folder.name,'MISSING_OR_INVALID_GATE',exc)
        for p in sorted((root/'gates').glob('*/per_user_*.csv')):
            try:
                binding=json.loads(p.with_suffix('.binding.json').read_text(encoding='utf-8'))
                if binding.get('sha256')!=file_sha(p):raise ValueError('Gate rank hash changed')
                validate_rank_measurements(pd.read_csv(p),p)
                result(dataset,str(p.relative_to(root)),'PASS')
            except (ValueError,KeyError,FileNotFoundError) as exc:result(dataset,str(p.relative_to(root)),'FAILED',exc)
    decision_path=output/'beauty/framing_decision.json'
    if decision_path.is_file():
        decision=json.loads(decision_path.read_text(encoding='utf-8'))
        family=decision.get('framing_family',[])
        official=['all_fixed','global_subset','group_subset','global_gate','unsupervised','softmax_null','independent_ID','early_no_sd','id_plus_meta','early_plus_meta']
        status_path=output/'beauty/selected_s2/status.json'
        s2=json.loads(status_path.read_text(encoding='utf-8')) if status_path.is_file() else {}
        no_gate=(decision.get('framing')=='F2' and s2.get('s3_eligible') is False
                 and decision.get('reason')==s2.get('decision')
                 and decision.get('s2_status_sha256')==file_sha(status_path))
        result('beauty','framing_family','NOT_APPLICABLE_NO_ELIGIBLE_S2' if no_gate else 'PASS' if family==official else 'UNVERIFIED',json.dumps(family))
    else:result('beauty','framing_decision','MISSING')
    blocked=[r for r in checks if r['status'] not in ['PASS','PRESENT','VERIFIED','VERIFIED_EFFECT','HUMAN_REVIEWED_FLAT','NOT_APPLICABLE_F2','NOT_APPLICABLE_NO_ELIGIBLE_S2']]
    return dict(status='PASS' if not blocked else 'INCOMPLETE',training=False,inference=False,
                source=str(output),checks=checks,issues=len(blocked),freeze_authorized=False)


def feature_definition(name):
    """Actual history_statistics/collect_ablations contract, without target fields."""
    exact={
        'feat_log_len':'ln(count of nonzero IDs in input history).',
        'feat_history_logpop_mean':'Mean ln(1+Train-base event count(item)) over input history.',
        'feat_last_logpop':'ln(1+Train-base event count(last input item)).',
        'feat_history_tail_ratio':'Mean Train-base tail flag over nonzero input-history IDs.'}
    if name in exact:return exact[name]
    if name.startswith('feat_state_') and name.removeprefix('feat_state_').isdigit():return 'Frozen ungated S1 context coordinate; detached, input-history only.'
    for source in ['category','brand','title']:
        prefix='feat_'+source+'_'
        if name==prefix+'top10_agreement':return 'Top-10 overlap of ungated full vs leave-one-source-out scores / top_n; no target lookup.'
        if name==prefix+'score_cosine':return 'Cosine of ungated full and leave-one-source-out full-catalog logits; no target lookup.'
        if name==prefix+'coverage':return 'Fraction of input-history items with the source available.'
        if name==prefix+'recency':return 'Available-source fraction over the last min(3,history length) items.'
        if name==prefix+'entropy':return 'Shannon -sum(p ln p) of available categorical nonzero labels; Title uses vector-byte identities, not semantic entropy.'
    return 'UNVERIFIED_FEATURE: consult source before using or claiming inference availability.'


def latex_table(frame):
    """Input fragment only; escaping does not require pandas/Jinja LaTeX extras."""
    mapping={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$',
             '#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}'}
    def escape(value):
        if value is None or (isinstance(value,float) and np.isnan(value)):return '--'
        text=f'{value:.6f}' if isinstance(value,(float,np.floating)) else str(value)
        return ''.join(mapping.get(c,c) for c in text.replace('\n',' '))
    lines=['% Generated measured table input; -- means missing, not zero.',
           r'\begin{tabular}{'+'l'*len(frame.columns)+'}',r'\hline',
           ' & '.join(escape(c) for c in frame.columns)+r' \\',r'\hline']
    lines.extend(' & '.join(escape(v) for v in row)+r' \\' for row in frame.itertuples(index=False,name=None))
    return '\n'.join(lines+[r'\hline',r'\end{tabular}',''])


def measured_figures(tables,dest,runtime):
    """Plots only complete measured seed summaries; no placeholder plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as exc:
        return [],dict(status='MISSING_PLOTTING_DEPENDENCY',detail=str(exc))
    generated=[]
    def finish(fig,name,caption):
        fig.text(.01,.015,caption,fontsize=8)
        fig.tight_layout(rect=(0,.065,1,1))
        for ext in ['svg','pdf']:
            p=dest/(name+'.'+ext);fig.savefig(p);generated.append(p)
        plt.close(fig)
    b=tables.get('B_three_datasets')
    if b is not None:
        b=b[(b['part']=='test')&(b['status']=='COMPLETE')].copy()
        order=['ID','C','B','T','CB','CT','BT','CBT']
        for dataset,frame in b.groupby('dataset'):
            if frame.empty:continue
            fig,ax=plt.subplots(figsize=(8,4));colors={'early':'#245b9e','late':'#be642b'}
            for i,kind in enumerate(['early','late']):
                f=frame[frame.mechanism==kind]
                if f.empty:continue
                x=np.array([order.index(v) for v in f.subset])+(-.18 if i==0 else .18)
                ax.bar(x,f['mean'],width=.34,yerr=f['sd'],capsize=2,label=kind,color=colors[kind])
            ax.set_xticks(range(len(order)),order);ax.set_xlabel('Metadata subset (ID = no metadata)')
            ax.set_ylabel('Test NDCG@10 (unitless)');ax.set_title(f'{dataset}: early vs late fusion')
            ax.legend(title='Mechanism');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
            finish(fig,'figure_B_'+dataset,f'Source: {runtime.name}/{dataset}/jobs; mean ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â± sample SD, seeds 42/43/44. Missing subsets omitted.')
    for key,frame in tables.items():
        if not key.endswith('_robustness') or frame.empty:continue
        dataset=key.removesuffix('_robustness')
        if 'evidence_status' in frame:
            frame=frame[frame.evidence_status=='PASS']
        for source,f in frame.groupby('source'):
            summary=f.groupby(['model','missing_rate']).agg(mean=('ndcg_10','mean'),sd=('ndcg_10','std'),n=('seed','nunique')).reset_index()
            summary=summary[summary.n==3]
            if summary.empty:continue
            fig,ax=plt.subplots(figsize=(7,4))
            for family,g in summary.groupby('model'):
                g=g.sort_values('missing_rate');ax.errorbar(g.missing_rate*100,g['mean'],yerr=g['sd'],marker='o',capsize=2,label=family)
            ax.set_xlabel(f'Available {source} items synthetically masked (%)')
            ax.set_ylabel('Test NDCG@10 (unitless)');ax.set_title(f'{dataset}: missing-{source} robustness')
            ax.legend(title='S1 family');ax.grid(alpha=.2)
            finish(fig,f'figure_robustness_{dataset}_{source}',f'Source: {runtime.name}/{dataset}/robustness.csv (evaluation v3); mean ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â± sample SD, 3 seeds. Natural missingness retained.')
    return generated,dict(status='GENERATED' if generated else 'NO_COMPLETE_MEASURED_DATA',figure_files=len(generated))


def write_memo_report(output,config,destination):
    """Export measured tables and explicit gaps outside the immutable run tree."""
    output=Path(output).resolve();dest=Path(destination).resolve()
    if dest==output or output in dest.parents or dest in output.parents:
        raise ValueError('Report destination must be separate from runtime output')
    dest.mkdir(parents=True,exist_ok=True)
    previous=dest/'package_manifest.json'
    if previous.is_file():
        old=json.loads(previous.read_text(encoding='utf-8'))
        owned=[dest/name for name in old.get('files',{})]
        if any(p.parent!=dest or p.suffix not in ['.csv','.md','.json','.tex','.svg','.pdf'] for p in owned):
            raise ValueError('Unsafe generated report manifest')
        for p in owned:
            if p.is_file():p.unlink()
    inputs={};tables={};gaps=[]
    def read_json(p):
        p=Path(p)
        if not p.is_file():return None
        inputs[str(p)]=file_sha(p)
        return json.loads(p.read_text(encoding='utf-8'))
    def csv(p):
        p=Path(p)
        if not p.is_file():return None
        inputs[str(p)]=file_sha(p)
        binding=next((q for q in [p.with_suffix('.binding.json'),p.with_suffix('.json')] if q.is_file()),None)
        meta=read_json(binding) if binding else {}
        if meta.get('sha256') and meta['sha256']!=inputs[str(p)]:
            raise ValueError('CSV cache binding changed: '+str(p))
        f=pd.read_csv(p)
        if {'rank','target_idx','ndcg@10'}.issubset(f):validate_rank_measurements(f,p)
        f.attrs['binding_status']='BOUND' if meta.get('sha256') else 'NO_ADJACENT_BINDING'
        return f
    def save(name,rows):
        f=pd.DataFrame(rows)
        if not f.empty:
            f.to_csv(dest/(name+'.csv'),index=False,lineterminator='\n');tables[name]=f
    jobs=[];records={}
    roots=[output]+[Path(v) for v in config.get('input_roots',{}).values()]
    for r in roots:
        for p in sorted(r.glob('*/jobs/*/completed.json')):
            key=(p.parent.parent.parent.name,p.parent.name)
            if key in records:continue
            rec=read_json(p);records[key]=(p.parent,rec)
            missing=[]
            for rel,expected in rec.get('artifacts',{}).items():
                q=(p.parent/rel).resolve()
                if not q.is_relative_to(p.parent.resolve()):raise ValueError('Artifact escapes source: '+str(q))
                if not q.exists():missing.append(rel);continue
                if file_sha(q)!=expected:raise ValueError('Changed result artifact: '+str(q))
            jobs.append(dict(dataset=key[0],job=key[1],seed=rec['spec']['seed'],
                val_ndcg_10=rec.get('val_ndcg_10'),test_ndcg_10=rec.get('test_ndcg_10'),
                best_epoch=rec.get('best_epoch'),best_epoch_le5=rec.get('best_epoch',999)<=5,
                source=str(p),missing_artifacts=json.dumps(missing),
                verification='INCOMPLETE_LOCAL_BUNDLE' if missing else 'ARTIFACT_HASHES_VERIFIED'))
    save('job_inventory',jobs)
    fairness=[]
    for dataset in config.get('datasets',['beauty','sports','toys']):
        for family in ['id_plus_meta','early_plus_meta']:
            for part in ['val','test']:
                measured=[r[1].get(part+'_ndcg_10') for (cohort,name),r in records.items()
                          if cohort==dataset and name.rsplit('_',1)[0]==family]
                measured=[v for v in measured if v is not None]
                if measured:fairness.append(dict(dataset=dataset,family=family,part=part,n_seeds=len(measured),
                    mean=float(np.mean(measured)),sd=float(np.std(measured,ddof=1)) if len(measured)>1 else None,
                    train_cohort='Train-base+eligible Train-meta'))
    save('fairness_summary',fairness)
    def metric(dataset,name,part):
        pair=records.get((dataset,name))
        return pair[1].get(part+'_ndcg_10') if pair else None
    b=[];adddrop=[]
    for dataset in config.get('datasets',['beauty','sports','toys']):
        for kind in ['early','late']:
            for lab in ['ID','C','B','T','CB','CT','BT','CBT']:
                for part in ['val','test']:
                    vals=[metric(dataset,f'id_{s}' if lab=='ID' else f'{kind}_{lab}_{s}',part) for s in [42,43,44]]
                    good=[x for x in vals if x is not None]
                    b.append(dict(dataset=dataset,mechanism=kind,subset=lab,part=part,n_seeds=len(good),
                        mean=float(np.mean(good)) if len(good)==3 else None,
                        sd=float(np.std(good,ddof=1)) if len(good)==3 else None,
                        status='COMPLETE' if len(good)==3 else 'MISSING_SEEDS'))
            for source,without in [('C','BT'),('B','CT'),('T','CB')]:
                for seed in [42,43,44]:
                    a,i,full,off=[metric(dataset,f'id_{seed}' if lab=='ID' else f'{kind}_{lab}_{seed}','test') for lab in [source,'ID','CBT',without]]
                    adddrop.append(dict(dataset=dataset,mechanism=kind,source=source,seed=seed,
                        add=a-i if a is not None and i is not None else None,
                        drop=full-off if full is not None and off is not None else None))
        for expected in ['group_analysis.json','robustness.csv','framing_decision.json']:
            if not (output/dataset/expected).is_file():gaps.append(dict(dataset=dataset,item=expected,status='MISSING'))
    save('B_three_datasets',b);save('add_drop_by_seed',adddrop)
    gates=[];utility=[];evidence=[];shared=[];group_rows=[];framing_rows=[];features=[];dataset_rows=[];history=[];oof_rows=[];oof_files=[]
    for dataset in config.get('datasets',['beauty','sports','toys']):
        root=output/dataset
        for name in ['robustness','robustness_diagnostics','gate_validation','gate_test','title_only_controls','late_rank_diagnostics']:
            f=csv(root/(name+'.csv'))
            if f is not None:
                if name=='robustness_diagnostics' and (root/'robustness_binding.json').is_file():
                    rb=read_json(root/'robustness_binding.json')
                    if rb.get('evaluation_version')!=3:continue
                if name in ['gate_validation','gate_test'] and 'mode' in f:
                    bad=[]
                    for i,row in f[f['mode']=='add_one'].iterrows():
                        rec=read_json(root/'gates'/f'add_one_{int(row.seed)}'/'completed.json') or {}
                        if rec.get('evaluation_version')!=2:bad.append(i)
                    f=f.drop(index=bad)
                if name=='robustness':
                    binding=read_json(root/'robustness_binding.json')
                    if not binding or binding.get('evaluation_version')!=3:
                        gaps.append(dict(dataset=dataset,item='robustness',status='UNVERIFIED_ROBUSTNESS_MISSING_V3_EVIDENCE'));continue
                    if binding.get('sha256')!=file_sha(root/'robustness.csv'):
                        raise ValueError('Robustness binding changed: '+dataset)
                    from .memo0711_robustness import inspect_evidence
                    inspection=inspect_evidence(root)
                    if inspection['status'] not in ['PASS','VERIFIED','VERIFIED_EFFECT','HUMAN_REVIEWED_FLAT']:
                        gaps.append(dict(dataset=dataset,item='robustness',status='REVIEW_REQUIRED',detail=json.dumps(inspection)))
                    f['evidence_status']=inspection['status']
                save(dataset+'_'+name,f.to_dict('records'))
        for name in ['framing_decision','baseline_comparisons','title_only_comparison','group_analysis','group_lineage','robustness_binding','robustness_review']:
            d=read_json(root/(name+'.json'))
            if d is not None:evidence.append(dict(dataset=dataset,kind=name,payload=json.dumps(d,ensure_ascii=False)))
        group_path=root/'group_analysis.json'
        group_data=read_json(group_path)
        lineage=None
        if (root/'group_lineage.json').is_file():lineage=inspect_group_lineage(root)
        if group_data:
            for row in group_data:
                group_rows.append(dict(dataset=dataset,mechanism=row['mechanism'],seed=row['seed'],
                    global_subset=row.get('global_subset'),global_test=row.get('global_test'),
                    group_val_selected_test=row.get('group_val_selected_test'),group_test_oracle=row.get('group_test_oracle'),
                    source_path=str(group_path),lineage_id=lineage.get('lineage_id') if lineage else None,
                    verification='LINEAGE_HASHES_VERIFIED' if lineage else 'UNVERIFIED_LINEAGE',
                    groups_sha256=(lineage or {}).get('files',{}).get('groups.csv'),
                    rules=json.dumps((lineage or {}).get('rules')),test_oracle_descriptive=True))
        if group_data and not lineage:gaps.append(dict(dataset=dataset,item='group_lineage',status='MISSING_LINEAGE'))
        decision=read_json(root/'framing_decision.json')
        if decision:
            if decision.get('framing_family')!=['all_fixed','global_subset','group_subset','global_gate','unsupervised','softmax_null','independent_ID','early_no_sd','id_plus_meta','early_plus_meta'] and decision.get('best_baseline'):
                gaps.append(dict(dataset=dataset,item='framing_family',status='UNVERIFIED_PRIMARY_FAMILY'))
            for scope,d in [('official',decision),('secondary_six_selectors',decision.get('secondary_six_selectors',{}))]:
                if not d:continue
                ci=d.get('comparison',{})
                framing_rows.append(dict(dataset=dataset,scope=scope,framing=d.get('framing'),
                    family=json.dumps(d.get('framing_family',d.get('family'))),best_baseline=d.get('best_baseline'),
                    mean_diff=ci.get('mean_diff'),ci_lower=ci.get('ci_lower'),ci_upper=ci.get('ci_upper'),
                    used_for_primary_decision=scope=='official',policy=decision.get('framing_policy'),source_path=str(root/'framing_decision.json')))
        hp=output/'historic_decisions'/dataset/'framing_decision.json'
        hd=read_json(hp);provenance=read_json(hp.parent/'provenance.json')
        if hd is not None:
            if provenance and provenance.get('source_sha256') and provenance['source_sha256']!=file_sha(hp):raise ValueError('Historical framing copy changed: '+str(hp))
            history.append(dict(dataset=dataset,recorded_framing=hd.get('framing'),recorded_best_baseline=hd.get('best_baseline'),
                recorded_family=json.dumps(hd.get('framing_family')),recorded_comparison=json.dumps(hd.get('comparison')),
                path=str(hp),sha256=file_sha(hp),provenance=json.dumps(provenance),historical_only=True,not_relabelled=True))
        meta_dir=Path(config.get('metadata_root','artifacts/metadata'))/dataset
        if not meta_dir.is_absolute():meta_dir=output.parent.parent/meta_dir
        statistics=read_json(meta_dir/'statistics.json');coverage=read_json(meta_dir/'coverage.json')
        if statistics:
            fields=['users','original_items','catalog_train_base_items','interactions','sparsity','mean_sequence_length','train_base_pairs','train_meta_contexts','validation_contexts','test_contexts']
            dataset_rows.append(dict(dataset=dataset,**{k:statistics.get(k) for k in fields},
                oov_counts=json.dumps(statistics.get('oov_counts')),coverage_catalog=json.dumps((coverage or {}).get('train_base_catalog')),
                source=str(meta_dir/'statistics.json'),sha256=file_sha(meta_dir/'statistics.json')))
        else:gaps.append(dict(dataset=dataset,item='dataset_statistics',status='MISSING_METADATA_ARTIFACT'))
        for feature_path in root.rglob('features_train_meta.csv'):
            columns=pd.read_csv(feature_path,nrows=0).columns
            feature_hash=file_sha(feature_path)
            source_files={str(p):file_sha(p) for p in [output.parent.parent/'src/taimma/evaluation/checkpoint2_utility.py',output.parent.parent/'src/taimma/experiments/checkpoint2_protocol.py',output.parent.parent/'src/taimma/training/selective_fusion.py'] if p.is_file()}
            for column in columns:
                if str(column).startswith('feat_'):
                    features.append(dict(dataset=dataset,scope=str(feature_path.parent.relative_to(root)),feature=column,
                        definition=feature_definition(column),inference_available=not feature_definition(column).startswith('UNVERIFIED_FEATURE'),target_used=False,
                        feature_csv=str(feature_path),feature_csv_sha256=feature_hash,source_sha256=json.dumps(source_files),
                        s3_recommendation_loss='full_catalog_cross_entropy'))
        for p in sorted((root/'gates').glob('*/per_user_*.csv')):
            part=p.stem.removeprefix('per_user_');f=csv(p)
            binding=read_json(p.with_suffix('.binding.json'))
            if binding and binding.get('sha256')!=file_sha(p):raise ValueError('Gate rank binding changed: '+str(p))
            if 'ndcg@10' not in f:raise ValueError('Missing rank metric: '+str(p))
            mode,seed=p.parent.name.rsplit('_',1)
            if mode=='add_one':
                completed=read_json(p.parent/'completed.json') or {}
                if completed.get('evaluation_version')!=2:
                    gaps.append(dict(dataset=dataset,item=p.parent.name,status='UNVERIFIED_OR_V1_DROPOUT_BUG'));continue
            gates.append(dict(dataset=dataset,mode=mode,seed=int(seed),part=part,ndcg_10=float(f['ndcg@10'].mean()),source=str(p),binding_status=f.attrs['binding_status']))
        # Include same-S1 subset/fairness/reference measurements in the gate table.
        for p in sorted((root/'evaluations').glob('*/*_*.csv')):
            stem=p.stem
            if not (stem.endswith('_val') or stem.endswith('_test')) or stem.startswith('missing_'):continue
            f=csv(p)
            if f is not None and 'ndcg@10' in f:
                job=p.parent.name;seed=job.rsplit('_',1)[-1]
                if seed.isdigit():gates.append(dict(dataset=dataset,mode=job.rsplit('_',1)[0]+':'+stem.rsplit('_',1)[0],seed=int(seed),part=stem.rsplit('_',1)[1],ndcg_10=float(f['ndcg@10'].mean()),source=str(p),binding_status=f.attrs['binding_status']))
        oof_manifests=sorted((root/'e1_oof_evidence').glob('*/manifest.json'))
        if oof_manifests:
            from .memo0711_e1_evidence import inspect_e1_evidence
            inspected=inspect_e1_evidence(root)
            if inspected['status']!='PASS':gaps.append(dict(dataset=dataset,item='e1_oof_evidence',status=inspected['status']))
            for manifest_path in oof_manifests:
                manifest=read_json(manifest_path);folder=manifest_path.parent
                for relative,h in manifest['files'].items():
                    p=(folder/relative).resolve()
                    if not p.is_relative_to(folder.resolve()) or not p.is_file() or file_sha(p)!=h:raise ValueError('OOF artifact changed')
                    oof_files.append(dict(dataset=dataset,metric=folder.name,path=str(p),sha256=h,bytes=p.stat().st_size,manifest=str(manifest_path),original_report_unchanged=manifest.get('original_report_unchanged')))
                summary=read_json(folder/'replay_summary.json')
                for row in summary.get('sources',[]):
                    oof_rows.append(dict(dataset=dataset,metric=folder.name,paired_predictions=str(folder/'paired_oof_predictions.csv') if (folder/'paired_oof_predictions.csv').is_file() else None,**row))
        elif (root/'selected_s2/status.json').is_file():gaps.append(dict(dataset=dataset,item='e1_oof_evidence',status='MISSING_OOF_PREDICTIONS'))
        for p in sorted(root.rglob('e0_e1_*.json')):
            if p.name.endswith('.cache.json'):continue
            report=read_json(p)
            for source,r in report.get('sources',{}).items():
                stability=r.get('stability',{});e1=r.get('e1',{});e0=r.get('e0',{})
                utility.append(dict(dataset=dataset,scope=str(p.parent.relative_to(root)),metric=p.stem.removeprefix('e0_e1_'),source=source,
                    epsilon=stability.get('epsilon'),label_stability_rho=stability.get('spearman_rho'),regime=stability.get('stability_regime'),
                    e0_passed=e0.get('e0_passed'),auc=e1.get('oof_auc_lr'),auc_ci=json.dumps(e1.get('bootstrap_95_ci')),
                    prediction_delta_spearman=e1.get('oof_spearman_rho'),prediction_delta_spearman_ci=json.dumps(e1.get('oof_spearman_ci')),
                    decision=report.get('decision'),source_path=str(p)))
        for p in sorted(root.rglob('*')):
            if p.is_file() and p.name in ['features_train_meta.csv','validation_same_model.csv']:
                shared.append(dict(dataset=dataset,path=str(p),bytes=p.stat().st_size,sha256=file_sha(p),copy_required=True))
    save('E1_OOF_replay_summary',oof_rows);save('E1_OOF_artifacts',oof_files)
    save('group_selection_by_seed',group_rows);save('framing_official_and_secondary',framing_rows);save('historical_framing',history);save('dataset_statistics',dataset_rows);save('feature_definitions',features)
    save('gate_and_reference_by_seed',gates);save('utility_E0_E1',utility);save('decision_evidence',evidence);save('shared_feature_artifacts',shared)
    if gates:
        summary=pd.DataFrame(gates).groupby(['dataset','mode','part']).ndcg_10.agg(['mean','std','count']).reset_index()
        save('gate_summary',summary.to_dict('records'))
    # Source controls remain historical, with paired-user CI over seed-mean differences.
    controls=[]
    for source in ['category','brand','title']:
        for kind in ['shuffled','random']:
            deltas=[];means=[];cohort=None
            for seed in [42,43,44]:
                names=[f'control_{kind}_{source}_{seed}',f'early_CBT_{seed}'];frames=[]
                for name in names:
                    pair=records.get(('beauty',name))
                    if pair:
                        directory,rec=pair;f=csv((directory/rec['checkpoint']).parent/'per_user_test.csv')
                        if f is not None:frames.append(f.sort_values('user_idx').reset_index(drop=True))
                if len(frames)!=2:continue
                keys=frames[0][['user_idx','target_idx']].to_numpy()
                if not np.array_equal(keys,frames[1][['user_idx','target_idx']].to_numpy()) or (cohort is not None and not np.array_equal(cohort,keys)):
                    raise ValueError('Unpaired source-control cohorts')
                cohort=keys;means.append(float(frames[0]['ndcg@10'].mean()));deltas.append(frames[0]['ndcg@10'].to_numpy()-frames[1]['ndcg@10'].to_numpy())
            row=dict(source=source,control=kind,n_seeds=len(deltas),status='COMPLETE' if len(deltas)==3 else 'MISSING_SEEDS')
            if len(deltas)==3:
                d=np.mean(deltas,axis=0);rng=np.random.default_rng(42);reps=config.get('bootstrap_reps',1000)
                lo,hi=np.percentile([d[rng.integers(len(d),size=len(d))].mean() for _ in range(reps)],[2.5,97.5])
                row.update(mean=float(np.mean(means)),sd=float(np.std(means,ddof=1)),delta_control_minus_real=float(d.mean()),ci_lower=float(lo),ci_upper=float(hi),replicates=reps)
            controls.append(row)
    save('historical_CBT_controls',controls)
    diagnostics=Path(config.get('input_roots',{}).get('diagnostics',output/'missing_diagnostics'))/'beauty'
    diagnostic_evidence=[]
    for p in sorted(diagnostics.rglob('K*.json')):
        if p.name.endswith('.cache.json'):continue
        d=read_json(p);diagnostic_evidence.append(dict(path=str(p),payload=json.dumps(d,ensure_ascii=False)))
    save('historical_K1_K4',diagnostic_evidence);save('completeness_gaps',gaps)
    exports=[]
    for name,f in tables.items():
        p=dest/(name+'.tex');p.write_text(latex_table(f),encoding='utf-8',newline='\n');exports.append(p)
    figures,plot_status=measured_figures(tables,dest,output);exports.extend(figures)
    plot_manifest=dest/'plot_status.json';plot_manifest.write_text(json.dumps(plot_status,indent=2),encoding='utf-8');exports.append(plot_manifest)
    notes=['# TAIMMA measured report package','',
        'Blank measurements mean missing or NOT_OPENED, never zero. SD is sample SD over seeds. B requires three seeds. Gate tables retain actual measured seed counts.',
        'S3 recommendation objective is full-catalog cross-entropy plus declared utility/sparsity terms; no post-result change to pairwise/BPR is implied. Feature definitions and source hashes document the actual inference-available psi.','Random controls retain empirical-marginal C/B and Gaussian unit-norm Title; no random reruns. Historical K1-K4 are not pooled with three-seed late means.',
        'Official framing preserves the memo-v1 ten-baseline family including fairness; six-selector comparison is secondary and cannot override the official conclusion. F1 is a Beauty validation decision. Group/global choices use validation; Test oracle is descriptive. Predictions-vs-delta Spearman differs from label-stability rho.',
        'Hash verification applies to artifacts actually present. INCOMPLETE_LOCAL_BUNDLE is not verification of absent server checkpoints.',
        'Robustness requires v3 effective-mask/logit/rank evidence; flat or identical-score outcomes require explicit review, not forced monotonic metrics. Robustness/add-one v1 outputs must not be adopted as corrected evidence. Consult repair/adoption provenance and readiness checks.','']
    for name,f in tables.items():
        notes.extend(['## '+name,'',f'Rows: {len(f)}. Data: `{name}.csv`; LaTeX input: `{name}.tex`.',''])
    notes.extend(['## Figures','',json.dumps(plot_status),*['- '+p.name for p in figures],''])
    (dest/'report.md').write_text('\n'.join(notes),encoding='utf-8',newline='\n')
    manifest=dict(training=False,inference=False,source_runtime=str(output),source_sha256=inputs,
        files={p.name:file_sha(p) for p in [dest/(name+'.csv') for name in tables]+[dest/'report.md']+exports})
    (dest/'package_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    return dest
