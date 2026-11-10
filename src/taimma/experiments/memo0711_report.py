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
            finish(fig,'figure_B_'+dataset,f'Source: {runtime.name}/{dataset}/jobs; mean ± sample SD, seeds 42/43/44. Missing subsets omitted.')
    for key,frame in tables.items():
        if not key.endswith('_robustness') or frame.empty:continue
        dataset=key.removesuffix('_robustness')
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
            finish(fig,f'figure_robustness_{dataset}_{source}',f'Source: {runtime.name}/{dataset}/robustness.csv (evaluation v2); mean ± sample SD, 3 seeds. Natural missingness retained.')
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
        f=pd.read_csv(p);f.attrs['binding_status']='BOUND' if meta.get('sha256') else 'NO_ADJACENT_BINDING'
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
    gates=[];utility=[];evidence=[];shared=[]
    for dataset in config.get('datasets',['beauty','sports','toys']):
        root=output/dataset
        for name in ['robustness','gate_validation','gate_test','title_only_controls','late_rank_diagnostics']:
            f=csv(root/(name+'.csv'))
            if f is not None:
                if name in ['gate_validation','gate_test'] and 'mode' in f:
                    bad=[]
                    for i,row in f[f['mode']=='add_one'].iterrows():
                        rec=read_json(root/'gates'/f'add_one_{int(row.seed)}'/'completed.json') or {}
                        if rec.get('evaluation_version')!=2:bad.append(i)
                    f=f.drop(index=bad)
                if name=='robustness':
                    binding=read_json(root/'robustness_binding.json')
                    if not binding or binding.get('evaluation_version')!=2:
                        gaps.append(dict(dataset=dataset,item='robustness',status='UNVERIFIED_OR_V1_MASK_BUG'));continue
                    if binding.get('sha256')!=file_sha(root/'robustness.csv'):
                        raise ValueError('Robustness binding changed: '+dataset)
                save(dataset+'_'+name,f.to_dict('records'))
        for name in ['framing_decision','baseline_comparisons','title_only_comparison','group_analysis','robustness_binding']:
            d=read_json(root/(name+'.json'))
            if d is not None:evidence.append(dict(dataset=dataset,kind=name,payload=json.dumps(d,ensure_ascii=False)))
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
        'Random controls retain empirical-marginal C/B and Gaussian unit-norm Title; no random reruns. Historical K1-K4 are not pooled with three-seed late means.',
        'F1 is a Beauty validation decision. Group/global choices use validation; Test oracle is descriptive. Predictions-vs-delta Spearman differs from label-stability rho.',
        'Hash verification applies to artifacts actually present. INCOMPLETE_LOCAL_BUNDLE is not verification of absent server checkpoints.',
        'Robustness/add-one v1 outputs must not be adopted as corrected evidence. Consult repair/adoption provenance and readiness checks.','']
    for name,f in tables.items():
        notes.extend(['## '+name,'',f'Rows: {len(f)}. Data: `{name}.csv`; LaTeX input: `{name}.tex`.',''])
    notes.extend(['## Figures','',json.dumps(plot_status),*['- '+p.name for p in figures],''])
    (dest/'report.md').write_text('\n'.join(notes),encoding='utf-8',newline='\n')
    manifest=dict(training=False,inference=False,source_runtime=str(output),source_sha256=inputs,
        files={p.name:file_sha(p) for p in [dest/(name+'.csv') for name in tables]+[dest/'report.md']+exports})
    (dest/'package_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    return dest
