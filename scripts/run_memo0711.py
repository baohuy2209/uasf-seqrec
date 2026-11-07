"""07/11 memo runner. Default plan never trains or runs inference."""
import argparse,json,os,sys,shutil
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['plan','check','b','beauty','fairness','robustness','title','rank','all','transfer','report','freeze','release-test'],default='plan')
    p.add_argument('--datasets',nargs='+',choices=['beauty','sports','toys'],default=['beauty','sports','toys'])
    p.add_argument('--config',default='configs/experiments/memo0711.yaml')
    p.add_argument('--early-root',default='runs/checkpoint2_full_20261106');p.add_argument('--late-root',default='runs/checkpoint2_late_20261106')
    p.add_argument('--diagnostics-root',default='runs/checkpoint2_20261106')
    p.add_argument('--output',default='runs/memo0711_v1');p.add_argument('--publish-dir')
    p.add_argument('--execute-training',action='store_true');p.add_argument('--execute-inference',action='store_true');p.add_argument('--freeze-tag')
    a=p.parse_args()
    if a.stage=='plan':
        print(json.dumps(dict(memo='2026-11-07',training=False,output=a.output,datasets=a.datasets,
            reuse='Beauty ID/early/late/SD seeds 42..44; missing original weights fail, never silently refit',
            new_fits=['B Sports/Toys only if absent','Beauty foundation 45/46','fairness 5 seeds','S3/baselines/ablations','T-only shuffled 3 seeds'],
            forbidden=['extra K4 seeds','random V2 reruns','UASF Test before freeze'],
            transfer='Explicit transfer stage, F1 Beauty and >=12/11; 3 seeds Sports/Toys'),indent=2));return
    import yaml
    from taimma.experiments.memo0711 import Memo0711,rank_diagnostics
    from taimma.experiments.checkpoint2 import exclusive
    from taimma.experiments.checkpoint2_protocol import write_json,sha
    cfg=yaml.safe_load((ROOT/a.config).read_text(encoding='utf-8'))
    cfg['input_roots']={k:str((ROOT/getattr(a,k+'_root')).resolve()) for k in ['early','late','diagnostics']}
    run=Memo0711(ROOT,cfg,a.output,execute_training=a.execute_training,execute_inference=a.execute_inference)
    if a.stage not in ['report','freeze','release-test']:
        for kind in ['early','late']:
            if not Path(cfg['input_roots'][kind]).is_dir():raise FileNotFoundError('Wrong input root; not a request to retrain: '+cfg['input_roots'][kind])
    if a.stage in ['b','beauty','fairness','title','all','transfer'] and not a.execute_training:p.error('This stage requires --execute-training; inspect plan first')
    if a.stage in ['robustness','release-test'] and not (a.execute_training or a.execute_inference):p.error('Requires --execute-inference')
    if a.stage=='rank':rank_diagnostics(cfg['input_roots'],run.output/'beauty')
    elif a.stage=='check':
        from taimma.experiments.table5_utility import check_references
        for k in ['early','late']:
            if not Path(cfg['input_roots'][k]).is_dir():raise FileNotFoundError('Wrong input root: '+cfg['input_roots'][k])
        print(json.dumps(check_references(cfg['input_roots']['early']),indent=2))
        for key in a.datasets:print(json.dumps(run.check_data(key),indent=2))
    elif a.stage=='freeze':
        if list(run.output.glob('*/memo_execution.lock')):raise RuntimeError('A study process is running; do not freeze')
        for key in cfg['datasets']:
            for seed in [42,43,44]:
                for mode in ['early','late']:
                    for lab in ['ID','C','B','T','CB','CT','BT','CBT']:
                        name=f'id_{seed}' if lab=='ID' else f'{mode}_{lab}_{seed}'
                        if run.completed(key,name) is None:raise RuntimeError('Missing mandatory diagnostic B: '+key+'/'+name)
            for name in ['group_analysis.json','robustness.csv']:
                if not (run.output/key/name).is_file():raise RuntimeError('Missing mandatory analysis: '+key+'/'+name)
            if key!='beauty' and not (run.output/key/'framing_decision.json').exists():
                beauty=json.loads((run.output/'beauty/framing_decision.json').read_text())
                if beauty['framing']=='F1':raise RuntimeError('F1 transfer validation must complete before freeze')
                write_json(run.output/key/'framing_decision.json',dict(framing='DIAGNOSTIC_ONLY_F2',test_used=False))
        run.freeze(a.freeze_tag)
    elif a.stage=='release-test':
        run.verify_freeze()
        for key in a.datasets:
            if (run.output/key/'selected_s2/status.json').exists() and list((run.output/key/'gates').glob('*/completed.json')):run.release(key)
    elif a.stage!='report':
        from contextlib import ExitStack
        with ExitStack() as stack:
            keys=a.datasets if a.stage in ['b','robustness','transfer','all'] else ['beauty']
            for key in sorted(set(keys)):stack.enter_context(exclusive(run.output/key/'memo_execution.lock'))
            if a.stage in ['b','all']:
                for key in a.datasets:run.grid_b(key)
            if a.stage in ['fairness']:run.fairness('beauty',list(range(42,47)))
            if a.stage in ['beauty','all']:run.uasf('beauty')
            if a.stage in ['robustness','all']:
                for key in a.datasets:run.robustness_current(key)
            if a.stage in ['title','all']:run.title_controls()
            if a.stage=='all':rank_diagnostics(cfg['input_roots'],run.output/'beauty')
            if a.stage=='transfer':
                for key in a.datasets:
                    if key!='beauty':run.uasf(key,transfer=True)
    # A registry includes only measured outputs, no invented Test values.
    import pandas as pd
    rows=[]
    for marker in run.output.glob('*/jobs/*/completed.json'):
        rec=run.completed(marker.parent.parent.parent.name,marker.parent.name)
        rows.append(dict(dataset=marker.parent.parent.parent.name,job=marker.parent.name,seed=rec['spec']['seed'],
            val_ndcg_10=rec['val_ndcg_10'],test_ndcg_10=rec.get('test_ndcg_10'),best_epoch=rec['best_epoch'],test_status=rec.get('test_status','HISTORICAL_DIAGNOSTIC')))
    if rows:
        df=pd.DataFrame(rows);df.to_csv(run.output/'results.csv',index=False,lineterminator='\n')
        df['family']=df.job.str.replace(r'_\d+$','',regex=True)
        summary=df.groupby(['dataset','family'])[['val_ndcg_10','test_ndcg_10']].agg(['mean','std','count'])
        summary.to_csv(run.output/'summary.csv',lineterminator='\n')
        note=['# Measured checkpoint results under memo 07 November',summary.to_string(),
            'Blank Test is NOT_OPENED, not zero. Gates are validation-only before freeze. Historical diagnostic Test values remain separate.']
        for path in run.output.glob('*/framing_decision.json'):
            note.extend([str(path.relative_to(run.output)),path.read_text()])
        (run.output/'report.md').write_text('\n\n'.join(note),encoding='utf-8',newline='\n')
    if a.publish_dir:
        dest=(ROOT/a.publish_dir).resolve()
        for srcroot in [run.output]+[Path(x) for x in cfg['input_roots'].values()]:
            if dest==srcroot or srcroot in dest.parents or dest in srcroot.parents:raise ValueError('Publish must be separate from all source/runtime folders')
        published={}
        for src in run.output.rglob('*'):
            if src.is_file() and src.suffix in ['.json','.csv','.md','.log'] and src.stat().st_size<10_000_000 and not src.name.endswith('.lock'):
                target=dest/src.relative_to(run.output);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,target);published[src.relative_to(run.output).as_posix()]=sha(src)
        write_json(dest/'published_manifest.json',published)
    print('Completed requested stage. No K4/random V2 fallback jobs were launched.')

if __name__=='__main__':main()
