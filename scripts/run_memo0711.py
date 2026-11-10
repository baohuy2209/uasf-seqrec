"""07/11 memo runner. Default plan never trains or runs inference."""
import argparse,json,os,sys,shutil
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))

def resolve_input_roots(repo, config, args):
    """Explicit CLI paths override YAML; omitted flags preserve configured paths."""
    defaults = dict(early='runs/checkpoint2_full_20261106',
                    late='runs/checkpoint2_late_20261106', diagnostics='runs/checkpoint2_20261106')
    return {key: str((Path(repo) / (getattr(args, key+'_root', None)
                    or config.get('input_roots', {}).get(key) or value)).resolve())
            for key, value in defaults.items()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['plan','check','adopt','b','beauty','fairness','robustness','title','rank','all','transfer','report','freeze','release-test'],default='plan')
    p.add_argument('--datasets',nargs='+',choices=['beauty','sports','toys'],default=['beauty','sports','toys'])
    p.add_argument('--config',default='configs/experiments/memo0711.yaml')
    p.add_argument('--early-root');p.add_argument('--late-root')
    p.add_argument('--diagnostics-root')
    p.add_argument('--reuse-memo',help='Verified completed memo output to adopt; old run remains unchanged')
    p.add_argument('--output',default='runs/memo0711_v2');p.add_argument('--publish-dir')
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
    cfg['input_roots']=resolve_input_roots(ROOT,cfg,a)
    run=Memo0711(ROOT,cfg,a.output,execute_training=a.execute_training,execute_inference=a.execute_inference)
    if a.stage not in ['report','freeze','release-test','adopt']:
        for kind in ['early','late']:
            if not Path(cfg['input_roots'][kind]).is_dir():raise FileNotFoundError('Wrong input root; not a request to retrain: '+cfg['input_roots'][kind])
    if a.stage in ['b','beauty','fairness','title','all','transfer'] and not a.execute_training:p.error('This stage requires --execute-training; inspect plan first')
    if a.stage in ['robustness','release-test'] and not (a.execute_training or a.execute_inference):p.error('Requires --execute-inference')
    if a.stage=='adopt' and not a.reuse_memo:p.error('adopt requires --reuse-memo runs/memo0711_v1')
    if a.reuse_memo and a.stage not in ['report','freeze','release-test','check']:
        run.adopt_memo((ROOT/a.reuse_memo).resolve(),a.datasets)
    if a.stage=='adopt':pass
    elif a.stage=='rank':rank_diagnostics(cfg['input_roots'],run.output/'beauty')
    elif a.stage=='check':
        from taimma.experiments.table5_utility import check_references
        for k in ['early','late']:
            if not Path(cfg['input_roots'][k]).is_dir():raise FileNotFoundError('Wrong input root: '+cfg['input_roots'][k])
        print(json.dumps(check_references(cfg['input_roots']['early']),indent=2))
        for key in a.datasets:print(json.dumps(run.check_data(key),indent=2))
    elif a.stage=='freeze':
        run.validate_ready()
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
            status_path=run.output/key/'selected_s2/status.json'
            if not status_path.is_file():
                framing_path=run.output/key/'framing_decision.json'
                framing=json.loads(framing_path.read_text()) if framing_path.is_file() else {}
                if key!='beauty' and framing.get('framing')=='DIAGNOSTIC_ONLY_F2':
                    print(key+': UASF Test not applicable under F2');continue
                raise FileNotFoundError('Missing S2 status for requested Test release: '+str(status_path))
            status=json.loads(status_path.read_text())
            if not status.get('s3_eligible'):
                print(key+': UASF Test not applicable; S2 decision='+str(status.get('decision')));continue
            if not list((run.output/key/'gates').glob('*/completed.json')):
                raise FileNotFoundError('No trained gates for requested Test release: '+key)
            run.release(key)
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
    if a.stage=='report':
        from taimma.experiments.memo0711_report import write_memo_report
        print(write_memo_report(run.output,cfg,ROOT/'results'/(run.output.name+'_tables')))
    if a.publish_dir:
        dest=(ROOT/a.publish_dir).resolve()
        for srcroot in [run.output]+[Path(x) for x in cfg['input_roots'].values()]:
            if dest==srcroot or srcroot in dest.parents or dest in srcroot.parents:raise ValueError('Publish must be separate from all source/runtime folders')
        published={};omitted_large={}
        for src in run.output.rglob('*'):
            if src.is_file() and src.suffix in ['.json','.csv','.md','.log'] and src.stat().st_size<10_000_000 and not src.name.endswith('.lock'):
                target=dest/src.relative_to(run.output);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,target);published[src.relative_to(run.output).as_posix()]=sha(src)
            elif src.is_file() and src.suffix in ['.json','.csv','.md','.log'] and src.stat().st_size>=10_000_000:
                omitted_large[src.relative_to(run.output).as_posix()]=dict(bytes=src.stat().st_size,sha256=sha(src),source=str(src),reason='Copy original to shared storage; excluded from lightweight Git bundle')
        write_json(dest/'published_manifest.json',published)
        write_json(dest/'large_artifacts_manifest.json',omitted_large)
    print('Completed requested stage. No K4/random V2 fallback jobs were launched.')

if __name__=='__main__':main()
