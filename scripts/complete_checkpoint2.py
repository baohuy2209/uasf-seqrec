"""Complete the November report without modifying historical output folders."""
import argparse,json,os,sys
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['plan','check','utility','all','report','freeze','release-test'],default='plan')
    p.add_argument('--config',default='configs/experiments/completion_0711.yaml')
    p.add_argument('--output')
    p.add_argument('--datasets',nargs='+',choices=['beauty','sports','toys'])
    for name in ['early','late','diagnostics']:p.add_argument('--'+name+'-root')
    p.add_argument('--execute-training',action='store_true');p.add_argument('--execute-inference',action='store_true')
    p.add_argument('--publish-dir',help='Copy lightweight summaries here only after all computations finish')
    p.add_argument('--freeze-tag',help='Existing reviewed Git tag at the clean commit to freeze')
    a=p.parse_args()
    if a.stage=='plan':
        print(json.dumps(dict(execution='DISABLED',phases=['adopt verified original checkpoints','uniform/Gaussian-covariance controls',
            'B on three cohorts','early-SD diagnostic utility and both E1 targets','validation-only backbone selection',
            'eligible S3/baselines/5 ablations on 5 Beauty and >=3 other seeds','missingness and group analyses',
            'results.csv, 7-table report, SHA256 manifests'],
            test='UASF test requires separate freeze/release-test on or after 13/11',
            missing_checkpoints='Supply original .pt files; ranks cannot recover CE labels'),indent=2));return 0
    import yaml
    cfg=yaml.safe_load((ROOT/a.config).read_text(encoding='utf-8'))
    datasets=a.datasets or cfg['datasets']
    if datasets[0]!='beauty' and a.stage=='all':p.error('Run Beauty first: the global backbone is chosen there')
    for name in ['early','late','diagnostics']:
        value=getattr(a,name+'_root') or cfg['input_roots'][name]
        cfg['input_roots'][name]=str((ROOT/value).resolve())
    if a.stage=='all' and not a.execute_training:p.error('all requires --execute-training')
    if a.stage=='utility' and not (a.execute_inference or a.execute_training):p.error('utility requires --execute-inference')
    from taimma.experiments.study_completion import StudyCompletion
    from taimma.experiments.checkpoint2 import exclusive
    run=StudyCompletion(ROOT,cfg,a.output or cfg['output'],execute_training=a.execute_training,execute_inference=a.execute_inference)
    if a.stage=='report':
        from taimma.experiments.completion_report import write_report
        for key in datasets:print(write_report(run.output/key,cfg))
    elif a.stage=='freeze':run.freeze(a.freeze_tag)
    elif a.stage=='release-test':
        from taimma.experiments.study_gates import release_test
        run.verify_freeze()
        for key in datasets:release_test(run,key);run.export(key)
    elif a.stage=='check':
        for key in datasets:print(json.dumps(run.check_data(key),indent=2))
        missing=[]
        for seed in [42,43]:
            folder=Path(cfg['input_roots']['early'])/'beauty/jobs'/f'early_sd_{seed}'
            rec=folder/'completed.json'
            if not rec.exists():missing.append(str(rec))
            else:
                path=folder/json.loads(rec.read_text())['checkpoint']
                if not path.exists():missing.append(str(path))
        print(json.dumps(dict(utility_original_checkpoint_missing=missing,training_executed=False),indent=2))
        return 2 if missing else 0
    else:
        with exclusive(run.output/'completion_execution.lock'):
            for key in datasets:
                if a.stage=='utility':
                    run.adopt(key,cfg['input_roots'],utility_only=True)
                    run.complete_utility(key);print(run.export(key))
                else:print(run.run_all(key,cfg['input_roots']))
    if a.publish_dir:
        import shutil
        dest=(ROOT/a.publish_dir).resolve()
        for original in cfg['input_roots'].values():
            protected=Path(original).resolve()
            if dest==protected or protected in dest.parents:raise ValueError('Publish cannot overwrite a historical input run')
        dest.mkdir(parents=True,exist_ok=True)
        if (run.output/'freeze.json').exists():shutil.copy2(run.output/'freeze.json',dest/'freeze.json')
        for key in datasets:
            for n in ['results.csv','mini_checkpoint.docx','mini_checkpoint.md','backbone_decision.json','framing_decision.json','SHA256_MANIFEST.json','utility_summary.csv','report_tables.csv',
                      'groups.json','groups.csv','group_analysis.json','dropout_profile.csv','robustness.csv','K3_validation_epochs.csv',
                      'gate_validation.csv','gate_test.csv','baseline_comparisons.json','protocol.lock.json','origin_git.json','s3_status.json','s3_test_status.json']:
                src=run.output/key/n
                if src.exists():
                    target=dest/key/n;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,target)
            for folder in ['diagnostic_early_sd','selected_s2']:
                for n in ['alignment.json','status.json','e0_e1_CE.json','e0_e1_LOG_RANK.json']:
                    src=run.output/key/folder/n
                    if src.exists():
                        target=dest/key/folder/n;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,target)
            for src in (run.output/key/'historical_diagnostics').glob('*'):
                if src.is_file() and src.suffix in ['.json','.csv']:
                    target=dest/key/'historical_diagnostics'/src.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,target)
            # Publish ranks/logs, not model weights or wide feature caches.
            names={'completed.json','config.json','metrics.json','history.json','epochs.csv','run_manifest.json',
                   'source_snapshot.json','per_user_val.csv','per_user_test.csv','run.log','import_provenance.json',
                   'id_dependency.json','late_verification.json'}
            for folder in ['jobs','gates']:
                for src in (run.output/key/folder).rglob('*'):
                    if src.is_file() and src.name in names:
                        target=dest/key/src.relative_to(run.output/key);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,target)
            for src in (run.output/key/'figures').glob('*.svg'):
                target=dest/key/'figures'/src.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,target)
    return 0


if __name__=='__main__':raise SystemExit(main())
