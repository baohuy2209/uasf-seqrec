"""Finalize the existing F2 study without retraining. Default: nonexecuting plan."""
from pathlib import Path
import argparse,json,os,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['plan','cpu','preflight','freeze','release-test','alpha'],default='plan')
    parser.add_argument('--run-root',default='runs/memo0711_v3')
    parser.add_argument('--freeze-tag')
    parser.add_argument('--early-release-reason',default='')
    parser.add_argument('--execute-inference',action='store_true')
    args=parser.parse_args()
    if args.stage=='plan':
        print(json.dumps(dict(training=False,run_root=args.run_root,stages=['cpu','preflight','freeze','release-test','alpha'],
            family='10 locked baselines plus UASF; all/global may share one display row',
            checkpoints='Only exact manifest-relative paths under the supplied memo run; no path search or refitting',
            early_release='Explicit recorded reason before 13/11; freeze/tag/SHA checks still mandatory'),indent=2));return 0
    from taimma.paths import resolve_project_path
    from taimma.experiments.memo0711_final import (checkpoint_inventory,gate_signature_audit,source_compatibility,reviewed_git,cpu_analysis,make_run,freeze,release,alpha_analysis)
    root=resolve_project_path(args.run_root)
    if not root.is_relative_to((ROOT/'runs').resolve()) or root==(ROOT/'runs').resolve():parser.error('Use the existing dedicated repo/runs/memo0711_v3 directory')
    if args.stage=='cpu':
        print(json.dumps(cpu_analysis(root),indent=2));return 0
    if args.stage=='preflight':
        result=checkpoint_inventory(root)
        result['gate_signature_audit']=gate_signature_audit(root)
        result['source_compatibility']=source_compatibility(ROOT,root)
        if result['gate_signature_audit']['status']!='PASS':result['status']='INCOMPLETE'
        if result['source_compatibility']['status']!='PASS':result['status']='INCOMPLETE'
        if result['status']=='PASS':
            try:
                result['git_readiness']=reviewed_git(ROOT,root)
                result['full_readiness']=make_run(ROOT,root).validate_ready()
            except Exception as exc:result.update(status='INCOMPLETE',readiness_error=str(exc))
        print(json.dumps(result,indent=2,ensure_ascii=True));return 0 if result['status']=='PASS' else 2
    if args.stage in ['release-test','alpha'] and not args.execute_inference:parser.error('Pass --execute-inference; training has no CLI flag')
    if args.stage=='freeze' and not args.freeze_tag:parser.error('Pass an existing reviewed --freeze-tag')
    run=make_run(ROOT,root,args.execute_inference)
    if args.stage=='freeze':
        from taimma.experiments.checkpoint2 import exclusive
        with exclusive(root/'finalization/freeze.lock'):
            result=freeze(run,args.freeze_tag,args.early_release_reason)
        print(json.dumps(dict(frozen=True,tag=result['git_tag'],framing=result['framing'],early_release_reason=result['early_release_authorization']),indent=2))
    elif args.stage=='release-test':print(json.dumps(release(run),indent=2))
    else:print(json.dumps(alpha_analysis(run),indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
