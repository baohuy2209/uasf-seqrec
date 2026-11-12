"""Inference-only finalization for the 12 November feedback; original runs are retained."""
from __future__ import annotations
from datetime import datetime, timezone, timedelta, date
from pathlib import Path
import hashlib
import json
import subprocess
from typing import Any
import numpy as np
import pandas as pd

SEEDS = tuple(range(42, 47))
BASELINES = ('all_fixed','global_subset','group_subset','global_gate','unsupervised',
             'softmax_null','independent_ID','early_no_sd','id_plus_meta','early_plus_meta')
METHODS = BASELINES + ('uasf',)
GATES = ('uasf','global_gate','unsupervised','softmax_null')


def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def write(path: Path, value: dict) -> None:
    from .checkpoint2_protocol import write_json
    write_json(path,value)


def inside(root: Path, relative: str) -> Path:
    p=(root/relative).resolve()
    if not p.is_relative_to(root.resolve()):raise ValueError('Artifact escapes root: '+relative)
    return p


def rank_csv(path: Path) -> pd.DataFrame:
    from .memo0711_report import validate_rank_measurements
    f=pd.read_csv(path,float_precision='round_trip')
    validate_rank_measurements(f,path)
    return f.sort_values('user_idx').reset_index(drop=True)


def paired_difference(left: dict, right: dict, reps: int=1000) -> dict:
    if set(left)!=set(right) or not left:raise ValueError('Paired seeds required')
    changes=[];keys=None
    for seed in sorted(left):
        a=left[seed].sort_values('user_idx');b=right[seed].sort_values('user_idx')
        k=a[['user_idx','target_idx']].to_numpy()
        if not np.array_equal(k,b[['user_idx','target_idx']].to_numpy()) or (keys is not None and not np.array_equal(keys,k)):
            raise ValueError('User/target pairing changed')
        keys=k;changes.append(a['ndcg@10'].to_numpy()-b['ndcg@10'].to_numpy())
    delta=np.mean(changes,axis=0);rng=np.random.default_rng(42)
    boot=[float(delta[rng.integers(len(delta),size=len(delta))].mean()) for _ in range(reps)]
    lo,hi=np.percentile(boot,[2.5,97.5])
    return dict(difference=float(delta.mean()),ci_lower=float(lo),ci_upper=float(hi),
                users=len(delta),seeds=sorted(left),resamples=reps,
                scope='unadjusted paired-user percentile CI, conditional on fitted models')


def checkpoint_inventory(root: Path) -> dict:
    """No torch import or fallback path search. Every checkpoint is manifest-bound."""
    entries=[];failures=[]
    required={root/'beauty'/kind/f'{name}_{seed}' for seed in SEEDS
              for kind,names in [('jobs',('id','early_CBT','early_sd','id_plus_meta','early_plus_meta')),('gates',GATES)]
              for name in names}
    discovered={p.parent for kind in ('jobs','gates') for p in root.glob(f'*/{kind}/*/completed.json')}
    for folder in sorted(required|discovered):
            kind,name=folder.parent.name,folder.name
            try:
                rec=read(folder/'completed.json')
                relative=rec['checkpoint'] if kind=='jobs' else rec['attempt']+'/gate.pt'
                path=inside(folder,relative)
                if relative not in rec['artifacts']:raise ValueError('Checkpoint lacks recorded SHA256')
                if not path.is_file():raise FileNotFoundError('Checkpoint missing: '+str(path))
                if sha(path)!=rec['artifacts'][relative]:raise ValueError('Checkpoint SHA256 mismatch: '+str(path))
                entries.append(dict(dataset=folder.parent.parent.name,kind=kind,name=name,path=str(path),sha256=sha(path),required_for_test=folder in required))
            except (FileNotFoundError,ValueError,KeyError) as exc:failures.append(str(exc))
    return dict(status='PASS' if not failures else 'INCOMPLETE',checkpoints=entries,failures=failures,
                training=False,inference=False,required_test_checkpoints=45,discovered_total=len(discovered))


def gate_signature_audit(root: Path) -> dict:
    """Check archived metadata compatibility even when local .pt files are absent."""
    from .checkpoint2_protocol import digest
    config=read(root/'beauty/protocol.lock.json')['config']
    status=read(root/'beauty/selected_s2/status.json')
    label=status['label_choice'];report=read(root/f'beauty/selected_s2/e0_e1_{label}.json')
    results=[]
    for seed in SEEDS:
        a=read(root/f'beauty/jobs/early_sd_{seed}/completed.json');b=read(root/f'beauty/jobs/id_{seed}/completed.json')
        model=digest(dict(early=a['artifacts'][a['checkpoint']],fallback=b['artifacts'][b['checkpoint']],policy='separate_ID_0711'))
        for mode in GATES:
            rec=read(root/f'beauty/gates/{mode}_{seed}/completed.json')
            expected=digest(dict(model=model,config=digest(config),mode=mode,label=label,report=report))
            results.append(dict(seed=seed,mode=mode,match=rec['signature']==expected))
    return dict(status='PASS' if all(r['match'] for r in results) else 'FAILED',
        checked_gates=len(results),results=results,checkpoint_bytes_verified=False)


def source_compatibility(repo: Path, root: Path) -> dict:
    """Reject changed model/loader/evaluator code before reusing frozen weights."""
    from .study_completion import COMPATIBLE_MODEL_FILES
    gate_files=['src/taimma/training/selective_fusion.py','src/taimma/experiments/study_gates.py',
                'src/taimma/evaluation/checkpoint2_utility.py','src/taimma/evaluation/e0_e1_protocol.py']
    current={rel:hashlib.sha256((repo/rel).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
             for rel in set(COMPATIBLE_MODEL_FILES+gate_files)}
    failures=[];checked=0
    for seed in SEEDS:
        for prefix in ('id','early_CBT','early_sd','id_plus_meta','early_plus_meta'):
            folder=root/f'beauty/jobs/{prefix}_{seed}';rec=read(folder/'completed.json')
            snapshot=read(inside(folder,str(Path(rec['checkpoint']).parent/'source_snapshot.json')))
            for rel in COMPATIBLE_MODEL_FILES:
                if snapshot.get(rel)!=current[rel]:failures.append(f'{prefix}_{seed}: {rel}')
            checked+=1
    protocol=read(root/'beauty/protocol.lock.json')['source_sha256']
    for rel in gate_files:
        if protocol.get(rel)!=current[rel]:failures.append('Gate dependency: '+rel)
    return dict(status='PASS' if not failures else 'FAILED',checked_test_foundations=checked,
                checked_gate_dependencies=len(gate_files),failures=failures,newline_policy='LF-normalized source hash')


def cpu_analysis(root: Path) -> dict:
    """Existing ranks and per-seed labels only; safe before freeze."""
    destination=root/'finalization';destination.mkdir(exist_ok=True)
    contrasts=[];inputs={}
    for dataset in ('beauty','sports','toys'):
        frames={name:{} for name in ('id','early_CBT','late_CBT')}
        for name in frames:
            for seed in (42,43,44):
                folder=root/dataset/'jobs'/f'{name}_{seed}';rec=read(folder/'completed.json')
                path=inside(folder,str(Path(rec['checkpoint']).parent/'per_user_test.csv'))
                frames[name][seed]=rank_csv(path);inputs[str(path.relative_to(root))]=sha(path)
        for name in ('early_CBT','late_CBT'):
            contrasts.append(dict(dataset=dataset,contrast=name+' minus ID',
                **paired_difference(frames[name],frames['id'])))
    labels=[]
    for seed in SEEDS:
        path=root/f'beauty/per_seed_s2/early_sd_{seed}/labels.csv';f=pd.read_csv(path)
        if f.user_idx.duplicated().any():raise ValueError('Duplicate label users')
        inputs[str(path.relative_to(root))]=sha(path)
        for source in ('category','brand','title'):
            a=f[source].to_numpy(float)
            if not np.isfinite(a).all() or not np.isin(a,[0,1]).all():raise ValueError('Expected original hard CE labels')
            labels.append(dict(seed=seed,source=source,contexts=len(a),positive_count=int(a.sum()),positive_fraction=float(a.mean())))
    pd.DataFrame(contrasts).to_csv(destination/'rq1_paired_ci.csv',index=False)
    pd.DataFrame(labels).to_csv(destination/'positive_labels.csv',index=False)
    result=dict(training=False,inference=False,inputs_sha256=inputs,framing='F2_UNCHANGED',
                label_definition='Original y = 1[Delta > epsilon]; no relabeling',contrasts=contrasts)
    write(destination/'cpu_analysis.json',result)
    return result


def allowed_runtime_path(relative: str) -> bool:
    """Allow only new post-freeze outputs, never tracked changes or source files."""
    p=Path(relative);parts=p.parts
    if len(parts)==2 and parts[0]=='finalization':
        return p.name in {'rq1_paired_ci.csv','positive_labels.csv','cpu_analysis.json','freeze.lock',
            'release.lock','alpha.lock','release_request.json','release_complete.json',
            'table5_validation_test.csv','test_paired_ci.csv','test_family_comparisons.json',
            'validation_alpha_per_user.csv','validation_alpha_summary.csv','alpha_complete.json'}
    if relative=='freeze.json':return True
    if len(parts)>=4 and parts[0]=='beauty' and parts[1]=='gate_features':return p.name in ('test.csv','test.json')
    if len(parts)>=4 and parts[0]=='beauty' and parts[1]=='evaluations':return p.name.endswith(('_test.csv','_test.json'))
    if len(parts)>=4 and parts[0]=='beauty' and parts[1]=='gates':return p.name in ('per_user_test.csv','per_user_test.binding.json')
    return False


def reviewed_git(repo: Path, root: Path, tag: str | None=None) -> dict:
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    raw=subprocess.check_output(['git','status','--porcelain=v1','-z','--untracked-files=all'],cwd=repo)
    prefix=root.relative_to(repo).as_posix()+'/'
    unexpected=[];generated=[]
    for record in raw.decode('utf-8').split('\0'):
        if not record:continue
        status,path=record[:2],record[3:]
        if status=='??' and path.startswith(prefix) and allowed_runtime_path(path[len(prefix):]):generated.append(path)
        else:unexpected.append(record)
    if unexpected:raise RuntimeError('Commit reviewed code/results first; unexpected working-tree changes:\n'+'\n'.join(unexpected))
    if tag:
        target=subprocess.check_output(['git','rev-parse',tag+'^{commit}'],cwd=repo,text=True).strip()
        if target!=commit:raise ValueError('Freeze tag must point to current reviewed HEAD')
    return dict(commit=commit,dirty_code=False,allowed_untracked_runtime=generated)


def make_run(repo: Path, root: Path, execute: bool=False) -> Any:
    from .memo0711 import Memo0711
    # Reuse the saved scientific config verbatim, including its signature.
    config=read(root/'beauty/protocol.lock.json')['config']
    class FinalRun(Memo0711):
        def run_job(self,*args,**kwargs):raise RuntimeError('Finalization never trains')
        def guard(self,training=False,inference=False):
            if training:raise RuntimeError('Finalization never trains')
            if inference:
                if not self.execute_inference:raise RuntimeError('Pass --execute-inference')
                self.verify_freeze();reviewed_git(self.repo,self.output)
                from .checkpoint2_protocol import check_training_environment
                check_training_environment(self.config['environment'])
        def verify_freeze(self):verify_freeze(self)
    return FinalRun(repo,config,str(root),execute_training=False,execute_inference=execute)


def verify_freeze(run: Any) -> dict:
    path=run.output/'freeze.json'
    if not path.is_file():raise RuntimeError('Run --stage freeze with the reviewed tag before opening Test or alpha analysis')
    lock=read(path)
    if lock.get('finalization_version')!=1:raise RuntimeError('Use the new finalization freeze; existing historical freeze is not overwritten')
    if lock['source_sha256']!=run.source_signature():raise ValueError('Source/config changed after freeze')
    if reviewed_git(run.repo,run.output)['commit']!=lock['git']['commit']:raise ValueError('Git HEAD changed after freeze')
    # Rehash on first use and whenever the file stat changes; avoid rehashing GBs per batch.
    cache=getattr(run,'_verified_final_files',{})
    for rel,h in lock['files'].items():
        p=inside(run.output,rel);st=p.stat();signature=(st.st_size,st.st_mtime_ns,st.st_ctime_ns)
        if cache.get(rel)!=signature:
            if sha(p)!=h:raise ValueError('Frozen artifact changed: '+rel)
            cache[rel]=signature
    run._verified_final_files=cache
    return lock


def freeze(run: Any, tag: str, early_reason: str='') -> dict:
    if not tag or not tag.strip():raise ValueError('An existing reviewed freeze tag is mandatory')
    if (run.output/'freeze.json').exists():return verify_freeze(run)
    now=datetime.now(timezone(timedelta(hours=7)))
    planned=date.fromisoformat(run.config['test_release_not_before'])
    if now.date()<planned and not early_reason.strip():raise RuntimeError('Before planned date: supply explicit --early-release-reason; config is not changed')
    git=reviewed_git(run.repo,run.output,tag)
    inventory=checkpoint_inventory(run.output)
    if inventory['status']!='PASS':raise RuntimeError('Missing/mismatched server checkpoints:\n'+'\n'.join(inventory['failures']))
    compatibility=source_compatibility(run.repo,run.output)
    if compatibility['status']!='PASS':raise ValueError('Model/gate source changed: '+str(compatibility['failures']))
    run.validate_ready()  # Full original data/group/robustness/OOF/validation lineage checks.
    framing=read(run.output/'beauty/framing_decision.json')
    if framing.get('framing')!='F2' or framing.get('framing_family')!=list(BASELINES):raise ValueError('Expected fixed F2 and locked ten-baseline family')
    if read(run.output/'beauty/selected_s2/status.json').get('label_choice')!='CE':raise ValueError('Expected the original CE supervision; no relabeling during finalization')
    files={p.relative_to(run.output).as_posix():sha(p) for p in sorted(run.output.rglob('*'))
           if p.is_file() and p.suffix in ('.pt','.json','.csv') and
           'finalization' not in p.relative_to(run.output).parts and p.name!='freeze.json'}
    lock=dict(finalization_version=1,frozen_at_utc=datetime.now(timezone.utc).isoformat(),files=files,
        source_sha256=run.source_signature(),git=git,git_tag=tag,datasets=run.config['datasets'],
        framing='F2',framing_locked=True,test_used_for_framing=False,baselines=list(BASELINES),methods=list(METHODS),
        checkpoint_inventory=inventory,source_compatibility=compatibility,planned_release_not_before=str(planned),
        early_release_authorization=early_reason.strip() or None)
    write(run.output/'freeze.json',lock)
    return lock


def validation_frames(run: Any, seed: int) -> dict:
    root=run.output/'beauty';job=f'early_sd_{seed}'
    selection=run._verify_subset_selection('beauty',seed)
    subsets={label:rank_csv(root/'evaluations'/job/f'{label}_val.csv') for label in ('ID','C','B','T','CB','CT','BT','CBT')}
    groups=pd.read_csv(root/'groups.csv');pieces=[]
    for gid,g in groups.groupby('val_group_id'):
        lab=selection['groups'].get(str(int(gid)),selection['global_subset'])
        pieces.append(subsets[lab][subsets[lab].user_idx.isin(g.user_idx)])
    frames=dict(all_fixed=subsets['CBT'],global_subset=subsets[selection['global_subset']],group_subset=pd.concat(pieces))
    for method,prefix in [('independent_ID','id'),('early_no_sd','early_CBT'),('id_plus_meta','id_plus_meta'),('early_plus_meta','early_plus_meta')]:
        frames[method]=rank_csv(root/f'evaluations/{prefix}_{seed}/CBT_val.csv')
    for mode in GATES:frames[mode]=rank_csv(root/f'gates/{mode}_{seed}/per_user_val.csv')
    return frames


def frozen_gate_test(run: Any, seed: int, mode: str, report: dict) -> pd.DataFrame:
    """Load the saved gate directly: no gate fitter or Train-meta forward path."""
    import torch
    from .memo0711 import NullGate
    from .study_gates import features_for
    from .checkpoint2 import rank_frame
    from .checkpoint2_protocol import digest
    from ..training.selective_fusion import SelectiveGate,gate_evaluate
    from ..training.trainer import set_seed
    verify_freeze(run)
    folder=run.output/f'beauty/gates/{mode}_{seed}';rec=read(folder/'completed.json')
    signature=digest(dict(model=run.model_hash('beauty',f'early_sd_{seed}'),config=run.protocol_hash,
        mode=mode,label='CE',report=report))
    if rec['signature']!=signature:raise ValueError('Saved gate/config signature mismatch: '+str(folder))
    out=folder/'per_user_test.csv';binding=out.with_suffix('.binding.json');expected=dict(gate=signature,part='test')
    if binding.exists():
        cached=read(binding)
        if cached['signature']!=expected or cached['sha256']!=sha(out):raise ValueError('Test gate cache changed')
        return rank_csv(out)
    run.guard(inference=True)
    set_seed(seed)
    adapter=run.load_job('beauty',f'early_sd_{seed}')
    state=torch.load(folder/rec['attempt']/'gate.pt',map_location=adapter.device,weights_only=True)
    columns=state['feature_columns'];regimes={s:report['sources'][s]['stability']['stability_regime'] for s in report['qualifying_sources']}
    gate=NullGate(len(columns),tau=run.config['gate_tau'],kappa=run.config['gate_kappa']) if mode=='softmax_null' else SelectiveGate(
        len(columns),'global' if mode=='global_gate' else 'unsupervised' if mode=='unsupervised' else 'contextual',regimes,
        tau=run.config['gate_tau'],kappa=run.config['gate_kappa'])
    gate.load_state_dict(state['model_state_dict']);gate.to(adapter.device)
    result=gate_evaluate(adapter,gate,run.dataset('beauty')['split'].test_samples,
        features_for(run,'beauty',f'early_sd_{seed}','test'),columns,
        np.array(state['feature_mean']),np.array(state['feature_scale']),run.config['batch_size'])
    frame=rank_frame(result);frame.to_csv(out,index=False,lineterminator='\n')
    write(binding,dict(signature=expected,sha256=sha(out)))
    return frame


def release(run: Any) -> dict:
    from .checkpoint2 import exclusive
    from .checkpoint2_protocol import aligned_rank_matrix
    from ..evaluation.statistics import paired_bootstrap_test,holm_bonferroni_correction,tost_equivalence_test
    from dataclasses import asdict
    lock=verify_freeze(run)
    if not run.execute_inference:raise RuntimeError('Pass --execute-inference')
    destination=run.output/'finalization';destination.mkdir(exist_ok=True)
    result_path=destination/'release_complete.json'
    if result_path.exists():
        result=read(result_path)
        if result['freeze_sha256']!=sha(run.output/'freeze.json'):raise ValueError('Release belongs to a different freeze')
        for rel,h in result['outputs_sha256'].items():
            if sha(inside(run.output,rel))!=h:raise ValueError('Released output changed: '+rel)
        return result
    with exclusive(destination/'release.lock'):
        request=destination/'release_request.json'
        intent=dict(freeze_sha256=sha(run.output/'freeze.json'),methods=list(METHODS),seeds=list(SEEDS),
                    split='test',framing='F2_UNCHANGED',selection='locked validation choices only')
        if request.exists() and read(request)!=intent:raise ValueError('Release request changed')
        if not request.exists():write(request,intent)
        status=read(run.output/'beauty/selected_s2/status.json')
        label=status['label_choice'];report=read(run.output/f'beauty/selected_s2/e0_e1_{label}.json')
        frames={part:{method:{} for method in METHODS} for part in ('val','test')}
        for seed in SEEDS:
            val=validation_frames(run,seed)
            test=run.subset_frames('beauty',seed,'test')
            for method,prefix in [('independent_ID','id'),('early_no_sd','early_CBT'),('id_plus_meta','id_plus_meta'),('early_plus_meta','early_plus_meta')]:
                test[method]=run.inference_ranks('beauty',f'{prefix}_{seed}','CBT','test')
            for mode in GATES:
                test[mode]=frozen_gate_test(run,seed,mode,report)
            for part,current in [('val',val),('test',test)]:
                if set(current)!=set(METHODS):raise ValueError('Incomplete release family')
                aligned_rank_matrix(current,getattr(run.dataset('beauty')['split'],part+'_samples'))
                for method,f in current.items():frames[part][method][seed]=f.sort_values('user_idx').reset_index(drop=True)
        # Aggregate only once the entire locked family is available.
        rows=[]
        for method in METHODS:
            row=dict(method=method,n_seeds=5)
            for part in ('val','test'):
                for metric in ('ndcg@10','ndcg@20','recall@10','recall@20'):
                    values=[float(frames[part][method][s][metric].mean()) for s in SEEDS]
                    row[f'{part}_{metric}_mean']=float(np.mean(values));row[f'{part}_{metric}_sd']=float(np.std(values,ddof=1))
            rows.append(row)
        pd.DataFrame(rows).to_csv(destination/'table5_validation_test.csv',index=False)
        contrasts=[];tests=[];equivalence={}
        for method in BASELINES:
            comparison=paired_difference(frames['test']['uasf'],frames['test'][method],run.config['bootstrap_reps'])
            contrasts.append(dict(baseline=method,**comparison))
            a=np.mean([frames['test']['uasf'][s]['ndcg@10'].to_numpy() for s in SEEDS],axis=0)
            b=np.mean([frames['test'][method][s]['ndcg@10'].to_numpy() for s in SEEDS],axis=0)
            tests.append(paired_bootstrap_test(a,b,method_b_name=method,n_bootstrap=run.config['bootstrap_reps']))
            equivalent,lower,upper,p=tost_equivalence_test(a,b)
            equivalence[method]=dict(equivalent=bool(equivalent),lower=lower,upper=upper,p=p,relative_margin=.01)
        pd.DataFrame(contrasts).to_csv(destination/'test_paired_ci.csv',index=False)
        write(destination/'test_family_comparisons.json',dict(family=list(BASELINES),
            holm=asdict(holm_bonferroni_correction(tests)),tost=equivalence,
            p_value_note='A recorded zero means zero exceedances in finite bootstrap draws, not an exact zero probability',
            framing='F2_UNCHANGED; Test never reruns framing selection'))
        same=all(frames['test']['all_fixed'][s].equals(frames['test']['global_subset'][s]) for s in SEEDS)
        outputs={p.relative_to(run.output).as_posix():sha(p) for p in destination.iterdir()
                 if p.is_file() and p.name in ('table5_validation_test.csv','test_paired_ci.csv','test_family_comparisons.json')}
        # Include every emitted Test cache to make a resumed release verifiable.
        for p in (run.output/'beauty').rglob('*test*'):
            if p.is_file() and p.suffix in ('.csv','.json'):outputs[p.relative_to(run.output).as_posix()]=sha(p)
        result=dict(status='COMPLETE',freeze_sha256=sha(run.output/'freeze.json'),methods=list(METHODS),
            rows=11,display_rows_if_identical_all_fixed_global=10 if same else 11,
            all_fixed_global_test_identical=same,outputs_sha256=outputs,
            framing='F2_UNCHANGED',training=False,completed_at_utc=datetime.now(timezone.utc).isoformat())
        write(result_path,result)
    return result


def alpha_analysis(run: Any) -> dict:
    """Post-freeze gate-only CPU forward on frozen validation features. No Test access."""
    import torch
    from ..training.selective_fusion import SelectiveGate,feature_matrix
    from .checkpoint2 import exclusive
    if not run.execute_inference:raise RuntimeError('Pass --execute-inference')
    verify_freeze(run)
    destination=run.output/'finalization';destination.mkdir(exist_ok=True)
    done=destination/'alpha_complete.json'
    if done.exists():
        rec=read(done)
        if rec['freeze_sha256']!=sha(run.output/'freeze.json'):raise ValueError('Alpha analysis freeze changed')
        for rel,h in rec['outputs_sha256'].items():
            if sha(inside(run.output,rel))!=h:raise ValueError('Posthoc alpha output changed')
        return rec
    with exclusive(destination/'alpha.lock'):
        rows=[];summaries=[];report=read(run.output/'beauty/selected_s2/e0_e1_CE.json')
        regimes={s:report['sources'][s]['stability']['stability_regime'] for s in report['qualifying_sources']}
        for seed in SEEDS:
            folder=run.output/f'beauty/gates/uasf_{seed}';rec=read(folder/'completed.json')
            state=torch.load(folder/rec['attempt']/'gate.pt',map_location='cpu',weights_only=True)
            features=pd.read_csv(run.output/f'beauty/gate_features/early_sd_{seed}/val.csv')
            if features.user_idx.duplicated().any():raise ValueError('Duplicate validation feature users')
            x=feature_matrix(features,state['feature_columns'],np.array(state['feature_mean']),np.array(state['feature_scale']))[0]
            gate=SelectiveGate(len(state['feature_columns']),regimes=regimes,
                               tau=run.config['gate_tau'],kappa=run.config['gate_kappa'])
            gate.load_state_dict(state['model_state_dict']);gate.eval()
            with torch.no_grad():
                alpha,beta=gate(torch.tensor(x),torch.tensor(features.group_idx.to_numpy(),dtype=torch.long),
                                torch.ones(len(features),3),threshold=True)
            a,b=alpha.numpy(),beta.numpy();off=(a==0).all(axis=1)
            for j,source in enumerate(('category','brand','title')):
                summaries.append(dict(seed=seed,source=source,users=len(features),mean_beta=float(b[:,j].mean()),
                    mean_alpha=float(a[:,j].mean()),median_alpha=float(np.median(a[:,j])),
                    p10_alpha=float(np.quantile(a[:,j],.1)),p90_alpha=float(np.quantile(a[:,j],.9)),
                    source_off_fraction=float((a[:,j]==0).mean()),all_off_fraction=float(off.mean())))
            for i,user in enumerate(features.user_idx):
                rows.append(dict(seed=seed,user_idx=int(user),all_off=bool(off[i]),
                    **{f'{kind}_{source}':float(array[i,j]) for kind,array in [('beta',b),('alpha',a)]
                       for j,source in enumerate(('category','brand','title'))}))
        pd.DataFrame(rows).to_csv(destination/'validation_alpha_per_user.csv',index=False)
        pd.DataFrame(summaries).to_csv(destination/'validation_alpha_summary.csv',index=False)
        outputs={p.relative_to(run.output).as_posix():sha(p) for p in
                 [destination/'validation_alpha_per_user.csv',destination/'validation_alpha_summary.csv']}
        result=dict(posthoc=True,split='validation',training=False,backbone_forward=False,
            used_for_selection=False,freeze_sha256=sha(run.output/'freeze.json'),outputs_sha256=outputs)
        write(done,result)
    return result
