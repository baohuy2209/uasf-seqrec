"""Read-only adoption of old runs and versioned completion of the November study.

No historical protocol lock, checkpoint, ranks, or source is rewritten. The
default CLI stage is a plan. Real computation requires explicit execution flags.
"""
from pathlib import Path
from dataclasses import asdict,replace
import copy,json,shutil,subprocess,sys
import numpy as np
import pandas as pd
import torch
from .checkpoint2 import Checkpoint2,subset_name,exclusive,rank_frame
from .late_checkpoint import LateCheckpoint,lock_json,normalize_meta_signature
from .checkpoint2_protocol import (SEEDS,SUBSETS,SOURCE_ORDER,sha,digest,write_json,context_signature,
    group_contexts,read_ranks,aligned_rank_matrix,grouped_subset_analysis,git_state)
from ..training.checkpoint2_trainer import ModelSpec,ModelAdapter,evaluate
from ..evaluation.checkpoint2_utility import collect_ablations
from metadata_pipeline.user_grouping import apply_group_rules
from .study_controls import build_controls

COMPATIBLE_MODEL_FILES=['src/taimma/models/'+n+'.py' for n in
 ['sasrec','early_fusion','late_fusion','source_dropout']]+[
 'src/taimma/training/checkpoint2_trainer.py','src/taimma/training/multimodal_trainer.py',
 'src/taimma/evaluation/evaluator.py','src/metadata_pipeline/loader.py']


def choose_backbone(id_val,early_val,sd_full,sd_off,joint_val=None):
    arrays=[np.asarray(a,float) for a in [id_val,early_val,sd_full,sd_off]]
    if any(a.shape!=(3,) or not np.isfinite(a).all() for a in arrays):raise ValueError('Exactly three validation seeds required')
    ident,early,full,off=[float(a.mean()) for a in arrays]
    def at_least(a,b):return bool(a>=b or np.isclose(a,b,rtol=1e-14,atol=0.))
    checks=dict(full_beats_id=full>ident,retains_98pct_early=at_least(full,.98*early),off_retains_95pct_id=at_least(off,.95*ident))
    selected='early_sd' if all(checks.values()) else None
    if not selected and joint_val is not None:
        joint=np.asarray(joint_val,float)
        if joint.shape!=(3,) or not np.isfinite(joint).all():raise ValueError('K4 selection requires three validation seeds')
        if joint.mean()>ident:selected='late_joint'
    return dict(backbone=selected,status='SELECTED' if selected else 'NO_ELIGIBLE_BACKBONE' if joint_val is not None else 'AWAITING_K4_SEEDS',
                checks=checks,mean_validation=dict(ID=ident,early=early,sd_full=full,sd_off=off,
                 joint=None if joint_val is None else float(np.mean(joint_val))),test_used=False)


def paired_summary(left,right,reps=1000):
    if set(left)!=set(right) or len(left)<3:raise ValueError('At least three matching seeds required')
    deltas=[];keys=None
    for s in sorted(left):
        _,users,m=aligned_rank_matrix({'left':left[s],'right':right[s]})
        current=list(zip(left[s].user_idx,left[s].target_idx))
        if keys is not None and current!=keys:raise ValueError('Seed cohorts differ')
        keys=current;deltas.append(m[:,0]-m[:,1])
    d=np.mean(deltas,axis=0);rng=np.random.default_rng(42)
    boots=[float(d[rng.integers(0,len(d),len(d))].mean()) for _ in range(reps)]
    return dict(delta_left_minus_right=float(d.mean()),ci95=np.percentile(boots,[2.5,97.5]).tolist(),
                seeds=sorted(left),resampling_unit='user_with_all_seed_predictions',replicates=reps)


class StudyCompletion(LateCheckpoint):
    def dataset(self,key):
        d=super().dataset(key)
        # Explicit, eligible-only S2 cohort; Val/Test retain every OOV miss.
        if not d.get('completion_meta_filtered'):
            d['split']=copy.copy(d['split'])
            d['split'].train_meta_samples=[s for s in d['split'].train_meta_samples if s.target>0]
            d['completion_meta_filtered']=True
        return d

    def prepare(self,key):
        d=super().prepare(key)
        lock_json(self.output/key/'utility_label_rule.json',dict(rule='CE if all three signs match validation; else LOG_RANK if all match; else UNRESOLVED',test_used=False))
        return d

    def run_job(self,key,name,spec,train_samples=None,id_job=None,control=None,max_wall_seconds=None):
        if (self.output/'freeze.json').exists():
            rec=self.completed(key,name)
            if rec is None:raise RuntimeError('Data are frozen; new model fitting is forbidden')
            return rec
        self.assert_protocol(key)
        if id_job:lock_json(self.job_path(key,name)/'id_dependency.json',dict(id_job=id_job,id_checkpoint_sha256=self.model_hash(key,id_job)))
        return Checkpoint2.run_job(self,key,name,spec,train_samples,id_job,control,max_wall_seconds)

    def import_job(self,key,name,source_root,source_name=None):
        """Copy and rehash verified artifacts; require original model weights."""
        target=self.job_path(key,name)
        if self.completed(key,name):return True
        source=Path(source_root)/key/'jobs'/(source_name or name)
        if not (source/'completed.json').exists():return False
        rec=json.loads((source/'completed.json').read_text(encoding='utf-8'))
        attempt=(source/rec['checkpoint']).parent
        if not (source/rec['checkpoint']).is_file():raise FileNotFoundError('Original .pt needed on server: '+str(source/rec['checkpoint']))
        cfg=json.loads((attempt/'config.json').read_text(encoding='utf-8'))
        environment=self.config.get('environment',{})
        if environment:
            origin=json.loads((attempt/'run_manifest.json').read_text(encoding='utf-8'))
            actual=dict(python=origin['python_version'],torch=origin['torch_version'],**origin['library_versions'])
            bad={k:(v,actual.get(k)) for k,v in environment.items() if k!='gpu_contains' and str(v)!=str(actual.get(k))}
            history=json.loads((attempt/'history.json').read_text(encoding='utf-8'))
            gpu=history.get('resource_usage',{}).get('gpu_name','')
            if bad or environment.get('gpu_contains','RTX 4060') not in gpu:
                raise ValueError('Historical checkpoint environment differs from the locked study: '+str(bad)+str(gpu))
        original=cfg['data_signature'];current=self.dataset(key)['signature']
        for field in ['raw_reviews_sha256','metadata_files','catalog_sha256','train_base_sha256','val_sha256','test_sha256']:
            if original[field]!=current[field]:raise ValueError('Historical dataset mismatch: '+field)
        snapshots=json.loads((attempt/'source_snapshot.json').read_text(encoding='utf-8'))
        for rel in COMPATIBLE_MODEL_FILES:
            import hashlib
            b=(self.repo/rel).read_bytes().replace(b'\r\n',b'\n')
            if hashlib.sha256(b).hexdigest()!=snapshots.get(rel):
                raise ValueError('Model/trainer changed; do not silently reuse: '+rel)
        verified={};missing=[]
        import hashlib
        for rel,h in rec['artifacts'].items():
            p=(source/rel).resolve()
            if not p.is_relative_to(source.resolve()):raise ValueError('Artifact escapes source job')
            if not p.exists():missing.append(rel);continue
            b=p.read_bytes()
            text_file=p.suffix.lower() in ['.json','.csv','.log','.txt','.md','.py','.yaml','.yml','.diff','.tex']
            if sha(p)!=h and not (text_file and hashlib.sha256(b.replace(b'\r\n',b'\n')).hexdigest()==h):
                raise ValueError('Historical artifact changed: '+str(p))
            verified[rel]=p
        if rec['checkpoint'] not in verified:raise ValueError('Unverified checkpoint')
        spec=ModelSpec(**{**rec['spec'],'sources':tuple(rec['spec']['sources'])})
        # Import only metadata-real ID/early/late/SD/K4; controls are versioned anew.
        if rec.get('control'):
            source_name,kind,seed=rec['control']
            if kind!='shuffled':raise ValueError('Old random controls do not match V2')
            original_csv=source.parent.parent/'training_contrasts/shuffled'/f'perm_{source_name}_seed_{seed}.csv'
            new_csv=self.output/key/'training_contrasts/shuffled'/original_csv.name
            if not new_csv.exists() or not pd.read_csv(original_csv).equals(pd.read_csv(new_csv)):
                raise ValueError('Permutation differs; cannot adopt the old shuffled fit')
        id_job=f'id_{spec.seed}' if spec.kind=='late' and spec.freeze_id else None
        samples=self.dataset(key)['split'].train_base_samples
        if name=='K2':
            split_record=json.loads((source.parent.parent/'diagnostics/K2_split.json').read_text())
            fit=set(split_record['fit_users']);samples=[s for s in self.dataset(key)['split'].train_meta_samples if s.user_idx in fit]
            if context_signature(samples)!=split_record['fit_context_sha256']:raise ValueError('K2 held-out fit cohort differs')
        fingerprint=digest(dict(spec=asdict(spec),train=context_signature(samples),
            config=self.protocol_hash,data=current,control=rec.get('control'),id_job=id_job))
        with exclusive(target/'execution.lock'):
            for rel,p in verified.items():
                dest=target/rel;dest.parent.mkdir(parents=True,exist_ok=True)
                if dest.exists() and sha(dest)!=sha(p):raise ValueError('Import target collision')
                if not dest.exists():shutil.copy2(p,dest)
            write_json(target/'import_provenance.json',dict(source=str(source),original_fingerprint=rec['fingerprint'],
                scientific_inputs_checked=True,missing_non_checkpoint_artifacts=missing))
            rec.update(fingerprint=fingerprint,artifacts={rel:sha(target/rel) for rel in verified})
            write_json(target/'completed.json',rec)
        return True

    def adopt(self,key,roots,utility_only=False):
        self.prepare(key)
        if utility_only:
            for seed in [42,43]:
                if not self.import_job(key,f'early_sd_{seed}',roots['early']):raise FileNotFoundError('Missing early SD reference '+str(seed))
            return
        labels=['ID','CBT'] if utility_only else [subset_name(s) for s in SUBSETS]
        for s in SEEDS:
            for lab in labels:
                if lab=='ID':self.import_job(key,f'id_{s}',roots['early'])
                else:
                    self.import_job(key,f'early_{lab}_{s}',roots['early'])
                    if not utility_only:self.import_job(key,f'late_{lab}_{s}',roots['late'])
            self.import_job(key,f'early_sd_{s}',roots['early'])
        if not utility_only:
            for job in ['late_reference_42','K2','K3','K4']:
                self.import_job(key,job,roots['diagnostics'])
            diagnostic=Path(roots['diagnostics'])/key/'diagnostics'
            if diagnostic.is_dir():
                dest=self.output/key/'historical_diagnostics';dest.mkdir(exist_ok=True)
                for p in diagnostic.iterdir():
                    if p.is_file() and p.suffix in ['.csv','.json']:shutil.copy2(p,dest/p.name)

    def inference_ranks(self,key,job,label,part='val',mask=None):
        self.guard(inference=True);d=self.dataset(key);samples=getattr(d['split'],part+'_samples')
        root=self.output/key/'evaluations'/job;p=root/f'{label}_{part}.csv';marker=p.with_suffix('.json')
        signature=dict(model=self.model_hash(key,job),contexts=context_signature(samples),label=label,
                       mask=None if mask is None else digest({s:v.tolist() for s,v in mask.items()}))
        if marker.exists():
            m=json.loads(marker.read_text())
            if m['signature']!=signature or sha(p)!=m['sha256']:raise ValueError('Rank cache changed')
            return read_ranks(p)
        adapter=self.load_job(key,job)
        if mask is not None:adapter.masks={s:v.to(adapter.device) for s,v in mask.items()}
        enabled=() if label=='ID' else tuple(s for s in SOURCE_ORDER if s[0].upper() in label) if label in [subset_name(s) for s in SUBSETS] else None
        result=evaluate(adapter,samples,self.config['batch_size'],enabled=enabled)
        root.mkdir(parents=True,exist_ok=True);rank_frame(result).to_csv(p,index=False,lineterminator='\n')
        write_json(marker,dict(signature=signature,sha256=sha(p)));return read_ranks(p)

    def select(self,key,complete_joint=True):
        self.id_grid(key)
        for s in SEEDS:
            self.run_job(key,f'early_CBT_{s}',ModelSpec('early',s))
            self.run_job(key,f'early_sd_{s}',ModelSpec('early',s,source_dropout=.2,all_off_probability=.2))
        def vals(prefix):return [self.completed(key,f'{prefix}_{s}')['val_ndcg_10'] for s in SEEDS]
        off=[self.inference_ranks(key,f'early_sd_{s}','ID')['ndcg@10'].mean() for s in SEEDS]
        prelim=choose_backbone(vals('id'),vals('early_CBT'),vals('early_sd'),off)
        joint=None
        if not prelim['backbone'] and complete_joint:
            for s in SEEDS:
                job='K4' if s==42 else f'late_joint_{s}'
                self.run_job(key,job,ModelSpec('late',s,freeze_id=False,source_dropout=.2,all_off_probability=.2))
            joint=[self.completed(key,'K4' if s==42 else f'late_joint_{s}')['val_ndcg_10'] for s in SEEDS]
        choice=choose_backbone(vals('id'),vals('early_CBT'),vals('early_sd'),off,joint)
        write_json(self.output/key/'backbone_decision.json',choice)
        return choice

    def utility(self,key,jobs,folder,selected=False):
        self.validate_independent_references(key,jobs)
        label=self.alignment(key,jobs,folder)
        root=self.output/key/folder;f=pd.read_csv(root/'utility_both_labels.csv')
        rules=json.loads((self.output/key/'groups.json').read_text())['rules']
        f['group_idx']=apply_group_rules(np.rint(np.exp(f.feat_log_len)),f.feat_history_tail_ratio,rules)[2]
        # Both targets are reported diagnostically; the locked rule alone selects one.
        outcomes={}
        for candidate,column in [('CE','delta_ce'),('LOG_RANK','delta_rank')]:
            data=f.copy();data['delta']=data[column];p=root/f'utility_{candidate}.csv'
            data.to_csv(p,index=False,lineterminator='\n')
            report=root/f'e0_e1_{candidate}.json';cache=report.with_suffix('.cache.json')
            signature=dict(input_sha256=sha(p),source_sha256=sha(self.repo/'src/taimma/evaluation/e0_e1_protocol.py'),reps=self.config['bootstrap_reps'])
            if report.exists():
                if not cache.exists() or json.loads(cache.read_text())!={'signature':signature,'sha256':sha(report)}:raise ValueError('E1 cache changed')
            else:
                subprocess.run([sys.executable,str(self.repo/'scripts/run_e0_e1_evaluation.py'),'--delta_file',str(p),
                    '--n_bootstrap',str(self.config['bootstrap_reps']),'--output_json',str(report)],cwd=self.repo,check=True)
                write_json(cache,dict(signature=signature,sha256=sha(report)))
            outcomes[candidate]=json.loads(report.read_text())
        status=dict(label_choice=label,selected_backbone=selected,
                    decision=outcomes[label]['decision'] if label in outcomes else 'UNRESOLVED',
                    s3_eligible=selected and label in outcomes and outcomes[label]['decision'] in ['GO','CONDITIONAL'],
                    diagnostic_only=not selected,reference_jobs=jobs,test_used=False)
        write_json(root/'status.json',status);return status

    def validate_independent_references(self,key,jobs):
        if len(jobs)!=2:raise ValueError('Exactly two independent reference systems required')
        records=[self.completed(key,j) for j in jobs]
        if any(r is None for r in records) or records[0]['spec']['seed']==records[1]['spec']['seed']:
            raise ValueError('Reference systems must use distinct declared seeds')
        weights=[]
        for job,rec in zip(jobs,records):
            state=torch.load(self.job_path(key,job)/rec['checkpoint'],map_location='cpu',weights_only=True)['model_state_dict']
            base={k.split('.',1)[1]:v for k,v in state.items() if k.startswith(('base_sasrec.','id_model.'))}
            if not base:raise ValueError('No ID backbone weights found in reference')
            weights.append(base)
        if weights[0].keys()!=weights[1].keys():raise ValueError('Reference ID architectures differ')
        if all(torch.equal(weights[0][k],weights[1][k]) for k in weights[0]):
            raise ValueError('Reference ID weights are identical; independent ID training required')

    def complete_utility(self,key):
        names=[f'early_sd_{s}' for s in [42,43]]
        for n in names:
            if not self.completed(key,n):raise FileNotFoundError('Need original checkpoint for '+n)
        return self.utility(key,names,'diagnostic_early_sd',selected=False)

    def grids(self,key):
        self.early_grid(key)
        for s in SEEDS:
            for sources in SUBSETS[1:]:
                self.run_job(key,f'late_{subset_name(sources)}_{s}',ModelSpec('late',s,sources=sources,
                    source_dropout=.2,source_dropout_mode='batch'),id_job=f'id_{s}')
                self.verify_frozen_id(key,f'late_{subset_name(sources)}_{s}')

    def controls_v2(self,key):
        old=Path(self.config['input_roots']['early'])/key/'training_contrasts'
        build_controls(self.dataset(key)['metadata_dir'],self.output/key/'training_contrasts',shuffled_source=old)
        for s in SEEDS:
            for source in SOURCE_ORDER:
                for kind in ['shuffled','random']:
                    if kind=='shuffled':self.import_job(key,f'control_{kind}_{source}_{s}',self.config['input_roots']['early'])
                    self.run_job(key,f'control_{kind}_{source}_{s}',ModelSpec('early',s),control=(source,kind,s))

    def dropout_profile(self,key):
        rows=[]
        for s in SEEDS:
            for label in ['CBT','BT','CT','CB','ID']:
                for part in ['val','test']:
                    f=self.inference_ranks(key,f'early_sd_{s}',label,part)
                    rows.append(dict(seed=s,subset=label,part=part,ndcg_10=f['ndcg@10'].mean()))
        pd.DataFrame(rows).to_csv(self.output/key/'dropout_profile.csv',index=False,lineterminator='\n')

    def id_grid_seed(self,key,seed):
        self.run_job(key,f'id_{seed}',ModelSpec('id',seed,sources=()))

    def selected_utility(self,key,choice):
        backbone=choice['backbone']
        if backbone is None:
            status=dict(s3_eligible=False,decision='NO_ELIGIBLE_BACKBONE',selected_backbone=False)
            write_json(self.output/key/'selected_s2/status.json',status)
            write_json(self.output/key/'framing_decision.json',dict(framing='F2',reason='No eligible backbone',test_used=False))
            return status
        jobs=[f'early_sd_{s}' if backbone=='early_sd' else 'K4' if s==42 else f'late_joint_{s}' for s in [42,43]]
        status=self.utility(key,jobs,'selected_s2',selected=True)
        if not status['s3_eligible']:write_json(self.output/key/'framing_decision.json',dict(framing='F2',reason=status['decision'],test_used=False))
        return status

    def freeze(self,tag=None):
        from datetime import datetime,timezone,timedelta,date
        today=datetime.now(timezone(timedelta(hours=7))).date()
        if today<date.fromisoformat(self.config['test_release_not_before']):raise RuntimeError('Do not release UASF test before the planned freeze date')
        if (self.output/'freeze.json').exists():self.verify_freeze();return
        git=git_state(self.repo,require_clean=True)
        if not tag:raise RuntimeError('Pass an existing reviewed Git tag with --freeze-tag')
        tagged=subprocess.check_output(['git','rev-parse',str(tag)+'^{commit}'],cwd=self.repo,text=True).strip()
        if tagged!=git['commit']:raise ValueError('Freeze tag must point to the current clean commit')
        # Each included cohort has an explicit validation-only framing decision.
        for key in self.config['datasets']:
            if not (self.output/key/'framing_decision.json').exists():raise RuntimeError('Missing validation framing decision: '+key)
        files={p.relative_to(self.output).as_posix():sha(p) for p in sorted(self.output.rglob('*'))
               if p.is_file() and p.suffix in ['.pt','.json','.csv'] and p.name not in
               ['freeze.json','SHA256_MANIFEST.json','results.csv','report_tables.csv','utility_summary.csv','gate_test.csv']
               and not p.name.endswith('_test.csv')}
        lock_json(self.output/'freeze.json',dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),files=files,git=git,git_tag=tag,
                  source_sha256=self.source_signature(),datasets=self.config['datasets'],test_used_for_framing=False))

    def verify_freeze(self):
        p=self.output/'freeze.json'
        if not p.exists():raise RuntimeError('Freeze the validation-only study before opening UASF test')
        m=json.loads(p.read_text())
        if m['source_sha256']!=self.source_signature():raise ValueError('Executable source changed after freeze')
        for rel,h in m['files'].items():
            if sha(self.output/rel)!=h:raise ValueError('Frozen artifact changed: '+rel)

    def run_all(self,key,roots):
        if (self.output/'freeze.json').exists():raise RuntimeError('Study frozen; use report or release-test, not all')
        self.adopt(key,roots)
        self.grids(key)
        if key=='beauty':self.controls_v2(key)
        if key=='beauty':
            choice=self.select(key)
            self.complete_utility(key)
            self.dropout_profile(key)
        else:
            p=self.output/'beauty/backbone_decision.json'
            if not p.exists():raise RuntimeError('Select the global backbone on Beauty validation first')
            choice=json.loads(p.read_text());choice=dict(choice,selection_dataset='beauty')
            write_json(self.output/key/'backbone_decision.json',choice)
            if choice['backbone']:
                from .study_gates import ensure_foundation
                for seed in SEEDS:ensure_foundation(self,key,choice['backbone'],seed)
        status=self.selected_utility(key,choice)
        self.group_analysis(key)
        self.robustness(key)
        if status['s3_eligible']:
            from .study_gates import fit_validation
            fit_validation(self,key,choice['backbone'],status)
        else:
            write_json(self.output/key/'s3_status.json',dict(status='SKIPPED_BY_PROTOCOL',reason=status['decision'],test_used=False))
        return self.export(key)

    def robustness(self,key):
        d=self.dataset(key);rows=[]
        for kind in ['early','late']:
            for s in SEEDS:
                for source in SOURCE_ORDER:
                    available=np.flatnonzero(d['masks'][source].numpy()>0)
                    order=np.random.default_rng(s).permutation(available)
                    for rate in self.config['missing_rates']:
                        masks={k:v.clone() for k,v in d['masks'].items()}
                        masks[source][order[:int(round(rate*len(order)))]]=0
                        label=f'missing_{source}_{rate:.2f}'
                        f=self.inference_ranks(key,f'{kind}_CBT_{s}',label,'test',masks)
                        rows.append(dict(dataset=key,mechanism=kind,seed=s,source=source,missing_rate=rate,ndcg_10=f['ndcg@10'].mean()))
        pd.DataFrame(rows).to_csv(self.output/key/'robustness.csv',index=False,lineterminator='\n')

    def group_analysis(self,key):
        split=self.dataset(key)['split'];groups=pd.read_csv(self.output/key/'groups.csv');rows=[]
        for kind in ['early','late']:
            for s in SEEDS:
                parts={}
                for part in ['val','test']:
                    parts[part]={}
                    for sources in SUBSETS:
                        lab=subset_name(sources);job=f'{kind}_{lab}_{s}' if sources else f'id_{s}'
                        rec=self.completed(key,job)
                        if rec is None:raise FileNotFoundError('Incomplete grid: '+job)
                        parts[part][lab]=read_ranks((self.job_path(key,job)/rec['checkpoint']).parent/f'per_user_{part}.csv')
                    aligned_rank_matrix(parts[part],getattr(split,part+'_samples'))
                summary,users=grouped_subset_analysis(parts['val'],parts['test'],groups)
                users.to_csv(self.output/key/f'group_{kind}_{s}.csv',index=False,lineterminator='\n')
                rows.append(dict(mechanism=kind,seed=s,**summary))
        write_json(self.output/key/'group_analysis.json',rows)

    def export(self,key):
        rows=[]
        for p in sorted((self.output/key/'jobs').glob('*/completed.json')):
            rec=self.completed(key,p.parent.name);attempt=(p.parent/rec['checkpoint']).parent
            h=json.loads((attempt/'history.json').read_text())
            row=dict(dataset=key,job=p.parent.name,kind=rec['spec']['kind'],seed=rec['spec']['seed'],
                     best_epoch=rec['best_epoch'],best_epoch_le5=rec['best_epoch']<=5,
                     val_ndcg_10=rec['val_ndcg_10'],test_ndcg_10=rec['test_ndcg_10'],
                     initial_val=h['initial_validation_metrics']['NDCG@10'],epoch1_val=h['val_ndcg_10'][0],
                     checkpoint_sha256=sha(p.parent/rec['checkpoint']))
            for part in ['val','test']:
                f=read_ranks(attempt/f'per_user_{part}.csv')
                aligned_rank_matrix({'job':f},getattr(self.dataset(key)['split'],part+'_samples'))
                rank=f['rank'].to_numpy();target=f.target_idx.to_numpy()
                recomputed=np.where((rank>=1)&(rank<=10)&(target>0),1/np.log2(1+np.maximum(rank,1)),0.)
                if not np.allclose(f['ndcg@10'],recomputed,rtol=0,atol=1e-14):raise ValueError('NDCG values disagree with ranks')
                if not np.isclose(f['ndcg@10'].mean(),row[part+'_ndcg_10'],atol=1e-14,rtol=0):raise ValueError('Metrics disagree with ranks')
            rows.append(row)
        pd.DataFrame(rows).to_csv(self.output/key/'results.csv',index=False,lineterminator='\n')
        curves=[]
        for name in ['late_reference_42','K3']:
            rec=self.completed(key,name)
            if rec:
                h=json.loads((self.job_path(key,name)/rec['checkpoint']).parent.joinpath('history.json').read_text())
                curves.append(dict(job=name,epoch=0,val_ndcg_10=h['initial_validation_metrics']['NDCG@10']))
                curves.extend(dict(job=name,epoch=i+1,val_ndcg_10=v) for i,v in enumerate(h['val_ndcg_10']))
        if curves:pd.DataFrame(curves).to_csv(self.output/key/'K3_validation_epochs.csv',index=False,lineterminator='\n')
        from .completion_report import write_report
        from .study_figures import write_figures
        write_figures(self.output/key)
        report=write_report(self.output/key,self.config)
        manifests={p.relative_to(self.output/key).as_posix():sha(p) for p in (self.output/key).rglob('*')
                   if p.is_file() and p.name!='SHA256_MANIFEST.json' and not p.name.endswith('.lock')}
        write_json(self.output/key/'SHA256_MANIFEST.json',manifests)
        return report
