"""Verified, read-only-source migration of memo S1 checkpoints and gate caches."""
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import shutil

from .checkpoint2 import exclusive
from .checkpoint2_protocol import context_signature, digest, sha, write_json
from .study_completion import COMPATIBLE_MODEL_FILES
from ..training.checkpoint2_trainer import ModelSpec


def training_samples(run, key, name):
    split = run.dataset(key)['split']
    samples = split.train_base_samples
    if name.startswith(('id_plus_meta_', 'early_plus_meta_')):
        samples = samples + split.train_meta_samples
    return samples


def checked_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Artifact escapes its job: ' + str(relative))
    return path


def verify_files(root, artifacts):
    verified = {}
    for relative, expected in artifacts.items():
        p = checked_path(root, relative)
        if not p.is_file():
            raise FileNotFoundError('Missing original artifact: ' + str(p))
        actual = sha(p)
        if actual != expected:
            text = p.suffix.lower() in {'.json', '.csv', '.log', '.txt', '.md', '.py', '.yaml', '.yml'}
            if not text or hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest() != expected:
                raise ValueError('Original artifact changed: ' + str(p))
        verified[relative] = p
    return verified


def adopt_s1(run, key, name, source_root):
    source = source_root / key / 'jobs' / name
    rec = json.loads((source / 'completed.json').read_text(encoding='utf-8'))
    if rec.get('status') != 'COMPLETED':
        raise ValueError('Job is not completed: ' + str(source))
    if name.startswith(('K', 'late_joint', 'control_random')):
        raise ValueError('Forbidden new-memo S1 job: ' + name)
    files = verify_files(source, rec['artifacts'])
    cp = checked_path(source, rec['checkpoint'])
    if rec['checkpoint'] not in files:
        raise ValueError('Checkpoint is not hash-bound: ' + str(cp))
    attempt = cp.parent
    for required in ['config.json', 'source_snapshot.json', 'run_manifest.json', 'history.json', 'per_user_val.csv']:
        relative = (attempt / required).relative_to(source).as_posix()
        if relative not in files:
            raise ValueError('Missing hash-bound evidence: ' + relative)
    cfg = json.loads((attempt / 'config.json').read_text(encoding='utf-8'))
    spec = ModelSpec(**{**rec['spec'], 'sources': tuple(rec['spec']['sources'])})
    if digest(cfg['spec']) != digest(asdict(spec)):
        raise ValueError('Recorded model specs disagree: ' + name)
    old_data = cfg['data_signature']; data = run.dataset(key)['signature']
    imported=(source/'import_provenance.json').is_file()
    protocol_path=source_root/key/'protocol.lock.json'
    if not protocol_path.is_file():raise FileNotFoundError('Source memo protocol lock missing')
    protocol=json.loads(protocol_path.read_text(encoding='utf-8'))
    # Historical imported configs retain the pre-normalization Meta cohort with
    # OOV rows; the memo lock binds its eligible-only S2 cohort instead.
    meta_evidence=protocol['data'] if imported else old_data
    for field in ['train_meta_sha256','train_meta_eligible','train_meta_oov','train_meta_total']:
        if meta_evidence.get(field)!=data.get(field):raise ValueError('Memo eligible Train-meta mismatch: '+field)
    for field in ['raw_reviews_sha256', 'metadata_files', 'catalog_sha256', 'train_base_sha256', 'val_sha256', 'test_sha256']:
        if old_data[field] != data[field]:
            raise ValueError('Memo data mismatch: ' + key + '/' + field)
    snapshots = json.loads((attempt / 'source_snapshot.json').read_text(encoding='utf-8'))
    for relative in COMPATIBLE_MODEL_FILES:
        current = hashlib.sha256((run.repo / relative).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        if snapshots.get(relative) != current:
            raise ValueError('S1 model/trainer changed: ' + relative)
    env = run.config.get('environment', {})
    origin = json.loads((attempt / 'run_manifest.json').read_text(encoding='utf-8'))
    actual = dict(python=origin['python_version'], torch=origin['torch_version'], **origin['library_versions'])
    if any(str(actual.get(k)) != str(v) for k, v in env.items() if k != 'gpu_contains'):
        raise ValueError('S1 runtime differs from locked environment: ' + name)
    if env.get('gpu_contains'):
        history = json.loads((attempt / 'history.json').read_text(encoding='utf-8'))
        if env['gpu_contains'] not in history.get('resource_usage', {}).get('gpu_name', ''):
            raise ValueError('S1 GPU differs from locked environment: ' + name)
    control = rec.get('control')
    if control:
        src, kind, seed = control
        if kind != 'shuffled':
            raise ValueError('Only existing shuffled control fits can be adopted')
        old = source_root / key / 'training_contrasts/shuffled' / f'perm_{src}_seed_{seed}.csv'
        new = run.output / key / 'training_contrasts/shuffled' / old.name
        if not old.is_file() or not new.is_file() or sha(old) != sha(new):
            raise ValueError('Training contrast mismatch: ' + name)
    id_job = f'id_{spec.seed}' if spec.kind == 'late' and spec.freeze_id else None
    source_config = protocol['config']
    for field in ['epochs','patience','batch_size','environment']:
        if source_config.get(field)!=run.config.get(field):raise ValueError('S1 training protocol changed: '+field)
    if not (source / 'import_provenance.json').is_file():
        original_fingerprint=digest(dict(spec=asdict(spec),train=context_signature(training_samples(run,key,name)),
            config=digest(source_config),data=protocol['data'],control=control,id_job=id_job))
        if rec['fingerprint']!=original_fingerprint or cfg.get('fingerprint')!=original_fingerprint:
            raise ValueError('Original memo training fingerprint mismatch: '+name)
    fingerprint = digest(dict(spec=asdict(spec), train=context_signature(training_samples(run, key, name)),
                              config=run.protocol_hash, data=data, control=control, id_job=id_job))
    old = run.completed(key, name)
    if old:
        if old['fingerprint'] != fingerprint or old['artifacts'][old['checkpoint']] != sha(cp):
            raise ValueError('Adoption destination collision: ' + name)
        return
    target = run.job_path(key, name)
    with exclusive(target / 'execution.lock'):
        for relative, p in files.items():
            dest = checked_path(target, relative); dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists() and sha(dest) != sha(p):
                raise ValueError('Adoption collision: ' + str(dest))
            if not dest.exists():
                shutil.copy2(p, dest)
        write_json(target / 'import_provenance.json', dict(source=str(source),
            original_completed_sha256=sha(source / 'completed.json'), original_fingerprint=rec['fingerprint'],
            train_context_sha256=context_signature(training_samples(run, key, name)), scientific_inputs_checked=True))
        rec.update(fingerprint=fingerprint, artifacts={rel:sha(target / rel) for rel in files})
        write_json(target / 'completed.json', rec)


def adopt_memo(run, source_root, datasets):
    source_root = Path(source_root).resolve()
    if source_root == run.output or source_root in run.output.parents or run.output in source_root.parents:
        raise ValueError('Use a separate output, never overwrite or nest inside the original memo')
    if not source_root.is_dir() or list(source_root.glob('*/memo_execution.lock')) or list(source_root.glob('*/jobs/*/execution.lock')):
        raise ValueError('Source memo must exist and have no active execution locks')
    manifest = dict(source=str(source_root), schema_version=2, jobs={}, gates={},
                    invalidated=['robustness', 'add_one'], training_executed=False)
    destination_manifest=run.output/'adoption_manifest.json'
    if destination_manifest.exists():
        previous=json.loads(destination_manifest.read_text(encoding='utf-8'))
        if previous.get('source')!=str(source_root) or previous.get('schema_version')!=2:
            raise ValueError('Adoption source/schema changed; choose another output')
        manifest=previous
    for key in datasets:
        markers = sorted((source_root / key / 'jobs').glob('*/completed.json'))
        if not markers:
            raise FileNotFoundError('No completed source jobs for ' + key)
        run.prepare(key)
        # Permutations must remain the exact same training controls.
        contrasts = source_root / key / 'training_contrasts'
        if contrasts.is_dir():
            dest = run.output / key / 'training_contrasts'
            for p in contrasts.rglob('*'):
                if p.is_file():
                    target = dest / p.relative_to(contrasts); target.parent.mkdir(parents=True, exist_ok=True)
                    if target.exists() and sha(target) != sha(p):raise ValueError('Contrast collision')
                    if not target.exists():shutil.copy2(p, target)
        # ID first, then dependent early/late models.
        markers.sort(key=lambda p:(not p.parent.name.startswith('id_'), p.parent.name))
        for marker in markers:
            adopt_s1(run, key, marker.parent.name, source_root)
        manifest['jobs'][key] = [p.parent.name for p in markers]
        manifest.setdefault('source_protocol_sha256',{})[key]=sha(source_root/key/'protocol.lock.json')
        # Posthoc utility is regenerated from unchanged models. Existing gates
        # are retained only when every exact signature dependency is identical.
        lock = source_root / key / 'protocol.lock.json'
        protocol = json.loads(lock.read_text()) if lock.exists() else {}
        gate_dependencies = ['src/taimma/training/selective_fusion.py', 'src/taimma/experiments/study_gates.py',
                             'src/taimma/evaluation/checkpoint2_utility.py', 'src/taimma/evaluation/e0_e1_protocol.py']
        compatible = digest(protocol.get('config', {})) == run.protocol_hash and all(
            protocol.get('source_sha256', {}).get(rel) == hashlib.sha256((run.repo/rel).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
            for rel in gate_dependencies)
        manifest['gates'][key] = []
        if compatible:
            # Bound feature caches and S2 evidence are inputs to existing gates.
            for folder in ['selected_s2', 'no_sd_s2', 'per_seed_s2', 'gate_features']:
                src = source_root / key / folder
                for p in src.rglob('*') if src.exists() else []:
                    if not p.is_file() or 'add_one' in p.name or p.suffix == '.lock':continue
                    target = run.output / key / folder / p.relative_to(src)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if target.exists() and sha(target) != sha(p):raise ValueError('Evidence collision: '+str(target))
                    if not target.exists():shutil.copy2(p,target)
            for marker in (source_root/key/'gates').glob('*/completed.json'):
                rec=json.loads(marker.read_text()); mode=rec['mode']; seed=rec['seed']
                if mode == 'add_one':continue
                folder='no_sd_s2' if mode=='wo_source_dropout' else 'selected_s2'
                status=json.loads((source_root/key/'selected_s2/status.json').read_text());label=status['label_choice']
                report=json.loads((source_root/key/f'{folder}/e0_e1_{label}.json').read_text())
                job=f'early_CBT_{seed}' if mode=='wo_source_dropout' else f'early_sd_{seed}'
                expected=digest(dict(model=run.model_hash(key,job),config=run.protocol_hash,mode=mode,label=label,report=report))
                if rec['signature'] != expected:continue
                files=verify_files(marker.parent,rec['artifacts'])
                target=run.output/key/'gates'/marker.parent.name
                for rel,p in files.items():
                    dest=checked_path(target,rel);dest.parent.mkdir(parents=True,exist_ok=True)
                    if dest.exists() and sha(dest)!=sha(p):raise ValueError('Gate adoption collision')
                    if not dest.exists():shutil.copy2(p,dest)
                write_json(target/'completed.json',rec)
                write_json(target/'import_provenance.json',dict(source=str(marker.parent),original_completed_sha256=sha(marker)))
                manifest['gates'][key].append(marker.parent.name)
    write_json(run.output/'adoption_manifest.json',manifest)
    run.reuse_memo_source=source_root
    return manifest
