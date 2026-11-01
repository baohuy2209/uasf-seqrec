"""Validate explicit reuse of completed ID stages after metadata-only changes."""
import ast
import json
import csv
import shutil
import zipfile
from pathlib import Path, PureWindowsPath

from ..data.split import compute_file_sha256

ID_SOURCES = (
    'src/taimma/models/sasrec.py', 'src/taimma/training/trainer.py',
    'src/taimma/evaluation/evaluator.py', 'src/taimma/data/split.py',
    'src/taimma/data/ingestion.py',
)


def model_definition(source):
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Checkpoint')
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'model')
    return ast.dump(method, include_attributes=False)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def reuse_completed_id_stages(previous_root, destination_root, current_lock, current_source):
    previous, destination = Path(previous_root).resolve(), Path(destination_root).resolve()
    if previous == destination:
        raise ValueError('Reuse requires a different output directory; original evidence is preserved')
    old_lock = read(previous/'protocol.lock.json')
    for key in ('seeds','max_epochs','patience','learning_rate','batch_size','main_mask_policy',
                'loo_reference_mask_policy','device'):
        if old_lock[key] != current_lock[key]:
            raise ValueError(f'Cannot reuse ID stages: changed {key}')
    old_meta = old_lock['metadata_manifest_sha256']
    new_meta = current_lock['metadata_manifest_sha256']
    if isinstance(old_meta,dict):old_meta=old_meta['beauty']
    if isinstance(new_meta,dict):new_meta=new_meta['beauty']
    if old_meta != new_meta:
        raise ValueError('Cannot reuse ID stages: Beauty input manifest changed')
    old_code = {k.replace('\\','/'):v for k,v in old_lock['code_sha256'].items()}
    new_code = {k.replace('\\','/'):v for k,v in current_lock['code_sha256'].items()}
    for name in ID_SOURCES:
        if old_code.get(name) is None or old_code[name] != new_code.get(name):
            raise ValueError(f'Cannot reuse ID stages: changed ID algorithm {name}')
    with zipfile.ZipFile(previous/'source_snapshot.zip') as archive:
        old_source = archive.read('scripts/run_checkpoint1.py').decode('utf-8-sig')
    if model_definition(old_source) != model_definition(current_source):
        raise ValueError('Cannot reuse ID stages: changed ID model construction')
    progress=read(previous/'progress.json')
    reused={}
    lineage={}
    allowed={'loo_d64','loo_d256','id_seed42','id_seed43','id_seed44'}
    for key in allowed.intersection(progress):
        entry=progress[key]
        folder=Path(entry['folder'])
        if not folder.is_dir():
            basename=PureWindowsPath(entry['folder']).name if '\\' in entry['folder'] else folder.name
            folder=previous/'runs'/basename
        entry=dict(entry,folder=str(folder.resolve()),checkpoint=str((folder/'checkpoint.pt').resolve()))
        if read(folder/'run_state.json')['execution_status'] != 'COMPLETED':
            raise ValueError(f'Cannot reuse incomplete stage {key}')
        index=read(folder/'artifact_index.json')
        expected=next(row['sha256'] for row in index['artifacts'] if row['path']=='checkpoint.pt')
        actual=compute_file_sha256(entry['checkpoint'])
        if expected != actual:
            raise ValueError(f'Cannot reuse changed checkpoint {key}')
        config=read(folder/'config.json')
        if config['sources'] or config['mask_policy'] != ('none' if key.startswith('loo_') else 'input_history_strict'):
            raise ValueError(f'Cannot reuse mismatched ID protocol {key}')
        entry=dict(entry,checkpoint_sha256=actual)
        test_file=folder/'per_user_test.csv'
        expected_test=next(row['sha256'] for row in index['artifacts'] if row['path']=='per_user_test.csv')
        if compute_file_sha256(str(test_file)) != expected_test:
            raise ValueError(f'Cannot reuse changed per-user evidence {key}')
        with test_file.open(encoding='utf-8-sig',newline='') as stream:
            records=list(csv.DictReader(stream))
        if not records:raise ValueError(f'Empty per-user evidence {key}')
        columns={name.lower():name for name in records[0]}
        entry['test_metrics_exact']={field:sum(float(row[columns[column]]) for row in records)/len(records)
                    for field,column in [('recall_10','recall@10'),('ndcg_10','ndcg@10')]}
        reused[key]=entry
        lineage[key]={'checkpoint':entry['checkpoint'],'sha256':actual,'source_run':str(folder)}
    if not reused:
        raise ValueError('No eligible completed ID/LOO stages found')
    for key in ('condition_3','backbone_dim'):
        if key in progress:reused[key]=progress[key]
    audit_files=['validation_target_audit.csv','test_target_audit.csv','same_checkpoint_mask_sensitivity.csv']
    if 'audit' in progress and all((previous/name).is_file() for name in audit_files):
        reused['audit']=progress['audit']
        for name in audit_files:shutil.copy2(previous/name,destination/name)
    if (previous/'throughput.json').is_file():shutil.copy2(previous/'throughput.json',destination/'throughput.json')
    if (previous/'results.csv').is_file():shutil.copy2(previous/'results.csv',destination/'results_inherited.csv')
    result={'source_root':str(previous),'source_protocol_sha256':compute_file_sha256(str(previous/'protocol.lock.json')),
            'rationale':'Only completed ID/LOO stages reused; ID source and model construction are unchanged. Incomplete residual fits are excluded.',
            'stages':lineage}
    (destination/'reused_stages.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return reused
