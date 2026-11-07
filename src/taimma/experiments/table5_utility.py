"""Table 5 only: reuse two early-SD references, never fit a model."""
from pathlib import Path
import csv
import json
import shutil

from .study_completion import StudyCompletion
from .checkpoint2 import exclusive
from .checkpoint2_protocol import sha, write_json

JOBS = ('early_sd_42', 'early_sd_43')


def check_references(early_root, key='beauty'):
    """Require both original weights before creating any analysis output."""
    evidence = []
    for name in JOBS:
        job = Path(early_root).resolve() / key / 'jobs' / name
        marker = job / 'completed.json'
        if not marker.is_file():
            raise FileNotFoundError(f'Missing original reference: {marker}; training is forbidden')
        rec = json.loads(marker.read_text(encoding='utf-8'))
        weight = (job / rec['checkpoint']).resolve()
        if not weight.is_relative_to(job):
            raise ValueError(f'Checkpoint escapes source job: {weight}')
        if not weight.is_file():
            raise FileNotFoundError(f'Missing original weights: {weight}; training is forbidden')
        expected = rec['artifacts'].get(rec['checkpoint'])
        if not expected or sha(weight) != expected:
            raise ValueError(f'Original checkpoint hash mismatch: {weight}')
        if rec['spec']['kind'] != 'early' or rec['spec']['seed'] != int(name.rsplit('_', 1)[1]):
            raise ValueError(f'Unexpected reference model: {marker}')
        evidence.append(dict(job=name, checkpoint=str(weight), sha256=expected))
    return evidence


class Table5Utility(StudyCompletion):
    def guard(self, training=False, inference=False):
        if training or self.execute_training:
            raise RuntimeError('Table 5 is inference-only; model training is forbidden')
        return super().guard(training=False, inference=inference)

    def run_job(self, *args, **kwargs):
        raise RuntimeError('Table 5 cannot create or train early/late/ID models')


def write_table(root):
    root = Path(root)
    diagnostic = root / 'diagnostic_early_sd'
    alignment = json.loads((diagnostic / 'alignment.json').read_text(encoding='utf-8'))
    status = json.loads((diagnostic / 'status.json').read_text(encoding='utf-8'))
    columns = ['source', 'delta_ce_train_meta', 'delta_log_rank_train_meta', 'delta_ndcg_val_full_minus_without']
    rows = []
    for short, source in [('C', 'category'), ('B', 'brand'), ('T', 'title')]:
        rows.append([short, alignment['mean_delta_ce'][source], alignment['mean_delta_rank'][source],
                     alignment['validation_ndcg_full_minus_without'][source]])
    with (root / 'table5_utility.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, lineterminator='\n'); writer.writerow(columns); writer.writerows(rows)
    text = '| Source | Delta CE Train-meta | Delta log-rank Train-meta | Delta NDCG Val full-minus-without |\n'
    text += '|---|---:|---:|---:|\n'
    text += ''.join('| ' + r[0] + ' | ' + ' | '.join(f'{x:+.6f}' for x in r[1:]) + ' |\n' for r in rows)
    text += (f"\nLabel: {status['label_choice']}; E0/E1: {status['decision']}. "
             'Diagnostic early SD only; this does not qualify a backbone or authorize S3. '
             'Means pool the same two reference systems (seeds 42/43); validation is recomputed '
             'on those systems and may differ from a previous single-seed or three-seed table. '
             'No test evaluation or model training is performed.\n')
    (root / 'table5_utility.md').write_text(text, encoding='utf-8', newline='\n')
    return text


def complete_table5(run, early_root, publish_dir=None):
    if run.execute_training:
        raise RuntimeError('Table 5 forbids --execute-training')
    evidence = check_references(early_root)
    with exclusive(run.output / 'table5_execution.lock'):
        run.prepare('beauty')
        for name in JOBS:
            if not run.import_job('beauty', name, early_root):
                raise FileNotFoundError(f'Cannot import {name}; training is forbidden')
        run.complete_utility('beauty')
        root = run.output / 'beauty'
        write_json(root / 'table5_inputs.json', dict(references=evidence, training_executed=False,
                   scope='diagnostic early SD; Train-meta/validation only'))
        text = write_table(root)
        files = [root / n for n in ['table5_utility.csv', 'table5_utility.md', 'table5_inputs.json']]
        # Full utility/psi evidence stays local; publish diagnostics and their cache bindings.
        files += [p for p in (root / 'diagnostic_early_sd').iterdir() if p.suffix in ('.json', '.csv')]
        write_json(root / 'table5_manifest.json', {p.relative_to(root).as_posix(): sha(p) for p in files})
        files.append(root / 'table5_manifest.json')
        if publish_dir:
            dest = Path(publish_dir).resolve()
            for protected in [run.output.resolve(), Path(early_root).resolve()] + [Path(p).resolve() for p in run.config['input_roots'].values()]:
                if dest == protected or protected in dest.parents or dest in protected.parents:
                    raise ValueError('Publish directory must be separate from runtime and source runs')
            for p in files:
                target = dest / 'beauty' / p.relative_to(root)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, target)
        return text
