"""Finish utility/E0/E1 from completed checkpoint1 runs without retraining."""
import argparse
import ast
import hashlib
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
import zipfile

from run_checkpoint1 import (Checkpoint, ROOT, SOURCES, Amazon2014Ingestor,
                             ExperimentLogger, SequenceSplitter, atomic_json,
                             compute_file_sha256, load_catalog, read_json, require, torch)

RECOVERY_SCRIPT = 'scripts/resume_checkpoint1_utility.py'


def without_utility(source):
    """Only the utility method may differ from the locked training driver."""
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Checkpoint')
    require(any(isinstance(node, ast.FunctionDef) and node.name == 'utility' for node in cls.body),
            'Locked checkpoint driver has no utility method')
    cls.body = [node for node in cls.body if not (isinstance(node, ast.FunctionDef) and node.name == 'utility')]
    return ast.dump(tree, include_attributes=False)


def verify_recovery_sources(root, project_root=ROOT):
    lock = read_json(root / 'protocol.lock.json')
    old = lock['code_sha256']
    driver = 'scripts/run_checkpoint1.py'
    with zipfile.ZipFile(root / 'source_snapshot.zip') as archive:
        original = archive.read(driver)
    require(hashlib.sha256(original).hexdigest() == old[driver], 'Locked driver snapshot checksum changed')
    require(without_utility(original.decode('utf-8-sig')) ==
            without_utility((project_root / driver).read_text(encoding='utf-8')),
            'Utility recovery requires unchanged training/evaluation driver code')
    current = {p.relative_to(project_root).as_posix(): compute_file_sha256(str(p))
               for folder in ['src', 'scripts', 'configs'] for p in (project_root / folder).rglob('*')
               if p.suffix in ('.py', '.yaml') and '__pycache__' not in p.parts}
    require(set(current) - set(old) <= {RECOVERY_SCRIPT}, 'Unexpected new experiment source during utility recovery')
    for name, sha in old.items():
        if name != driver:
            require(current.get(name) == sha, f'Utility recovery requires unchanged source: {name}')
    for key, sha in lock['metadata_manifest_sha256'].items():
        require(compute_file_sha256(str(project_root / 'artifacts/metadata' / key / 'manifest.json')) == sha,
                f'Utility recovery metadata changed: {key}')
    return lock, current


def recover(output, device='cuda', threads=2, utility_only=False):
    root = (ROOT / output).resolve()
    require(root.is_dir(), f'Missing checkpoint output: {root}')
    lock, signature = verify_recovery_sources(root)
    require(device == lock['device'], 'Use the original experiment device for utility recovery')
    if device == 'cuda':
        require(torch.cuda.is_available(), 'CUDA unavailable')
    torch.set_num_threads(threads)
    progress = read_json(root / 'progress.json')
    require(progress.get('diagnostics') == 'COMPLETED', 'Finish training and diagnostics before utility recovery')
    require(read_json(root / 'reproducibility.json')['status'] == 'PASS', 'Original reproducibility check must pass')
    for seed in lock['seeds'][:2]:
        for prefix in ['id_seed', 'late_CBT_seed']:
            key = f'{prefix}{seed}'
            entry = progress[key]
            folder = Path(entry['folder'])
            require(read_json(folder / 'run_state.json')['execution_status'] == 'COMPLETED', f'Incomplete reference: {key}')
            require(read_json(folder / 'run_manifest.json')['torch_version'] == torch.__version__,
                    f'Use the original PyTorch build for reference {key}')
            index = read_json(folder / 'artifact_index.json')
            expected = next(row['sha256'] for row in index['artifacts'] if row['path'] == 'checkpoint.pt')
            actual = compute_file_sha256(entry['checkpoint'])
            require(actual == expected and actual == entry.get('checkpoint_sha256'), f'Changed reference: {key}')
            config = read_json(folder / 'config.json')
            require(config['seed'] == seed and config['mask_policy'] == lock['main_mask_policy'] and
                    config['sources'] == ([] if prefix == 'id_seed' else list(SOURCES)), f'Changed reference config: {key}')
    runner = Checkpoint.__new__(Checkpoint)
    runner.root = root
    runner.device = torch.device(device)
    runner.args = SimpleNamespace(seeds=lock['seeds'], batch_size=lock['batch_size'])
    runner.progress = progress
    runner.data_mode = 'real'
    runner.backbone_dim = progress['backbone_dim']
    runner.ingestor = Amazon2014Ingestor(str(ROOT / 'data/raw'), 'Beauty')
    runner.sequences, _ = runner.ingestor.build_interaction_sequences(include_metadata=False)
    runner.split = SequenceSplitter().split(runner.sequences)
    runner.attrs, runner.masks = load_catalog(ROOT / 'artifacts/metadata/beauty')
    # Validate the input hashes and raw-to-model mapping using the existing audit.
    runner.audit()
    provenance = root / 'utility_recovery'
    provenance.mkdir(exist_ok=True)
    if not (provenance / 'source_snapshot.zip').exists():
        ExperimentLogger._snapshot_source(provenance)
    else:
        require(read_json(provenance / 'source_snapshot.json') == signature, 'Utility recovery source changed')
    atomic_json(provenance / 'lineage.json', dict(
        original_protocol_sha256=compute_file_sha256(str(root / 'protocol.lock.json')),
        code_sha256=signature, scope='Utility cohort selection and E0/E1 only; no training',
        reference_seeds=lock['seeds'][:2]))
    destination = runner.utility()
    if utility_only:
        return destination
    report = root / 'e0_e1.json'
    if not report.exists():
        subprocess.run([sys.executable, str(ROOT / 'scripts/run_e0_e1_evaluation.py'),
                        '--delta_file', str(destination), '--n_bootstrap', '1000', '--seed', '42',
                        '--output_json', str(report)], check=True, cwd=ROOT)
    runner.progress['checkpoint_experiments'] = 'COMPLETED'
    runner.save_progress()
    print(f'Checkpoint experiments completed: {report}', flush=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--utility-only', action='store_true', help='Save utility labels without running E0/E1')
    args = parser.parse_args()
    require(args.threads > 0, 'Threads must be positive')
    recover(args.output, args.device, args.threads, args.utility_only)


if __name__ == '__main__':
    main()
