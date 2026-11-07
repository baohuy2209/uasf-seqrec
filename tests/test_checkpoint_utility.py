"""Utility cohort selection, paired references, and recovery source safeguards."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from taimma.data.ingestion import InteractionSequence
from taimma.data.split import SequenceSplitter, compute_file_sha256
from taimma.models.sasrec import SASRecModel
from taimma.models.late_fusion import SourceSequentialEncoder, ResidualLateFusionModel
from resume_checkpoint1_utility import verify_recovery_sources


@pytest.fixture
def utility_runner(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('utility_driver_test', ROOT / 'scripts/run_checkpoint1.py')
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    monkeypatch.setattr(driver, 'ROOT', tmp_path)
    runner = driver.Checkpoint.__new__(driver.Checkpoint)
    runner.root = tmp_path / 'run'
    runner.root.mkdir()
    runner.device = torch.device('cpu')
    runner.args = SimpleNamespace(seeds=[42, 43, 44], batch_size=1)
    runner.split = SequenceSplitter(max_seq_len=5).split([
        InteractionSequence('U0', list('ABXYZ'), list(range(5))),
        InteractionSequence('U1', list('BCAXY'), list(range(5))),
        InteractionSequence('U2', list('CABYZ'), list(range(5))),
    ])
    runner.attrs = {'category': torch.tensor([[1], [2], [1]]),
                    'brand': torch.tensor([1, 2, 3]), 'title': torch.eye(3)}
    runner.masks = {source: torch.ones(3) for source in runner.attrs}
    runner.progress = {}
    models = {}
    for seed in [42, 43]:
        checkpoint = tmp_path / f'reference{seed}.pt'
        model = ResidualLateFusionModel(
            SASRecModel(3, max_seq_len=5, embedding_dim=8, num_heads=2),
            {source: SourceSequentialEncoder(source, 3, feature_dim=8, max_seq_len=5,
                num_heads=2, is_dense_feature=source == 'title') for source in runner.attrs},
            sources_enabled=tuple(runner.attrs))
        torch.save(model.state_dict(), checkpoint)
        runner.progress[f'late_CBT_seed{seed}'] = {'checkpoint': str(checkpoint), 'seed': seed}
        models[seed] = model
    runner.load_residual = lambda entry: models[entry['seed']]
    manifest = tmp_path / 'artifacts/metadata/beauty/manifest.json'
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{}')
    return runner, models


def test_utility_excludes_oov_with_identical_paired_cohorts_and_resumes(utility_runner):
    runner, models = utility_runner
    original_split = copy.deepcopy(runner.split)
    original_states = {seed: copy.deepcopy(model.state_dict()) for seed, model in models.items()}
    destination = runner.utility()
    frame = pd.read_csv(destination)
    assert len(frame) == 2 * 2 * 3
    assert set(frame.user_idx) == {1, 2}
    assert (frame.target_idx > 0).all()
    for source, group in frame.groupby('source_name'):
        first = group[group.reference_seed == 42].sort_values('context_id')
        repeat = group[group.reference_seed == 43].sort_values('context_id')
        assert first.context_id.tolist() == repeat.context_id.tolist()
        assert first.user_idx.tolist() == repeat.user_idx.tolist()
        assert len(first) == len(repeat) == 2
    audit = pd.read_csv(runner.root / 'utility_cohort.csv', keep_default_na=False)
    assert audit.included.tolist() == [False, True, True]
    assert audit.exclusion_reason.tolist() == ['target_oov', '', '']
    assert audit.iloc[0].raw_target_asin == 'X'
    summary = json.loads((runner.root / 'utility_cohort.json').read_text())
    assert summary['total_contexts'] == 3
    assert summary['included_contexts'] == 2
    assert summary['excluded_oov_contexts'] == 1
    assert runner.split == original_split
    for seed, model in models.items():
        assert all(torch.equal(value, model.state_dict()[name]) for name, value in original_states[seed].items())
        assert all(param.grad is None for param in model.parameters())
    digest = compute_file_sha256(str(destination))
    runner.load_residual = lambda _: pytest.fail('Completed utility must reuse saved labels')
    assert runner.utility() == destination
    assert compute_file_sha256(str(destination)) == digest


def test_utility_rejects_corrupt_ids_instead_of_silently_dropping_them(utility_runner):
    runner, _ = utility_runner
    runner.split.train_meta_samples[0].target = runner.split.num_items + 1
    with pytest.raises(ValueError, match='Invalid Train-meta target ID'):
        runner.utility()


def test_all_oov_contexts_stop_with_audit_and_without_inference(utility_runner):
    runner, _ = utility_runner
    for sample in runner.split.train_meta_samples:
        sample.target = 0
    runner.load_residual = lambda _: pytest.fail('No model inference needed for an empty eligible cohort')
    with pytest.raises(ValueError, match='No in-catalog Train-meta targets'):
        runner.utility()
    assert json.loads((runner.root / 'utility_cohort.json').read_text())['excluded_oov_contexts'] == 3
    assert not (runner.root / 'utility.csv').exists()


def test_interrupted_utility_promotion_preserves_cohort_lineage(utility_runner):
    runner, _ = utility_runner
    destination = runner.utility()
    sidecar = destination.with_suffix('.csv.manifest.json')
    sidecar.rename(runner.root / 'utility.partial.csv.manifest.json')
    assert runner.utility() == destination
    assert sidecar.exists()


@pytest.fixture
def recovery_sources(tmp_path):
    project = tmp_path / 'project'
    run = tmp_path / 'run'
    run.mkdir()
    source = ('class Checkpoint:\n'
              '    def model(self): return 1\n'
              '    def utility(self): return 0\n')
    files = {'scripts/run_checkpoint1.py': source, 'src/model.py': 'MODEL = 1\n'}
    for name, content in files.items():
        path = project / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, newline='\n')
    manifest = project / 'artifacts/metadata/beauty/manifest.json'
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{}')
    lock = {'code_sha256': {name: hashlib.sha256(value.encode()).hexdigest() for name, value in files.items()},
            'metadata_manifest_sha256': {'beauty': compute_file_sha256(str(manifest))}}
    (run / 'protocol.lock.json').write_text(json.dumps(lock))
    with zipfile.ZipFile(run / 'source_snapshot.zip', 'w') as archive:
        archive.writestr('scripts/run_checkpoint1.py', source)
    (project / 'scripts/run_checkpoint1.py').write_text(source.replace('return 0', 'return 2'), newline='\n')
    (project / 'scripts/resume_checkpoint1_utility.py').write_text('# recovery entry point\n')
    return run, project


def test_recovery_allows_only_utility_driver_changes(recovery_sources):
    run, project = recovery_sources
    lock, current = verify_recovery_sources(run, project)
    assert lock['code_sha256']['scripts/run_checkpoint1.py'] != current['scripts/run_checkpoint1.py']
    assert (run / 'protocol.lock.json').exists()


@pytest.mark.parametrize('change', ['model', 'driver', 'new_source', 'metadata', 'snapshot'])
def test_recovery_rejects_changes_outside_utility(recovery_sources, change):
    run, project = recovery_sources
    if change == 'model':
        (project / 'src/model.py').write_text('MODEL = 2\n')
    elif change == 'driver':
        path = project / 'scripts/run_checkpoint1.py'
        path.write_text(path.read_text().replace('return 1', 'return 2'))
    elif change == 'new_source':
        (project / 'src/new.py').write_text('MODEL = 2\n')
    elif change == 'metadata':
        (project / 'artifacts/metadata/beauty/manifest.json').write_text('{"changed": true}')
    else:
        with zipfile.ZipFile(run / 'source_snapshot.zip', 'w') as archive:
            archive.writestr('scripts/run_checkpoint1.py', 'class Checkpoint: pass\n')
    with pytest.raises(ValueError):
        verify_recovery_sources(run, project)
