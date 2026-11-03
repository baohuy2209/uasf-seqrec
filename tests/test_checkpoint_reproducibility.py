"""CUDA repeatability and strict checkpoint reproducibility regression tests."""
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from taimma.data.split import SequenceSample
from taimma.experiments.logger import ExperimentLogger
from taimma.training.trainer import set_seed


@pytest.fixture
def driver():
    spec = importlib.util.spec_from_file_location('checkpoint_repro_test', ROOT / 'scripts/run_checkpoint1.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    yield module
    torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    torch.set_num_threads(threads)


def test_seeding_requires_deterministic_kernels(driver):
    torch.use_deterministic_algorithms(False)
    set_seed(42)
    assert torch.are_deterministic_algorithms_enabled()
    assert not torch.is_deterministic_algorithms_warn_only_enabled()
    assert os.environ['CUBLAS_WORKSPACE_CONFIG'] in (':4096:8', ':16:8')


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_full_training_repeats_exactly_and_resumes(driver, tmp_path, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    runner = driver.Checkpoint.__new__(driver.Checkpoint)
    runner.args = SimpleNamespace(batch_size=64, epochs=3, patience=2, train_job=None)
    runner.device = torch.device(device)
    runner.root = tmp_path
    runner.data_mode = 'synthetic_fixture'
    runner.progress = {}
    runner.backbone_dim = 8
    generator = torch.Generator().manual_seed(11)
    histories = torch.randint(1, 129, (128, 50), generator=generator).tolist()
    # 64 * 50 tokens per batch exercise CUDA embedding-gradient reductions.
    train = [SequenceSample(i, row, i % 128 + 1, 50) for i, row in enumerate(histories)]
    val = [SequenceSample(i, row, (i + 7) % 128 + 1, 50) for i, row in enumerate(histories[:8])]
    test = [SequenceSample(i, row, (i + 13) % 128 + 1, 50) for i, row in enumerate(histories[:8])]
    runner.split = SimpleNamespace(num_items=128, train_base_samples=train,
                                   val_samples=val, test_samples=test, seen_train_items_per_user={})
    runner.logger = ExperimentLogger(str(tmp_path / 'runs'), str(tmp_path / 'results.csv'))
    runner.reproducibility()
    report = json.loads((tmp_path / 'reproducibility.json').read_text())
    assert report['status'] == 'PASS'
    assert report['deterministic_algorithms'] is True
    assert all(value == 'PASS' for value in report['comparisons'].values())
    entries = runner.progress.copy()
    original_history, repeated_history = [json.loads((Path(entries[key]['folder']) / 'history.json').read_text())
        for key in ['id_seed42', 'id_seed42_repeat']]
    for name in ['train_losses', 'val_metrics_by_epoch', 'best_epoch']:
        assert original_history[name] == repeated_history[name]
    for entry in entries.values():
        config = json.loads((Path(entry['folder']) / 'config.json').read_text())
        assert config['deterministic_algorithms'] is True
        assert config['cublas_workspace_config'] == os.environ['CUBLAS_WORKSPACE_CONFIG']
    torch.use_deterministic_algorithms(False)
    runner.reproducibility()
    assert runner.progress == entries
    assert len(list((tmp_path / 'runs').iterdir())) == 2
    assert json.loads((tmp_path / 'reproducibility.json').read_text())['deterministic_algorithms'] is True


@pytest.mark.parametrize('mismatch', ['per_user_val.csv', 'per_user_test.csv', 'weights', 'epoch'])
def test_mismatches_are_recorded_and_still_fail(driver, tmp_path, mismatch):
    runner = driver.Checkpoint.__new__(driver.Checkpoint)
    runner.root = tmp_path
    runner.split = None
    runner.backbone_dim = 8
    entries = {}
    for key in ['id_seed42', 'id_seed42_repeat']:
        folder = tmp_path / key
        folder.mkdir()
        repeated = key.endswith('_repeat')
        for name in ['per_user_val.csv', 'per_user_test.csv']:
            pd.DataFrame({'user_idx': [0], 'target_idx': [1],
                          'rank': [2 if repeated and mismatch == name else 1]}).to_csv(folder / name, index=False)
        torch.save({'epoch': 2 if repeated and mismatch == 'epoch' else 1,
                    'model_state_dict': {'weight': torch.tensor([2. if repeated and mismatch == 'weights' else 1.])}},
                   folder / 'checkpoint.pt')
        entries[key] = {'folder': str(folder), 'checkpoint': str(folder / 'checkpoint.pt')}
    runner.train = lambda key, *args: entries[key]
    (tmp_path / 'reproducibility.json').write_text('{"status": "PASS"}')
    with pytest.raises(ValueError, match='Reproducibility failed; details saved'):
        runner.reproducibility()
    report = json.loads((tmp_path / 'reproducibility.json').read_text())
    assert report['status'] == 'FAIL'
    comparison = {'weights': 'model_weights', 'epoch': 'best_epoch'}.get(mismatch, mismatch)
    assert report['comparisons'][comparison] != 'PASS'
