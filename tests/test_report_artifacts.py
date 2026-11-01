"""Report-artifact contracts; all numbers here are explicitly test fixtures."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from taimma.data.ingestion import InteractionSequence
from taimma.data.split import SequenceSplitter
from taimma.evaluation.evaluator import IndependentEvaluator
from taimma.evaluation.utility import UtilityRecord, save_utility_artifact
from taimma.experiments.logger import ExperimentLogger
from taimma.experiments.reporting import dataset_statistics, context_table, runtime_snapshot, validate_rank_format


def fixture_split():
    sequences = [InteractionSequence(str(i), list('ABCDEF'), list(range(6))) for i in range(10)]
    return sequences, SequenceSplitter(max_seq_len=5).split(sequences)


def test_missing_metadata_is_unknown_in_table1_and_per_user():
    sequences, split = fixture_split()
    stats = dataset_statistics(sequences, split, {}, False, 'Fixture')
    assert stats['interactions'] == 60 and stats['mean_sequence_length'] == 6
    assert stats['metadata_coverage_original_items']['brand'] is None
    assert stats['category_vocab_train_base'] is None
    ctx = context_table(split.test_samples, split, {}, False)
    assert ctx['target_metadata_has_title'].isna().all()
    assert len(set(ctx.analysis_length_quintile)) == 5
    assert ctx['target_is_oov'].all()


def test_available_metadata_coverage_and_train_base_vocab():
    sequences, split = fixture_split()
    meta = {'A': {'categories': ['x', 'y'], 'brand': 'one', 'title': 'text'},
            'F': {'categories': ['heldout-only'], 'brand': 'heldout-only'}}
    stats = dataset_statistics(sequences, split, meta, True, 'Fixture')
    assert stats['metadata_coverage_original_items']['category'] == pytest.approx(2/6)
    assert stats['metadata_coverage_train_base_catalog']['category'] == pytest.approx(1/3)
    assert stats['category_vocab_train_base'] == 2
    assert stats['brand_vocab_train_base'] == 1


def test_cpu_resource_measurements_never_fabricate_gpu_memory():
    result = runtime_snapshot('cpu')
    assert result['gpu_peak_allocated_bytes'] is None
    assert result['gpu_status'] == 'not_applicable_cpu'
    assert result['ram_status']


def test_gpu_measurement_schema_with_explicit_mock(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda device: None)
    monkeypatch.setattr(torch.cuda, 'get_device_properties', lambda device: SimpleNamespace(name='FIXTURE_GPU', total_memory=10000))
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda device: 123)
    monkeypatch.setattr(torch.cuda, 'max_memory_reserved', lambda device: 456)
    result = runtime_snapshot('cuda:0')
    assert result['gpu_peak_allocated_bytes'] == 123
    assert result['gpu_peak_reserved_bytes'] == 456
    assert result['gpu_name'] == 'FIXTURE_GPU'


def test_registry_migration_preserves_history_and_backs_up_bytes(tmp_path):
    registry = tmp_path / 'results.csv'
    old = b'run_id,status,test_ndcg_10\r\nhistorical,BLOCKED,0.125\r\n'
    registry.write_bytes(old)
    run = tmp_path / 'new_run'
    run.mkdir()
    (run / 'config.json').write_text(json.dumps({'mechanism': 'id_only', 'sources': [], 'control': 'none', 'missing_rate': 0}))
    result = IndependentEvaluator(3).evaluate_ranks(np.array([[.1, .2, .3]]), [3], [0])
    logger = ExperimentLogger(str(tmp_path), str(registry))
    logger.register_result('new', 'Fixture', 'SASRec_ID', 42, 1, 1, result, result, .5, str(run / 'checkpoint.pt'), 'SMOKE_TEST')
    with registry.open(newline='') as f:
        rows = list(csv.DictReader(f))
    assert rows[0]['run_id'] == 'historical' and rows[0]['status'] == 'BLOCKED'
    assert rows[0]['test_ndcg_10'] == '0.125'
    assert rows[0]['git_commit'] == '' and rows[0]['gpu_peak_allocated_bytes'] == ''
    assert rows[1]['mechanism'] == 'id_only' and rows[1]['registry_schema_version'] == '2'
    assert next(tmp_path.glob('results.csv.schema-before.*.bak')).read_bytes() == old
    before = registry.read_bytes()
    with pytest.raises(ValueError, match='already registered'):
        logger.register_result('new', 'Fixture', 'SASRec_ID', 42, 1, 1, result, result, .5, str(run / 'checkpoint.pt'))
    assert registry.read_bytes() == before


def test_failed_execution_keeps_log_state_and_checksum_index(tmp_path):
    logger = ExperimentLogger(str(tmp_path), str(tmp_path / 'results.csv'))
    _, directory = logger.create_run_dir('fixture')
    with pytest.raises(RuntimeError, match='deliberate fixture failure'):
        with logger.capture_output(directory):
            print('started fixture operation')
            raise RuntimeError('deliberate fixture failure')
    run = Path(directory)
    assert json.loads((run / 'run_state.json').read_text())['execution_status'] == 'FAILED'
    assert 'deliberate fixture failure' in (run / 'run.log').read_text(encoding='utf-8')
    index = json.loads((run / 'artifact_index.json').read_text())
    assert any(row['path'] == 'source_snapshot.zip' for row in index['artifacts'])
    for row in index['artifacts']:
        assert hashlib.sha256((run / row['path']).read_bytes()).hexdigest() == row['sha256']


def test_parquet_request_fails_explicitly_without_engine(monkeypatch):
    monkeypatch.setattr(importlib.util, 'find_spec', lambda name: None)
    validate_rank_format('csv')
    with pytest.raises(ImportError, match='Parquet requested'):
        validate_rank_format('both')


def test_utility_records_are_saved_with_physical_lineage(tmp_path):
    manifest = tmp_path / 'dataset_manifest.json'
    manifest.write_text('{"fixture": true}')
    checkpoint = tmp_path / 'reference.pt'
    checkpoint.write_bytes(b'explicit test fixture, not a research checkpoint')
    record = UtilityRecord('context', 0, 1, 'category', 7, .25, 1.25, 1.5, True, 3,
                           {'feat_log_len': float(np.log(3)), 'feat_coverage': 1.})
    path, sidecar = save_utility_artifact(tmp_path / 'utility.csv', [record], manifest, {7: str(checkpoint)})
    metadata = json.loads(Path(sidecar).read_text())
    assert metadata['file_sha256'] == hashlib.sha256(Path(path).read_bytes()).hexdigest()
    assert metadata['reference_checkpoints']['7']['sha256'] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert pd.read_csv(path).iloc[0]['delta'] == .25
    with pytest.raises(FileExistsError):
        save_utility_artifact(path, [record], manifest, {7: str(checkpoint)})


def test_table1_collection_from_explicit_three_dataset_fixtures(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('table1_fixture', ROOT / 'scripts/collect_dataset_statistics.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    raw = tmp_path / 'raw'
    raw.mkdir()
    for key in ('Beauty', 'Sports_and_Outdoors', 'Toys_and_Games'):
        rows = [dict(reviewerID='fixture-user', asin=str(i), unixReviewTime=i, overall=5) for i in range(6)]
        (raw / (key + '_5.json')).write_text('\n'.join(json.dumps(row) for row in rows), encoding='utf-8')
    out = tmp_path / 'table1'
    monkeypatch.setattr(sys, 'argv', ['collect_dataset_statistics.py', '--raw_root', str(raw), '--output_dir', str(out)])
    module.main()
    table = pd.read_csv(out / 'table1.csv')
    assert len(table) == 3 and table['coverage_brand'].isna().all()
    assert not table['reference_users_items_match'].any()
    assert json.loads((out / 'manifest.json').read_text())['training_performed'] is False
