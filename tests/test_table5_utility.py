"""Inference-only orchestration tests; no research model is executed."""
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from taimma.experiments.table5_utility import Table5Utility, check_references, complete_table5
from taimma.experiments.checkpoint2_protocol import sha, write_json


def originals(tmp_path):
    early = tmp_path / 'early'
    for seed in [42, 43]:
        job = early / 'beauty/jobs' / f'early_sd_{seed}'
        weight = job / 'attempt_001/checkpoint.pt'
        weight.parent.mkdir(parents=True)
        weight.write_bytes(f'synthetic {seed}'.encode())
        write_json(job / 'completed.json', dict(checkpoint='attempt_001/checkpoint.pt',
                   artifacts={'attempt_001/checkpoint.pt': sha(weight)}, spec=dict(kind='early', seed=seed)))
    return early


def test_requires_both_original_weights_and_rejects_changed_hash(tmp_path):
    early = originals(tmp_path)
    assert len(check_references(early)) == 2
    weight = early / 'beauty/jobs/early_sd_43/attempt_001/checkpoint.pt'
    weight.write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash mismatch'):
        check_references(early)
    weight.unlink()
    with pytest.raises(FileNotFoundError, match='training is forbidden'):
        check_references(early)


def test_training_is_unconditionally_blocked(tmp_path):
    run = Table5Utility(tmp_path, {}, 'runs/completion_utility', device='cpu', execute_inference=True)
    with pytest.raises(RuntimeError, match='training is forbidden'):
        run.guard(training=True)
    with pytest.raises(RuntimeError, match='cannot create or train'):
        run.run_job('beauty', 'late_C_42', None)
    run.execute_training = True
    with pytest.raises(RuntimeError, match='forbids'):
        complete_table5(run, tmp_path / 'missing')


def test_table5_only_imports_two_references_and_publishes_no_weights(tmp_path):
    early = originals(tmp_path)
    output = tmp_path / 'runtime'
    calls = []

    def prepare(key):
        calls.append(('prepare', key)); (output / key).mkdir(parents=True)

    def adopt(key, name, source):
        calls.append(('import', name)); return True

    def utility(key):
        calls.append(('utility', key))
        diagnostic = output / key / 'diagnostic_early_sd'
        values = dict(category=.01, brand=.02, title=.03)
        write_json(diagnostic / 'alignment.json', dict(mean_delta_ce=values, mean_delta_rank=values,
                   validation_ndcg_full_minus_without=values))
        write_json(diagnostic / 'status.json', dict(label_choice='CE', decision='NO_GO'))
        (diagnostic / 'checkpoint.pt').write_bytes(b'never publish this')

    run = SimpleNamespace(output=output, execute_training=False, prepare=prepare,
                          import_job=adopt, complete_utility=utility, config={'input_roots': {'early': str(early)}})
    dest = tmp_path / 'published'
    text = complete_table5(run, early, dest)
    assert calls == [('prepare', 'beauty'), ('import', 'early_sd_42'), ('import', 'early_sd_43'), ('utility', 'beauty')]
    assert 'NO_GO' in text
    assert (dest / 'beauty/table5_utility.csv').is_file()
    assert not list(dest.rglob('*.pt'))
    manifest = json.loads((dest / 'beauty/table5_manifest.json').read_text())
    assert all(sha(dest / 'beauty' / rel) == expected for rel, expected in manifest.items())
    assert not (output / 'table5_execution.lock').exists()


def test_missing_reference_stops_before_any_prepare(tmp_path):
    def unexpected(*args):
        pytest.fail('Must not prepare or train when original checkpoints are missing')
    run = SimpleNamespace(output=tmp_path / 'runtime', execute_training=False, prepare=unexpected)
    with pytest.raises(FileNotFoundError):
        complete_table5(run, tmp_path / 'missing')
    assert not run.output.exists()
