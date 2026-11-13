"""Publication checks use small synthetic summaries, never research measurements."""
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from taimma.paper_release import (BASELINES, METHODS, _original, _paired, _path,
                                 _validate_table, _check_cpu_tables, export_bundle, verify_bundle)


def encoded(value):
    return (json.dumps(value, sort_keys=True) + "\n").encode()


def table():
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=["method", "n_seeds", "val_ndcg@10_mean", "test_ndcg@10_mean"])
    writer.writeheader()
    for method in METHODS:
        writer.writerow(dict(method=method, n_seeds=5, **{"val_ndcg@10_mean": 0.2, "test_ndcg@10_mean": 0.1}))
    return stream.getvalue().encode()


@pytest.fixture
def bundle(tmp_path):
    files = {"run/finalization/table5_validation_test.csv": table()}
    frozen_files = {}
    for dataset in ("beauty", "sports", "toys"):
        hashes = {}
        for name in ("statistics.json", "coverage.json"):
            data = encoded({"fixture": True})
            files[f"metadata/{dataset}/{name}"] = data
            hashes[name] = hashlib.sha256(data).hexdigest()
        rel = f"{dataset}/protocol.lock.json"
        files["run/" + rel] = encoded({"data": {"metadata_files": hashes}})
        frozen_files[rel] = hashlib.sha256(files["run/" + rel]).hexdigest()
    freeze = encoded(dict(framing="F2", test_used_for_framing=False,
                          baselines=list(BASELINES), files=frozen_files))
    files["run/freeze.json"] = freeze
    freeze_hash = hashlib.sha256(freeze).hexdigest()
    files["run/finalization/release_complete.json"] = encoded(dict(
        status="COMPLETE", training=False, freeze_sha256=freeze_hash,
        methods=list(METHODS), outputs_sha256={"finalization/table5_validation_test.csv":
            hashlib.sha256(files["run/finalization/table5_validation_test.csv"]).hexdigest()}))
    files["run/finalization/alpha_complete.json"] = encoded(dict(
        posthoc=True, training=False, used_for_selection=False, split="validation",
        freeze_sha256=freeze_hash, outputs_sha256={}))
    for rel, data in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    manifest = dict(schema_version=1, version="fixture", files={
        rel: hashlib.sha256(data).hexdigest() for rel, data in files.items()})
    (tmp_path / "release_manifest.json").write_bytes(encoded(manifest))
    return tmp_path


def test_summary_verifies_without_torch_or_git(bundle):
    assert verify_bundle(bundle)["status"] == "PASS"
    code = "import sys; sys.path.insert(0, sys.argv[1]); from pathlib import Path; from taimma.paper_release import verify_bundle; verify_bundle(Path(sys.argv[2])); assert 'torch' not in sys.modules"
    result = subprocess.run([sys.executable, "-S", "-c", code, str(ROOT / "src"), str(bundle)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_tampering_is_rejected_even_if_top_manifest_is_updated(bundle):
    rel = "run/finalization/table5_validation_test.csv"
    p = bundle / rel
    p.write_bytes(p.read_bytes().replace(b"0.1", b"0.3"))
    m = json.loads((bundle / "release_manifest.json").read_text())
    m["files"][rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    (bundle / "release_manifest.json").write_bytes(encoded(m))
    with pytest.raises(ValueError, match="original scientific manifest"):
        verify_bundle(bundle)


def test_known_lf_restore_does_not_accept_a_changed_value(tmp_path):
    p = tmp_path / "file.csv"
    expected = hashlib.sha256(b"value\n1\n").hexdigest()
    p.write_bytes(b"value\r\n1\r\n")
    assert _original(p, expected) == b"value\n1\n"
    p.write_bytes(b"value\r\n2\r\n")
    with pytest.raises(ValueError, match="differs"):
        _original(p, expected)


@pytest.mark.parametrize("relative", ["../outside", "/outside", "C:/outside", "a\\..\\outside"])
def test_manifest_cannot_escape_destination(tmp_path, relative):
    with pytest.raises(ValueError):
        _path(tmp_path, relative)


def test_full_family_and_seed_count_are_required():
    _validate_table(table())
    with pytest.raises(ValueError, match="five seeds"):
        _validate_table(table().replace(b",5,", b",3,"))


def test_pairing_checks_targets_and_preserves_constant_effect():
    a = [{0: (1, 0.4), 1: (2, 0.3)}] * 3
    b = [{0: (1, 0.2), 1: (2, 0.1)}] * 3
    assert _paired(a, b) == pytest.approx((0.2, 0.2, 0.2))
    a[1] = {0: (9, 0.4), 1: (2, 0.3)}
    with pytest.raises(ValueError, match="target pairing"):
        _paired(a, b)


def test_export_refuses_existing_output_before_reading_inputs(tmp_path):
    destination = tmp_path / "published"
    destination.mkdir()
    with pytest.raises(FileExistsError):
        export_bundle(tmp_path, tmp_path / "run", tmp_path / "early", tmp_path / "metadata", destination)


def test_cpu_additions_must_be_bound_to_frozen_inputs(tmp_path):
    files = {"run/finalization/cpu_analysis.json": encoded(dict(
        training=False, inference=False, inputs_sha256={"unfrozen.csv": "0" * 64}))}
    with pytest.raises(ValueError, match="not bound"):
        _check_cpu_tables(tmp_path, {"files": {}}, files, {})


def test_logger_discloses_missing_git_but_keeps_source_snapshot(tmp_path, monkeypatch):
    from taimma.experiments.logger import ExperimentLogger
    import taimma.experiments.logger as logger

    def without_git(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 128, b"", b"fatal: not a git repository")

    monkeypatch.setattr(logger.subprocess, "run", without_git)
    ExperimentLogger._snapshot_source(tmp_path)
    note = json.loads((tmp_path / "git_snapshot.json").read_text())
    assert note == {"available": False, "reason": "source_archive"}
    assert (tmp_path / "source_snapshot.zip").is_file()
    assert json.loads((tmp_path / "source_snapshot.json").read_text())
