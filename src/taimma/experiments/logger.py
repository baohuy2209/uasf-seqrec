"""
logger.py - Experiment artifact management, run manifest serialization, and results registry.

Requirements:
  - Strict Data-First: records actual on-disk artifact paths and physical execution timings.
  - Generates ISO timestamped run directories under runs/<run_id>/.
  - Safe, concurrent-friendly registry writing to results/results.csv.
"""

import csv
import time
import sys
import traceback
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import datetime
import json
import os
import platform
import subprocess
import uuid
import importlib.metadata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import torch
from ..evaluation.evaluator import EvaluationResult, UserRankRecord
from .reporting import write_json, write_rank_table, write_artifact_index, validate_rank_format


class ExperimentLogger:
    """Manages run directories, execution manifests, and results.csv registry."""

    def __init__(self, runs_root: str = "runs", results_csv: str = "results/results.csv"):
        self.runs_root = runs_root
        self.results_csv = results_csv

    @staticmethod
    def _get_git_info() -> Dict[str, Any]:
        try:
            repo = next(p for p in Path(__file__).resolve().parents if (p / ".git").exists())
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, stderr=subprocess.DEVNULL).decode("ascii").strip()
            status_out = subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, stderr=subprocess.DEVNULL).decode("utf-8").strip()
            return {"commit": commit, "dirty": len(status_out) > 0}
        except Exception:
            return {"commit": "unknown", "dirty": None}

    def create_run_dir(self, experiment_name: str) -> Tuple[str, str]:
        """Create ISO timestamped run directory."""
        iso_now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_id = f"{experiment_name}_{iso_now}_{uuid.uuid4().hex[:8]}"
        run_path = os.path.join(self.runs_root, run_id)
        os.makedirs(run_path, exist_ok=False)
        return run_id, run_path

    def save_run_artifacts(
        self,
        run_path: str,
        run_id: str,
        config: Dict[str, Any],
        val_result: EvaluationResult,
        test_result: Optional[EvaluationResult],
        train_time: float,
        best_epoch: int,
        total_epochs: int,
        actual_device: Optional[str] = None,
        context_tables: Optional[Dict[str, pd.DataFrame]] = None,
        rank_format: str = 'csv',
    ) -> Dict[str, Any]:
        """Save config, manifest, metrics, and per-user evaluation records."""
        validate_rank_format(rank_format)
        context_tables = context_tables or {}
        config = dict(config)
        if "device" in config:
            config["device"] = str(config["device"])
        for name in ("config.json", "metrics.json", "run_manifest.json", "per_user.csv", "per_user_val.csv", "per_user_test.csv"):
            if os.path.exists(os.path.join(run_path, name)):
                raise FileExistsError(f"Refusing to overwrite run artifact: {name}")
        # 1. Save config
        config_file = os.path.join(run_path, "config.json")
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        # 2. Save metrics
        metrics = {
            "run_id": run_id,
            "best_epoch": best_epoch,
            "total_epochs": total_epochs,
            "train_time_sec": round(train_time, 2),
            "validation": val_result.to_dict(),
            "test": test_result.to_dict() if test_result else None,
        }
        metrics_file = os.path.join(run_path, "metrics.json")
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        # 3. Save run manifest
        device_used = actual_device or config.get("device") or "unknown"
        git_info = self._get_git_info()
        manifest = {
            "run_id": run_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "git_commit": git_info["commit"],
            "git_dirty": git_info["dirty"],
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": str(device_used),
            "library_versions": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn", "pandas")},
            "metrics": metrics,
            "rank_format": rank_format,
            "source_snapshot_path": "source_snapshot.zip" if (Path(run_path) / 'source_snapshot.zip').exists() else None,
        }
        manifest_file = os.path.join(run_path, "run_manifest.json")
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        if val_result and val_result.per_user_records:
            write_rank_table(run_path, 'per_user_val', val_result.per_user_records,
                             context_tables.get('validation'), rank_format)
        if test_result and test_result.per_user_records:
            write_rank_table(run_path, 'per_user_test', test_result.per_user_records,
                             context_tables.get('test'), rank_format)
            write_rank_table(run_path, 'per_user', test_result.per_user_records,
                             context_tables.get('test'), rank_format)
        elif val_result and val_result.per_user_records:
            write_rank_table(run_path, 'per_user', val_result.per_user_records,
                             context_tables.get('validation'), rank_format)

        return manifest

    @contextmanager
    def capture_output(self, run_path):
        """Capture stdout/stderr and distinguish execution from scientific status."""
        path = Path(run_path)
        start = time.perf_counter()
        state = {'execution_status': 'RUNNING', 'started_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
        state_path = path / 'run_state.json'
        write_json(state_path, state)
        self._snapshot_source(path)
        class Tee:
            def __init__(self, original, log):
                self.original, self.log = original, log
            def write(self, value):
                self.original.write(value)
                self.log.write(value)
                return len(value)
            def flush(self):
                self.original.flush()
                self.log.flush()
        with (path / 'run.log').open('x', encoding='utf-8') as log:
            with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
                try:
                    yield
                except BaseException as exc:
                    state.update(execution_status='FAILED', exception_type=type(exc).__name__, error=str(exc))
                    traceback.print_exc()
                    raise
                else:
                    state['execution_status'] = 'COMPLETED'
                finally:
                    state['ended_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
                    state['wall_sec'] = time.perf_counter() - start
                    with state_path.open('w', encoding='utf-8') as f:
                        json.dump(state, f, indent=2, allow_nan=False)
                    log.flush()
                    write_artifact_index(path)

    @staticmethod
    def _snapshot_source(folder):
        import zipfile
        import hashlib
        root = Path(__file__).resolve().parents[3]
        files = sorted([p for name in ('src', 'scripts', 'configs') for p in (root / name).rglob('*')
                        if p.is_file() and p.suffix in ('.py', '.yaml') and '__pycache__' not in p.parts])
        hashes = {}
        with zipfile.ZipFile(folder / 'source_snapshot.zip', 'x', compression=zipfile.ZIP_DEFLATED) as z:
            for file in files:
                data = file.read_bytes()
                relative = file.relative_to(root).as_posix()
                z.writestr(relative, data)
                hashes[relative] = hashlib.sha256(data).hexdigest()
        write_json(folder / 'source_snapshot.json', hashes)
        decisions = root / 'docs/decision_log.md'
        if decisions.is_file():
            with (folder / 'decision_log_snapshot.md').open('xb') as out:
                out.write(decisions.read_bytes())


    def register_result(
        self, run_id: str, dataset: str, model: str, seed: int, epochs: int,
        best_epoch: int, val_result: EvaluationResult, test_result: Optional[EvaluationResult],
        train_time: float, checkpoint_path: str, status: str = 'COMPLETED',
    ):
        """Atomically append the report schema, preserving legacy values and backup.

        Unknown historical fields remain empty. No historical status, timing or
        metric is reconstructed. A sibling lock rejects concurrent writers.
        """
        import tempfile
        import shutil
        path = Path(self.results_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        run_path = Path(checkpoint_path).parent.resolve()
        def read_optional(name):
            file = run_path / name
            return json.loads(file.read_text(encoding='utf-8')) if file.is_file() else {}
        config, manifest, report = (read_optional(name) for name in ('config.json', 'run_manifest.json', 'report_data.json'))
        resources = report.get('costs', {}).get('training_resources', {})
        costs = report.get('costs', {})
        def artifact(stem):
            return str(run_path / stem) if (run_path / stem).is_file() else None
        row = dict(run_id=run_id, timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   dataset=dataset, model=model, seed=seed, epochs=epochs, best_epoch=best_epoch,
                   val_ndcg_10=val_result.ndcg_10, val_recall_10=val_result.recall_10,
                   val_ndcg_20=val_result.ndcg_20, val_recall_20=val_result.recall_20,
                   test_ndcg_10=test_result.ndcg_10 if test_result else None,
                   test_recall_10=test_result.recall_10 if test_result else None,
                   test_ndcg_20=test_result.ndcg_20 if test_result else None,
                   test_recall_20=test_result.recall_20 if test_result else None,
                   train_time_sec=train_time, checkpoint_path=str(Path(checkpoint_path).resolve()), status=status,
                   registry_schema_version=2, git_commit=manifest.get('git_commit'), git_dirty=manifest.get('git_dirty'),
                   data_mode=report.get('data_mode'), mechanism=config.get('mechanism'),
                   sources=json.dumps(config['sources']) if 'sources' in config else None,
                   control=config.get('control'), missing_rate=config.get('missing_rate'),
                   wall_to_reporting_sec=costs.get('wall_to_reporting_sec'),
                   mean_epoch_train_sec=costs.get('mean_epoch_train_sec'), mean_epoch_wall_sec=costs.get('mean_epoch_wall_sec'),
                   stop_reason=costs.get('stop_reason'), gpu_name=resources.get('gpu_name'),
                   gpu_peak_allocated_bytes=resources.get('gpu_peak_allocated_bytes'),
                   gpu_peak_reserved_bytes=resources.get('gpu_peak_reserved_bytes'),
                   process_peak_rss_bytes=resources.get('process_peak_rss_bytes'), host_ram_bytes=resources.get('host_ram_bytes'),
                   actual_device=manifest.get('device'), run_path=str(run_path),
                   config_path=artifact('config.json'), manifest_path=artifact('run_manifest.json'),
                   report_data_path=artifact('report_data.json'), dataset_statistics_path=artifact('dataset_statistics.json'),
                   per_user_val_path=artifact('per_user_val.csv') or artifact('per_user_val.parquet'),
                   per_user_test_path=artifact('per_user_test.csv') or artifact('per_user_test.parquet'))
        lock = path.with_name(path.name + '.lock')
        try:
            handle = lock.open('x', encoding='utf-8')
        except FileExistsError as exc:
            raise RuntimeError(f'Registry is locked by another writer: {lock}') from exc
        tmp = None
        try:
            with handle:
                handle.write(str(os.getpid()))
                handle.flush()
                old_rows, header = [], []
                if path.is_file() and path.stat().st_size:
                    with path.open(newline='', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        header, old_rows = reader.fieldnames or [], list(reader)
                    if len(header) != len(set(header)) or 'run_id' not in header or any(None in record for record in old_rows):
                        raise ValueError(f'Malformed registry; original file preserved: {path}')
                if any(record['run_id'] == run_id for record in old_rows):
                    raise ValueError(f'Run already registered: {run_id}')
                fields = header + [key for key in row if key not in header]
                if header and fields != header:
                    backup = path.with_name(path.name + '.schema-before.' + uuid.uuid4().hex + '.bak')
                    with path.open('rb') as source, backup.open('xb') as dest:
                        shutil.copyfileobj(source, dest)
                fd, tmp_name = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
                tmp = Path(tmp_name)
                with os.fdopen(fd, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(old_rows)
                    writer.writerow(row)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, path)
        finally:
            if tmp is not None:
                tmp.unlink(missing_ok=True)
            lock.unlink(missing_ok=True)
