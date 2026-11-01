"""
test_migration_contracts.py - Verification tests for standalone repository migration.

Validates:
  1. Package identity and src/taimma discovery.
  2. Repository-relative path contract and root independence.
  3. Dataset configuration paths (raw, processed, splits).
  4. Real dataset file resolution and fail-fast behavior on missing inputs.
  5. CLI help and entrypoint validity.
"""

import os
import subprocess
import sys
from pathlib import Path
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import taimma
from taimma.paths import PROJECT_ROOT, resolve_project_path
from taimma.data.ingestion import Amazon2014Ingestor
from taimma.models.sasrec import SASRecModel
from taimma.evaluation.evaluator import IndependentEvaluator


def test_package_identity_and_location():
    """taimma package must resolve directly inside the destination src directory."""
    pkg_file = Path(taimma.__file__).resolve()
    assert pkg_file == (SRC_DIR / "taimma" / "__init__.py").resolve()
    assert PROJECT_ROOT == REPO_ROOT.resolve()


def test_public_interface_imports():
    """Core public symbols must be directly importable from the taimma package."""
    assert Amazon2014Ingestor is not None
    assert SASRecModel is not None
    assert IndependentEvaluator is not None


def test_resolve_project_path_relative_and_absolute(tmp_path):
    """Relative paths must anchor to REPO_ROOT; absolute paths must be preserved."""
    rel = "data/raw/Beauty/Beauty_5.json"
    resolved_rel = resolve_project_path(rel)
    assert resolved_rel == (REPO_ROOT / rel).resolve()

    # Absolute path test
    abs_path = (tmp_path / "custom_data").resolve()
    resolved_abs = resolve_project_path(abs_path)
    assert resolved_abs == abs_path


def test_dataset_configs_aligned():
    """All three active dataset configuration files must point to data/raw, data/processed, data/splits."""
    configs_dir = REPO_ROOT / "configs" / "datasets"
    expected = {
        "beauty_2014.yaml": ("Beauty_2014", "data/raw", "data/processed/beauty_2014", "data/splits/beauty_2014"),
        "sports_and_outdoors_2014.yaml": ("Sports_and_Outdoors_2014", "data/raw", "data/processed/sports_and_outdoors_2014", "data/splits/sports_and_outdoors_2014"),
        "toys_and_games_2014.yaml": ("Toys_and_Games_2014", "data/raw", "data/processed/toys_and_games_2014", "data/splits/toys_and_games_2014"),
    }
    for filename, (ds_name, raw_dir, proc_dir, splits_dir) in expected.items():
        cfg_file = configs_dir / filename
        assert cfg_file.is_file(), f"Missing config file: {filename}"
        with cfg_file.open(encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        assert cfg["dataset_name"] == ds_name
        assert cfg["raw_data_dir"] == raw_dir
        assert cfg["processed_dir"] == proc_dir
        assert cfg["splits_dir"] == splits_dir


def test_real_dataset_review_files_exist_and_resolve():
    """Verify that Amazon2014Ingestor resolves real review files for all three categories."""
    categories = ["Beauty", "Sports_and_Outdoors", "Toys_and_Games"]
    for cat in categories:
        ingestor = Amazon2014Ingestor(raw_data_dir="data/raw", dataset_key=cat)
        reviews_path, meta_path = ingestor.resolve_paths()
        assert Path(reviews_path).is_file(), f"Failed to locate review file for {cat}"
        if meta_path is not None:
            assert Path(meta_path).is_file(), f"Invalid metadata path for {cat}"


def test_missing_raw_directory_fails_fast_without_fallback():
    """Invalid raw directory must raise FileNotFoundError and not fall back to old repo."""
    ingestor = Amazon2014Ingestor(raw_data_dir="non_existent_folder_abc", dataset_key="Beauty")
    with pytest.raises(FileNotFoundError) as exc_info:
        ingestor.resolve_paths()
    msg = str(exc_info.value)
    assert "[TAIMMA Data Error]" in msg
    assert "data-handle-pipeline" not in msg


def test_cli_help_independent_of_cwd(tmp_path):
    """Scripts must print --help cleanly when invoked from any arbitrary directory."""
    scripts = [
        "scripts/run_pilot.py",
        "scripts/collect_dataset_statistics.py",
        "scripts/run_e0_e1_evaluation.py",
        "scripts/download_amazon2014.py",
    ]
    for rel_script in scripts:
        script_path = str(REPO_ROOT / rel_script)
        res = subprocess.run(
            [sys.executable, script_path, "--help"],
            cwd=str(tmp_path),
            capture_output=True,
            text=True,
        )
        assert res.returncode == 0, f"Script {rel_script} failed --help in external cwd: {res.stderr}"
        assert "usage:" in res.stdout.lower()
