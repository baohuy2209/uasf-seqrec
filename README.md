# UASF Sequential Recommendation

Standalone TAIMMA research code for sequential recommendation with Category, Brand, and Title information. The Python package is `taimma`; the repository is `uasf-seqrec`.

The experiment entry points currently run a **Beauty SASRec ID-only pilot**, collect statistics for three Amazon 2014 datasets, and evaluate E0/E1 from an existing utility-delta file. Early fusion, residual late fusion, and UASF implementations are included; their presence does not imply that those experiments have been executed.

## 1. Requirements

- Git and **64-bit Python 3.12**.
- Windows PowerShell for CPU execution, or Linux/WSL2 with a working NVIDIA GPU for CUDA execution.
- Real dataset files for real experiments. The installation check and synthetic smoke test do not require data.

Create an environment in this checkout. Do not copy one from another repository or share an environment between Windows and WSL. See the [Python venv documentation](https://docs.python.org/3.12/library/venv.html).

`requirements.txt` was distilled from the original repository's dependency files, actual imports, and installed versions. It lists the runtime libraries needed here. PyTorch is selected separately for CPU/CUDA; the other direct dependencies are pinned. This is not a complete transitive lockfile. Browser automation, graph-learning packages, Jupyter, `torchvision`, and `torchaudio` are unnecessary for the supplied scripts.

## 2. Windows CPU installation

For a new checkout:

```powershell
git clone https://github.com/baohuy2209/uasf-seqrec.git
Set-Location uasf-seqrec
```

For the existing local checkout, start with `Set-Location "D:\Github\uasf-seqrec"` instead. Then:

```powershell
$env:PYTHONUTF8 = "1"
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m pip check
```

If `py` is unavailable, confirm `python --version` is 3.12 and use `python -m venv .venv`. Commands invoke the environment's Python directly; activation and PowerShell execution-policy changes are unnecessary.

Verify the libraries and command entry point:

```powershell
.\.venv\Scripts\python.exe -c "import torch, numpy, pandas, scipy, sklearn, yaml, psutil; print('Imports OK'); print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
.\.venv\Scripts\python.exe scripts/run_pilot.py --help
```

`CUDA available: False` is expected for this CPU installation.

## 3. Linux or WSL2 GPU installation

In a Linux/WSL terminal, change to your checkout. For the existing Windows location:

```bash
cd /mnt/d/Github/uasf-seqrec
python3.12 --version
nvidia-smi

python3.12 -m venv .venv-wsl
.venv-wsl/bin/python -m pip install --upgrade pip
.venv-wsl/bin/python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
.venv-wsl/bin/python -m pip install -r requirements.txt
.venv-wsl/bin/python -m pip install -e .
.venv-wsl/bin/python -m pip check

.venv-wsl/bin/python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA runtime:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'unavailable')"
```

On Ubuntu, if environment creation reports missing `ensurepip`, install the matching `python3.12-venv` package and retry. Resolve GPU/driver availability before requesting CUDA. The runner fails explicitly if CUDA is unavailable.

**2.6.0+cu124** matches the recorded GPU pilot environment. **2.13.0+cpu** is the observed Windows reference build. These are two documented configurations, not a claim that every intermediate version has been tested. Installation commands follow the [official PyTorch archive](https://pytorch.org/get-started/previous-versions/). For another GPU platform, choose its official PyTorch build before installing `requirements.txt`.

For Linux CPU execution, install `torch==2.13.0` from the CPU index in a Linux environment, then install `requirements.txt` and choose `--device cpu`.

## 4. Smoke test without real data

From the checkout root:

```powershell
.\.venv\Scripts\python.exe scripts/run_pilot.py `
  --smoke_test `
  --epochs 5 `
  --patience 5 `
  --device cpu `
  --rank_format csv `
  --verify_reproducibility `
  --runs_dir .tmp/smoke/runs `
  --results_csv .tmp/smoke/results.csv
```

For Linux, replace the interpreter with `.venv-wsl/bin/python` and use Bash continuations (`\`) or put the arguments on one line. This uses only a small built-in fixture and separate `.tmp/smoke/` outputs. It verifies execution; it is not real research data or a scientific pilot PASS.

If your checkout includes `tests/`, install the test extra and run the research suite:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
.\.venv\Scripts\python.exe -m pytest tests/ -q
```

Use `.venv-wsl/bin/python` for the equivalent Linux commands. Some shared checkouts may omit `tests/`; use the smoke check in that case, without claiming full test coverage. The obsolete `notebooks/taimma/tests/` command does not apply to this repository.

## 5. Real Amazon 2014 datasets

Use **Amazon Reviews 2014 / 5-core**. Common local filenames are:

| Dataset key | Local review file |
| --- | --- |
| `Beauty` | `data/raw/Beauty/Beauty_5.json` |
| `Sports_and_Outdoors` | `data/raw/Sports and Outdoors/Sports_and_Outdoors_5.json` |
| `Toys_and_Games` | `data/raw/Toys and Games/Toys_and_Games_5.json` |

The loader also accepts official `reviews_<category>_5.json.gz` files directly under `data/raw/` or an appropriate category directory. Compressed files do not need extraction. Do not add an `amazon_reviews/2014/` directory layer.

Data and generated results are ignored by Git, so a fresh clone needs a local data copy or download. If reviews already exist, skip downloading them again. For a fresh data directory:

```powershell
.\.venv\Scripts\python.exe scripts/download_amazon2014.py --dataset Beauty --output_dir data/raw
.\.venv\Scripts\python.exe scripts/download_amazon2014.py --dataset Sports_and_Outdoors --output_dir data/raw
.\.venv\Scripts\python.exe scripts/download_amazon2014.py --dataset Toys_and_Games --output_dir data/raw
```

The downloader retrieves reviews and metadata from Stanford SNAP. Downloads may be large. Existing files with the exact destination name are skipped; it does not recognize an uncompressed review in another directory as equivalent to a compressed download destination.

Expected original user/item counts: Beauty **22,363 / 12,101**, Sports **35,598 / 18,357**, Toys **19,412 / 11,924**. The Train-base candidate catalog can be smaller than the full item set.

Unpruned `ratings_*.csv` files are optional diagnostics; experiments use 5-core reviews. Missing metadata permits ID-only training but leaves metadata-dependent experiments blocked. Title experiments additionally require real encoded Title features. Missing coverage is unknown, not a measured zero.

## 6. Table 1 statistics without training

Choose a new output directory each time:

```powershell
$reportStamp = Get-Date -Format "yyyyMMdd_HHmmss"
.\.venv\Scripts\python.exe scripts/collect_dataset_statistics.py `
  --raw_root data/raw `
  --output_dir "results/table1_$reportStamp"
```

This reads all three real datasets and saves `table1.csv`, per-dataset JSON files, and a source manifest. It does not train a model. Missing metadata coverage/vocabulary remains null.

## 7. Real Beauty pilot with training and evaluation

### Windows CPU

```powershell
.\.venv\Scripts\python.exe -u scripts/run_pilot.py `
  --real_data `
  --config configs/experiments/pilot_beauty.yaml `
  --dataset_dir data/raw `
  --device cpu `
  --rank_format csv `
  --verify_reproducibility
```

### Linux or WSL GPU

```bash
.venv-wsl/bin/python -u scripts/run_pilot.py \
  --real_data \
  --config configs/experiments/pilot_beauty.yaml \
  --dataset_dir data/raw \
  --device cuda \
  --rank_format csv \
  --verify_reproducibility
```

The YAML supplies seed **42**, batch size **256**, learning rate **0.001**, maximum **100 epochs**, and patience **10**. CLI values override YAML. Select the device explicitly.

The runner trains, restores the checkpoint selected by validation NDCG@10, and evaluates validation/test NDCG and Recall at 10 and 20. It also evaluates Random, Popularity, and shuffled targets. There is no separate checkpoint-evaluation CLI.

`--verify_reproducibility` performs a second full training run with the same seed, approximately doubling training work. It verifies repeatability, not variation across independent seeds. The pilot validates Beauty only; changing `--dataset_dir` does not turn it into a Sports/Toys training runner.

### Output locations

- `runs/<unique_run_id>/`: checkpoint, config, metrics, baselines, per-user ranks, epoch history, logs, resource measurements, source snapshot, and checksum index.
- `data/processed/beauty_2014/<unique_run_id>/`: samples, mappings, and provenance for a new real pilot.
- `results/results.csv`: experiment registry. The smoke command uses a separate registry.
- `repeat/` within a run: repeated-training evidence when requested.

`COMPLETED` describes execution. Scientific acceptance can remain `PENDING` or `REVIEW` because of missing metadata, a reference comparison, or a sanity check. Preserve these states.

Copied historical artifacts retain their original paths, commit IDs, and checksums. They are archival evidence; new runs use this checkout. Do not bulk-rewrite historical manifests or registry rows.

## 8. Real E0/E1 evaluation

This requires a **real utility-delta artifact from two independent late-fusion reference seeds**. The ID-only pilot does not generate it. Supply an actual CSV, Parquet, or NPZ file:

```powershell
$deltaFile = Read-Host "Path to the real two-seed delta artifact"
if (-not (Test-Path -LiteralPath $deltaFile)) { throw "Missing delta file: $deltaFile" }
$evalStamp = Get-Date -Format "yyyyMMdd_HHmmss"
.\.venv\Scripts\python.exe scripts/run_e0_e1_evaluation.py `
  --delta_file $deltaFile `
  --n_bootstrap 1000 `
  --seed 42 `
  --output_json "results/e0_e1_$evalStamp.json"
```

The loader validates aligned contexts, paired seeds, actual sequence lengths, and explicit context features. Long CSV/Parquet includes `user_idx`, `source_name`, `reference_seed`, `delta`, `seq_len`, and `feat_*` columns; `context_id` distinguishes multiple contexts per user. Both seeds must cover the same contexts for each source. Wide and NPZ schemas differ; inspect `load_delta_dataset` before constructing an artifact.

The explicit `--synthetic` mode tests the procedure only. Keep those outputs separate and never present them as real E0/E1 findings.

## 9. Paths, imports, and troubleshooting

Examples start from the repository root. Relative configuration, input, and output paths are anchored to that root even when an absolute script path is invoked from elsewhere. Absolute paths are honored. Quote paths containing spaces and use Windows paths on Windows or Linux paths in WSL.

`pyproject.toml` installs the distribution `uasf-seqrec` while preserving the `taimma` Python namespace. The editable installation above enables imports from any working directory; scripts also support direct execution from the checkout. Verify the module origin without setting PYTHONPATH:

```powershell
.\.venv\Scripts\python.exe -c "import taimma; print(taimma.__file__)"
```

The module path must belong to this checkout. Raw `data/` files are inputs; the Python loader package is `src/taimma/data/`.

- **Parquet:** CSV is the default. For Parquet ranks or deltas, install `pyarrow` with the same environment's Python, for example `.\.venv\Scripts\python.exe -m pip install pyarrow`. Missing support raises an error rather than changing formats silently.
- **Missing imports:** use the same interpreter for installation and execution; check its `-m pip --version` and reinstall `requirements.txt` as needed.
- **CUDA unavailable:** check `nvidia-smi` and `torch.cuda.is_available()`, or explicitly use the CPU setup and `--device cpu`.
- **Missing dataset:** place the actual 2014 5-core files under `data/raw/`, or supply a valid dataset root. Another dataset is not substituted.
- **Output exists:** select a new timestamp or filename. Existing outputs are protected.
- **Unicode console errors on Windows:** set `$env:PYTHONUTF8 = "1"` before invoking Python, especially when the checkout path contains accents.
- **Missing Python wheels or syntax errors:** verify 64-bit Python 3.12 and create a new environment rather than copying one.
- **Exact environment record:** save `python -m pip freeze` from the actual execution environment alongside new experiment records. Different hardware or PyTorch builds need not produce bitwise-identical results.

## Scientific protocol

The project uses chronological per-user Train-base/Train-meta/validation/test splits with `n >= 5`, maximum input length 50, Train-base candidate catalogs, full-catalog cross-entropy and ranking, documented Train-base seen-item masking, deterministic ties, and OOV evaluation targets retained as misses. Checkpoints are selected using validation NDCG@10. Missing scientific evidence remains pending.
