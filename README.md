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

The project uses chronological per-user Train-base/Train-meta/validation/test splits with `n >= 5`, maximum input length 50, Train-base candidate catalogs, full-catalog cross-entropy and ranking, strict per-prediction input-history masking (including repeated targets as misses), deterministic ties, and OOV evaluation targets retained as misses. Checkpoints are selected using validation NDCG@10. Missing scientific evidence remains pending.


## Checkpoint 1 execution after the 29 October memo

Use a separate CUDA environment on Windows if WSL cannot start. The existing CPU
`.venv` remains available for tests. A clean CUDA setup selects PyTorch first:

```powershell
py -3.12 -m venv .venv-cuda
.\.venv-cuda\Scripts\python.exe -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
.\.venv-cuda\Scripts\python.exe -m pip install -r requirements.txt
.\.venv-cuda\Scripts\python.exe -m pip install -e .
.\.venv-cuda\Scripts\python.exe -m pytest -q tests
.\.venv-cuda\Scripts\python.exe -u scripts/run_checkpoint1.py --device cuda --stage all --workers 2
```

The driver uses real Beauty data and verified metadata. It audits repeated/OOV
targets and four named masking policies on the same historical checkpoint, then
runs standard LOO. It tries d=256 only if d=64 fails the fixed Recall/NDCG bounds;
it stops the diagnostic grid if both fail. The sequential DIF-SR reference uses
no history mask; the main four-segment experiments use `input_history_strict`.
Validation selection, test, baselines and multimodal evaluation share the named
main policy. Masking affects ranking scores, not the model's input sequence.

After the gate passes, the driver measures one real epoch with CUDA process counts
up to `--workers`; trains and verifies two ID seed 42 runs; fits independent ID backbones
and seven late-fusion subsets for seeds
42/43/44; stores unmasked float32 ID scores for Train-meta/validation/test; fits
the first early-fusion CBT model; evaluates inference
missingness/permutation; creates paired utility labels and runs real E0/E1.
Residual final projections start at zero and ID parameters stay frozen.

For new runs, every model trains from epoch 1 and selects `checkpoint.pt` using
the highest full-precision validation NDCG@10 among trained epochs only. The
pre-training evaluation is recorded in `initial_validation_metrics` and as a
separate `epochs.csv` row with `phase=initial_validation`, `epoch=0`,
`trained=False`, and no training loss. It is never eligible for selection.
`total_epochs` counts training epochs only. Ties keep the earlier trained epoch.

Console output includes every training epoch. `epochs.csv` and `history.json`
are updated atomically after each completed epoch, including non-improving
epochs, so completed epoch logs remain available if a later epoch fails.
The CSV marks the selected checkpoint; `metrics.json` records the selection
scope, exact selected score, selected epoch, and separate initial validation.
Checkpoint writes also use a temporary file before replacing the previous best.
Existing runs retain their historical checkpoints and selection policy; use a
fresh output directory for this trained-epoch-only policy.

Seeding enables deterministic PyTorch algorithms and configures the cuBLAS
workspace before CUDA initialization. Checkpoint 1 compares exact validation/test
ranks, model weights and best epoch before starting the residual grid, saving
`reproducibility.json` on both success and failure. Runs trained before this
determinism fix require a fresh `--output`; their existing checkpoints cannot
be made reproducible retroactively.

Default output: `runs/checkpoint1_20261101_batch`. Repeating the same command resumes
completed stages; changed experiment code/config/metadata requires a new
`--output`. Failed attempts retain their logs. Use `--stage loo` to stop after
the comparison or `--stage grid` after the training grid. Diagnostic robustness
results currently vary one metadata source at a time, preserving the others.

For limited host RAM, use `--workers 1`. This runs training in the main process
and skips the parallel throughput benchmark. Higher worker counts are benchmarked
only within the requested limit and available host RAM. Failed workers stop the
current measurement, all child processes are cleaned up, and a previously
successful lower count is used if available.

```bash
python3 -u scripts/run_checkpoint1.py --device cuda --output runs/checkpoint1_memorysafe --stage all --workers 1
```

Utility labels use only Train-meta contexts whose target is in the Train-base
catalog. Both reference seeds use the same selected contexts. The full selection
and OOV exclusions are recorded in `utility_cohort.csv` and `utility_cohort.json`;
validation/test still retain OOV targets as misses.

If a completed training run stopped at utility because of an OOV target, resume
utility and E0/E1 using its existing checkpoints:

```bash
python scripts/resume_checkpoint1_utility.py --device cuda --output runs/checkpoint1_20261103_deterministic
```

This recovery command checks the locked source, metadata and reference checkpoint
hashes. Only the driver's utility method may differ from the original source.
It saves its source snapshot under `utility_recovery/`, keeps the original
protocol lock, and performs no training. `--utility-only` stops after saving labels.

After all stages succeed, create the measured tables and provisional budget:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[analysis]"
.\.venv-cuda\Scripts\python.exe analysis/checkpoint1_postprocess.py
.\.venv\Scripts\python.exe analysis/plot_checkpoint1.py
.\.venv\Scripts\python.exe analysis/build_checkpoint1_report.py
```

The last command needs reportlab/pypdf and Unicode fonts. The report enforces the
two-page limit. Scientific acceptance is separate from execution completion:
read `e0_e1.json`, including confidence intervals and the decision regime.
Derived tables go to `reports/`; training checkpoints and raw score caches stay
in the run folder. Analysis source hashes are recorded independently of the
locked experiment source. No results are automatically published or emailed.


### Prepare now and run later without GPU

`--prepare-only` does not initialize CUDA, train, evaluate, or generate utility
labels. It validates the real data setup and writes a resumable execution plan.
For this checkout, the earlier LOO and ID seed 42 have finished. Reuse them
explicitly while discarding the interrupted metadata fit:

```powershell
.\.venv\Scripts\python.exe scripts/run_checkpoint1.py --prepare-only --device cuda --output runs/checkpoint1_20261101_batch --reuse-completed-from runs/checkpoint1_20261101
```

Reuse verifies the unchanged ID algorithms/model construction, input manifest,
completed run status, and checkpoint/per-user evidence hashes. It never reuses
an incomplete residual fit. Keep the source run directory because imported
checkpoints and ID-score arrays remain there. The transfer is recorded in
`reused_stages.json`.

When CUDA is ready, use the same destination, without `--prepare-only`:

```powershell
.\.venv-cuda\Scripts\python.exe -u scripts/run_checkpoint1.py --device cuda --output runs/checkpoint1_20261101_batch --stage all --workers 2
```

For a fresh experiment, choose a new output directory and omit
`--reuse-completed-from`. Completed stages are retained; robustness resumes
at saved measurement boundaries and utility writes are promoted atomically.
The first interrupted CBT profiling attempt remains in the old run directory
and is not a completed experimental result.
