# UASF Sequential Recommendation

The 03 November supervisor-feedback pipeline is documented in [CHECKPOINT2.md](CHECKPOINT2.md).
`python scripts/run_checkpoint2.py` prints a nonexecuting plan. Training/inference
require explicit flags, a clean working tree, and the recorded RTX4060 environment.
Historical Checkpoint 1 results are preserved; no new experimental results are
claimed by the implementation alone.

Standalone TAIMMA research code for sequential recommendation with Category, Brand, and Title information. The Python package is `taimma`; the repository is `uasf-seqrec`.

The repository includes the Beauty pilot, a resumable **Checkpoint 1** pipeline, three-dataset metadata preparation, residual late fusion, early fusion, and E0/E1 evaluation. The completed result bundle `runs/checkpoint1_version1` records the 3 November 2026 Beauty experiments. **UASF S3 training, group-gate evaluation, full early-fusion diagnostics, and training-time controls are not completed in that bundle.**

For the recorded results, file guide, and storage/reproduction limits, see [Checkpoint 1 recorded results](#checkpoint-1-recorded-results-3-november-2026). For a new experiment, follow [Checkpoint 1 prerequisites and a fresh run](#checkpoint-1-prerequisites-and-a-fresh-run).

## 1. Requirements

- Git and **64-bit Python 3.12**.
- Windows PowerShell for CPU execution, or Linux/WSL2 with a working NVIDIA GPU for CUDA execution.
- Real dataset files for real experiments. The installation check and synthetic smoke test do not require data.

Create an environment in this checkout. Do not copy one from another repository or share an environment between Windows and WSL. See the [Python venv documentation](https://docs.python.org/3.12/library/venv.html).

`requirements.txt` was distilled from the original repository's dependency files, actual imports, and installed versions. It lists the runtime libraries needed here. PyTorch is selected separately for CPU/CUDA; the other direct dependencies are pinned. This is not a complete transitive lockfile. Metadata preparation additionally uses `sentence-transformers` and `tabulate`, both listed in the current requirements. Browser automation, graph-learning packages, Jupyter, `torchvision`, and `torchaudio` are unnecessary for the supplied command-line scripts. Install the optional `analysis` extra for the supplied plotting/PDF scripts.

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

**2.6.0+cu124** matches the historical 28 October GPU pilot. **2.13.0+cpu** is the observed Windows reference build. The later `checkpoint1_version1` was recorded on **Python 3.12.16, PyTorch 2.13.0+cu130, CUDA runtime 13.0, and an RTX 4060**. The commands above describe a historical setup, not exact reproduction of that later run; use its recorded library/hardware versions when checking determinism. These are documented configurations, not a claim that every intermediate version has been tested. Installation commands follow the [official PyTorch archive](https://pytorch.org/get-started/previous-versions/). For another GPU platform, choose its official PyTorch build before installing `requirements.txt`.

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

Raw `data/`, model weights (`*.pt`), and large score arrays (`*.npy`, `*.npz`) are ignored by Git; metadata arrays under `artifacts/` have an explicit exception. Text logs and result tables can be versioned. A fresh clone needs a local raw-data copy or download and does not include ignored checkpoints/cache files. If reviews already exist, skip downloading them again. For a fresh data directory:

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
# Run in the original execution environment, with checkpoint/cache files present.
# Replace the interpreter if the original run used Linux instead of Windows.
.\.venv-cuda\Scripts\python.exe -m pip install -e ".[analysis]"
.\.venv-cuda\Scripts\python.exe analysis/checkpoint1_postprocess.py --run runs/checkpoint1_version1
.\.venv-cuda\Scripts\python.exe analysis/plot_checkpoint1.py --run runs/checkpoint1_version1 --output .tmp/checkpoint1_version1_figures
.\.venv-cuda\Scripts\python.exe analysis/build_checkpoint1_report.py --run runs/checkpoint1_version1 --output reports/checkpoint1_version1 --figures .tmp/checkpoint1_version1_figures
```

`checkpoint1_postprocess.py` loads model weights and follows the original paths in
`progress.json`; it is **not a log-only analysis command**. A Windows clone of the
text result bundle alone is insufficient. Do not rewrite historical paths or
checksums just to make this command pass.

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


## Checkpoint 1 prerequisites and a fresh run

Use the same prepared environment for dependency installation and execution.
The examples below assume `python` is that environment's interpreter and the
working directory is the repository root.

1. Install the appropriate PyTorch build first, then `requirements.txt` and `pip install -e .`.
2. Obtain all three real Amazon 2014/5-core review files and metadata files.
3. Prepare the metadata and deterministic item-level perturbations. Title encoding may download the sentence encoder on first use; this is not a synthetic fallback.

```bash
python -m metadata_pipeline.prepare --config configs/metadata.yaml --dataset all
python -m metadata_pipeline.build_perturbations --config configs/metadata.yaml --dataset all
python -m metadata_pipeline.report --output reports/table_1_metadata.csv
```

Metadata preparation protects an existing output directory unless `--overwrite`
is explicitly supplied. Do not overwrite metadata already referenced by a
completed run. Review the manifests before regenerating perturbations as well.

4. Choose a **new** output directory. A memory-conservative complete run is:

```bash
python -u scripts/run_checkpoint1.py --device cuda --output runs/checkpoint1_NEW_RUN --stage all --workers 1 --epochs 100 --patience 20 --batch_size 256 --seeds 42 43 44 --threads 2
```

Replace `checkpoint1_NEW_RUN` with a unique name. `--workers 1` explicitly skips
the parallel throughput benchmark. `--stage loo` stops after the LOO comparison;
`--stage grid` stops after the training grid and baseline sanity checks, before
robustness/utility/E0/E1. Both are partial runs. Default `all` also generates
utility and runs E0/E1. The same unchanged command can resume completed stages
only if their checkpoint/cache dependencies remain available and the locked
source/config/metadata/environment checks pass.

**Do not target `runs/checkpoint1_version1` with the current checkout to resume the
archived run.** The archived source snapshot differs from the current checkout
in `src/taimma/data/__init__.py`, `ingestion.py`, and `split.py`, in addition to
Windows line-ending differences. Use a separate checkout restored to the recorded
snapshot and the original execution environment for exact historical reproduction.
For a new run, preserve the archive and use a new output name.

## Checkpoint 1 recorded results 3 November 2026

Source: [`runs/checkpoint1_version1`](runs/checkpoint1_version1), with the
scientific decision stored in [`e0_e1.json`](runs/checkpoint1_version1/e0_e1.json).
These values describe existing real experiments, not a new training run performed
while updating this README. Metrics use the [0,1] scale.

| Recorded result | Value and scope |
| --- | --- |
| LOO pilot condition 3 | PASS: test Recall@10 0.078880; NDCG@10 0.042269; seed 42, d=64, no history masking |
| Main Beauty protocol | Four segments; 109,050 train pairs; 11,998 catalog items; strict input-history masking; 22,363 validation/test users including OOV misses |
| ID-only test NDCG@10 | 0.047851 ± 0.000756, sample SD across seeds 42/43/44 |
| Late all-fixed CBT test NDCG@10 | 0.044826 ± 0.000909, 6.32% below ID; all seven late subsets are below ID in their three-seed mean |
| Early CBT test NDCG@10 | 0.053815, seed 42 only; 14.26% above same-seed ID, not a completed multi-seed early grid |
| Robustness | 90 inference measurements: 9 missing rates × 3 individual sources × 3 seeds, plus 3 permutations × 3 seeds |
| Utility | 22,240 matched Train-meta contexts × 3 sources × 2 independent reference seeds = 133,440 rows; 123 OOV targets excluded from CE utility with an explicit audit |
| E0 | All three sources pass the locked positive/negative fraction criterion |
| E1 | Brand only: LR OOF AUC 0.551050, 95% CI [0.541205, 0.561112], length-only AUC 0.487927; averaged labels because stability rho=0.204658 |
| Decision | CONDITIONAL; not GO for contextual UASF. Group-level gate performance and S3 recommendation gains are not demonstrated yet |
| Cost | 27 fits total 5,252.18 seconds = 1.45894 GPU job-hours for train + validation; excludes complete inference/cache/utility/E0/E1 overhead |
| Execution environment | RTX 4060, Python 3.12.16, PyTorch 2.13.0+cu130, CUDA runtime 13.0; sequential execution, no parallel throughput measurement |

All 21 residual fits selected a trained epoch whose validation NDCG@10 was below
the initial ID-equivalent evaluation. The recorded policy is
`trained_epochs_only`; epoch 0 is logged but cannot be selected. This is an
observed limitation to investigate, not evidence that residual training improved
ID. Do not retrospectively change checkpoint selection after reading test results.

The Brand E1 result is marginally above the AUC threshold and uses averaged
contextual labels. `gating_regime=group_level` in the decision is the prescribed
next step; it is not proof that a group gate has been trained or evaluated.
Category/Title did not pass E1 with the current feature set.

### Where each result can be checked

| Question | Evidence within `runs/checkpoint1_version1` |
| --- | --- |
| All 27 training runs and metrics | `results.csv`; `runs/*/metrics.json` |
| Every epoch and checkpoint choice | `runs/*/epochs.csv`, `history.json`, `run.log`, and `decision_log_snapshot.md` |
| Recompute ranking metrics | `runs/*/per_user_val.csv` and `per_user_test.csv`; average the `ndcg@10` column |
| Full late subset matrix | `diagnostic_B_per_seed.csv`, `diagnostic_B_summary.csv` |
| Add/drop between separately trained subsets | `marginal_utility_seed42.csv`, `marginal_utility_seed43.csv`, `marginal_utility_seed44.csv` |
| Inference robustness | `robustness.csv`; perturbations in `artifacts/metadata/beauty/perturbations/` |
| Utility eligibility and OOV exclusions | `utility_cohort.csv`, `utility_cohort.json` |
| Frozen-reference CE deltas | `utility.csv`, `utility.csv.manifest.json`, `delta_distribution.csv` |
| Label stability, E0/E1 and decision | `e0_e1.json` |
| Reproducibility and cached-score lineage | `reproducibility.json`, `score_cache_integrity/` |
| Historical source version | `protocol.lock.json`; `runs/*/source_snapshot.zip`, `source_snapshot.json`, `working_tree.diff` |
| Dataset statistics and coverage | `artifacts/metadata/{beauty,sports,toys}/{statistics.json,coverage.json,manifest.json}` |

The historical `same_checkpoint_mask_sensitivity.csv` uses the October 28 pilot
checkpoint, not the newly trained ID seed 42. Never label it as a same-checkpoint
comparison of LOO versus four-segment training. Diagnostic B add/drop compares
separately trained subset models; utility delta turns a source off on the **same
frozen all-source reference** without retraining.

### Large files intentionally kept outside Git

The group deliberately ignores **checkpoint weights (`*.pt`) and raw ID-score
caches (`*.npy`/`*.npz`) because they are large**. This is a storage decision;
their absence from Git does not mean that training was not executed. The shared
`checkpoint1_version1` directory contains the text logs/results and manifests,
not its 27 model checkpoint files or nine raw ID-score matrices.

Per-user ranks permit metric recomputation and many statistical analyses without
the weights. Weight-level reproducibility, frozen-ID integrity checks, cached-score
checks, recovery, and new inference **require the corresponding files from the
original execution environment or a separate archive**. Their expected SHA256
values are recorded in manifests. Do not create dummy files or claim that these
checks were independently rerun from the Git clone alone. No external archive
download location is currently declared here.

Original run manifests record Linux absolute paths and `git_dirty=true`, together
with the base commit `5fd0ce8e9ec3f2d280a724220ac9cb3d8692a94f`, source snapshots,
and diffs. The commit alone is insufficient to reproduce the exact source.
Text artifacts may have CRLF line endings after a Windows Git checkout, changing
their byte SHA256 relative to the original LF files. Preserve historical hashes;
recover original bytes or explicitly document line-ending normalization when
auditing them, rather than silently replacing checksums.

### Implementation file guide for supervisor review

| File or module | Role and outputs to inspect |
| --- | --- |
| `scripts/run_checkpoint1.py` | Main orchestration, protocol lock, LOO gate, independent ID/residual seeds, subset grid, robustness, utility and E0/E1 |
| `scripts/checkpoint1_early.py` | First jointly trained early-fusion CBT run and timing |
| `scripts/benchmark_checkpoint1.py` | Optional CUDA worker benchmark; skipped when workers=1 |
| `scripts/resume_checkpoint1_utility.py` | Checkpoint-dependent recovery of utility/E0/E1, no retraining |
| `scripts/run_e0_e1_evaluation.py` | Validate paired contexts, compute E0/E1 and serialize the decision |
| `src/taimma/data/ingestion.py`, `split.py` | Chronological user sequences, catalog and four-segment/LOO splits; inspect the archived source for this historical run |
| `src/metadata_pipeline/prepare.py`, `contracts.py` | Metadata, Train-base vocabularies, real frozen Title encoding and input contracts |
| `src/metadata_pipeline/loader.py`, `build_perturbations.py` | Item-aligned metadata, common history/candidate masks and fixed-missingness permutations |
| `src/metadata_pipeline/report.py`, `validate.py` | Three-dataset Table 1, coverage and artifact validation |
| `src/taimma/models/sasrec.py` | Causal ID backbone and full-catalog scores |
| `src/taimma/models/late_fusion.py` | Frozen ID plus source-only residual encoders, zero final projection and source dropout |
| `src/taimma/models/early_fusion.py` | Metadata addition to sequence and candidate embeddings, joint optimization |
| `src/taimma/training/trainer.py`, `multimodal_trainer.py` | Training, trained-epoch validation selection, atomic history/checkpoint writes |
| `src/taimma/evaluation/evaluator.py` | Shared full ranking, strict input masking, deterministic ties and OOV misses |
| `src/taimma/evaluation/utility.py`, `e0_e1_protocol.py` | Frozen-reference CE delta, history-only features, epsilon/stability, grouped OOF and bootstrap decision logic |
| `src/taimma/experiments/logger.py`, `reporting.py`, `checkpoint_state.py` | Registry, provenance, resource scopes and verified completed-stage reuse |
| `analysis/checkpoint1_postprocess.py` | Weight-dependent frozen-ID audit, registry consolidation and provisional budget |
| `analysis/plot_checkpoint1.py`, `build_checkpoint1_report.py` | Figures and the existing two-page PDF report generator; pass the correct --run explicitly |
| `analysis/audit_checkpoint_selection.py` | Checkpoint-dependent validation selection audit |

This guide identifies files involved in Checkpoint 1; it does not assert that all
of them were first created in one commit or that every implemented model has
been experimentally evaluated. For this run's actual changes relative to its
base commit, read `working_tree.diff` and the complete source snapshot.
