# Reproduction and artifact guide

## 1. Inspect the official results

```bash
python scripts/paper_results.py verify
```

The committed `results/paper/` is a compact summary package. Its manifest checks
the selected original artifacts against the original freeze/release bindings.
`tables/source_composition.csv` is aggregated from the recorded job registry.
`tables/correspondence.csv` is recomputed from existing paired prediction CSVs
with 1,000 NumPy user resamples (seed 42), averaging the three matched-seed
differences within each user before resampling. No models are loaded.

This package does not contain all files named in the original `freeze.json`.
That file is retained as provenance, not as a claim that weights and per-user
archives are part of the Git source release. The standalone verifier checks
the package's selected summaries; the historical inference guards check the
complete original run and are intentionally stricter.

The source release changes the package version from 0.1.0 to 1.0.0 and updates
logger provenance for source ZIPs without `.git`. The exporter accepts the
version substitution only when restoring the old line reproduces its original
checksum; the logger exception is pinned to the exact audited old/new checksums.
Both are disclosed in `publication_support_changes`. It rejects other changes
to frozen source files. New publication tools and the portable example profile
are additional files, outside the original scientific freeze.

In a source archive the logger records `git_snapshot.json` with `available=false`
and still saves a complete source snapshot and hashes. It does not invent a Git
commit or a clean working tree. Unexpected Git errors in a real checkout remain
errors, and the historical research execution guards still require reviewed Git.

## 2. Obtain data and build metadata in a fresh workspace

Obtain the original Amazon Reviews 2014 5-core review and metadata files separately.
Expected layout (the loader also accepts official gzip filenames):

```text
data/raw/Beauty/Beauty_5.json
data/raw/Sports and Outdoors/Sports_and_Outdoors_5.json
data/raw/Toys and Games/Toys_and_Games_5.json
data/raw/meta_Beauty.json.gz
data/raw/meta_Sports_and_Outdoors.json.gz
data/raw/meta_Toys_and_Games.json.gz
```

After installing the runtime dependencies, generate metadata only in a fresh
artifact directory, or retain your existing verified artifacts:

```bash
python -m metadata_pipeline.prepare --dataset all --raw_root data/raw --output_dir artifacts/metadata
```

This extracts dataset-specific mappings and frozen Title embeddings. It is a
data preparation step, not a recommender training run. Do not use `--overwrite`
on the artifacts belonging to the original study.

## 3. Understand the staged execution

The recorded study reused verified foundation models from earlier executions.
The memo runner deliberately fails if an expected adopted checkpoint is absent;
it does not silently substitute a newly trained model.

| Stage | Driver / guide | Role |
| --- | --- | --- |
| Early foundations and correspondence controls | `scripts/run_checkpoint2.py`, [Checkpoint 2](../CHECKPOINT2.md) | ID, seven early subsets, train-time controls, early SD |
| Residual late foundations | `scripts/run_late_checkpoint.py`, [late guide](../README_LATE_CHECKPOINT.md) | Frozen ID plus residual source encoders |
| Preliminary K1–K4 | `scripts/run_checkpoint2.py`, [Checkpoint 2](../CHECKPOINT2.md) | Historical Beauty diagnostics, kept separate from the main grid |
| Memo study | `scripts/run_memo0711.py`, [memo guide](../README_MEMO0711.md) | Three-dataset diagnostics, Beauty five-seed gates, controls, F2 decision |
| Original final evaluation | `scripts/finalize_memo0711.py`, [final guide](../README_FINAL_TEST.md) | Existing weights only, locked-family Test and posthoc validation alpha |

Use `--help` and `--stage plan` first. Those commands do not start training.
The original training environment is recorded in the published protocol locks;
the execution guards require that environment. New reproduction outputs must
use new directories and new protocol locks, separate from the original frozen run.

The portable profile `configs/experiments/paper_public.yaml` preserves the
scientific settings and replaces machine-specific input roots with checkout-relative
paths. Populate those foundation directories first, or override `--early-root`,
`--late-root`, and `--diagnostics-root` explicitly.

```bash
python scripts/run_memo0711.py --stage plan --config configs/experiments/paper_public.yaml --output runs/reproduction_memo0711
```

After preparing the required foundations and reviewing the plan, the corresponding
memo command is below. It **does train** and is provided for independent
reproduction, not for inspecting the published tables:

```bash
python scripts/run_memo0711.py --stage all --config configs/experiments/paper_public.yaml --output runs/reproduction_memo0711 --execute-training
```

The study's no-additional-training decision applies to the recorded experiment.
This source release does not launch any such command. F1/F2 and Test restrictions
remain enforced by the existing runners.

## 4. Export summaries from an existing original archive

With the completed original memo run, its early-control archive, and verified
metadata available locally, export to a new directory:

```bash
python scripts/paper_results.py export --run-root runs/memo0711_v3 --early-root runs/checkpoint2_full_20261106 --metadata-root artifacts/metadata --destination results/paper_reexport
python scripts/paper_results.py verify --destination results/paper_reexport
```

Export requires NumPy for the correspondence bootstrap. It only reads recorded
measurements and copies/aggregates them. It never trains, invokes inference,
relabels utility, or changes a protocol/hash index. Existing destinations are
refused. A source file differing from its original hash is rejected.

On Windows, export can restore LF bytes only if the restored bytes match the
original recorded checksum exactly. It does not relax checksum verification.
The public summary files use `-text` Git attributes to retain those exact bytes.

## 5. Statistical scopes

- Source-composition, correspondence, and robustness use seeds 42–44.
- Main Beauty selection and training-reference comparisons use seeds 42–46.
- All-source/global share one display row because predictions are identical;
  all 11 method labels remain in the machine-readable Test family.
- Same-backbone group subsets are different from the group selection over
  separately trained configurations. Keep their tables separate.
- Oracle Test selection is a descriptive upper bound within the declared
  group-subset family; it is never used for model/backbone selection.
- Bootstrap intervals condition on the fitted models. The resampling unit is
  user, retaining matched-seed predictions together; it is not user–seed.
- Alpha is posthoc validation analysis. Its all-off rate is 11/111,815 user–seed
  contexts, not a count of 111,815 independent users.
- A stored bootstrap p-value of zero means zero exceedances among finite draws,
  not an exact zero probability. Do not print `p = 0` as an inferential claim.

## 6. Local historical records

Cleaning the public Git tree does not delete the maintainer's `runs/`, raw data,
metadata, or reports. Those folders are ignored in the public source tree.
Old server paths in preserved historical documents and manifests remain unchanged.
They are provenance, not portable installation commands.

This cleanup changes the current Git tree, without rewriting commit history.
Earlier commits retain artifacts that were previously tracked. Source archives
built from the cleaned tree exclude those local artifacts.
