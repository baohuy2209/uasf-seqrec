# Useful or Redundant? A Controlled Study of Item Metadata in Sequential Recommendation

Official research code for the TAIMMA study of Category, Brand, and Title on
Amazon Reviews 2014 Beauty, Sports and Outdoors, and Toys and Games.
The Python distribution is **uasf-seqrec** and its packages are `taimma` and
`metadata_pipeline`. This source release is version **1.0.0**.

The study compares all eight source combinations under early addition and
frozen-ID residual late fusion, correspondence controls, single-source
missingness, and utility-supervised selective fusion (UASF). The final framing
is **F2**, a controlled empirical study. UASF does not outperform the ungated
full-source model or the validation-selected group baseline in the evaluated
Beauty setting. Its transfer to Sports/Toys was not performed under F2.

## Quick verification of the published results

From a source checkout, using Python 3.12:

```bash
python scripts/paper_results.py verify
```

This command uses only the Python standard library. It verifies the checksums,
freeze binding, complete 11-label comparison family, and summary-table schema
in [`results/paper/`](results/paper/). It requires no raw data, model weights,
GPU, or Git history and does not train or run inference. It verifies published
summary evidence; it does not claim to replay model inference from those tables.

## Recorded findings

Early full-source fusion improves Test NDCG@10 over ID-only by **15.29%**, **11.63%**,
and **13.85%** on Beauty, Sports, and Toys, respectively (seeds 42–44).
Every evaluated nonempty frozen-ID residual late configuration is below its
matched ID reference, **63/63** comparisons. Source value is compositional.
On Beauty, shuffling Brand removes the observed full-source gain, while Title
helps alone but adds little in the full-source configuration. Source dropout
reduces degradation under random single-source missingness, with a clean-score
cost on Sports and Toys.

Beauty selection comparisons use five seeds, 42–46. The values below are
NDCG@10 **×100**, mean ± sample SD:

| Configuration | Validation | Frozen Test |
| --- | ---: | ---: |
| All-source / global subset | 6.932 ± 0.096 | 5.551 ± 0.039 |
| Group subset | 6.945 ± 0.100 | 5.475 ± 0.085 |
| Global weights | 6.802 ± 0.102 | 5.382 ± 0.018 |
| Sigmoid without utility | 6.839 ± 0.067 | 5.416 ± 0.039 |
| Softmax with ID/null | 6.457 ± 0.060 | 5.124 ± 0.026 |
| UASF | 6.583 ± 0.069 | 5.226 ± 0.056 |
| Independent Train-base ID | 6.031 ± 0.065 | 4.801 ± 0.062 |
| Early without source dropout | 7.006 ± 0.080 | 5.514 ± 0.020 |
| ID + Meta | 6.916 ± 0.031 | 5.484 ± 0.038 |
| Early + Meta | 7.755 ± 0.056 | 6.140 ± 0.075 |

All-source and global subset have identical predictions and share one display
row. The machine-readable family retains all ten baselines plus UASF. Group
selection has the highest validation mean within the shared-backbone family;
all-source/global has the highest Test mean in that family. These Test outcomes
did not change F2, thresholds, model choices, or the label definition.

Authoritative tables are the [complete Val/Test table](results/paper/run/finalization/table5_validation_test.csv),
[paired Test intervals](results/paper/run/finalization/test_paired_ci.csv),
[source-composition table](results/paper/tables/source_composition.csv), and
[correspondence controls](results/paper/tables/correspondence.csv).
Metric columns in CSVs use the original 0–1 scale unless stated otherwise.

## Installation and tests

Use a separate virtual environment for this checkout:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# Install the appropriate CPU/CUDA PyTorch build for your platform first.
python -m pip install -r requirements.txt
python -m pip install -e ".[test,analysis]"
python -m pytest tests -q
```

The published experiments record Python **3.12.16**, PyTorch **2.13.0+cu130**,
NumPy **2.5.1**, pandas **3.0.5**, SciPy **1.18.0**, scikit-learn **1.9.0**, and
an NVIDIA GeForce **RTX 4060**. See each published `protocol.lock.json` for its
recorded environment. A generic installation is suitable for code checks;
the research runners enforce their recorded environment for execution.
`requirements.txt` is a direct-dependency specification, not a transitive lockfile.

Tests use small fixtures and temporary output folders. They do not establish
bitwise reproducibility on a different GPU/software stack. Some CUDA tests skip
when CUDA is unavailable.

## Data and experimental protocol

Obtain Amazon Reviews **2014 / 5-core** reviews and source metadata separately,
under their original providers' terms. No raw reviews or item-level metadata
are distributed in this source release.

For each user's chronological sequence of length `n >= 5`:

| Partition | Position | Purpose |
| --- | --- | --- |
| Train-base | 1 through n−3 | Recommender training and candidate vocabulary |
| Train-meta | n−2 | Utility labels and gate training |
| Validation | n−1 | Checkpoint and subset selection |
| Test | n | Frozen evaluation |

Ranking uses the full Train-base candidate catalog and `input_history_strict`:
non-padding input-history items are masked, including repeated targets. OOV
validation/Test targets remain in the denominator as misses. Train-meta OOV
targets are excluded from utility-label generation and gate training.

Category uses pooled categorical embeddings, Brand a categorical embedding,
and Title a learned projection of frozen `all-MiniLM-L6-v2` sentence embeddings.
Source weights apply to both history encoding and candidate representations.
All-off UASF contexts use the independent same-seed Train-base ID model.

Reported 95% paired-user intervals average matched-seed differences within user
before 1,000 user resamples. They condition on the fitted models. The displayed
intervals are unadjusted; p-values for the ten locked UASF–baseline Test
comparisons have Holm correction. They are not confidence intervals over new
training seeds or new datasets.

## Reproducing or inspecting the study

Read [the reproduction guide](docs/REPRODUCIBILITY.md) before running experiments.
The recorded study adopted foundations from earlier runs. A fresh checkout
does not contain those weights or the providers' datasets. The guide identifies
the foundation, memo, and finalization stages and the portable path profile.

```bash
# Safe, nonexecuting plans:
python scripts/run_memo0711.py --stage plan --output runs/reproduction_memo0711
python scripts/finalize_memo0711.py --stage plan
```

Training and inference require explicit flags. The old training runner's
`--stage all --execute-training` is not a command for inspecting final results.
Do not refit, relabel, tune on Test, or overwrite a frozen run to reproduce a table.

The original scientific freeze tag is `taimma-f2-freeze-20261112-r2`.
This source release adds publication tools and documentation; it is distinct
from that historical freeze. Historical inference guards intentionally reject
changed source/HEAD or changed artifact bytes. The lightweight results verifier
above is intended for inspecting the published archive after release.

## Repository map

```text
src/taimma/              Models, data splitting, training, evaluation, experiments
src/metadata_pipeline/   Metadata extraction, controls, grouping, validation
scripts/                Experiment, figure, finalization, and publication CLIs
configs/                Scientific configs, including unchanged historical configs
tests/                  CPU fixtures and regression tests
results/paper/          Verified, compact publication summaries and provenance
docs/                   Reproduction and release documentation
```

The current early-UASF execution path is `Memo0711.gate_run` →
`SelectiveGate` / `train_gate` → `ModelAdapter` / `SeparateID`.
`src/taimma/models/uasf.py` retains the earlier late-fusion UASF implementation
for historical work; it is not the early-UASF implementation used for the final table.

Local `data/`, `artifacts/metadata/`, `runs/`, and historical reports are excluded
from the source release. They are retained on the maintainer's machine.
The public summary bundle is intentionally not a complete inference archive.

## Figures

Figure scripts read existing CSV/JSON measurements and save vector PDFs.
See [the figure guide](README_PAPER_FIGURES.md). They do not generate new
experimental measurements. Keep the three-seed diagnostic figures separate
from the five-seed Beauty selection table.

## Citation and license

Software citation metadata is provided in [CITATION.cff](CITATION.cff).
No paper DOI, archive DOI, or publication date is invented in this checkout.
Original source code is released under the [MIT License](LICENSE).
Dataset, pretrained-model, and other third-party rights remain with their owners;
see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Historical documentation: [Checkpoint 1](docs/history/README_CHECKPOINT1.md),
[Checkpoint 2](CHECKPOINT2.md), [late rerun](README_LATE_CHECKPOINT.md),
[memo protocol](README_MEMO0711.md), and [finalization](README_FINAL_TEST.md).
These documents preserve the original execution context, including old paths.
