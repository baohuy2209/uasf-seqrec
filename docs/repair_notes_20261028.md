# TAIMMA implementation repairs on October 28

These changes repair the independently reproduced runtime and protocol defects.
They do not certify the real pilot or claim that Diagnostic B/UASF experiments
have been completed. Existing checkpoints, metrics and registry rows are preserved.

## Execution and artifacts

Both runners require an explicit real-input or synthetic-fixture mode. Pilot
flags `--real_data` and `--smoke_test` are mutually exclusive. An experiment YAML
must exist and be a mapping; explicit CLI arguments take precedence. Dataset and
model config paths resolve relative to `notebooks/taimma`, and a relative raw
data location resolves from the repository. The pilot is explicitly Beauty-only.

Pilot supports `--runs_dir` and `--results_csv` for isolated verification. Model
architecture, evaluation masking, batch sizes, optimizer settings and seed are
recorded in the effective config. Requesting unavailable CUDA raises an error;
use `--device cpu` explicitly instead of silently changing devices.

Run names include microseconds and a unique suffix. Existing run artifacts and
processed artifacts are never overwritten. Real pilot splits and mappings are
saved in `data/taimma/processed/beauty_2014/<run_id>` with source/split hashes.
Validation/test ranks and complete histories are saved separately. A repeat
training pass gets its own checkpoint and reports under `<run>/repeat/`.

`--verify_reproducibility` compares complete histories, best epoch, validation
and test metrics, and keyed per-user ranks. The tolerance is explicit via
`--repro_tolerance` (default absolute 1e-6, relative tolerance zero). This compares
same-seed runs; it does not replace the required multi-seed research analysis.

Real pilot benchmark comparison remains PENDING until a matched reference
implementation/config/commit is actually verified. Shuffling reports REVIEW:
it preserves empirical target popularity and has no invented acceptance cutoff.
A failed sanity check takes precedence over pending checks. Fixture runs are
recorded as SMOKE_TEST and cannot certify research acceptance.

## Delta interchange

CSV/parquet wide tables require `user_idx`, real `seq_len`, explicit `feat_*`
columns, and complete pairs `delta_<source>_seed1/seed2`. Supply `context_id`
when a user has multiple contexts. Long tables instead require `source_name`,
`reference_seed`, `delta` and exactly two reference seeds. Sources are Category,
Brand and Title. Every source/seed must cover the same context/user cohort.

Long-format features may differ between sources but must agree across seeds for
the same source/context. User IDs, lengths, features and deltas are aligned by
one sorted `(user_idx, context_id)` index. Duplicates, missing fields, inconsistent
lengths, incomplete seed cohorts and nonfinite values raise errors. Lengths are
never replaced by constants. Feature column zero is always log of the supplied
real sequence length; remaining named feature columns have a deterministic order.

NPZ input requires `context_features`, `user_ids`, `seq_lens` and complete delta
seed pairs; its first feature must be log(seq_lens). `feature_names` and
`group_indices` may be provided. Array rows must already represent the same
context across the two seed slots; preserve the external split/checkpoint lineage.

The utility producer now creates stable context hashes (or accepts caller-supplied
context IDs) and exports actual history lengths and source features, including
Shannon entropy. It does not reset context IDs to a batch-local row number.
Callers with distinct repeated observations of identical user/history/target
must supply globally unique context IDs. Real experiments must retain checkpoint,
dataset, split and reference-seed provenance alongside the resulting delta file.

## E0/E1 and fallback

Noise thresholds and label stability are measured from the two actual reference
seeds. Stability includes a user-clustered Spearman confidence interval. E0
resamples users when multiple contexts are supplied; quintile extremes are
disjoint and equal-length groups do not certify a length effect.

Contextual evaluation uses seed 1 when rho >= 0.3. The intermediate regime uses
two-seed averaged deltas consistently for E0/E1. `make_utility_labels` supplies
hard labels or the prescribed sigmoid soft labels with an explicitly supplied
positive temperature for later S3 training; this is not a completed S3 trainer.

For rho < 0.1, a real group fallback requires `group_idx` in CSV/parquet or
`group_indices` in NPZ. These must be integer IDs 0..9 encoding
`2 * length_quintile + history_head_tail`, based only on inference-available
history. The producer of these groups must lock and record its Train-base
quintile/head-tail rules. Do not group by the unknown future target's popularity.
The runner does not fabricate a group definition when this information is absent.

Group E1 predicts held-out users using group label means fitted only on training
users. It is compared with the length-only baseline under the same user folds.
Unseen training groups, single-class folds or insufficient users are recorded
as NOT_EVALUABLE with undefined metrics represented by JSON null, not artificial
AUC=0.5. AUC and OOF Spearman include user-clustered bootstrap confidence intervals.

The decision engine requires same-source evidence and the correct evaluated
regime. Missing or undefined evidence produces PENDING rather than a scientific
NO-GO. CONDITIONAL requires evaluated averaging/group fallback and recommends
the group-level gate specified in the October 26 memo. The JSON report records
the actual data mode, file checksum, seed, bootstrap count and per-source evidence.
Existing JSON reports are not overwritten.

## Metadata and controls

Category label embeddings are averaged over non-padding labels in both early
and late fusion, for histories and candidates. Brand values are case-folded and
whitespace-normalized; Unknown/empty are missing, Generic is retained. The Title
control preserves full covariance with explicit 1e-6 diagonal stabilization,
including rank-deficient covariance; fewer than two titles is an explicit error.

Popularity and head/tail require complete Train-base events. Truncated prefix
samples cannot safely reconstruct frequencies and no longer provide a fallback.
Context rank degradation uses the same evaluator/tie/masking policy as ranking.
`save_control_artifact` persists shared masks/permutations/random controls with
dataset/source/type/rate/seed in the name and a checksum manifest, without overwrite.

## Verification and remaining work

The regression suite covers both complete fixture entrypoints, repeat artifacts,
CLI/config precedence, delta row/source alignment and invalid inputs, utility
producer-to-loader interchange, multilabel late fusion, all cutoff OOV misses,
device JSON serialization, immutable artifacts, full covariance, undefined AUC,
and held-out-user group fallback.

Real metadata files, frozen Title embeddings, matched benchmark reproduction,
real multi-seed pilot acceptance, Diagnostic B grids, saved real controls/deltas,
S3/baseline/ablation experiments, final statistics and the paper remain research
prerequisites. No synthetic fixture is substituted for these assets or results.
