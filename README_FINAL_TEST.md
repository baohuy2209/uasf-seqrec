# Final F2 evaluation — feedback 12 November

This runner finishes the existing study without training. It keeps the original protocol/configuration and validation decision unchanged. Default stage `plan` does nothing. Do not use `--stage all` on the old training runner for this task.

## Scope

1. CPU: paired-user CIs for early CBT minus ID and late CBT minus ID, three datasets, seeds 42–44; original positive-label fractions per source and seed.
2. Freeze the reviewed code/config, groups, validation selections, baseline family, data/cache manifests and checkpoint SHA256 values. A Git tag must already point to the current reviewed commit.
3. Release all ten locked baseline labels plus UASF, Beauty seeds 42–46, together. The displayed ten-row table combines all-fixed/global-subset only if their outputs are actually identical. No method is dropped from the machine-readable 11-label family.
4. Optional posthoc validation alpha distribution from saved UASF gates and cached validation features. No target labels enter the gate. No backbone forward or training is needed for this analysis.

Framing remains F2 irrespective of Test rankings. The new runner never calls a fitter; its run class rejects training. Gate Test evaluation loads saved feature columns/scaler directly from `gate.pt`, avoiding unnecessary Train-meta inference. All-off routing remains the separate same-seed Train-base ID model.

## Server paths — preflight first

Run from the checkout containing your completed v3. For the existing server setup:

```bash
cd /home/ubuntu/projects/uasf-seqrec
source .venv/bin/activate
git pull --ff-only origin main

python scripts/finalize_memo0711.py --stage preflight --run-root runs/memo0711_v3
```

`preflight` is CPU/read-only: no optimizer, no model forward, no GPU evaluation. It enumerates every completed S1 and gate checkpoint in the run and verifies its exact manifest-relative path and SHA256; 45 are the direct Test dependencies and the other archived models are checked too. When those exist, it also runs the original full `validate_ready` checks. The Windows copy without `.pt` will correctly report `INCOMPLETE`; it is not a request to retrain.

The code reads `runs/memo0711_v3/beauty/protocol.lock.json` to preserve the original scientific config/signatures. It loads checkpoints only under `--run-root`, using `completed.json` (`checkpoint` or `attempt/gate.pt`). It does not depend on stale absolute `input_roots` from another worktree, search for a vaguely similar checkpoint, change protocol locks, or copy weights from a different experiment. If v3 is elsewhere, change `--run-root` to that actual dedicated directory under the checkout's `runs/`.

## CPU additions

```bash
python scripts/finalize_memo0711.py --stage cpu --run-root runs/memo0711_v3
```

This reads 27 existing diagnostic rank CSVs and five original label CSVs and writes only `runs/memo0711_v3/finalization/`. It does not open UASF Test. Before freeze it is permissible; after freeze it does not alter the frozen input files either.

## Commit before tag; early scheduling is explicit

Commit and push the reviewed finalization code first. Inspect `git status --short`; pre-existing source edits or outputs outside the narrow new-runtime allowlist must also be resolved. Do not `reset --hard`, force-add checkpoints, or hide code changes in Git exclusions.

Before server freeze, stage only the reviewed code/docs/tests, not `runs/memo0711_v3/finalization/`. The local CPU outputs there are allowed untracked and can be regenerated on the server. Publish results in a separate commit only after release and optional alpha are finished.

On the server, after a successful preflight and reviewing the pulled commit:

```bash
git tag -a taimma-f2-freeze-20261112 -m "Freeze reviewed F2 validation study before final Test"
git push origin taimma-f2-freeze-20261112

python -u scripts/finalize_memo0711.py \
  --stage freeze \
  --run-root runs/memo0711_v3 \
  --freeze-tag taimma-f2-freeze-20261112 \
  --early-release-reason "User requested advancement from 13 November to 12 November after feedback; no changes to models, labels, hyperparameters or F2"
```

The supervisor scheduled 13 November; the user's instruction advances execution. The explicit reason is preserved in `freeze.json`. This is not recorded as supervisor approval of the earlier date. The original `test_release_not_before` config value is never edited. On/after the planned date the reason flag can be omitted. Without the flag an early freeze is refused.

If the tag already exists, inspect its commit and reuse it only if correct; do not overwrite/move it. A historical incompatible `freeze.json` is not overwritten. The runner will stop and require inspection.

## One frozen release, resumable after interruption

```bash
python -u scripts/finalize_memo0711.py \
  --stage release-test \
  --run-root runs/memo0711_v3 \
  --execute-inference
```

Use the locked RTX 4060 environment for this stage. No `--execute-training` flag exists. Validation subset choices, gate checkpoints/scalers, tau/kappa and losses remain unchanged. Existing diagnostic ranks may be reused when their bindings match; new Test ranks are cached with the frozen model signature.

All 11 method labels must finish before the final table and completion marker are published. If interrupted, rerun the same command: verified caches resume the same fixed evaluation. After `release_complete.json` exists, rerunning verifies output hashes and returns without inference. If a lock survives a hard kill, inspect its PID and the server process before removing that specific stale lock; never delete checkpoints or entire runs to resume.

Do not modify source/config or create a new commit between freeze, release and optional alpha. Only expected new untracked finalization/Test cache files are permitted while running; tracked modifications and unrelated untracked files still block. This avoids the historical self-generated-lock/dirty-tree error without ignoring experiment outputs globally.

### Final outputs

Under `runs/memo0711_v3/finalization/`:

- `rq1_paired_ci.csv`: six primary RQ1 contrasts with 1,000 paired-user percentile intervals, seeds held together per user. These intervals are unadjusted and conditional on the fitted models.
- `positive_labels.csv`: original hard-label positive fractions, five seeds, C/B/T. Not the E0 fraction of averaged reference utilities.
- `table5_validation_test.csv`: complete Val/Test means and sample SD for NDCG/Recall @10/20 for every frozen method label.
- `test_paired_ci.csv`: UASF minus each of the ten baselines, exact user/target/seed pairing.
- `test_family_comparisons.json`: existing locked bootstrap/Holm family and TOST (1% relative margin). A finite bootstrap count of zero is not an exact probability zero.
- `release_complete.json`: use the table only when this says `COMPLETE`; it binds the freeze and every exported Test cache/table by SHA256.

No Test-based model selection, relabeling, threshold tuning, framing recomputation, extra K4, transfer fitting, or random-control refitting is performed. Reporting a different Test ordering does not change F2.

## Optional posthoc gate weights

After freeze, preferably after the final Test table:

```bash
python -u scripts/finalize_memo0711.py \
  --stage alpha \
  --run-root runs/memo0711_v3 \
  --execute-inference
```

This gate-only forward uses CPU and the frozen cached validation features, gate state and training scaler. It writes `validation_alpha_per_user.csv`, `validation_alpha_summary.csv`, and `alpha_complete.json`. Both continuous beta and post-threshold alpha are retained, with source-off and all-off frequencies. This is posthoc description, not a test of new labels or a tuning step. No claim about the hypothesis is made until these outputs are inspected.

## Tests

```bash
python -m pytest tests/test_memo0711_final.py tests/test_memo0711.py tests/test_memo0711_repairs.py -q
```

Original training/model files are untouched. Real checkpoint byte verification and final Test execution remain server tasks; local CPU fixtures do not establish that missing server files exist.
