# Frozen-ID late checkpoint rerun

This entry point reruns only the late reference. It uses the existing model and
the unchanged checkpoint2 training/evaluation loop. It never launches early,
training contrasts, K1--K4, S2/E1, or S3.

Per dataset: ID seeds 42/43/44 plus seven nonempty metadata subsets per seed,
24 fits in total. The late reference freezes the selected same-seed ID model,
uses Train-base, residual dropout 0.2 per batch, lr 0.001, and no weight decay.
Selection is maximum full-precision validation NDCG@10 among trained epochs,
starting at epoch 1; pre-training validation is logged separately.

## Keep the running early checkout unchanged

Commit and push the reviewed patch from the Windows checkout first. Include the
three existing Python files in `src/taimma/data/`; the old `data/` ignore pattern
hid these source dependencies. Their contents are not changed by this patch.
The corrected `/data/` rule still ignores raw datasets at repository root.

On the server, fetch and create a separate checkout. Fetching and adding a
worktree do not check out a different commit in the active early directory.
Use the exact hash of the reviewed late commit if origin/main has later changes.

```bash
cd /home/ubuntu/projects/uasf-seqrec
git fetch origin
git worktree add --detach /home/ubuntu/projects/uasf-seqrec-late-20261106 origin/main
cd /home/ubuntu/projects/uasf-seqrec-late-20261106
mkdir -p data
ln -s /home/ubuntu/projects/uasf-seqrec/data/raw data/raw
source /home/ubuntu/projects/uasf-seqrec/.venv/bin/activate
python scripts/run_late_checkpoint.py --stage check-data --datasets beauty
```

The new worktree must contain the exact metadata artifacts, including
`title_embeddings.npy`, referenced by each dataset manifest. Data checks verify
these hashes; do not regenerate metadata to bypass a mismatch. No dependency
installation or library upgrade is needed. Use the existing locked RTX4060
environment. A missing runtime data symlink or missing artifacts is a data
provisioning issue, not a reason to train with a different mapping.

If there is one GPU and early is still using it, run the training command after
early finishes. A separate checkout isolates files and protocol locks, not GPU
memory. Do not pull/checkout/commit in the active early working tree mid-run.

## Run Beauty only

```bash
python -u scripts/run_late_checkpoint.py --stage run --datasets beauty \
  --output runs/checkpoint2_late_20261106 --execute-training
```

For the complete three-dataset late benchmark, use this command instead. With
the same output, already completed Beauty jobs are verified and skipped.

```bash
python -u scripts/run_late_checkpoint.py --stage run --datasets beauty sports toys \
  --output runs/checkpoint2_late_20261106 --execute-training
```

## Cache and interruptions

Completed jobs are reused only when source/config/data/runtime, artifact hashes
and same-seed ID checkpoint binding agree. Changing scientific inputs requires
a fresh output. A clean results-only commit can change HEAD without invalidating
the source-based protocol. Keep `.pt` checkpoints on the server: the late jobs
need their ID checkpoint, and the cache validates all completed artifacts.

An interrupted incomplete job restarts from epoch 1 in a new attempt; this is
not optimizer-state resume. A process killed without cleanup can leave
`late_execution.lock` or a job `execution.lock`. Confirm no process owns it
before manually removing only that stale lock. Never delete completed attempts.

New output paths matching `/runs/checkpoint2_late_*/` are ignored to prevent
generated logs from tripping clean-tree guards during execution. Keep them on
disk; selected lightweight report/CSV/JSON outputs can be force-added later.
Do not commit results while another active checkout relies on a fixed HEAD.

## Outputs

Under `runs/checkpoint2_late_20261106/<dataset>/`:

- `jobs/*/attempt_*/`: checkpoint, history, run log, config, validation/test ranks,
  source snapshot and manifest.
- `jobs/late_*/id_dependency.json`: exact same-seed ID checkpoint hash.
- `jobs/late_*/late_verification.json`: frozen-ID identity and checkpoint selection.
- `late_per_seed.csv`, `late_summary.csv`: 8 subsets, NDCG@10 mean/sample SD over
  3 seeds, with per-seed Recall@10.
- `late_add_drop.json`: Add(source) = single-source minus ID; Drop(source) =
  full minus without-source. Paired-user percentile CI retains the 3 seed
  predictions together and is conditional on those training seeds.
- `late_group_analysis.json`, `group_late_<seed>.csv`: global subset selected on
  validation, validation-selected group subsets, and descriptive test oracle.
  These use only this new late grid, never legacy checkpoint1 ranks.
- `protocol.lock.json`, `origin_git.json`, `groups*.json`, `late_report.json`:
  immutable scientific input bindings and the completed result summary.

## Compare against the completed early grid

After early has finished, this reads ranks without training or model inference:

```bash
python scripts/run_late_checkpoint.py --stage report --datasets beauty \
  --output runs/checkpoint2_late_20261106 \
  --early-output /home/ubuntu/projects/uasf-seqrec/runs/checkpoint2_full_20261106
```

`early_minus_late.json` checks dataset/catalog/context signatures and artifact
hashes, pairs matching users/targets/seeds and reports early minus late per
subset. This does not edit early outputs or rewrite the existing mini report.
Keep checkpoint1 and K1--K4 as separately labelled historical/diagnostic results.
