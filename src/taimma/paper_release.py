"""Export and verify the published summary bundle; never fit or load a model."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import statistics
from typing import Any

VERSION = "1.0.0"
DATASETS = ("beauty", "sports", "toys")
SOURCES = ("category", "brand", "title")
SUBSETS = ("ID", "C", "B", "T", "CB", "CT", "BT", "CBT")
BASELINES = ("all_fixed", "global_subset", "group_subset", "global_gate",
             "unsupervised", "softmax_null", "independent_ID", "early_no_sd",
             "id_plus_meta", "early_plus_meta")
METHODS = BASELINES + ("uasf",)
# Audited publication-only change to source-archive provenance, not numerical code.
LOGGER_RELEASE = ("500e38d9dd8fae0d1fbf333b8a52ba873a42cff585e9c1c3fcfd370dd2d88970",
                  "f271f84a049a941c6c31029ff12163e6de8a5eaab3466e713cdd3c120c1c6c90")


def _path(root: Path, relative: str) -> Path:
    part = PurePosixPath(relative)
    if not part.parts or part.is_absolute() or ".." in part.parts or "\\" in relative or ":" in relative:
        raise ValueError(f"Unsafe manifest path: {relative}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Manifest escapes its root: {relative}")
    return path


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _original(path: Path, expected: str) -> bytes:
    data = path.read_bytes()
    if _sha(data) == expected:
        return data
    # Restore known server bytes only when they match the recorded checksum.
    lf = data.replace(b"\r\n", b"\n")
    if _sha(lf) == expected:
        return lf
    raise ValueError(f"Content differs from the recorded artifact: {path}")


def _json(data: bytes) -> Any:
    return json.loads(data.decode("utf-8"))


def _dump(value: Any) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _csv(rows: list[dict[str, Any]]) -> bytes:
    if not rows:
        raise ValueError("Cannot publish an empty table")
    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue().encode("utf-8")


def _rows(data: bytes) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(data.decode("utf-8"))))


def _composition(registry: bytes) -> bytes:
    records = {(row["dataset"], row["job"]): row for row in _rows(registry)}
    rows = []
    for dataset in DATASETS:
        for family in ("early", "late"):
            for subset in SUBSETS:
                jobs = [f"id_{s}" if subset == "ID" else f"{family}_{subset}_{s}" for s in (42, 43, 44)]
                values = [float(records[dataset, job]["test_ndcg_10"]) for job in jobs]
                if not all(math.isfinite(v) and 0 <= v <= 1 for v in values):
                    raise ValueError(f"Invalid diagnostic scores: {dataset}/{family}/{subset}")
                rows.append(dict(dataset=dataset, mechanism=family, subset=subset,
                                 n_seeds=3, mean=statistics.mean(values), sd=statistics.stdev(values)))
    return _csv(rows)


def _ranks(run: Path, dataset: str, job: str, subset: str,
           inputs: dict[str, str], prefix: str) -> dict[int, tuple[int, float]]:
    folder = _path(run, f"{dataset}/jobs/{job}")
    record = _json((folder / "completed.json").read_bytes())
    checkpoint = PurePosixPath(record["checkpoint"])
    rel = (checkpoint.parent / "per_user_test.csv").as_posix()
    if rel in record["artifacts"] and (folder / rel).is_file():
        path = _path(folder, rel)
        expected = record["artifacts"][rel]
    else:
        rel = f"{dataset}/evaluations/{job}/{subset}_test.csv"
        path = _path(run, rel)
        binding = _json(path.with_suffix(".json").read_bytes())
        expected = binding["sha256"]
    data = _original(path, expected)
    inputs[f"{prefix}/{path.relative_to(run).as_posix()}"] = expected
    result = {}
    for row in _rows(data):
        user, target, rank = int(row["user_idx"]), int(row["target_idx"]), int(row["rank"])
        score = float(row["ndcg@10"])
        calculated = 1 / math.log2(rank + 1) if 1 <= rank <= 10 else 0.0
        if user in result or rank < 1 or not math.isclose(score, calculated, abs_tol=1e-12):
            raise ValueError(f"Invalid per-user rank record: {path}")
        if target == 0 and score != 0:
            raise ValueError(f"OOV target scored as a hit: {path}")
        result[user] = (target, score)
    if not result:
        raise ValueError(f"Empty rank archive: {path}")
    return result


def _paired(control: list[dict], authentic: list[dict]) -> tuple[float, float, float]:
    import numpy as np

    keys = sorted(authentic[0])
    differences = []
    targets = [authentic[0][key][0] for key in keys]
    for a, b in zip(control, authentic, strict=True):
        if sorted(a) != keys or sorted(b) != keys:
            raise ValueError("Control/reference users differ")
        if [a[k][0] for k in keys] != targets or [b[k][0] for k in keys] != targets:
            raise ValueError("Control/reference target pairing differs")
        differences.append([a[k][1] - b[k][1] for k in keys])
    delta = np.mean(differences, axis=0)
    rng = np.random.default_rng(42)
    means = [float(delta[rng.integers(len(delta), size=len(delta))].mean()) for _ in range(1000)]
    low, high = np.percentile(means, [2.5, 97.5])
    return float(delta.mean()), float(low), float(high)


def _correspondence(run: Path, early: Path, inputs: dict[str, str]) -> bytes:
    authentic = [_ranks(run, "beauty", f"early_CBT_{s}", "CBT", inputs, "memo") for s in (42, 43, 44)]
    rows = []
    for source in SOURCES:
        for kind in ("shuffled", "random"):
            control = [_ranks(early, "beauty", f"control_{kind}_{source}_{s}", "CBT", inputs, "early") for s in (42, 43, 44)]
            mean, low, high = _paired(control, authentic)
            rows.append(dict(model="CBT", source=source, replacement=kind,
                             difference=mean, ci_lower=low, ci_upper=high, n_seeds=3,
                             resamples=1000, direction="control_minus_authentic"))
    reference = [_ranks(run, "beauty", f"early_T_{s}", "T", inputs, "memo") for s in (42, 43, 44)]
    shuffled = [_ranks(run, "beauty", f"title_shuffled_{s}", "T", inputs, "memo") for s in (42, 43, 44)]
    mean, low, high = _paired(shuffled, reference)
    rows.append(dict(model="T-only", source="title", replacement="shuffled",
                     difference=mean, ci_lower=low, ci_upper=high, n_seeds=3,
                     resamples=1000, direction="control_minus_authentic"))
    return _csv(rows)


def _check_cpu_tables(run: Path, freeze: dict, files: dict[str, bytes], inputs: dict[str, str]) -> None:
    """CPU additions were not themselves frozen; verify their frozen inputs and recompute."""
    cpu = _json(files["run/finalization/cpu_analysis.json"])
    if cpu.get("training") is not False or cpu.get("inference") is not False:
        raise ValueError("CPU table export must not be a model execution")
    for rel, expected in cpu["inputs_sha256"].items():
        if freeze["files"].get(rel) != expected:
            raise ValueError(f"CPU input is not bound to the original freeze: {rel}")
        _original(_path(run, rel), expected)
        inputs["memo/" + rel] = expected
    intervals = _rows(files["run/finalization/rq1_paired_ci.csv"])
    required = {(d, f"{m}_CBT minus ID") for d in DATASETS for m in ("early", "late")}
    if len(intervals) != 6 or {(r["dataset"], r["contrast"]) for r in intervals} != required:
        raise ValueError("Six primary RQ1 intervals are required")
    for row in intervals:
        dataset = row["dataset"]
        family = "early" if row["contrast"] == "early_CBT minus ID" else "late"
        lhs = [_ranks(run, dataset, f"{family}_CBT_{s}", "CBT", inputs, "memo") for s in (42, 43, 44)]
        rhs = [_ranks(run, dataset, f"id_{s}", "ID", inputs, "memo") for s in (42, 43, 44)]
        computed = _paired(lhs, rhs)
        if any(not math.isclose(a, float(row[k]), abs_tol=1e-12) for a, k in zip(computed, ("difference", "ci_lower", "ci_upper"))):
            raise ValueError(f"RQ1 interval differs from frozen ranks: {dataset}/{family}")
    positives = {(int(r["seed"]), r["source"]): r for r in _rows(files["run/finalization/positive_labels.csv"])}
    if set(positives) != {(s, source) for s in range(42, 47) for source in SOURCES}:
        raise ValueError("Positive-label summaries require all five seeds and three sources")
    for seed in range(42, 47):
        rel = f"beauty/per_seed_s2/early_sd_{seed}/labels.csv"
        rows = _rows(_original(run / rel, freeze["files"][rel]))
        for source in SOURCES:
            positive = sum(float(r[source]) == 1.0 for r in rows)
            recorded = positives[seed, source]
            if (int(recorded["contexts"]) != len(rows) or int(recorded["positive_count"]) != positive
                    or not math.isclose(float(recorded["positive_fraction"]), positive / len(rows), abs_tol=1e-12)):
                raise ValueError(f"Positive-label fraction differs: {seed}/{source}")


def _validate_table(data: bytes) -> None:
    rows = _rows(data)
    if len(rows) != len(METHODS) or {r["method"] for r in rows} != set(METHODS):
        raise ValueError("The published Test family must contain 10 baselines plus UASF")
    for row in rows:
        if int(row["n_seeds"]) != 5:
            raise ValueError("Beauty selection comparisons require all five seeds")
        for name, value in row.items():
            if name.endswith(("_mean", "_sd")):
                number = float(value)
                if not math.isfinite(number) or number < 0 or (name.endswith("_mean") and number > 1):
                    raise ValueError(f"Invalid summary value: {name}")


def _validate_release(files: dict[str, bytes]) -> None:
    release = _json(files["run/finalization/release_complete.json"])
    freeze_bytes = files["run/freeze.json"]
    freeze = _json(freeze_bytes)
    if release.get("status") != "COMPLETE" or release.get("training") is not False:
        raise ValueError("The original frozen Test release is not complete")
    if _sha(freeze_bytes) != release["freeze_sha256"]:
        raise ValueError("Release/freeze binding differs")
    if freeze.get("framing") != "F2" or freeze.get("test_used_for_framing") is not False:
        raise ValueError("Published framing must remain the locked validation F2 decision")
    if freeze.get("baselines") != list(BASELINES) or release.get("methods") != list(METHODS):
        raise ValueError("Locked comparison family differs")
    _validate_table(files["run/finalization/table5_validation_test.csv"])
    alpha = _json(files["run/finalization/alpha_complete.json"])
    if (alpha.get("used_for_selection") is not False or alpha.get("split") != "validation"
            or alpha.get("posthoc") is not True or alpha.get("training") is not False
            or alpha.get("freeze_sha256") != release["freeze_sha256"]):
        raise ValueError("Alpha analysis must remain posthoc validation evidence")
    for rel, data in files.items():
        if not rel.startswith("run/"):
            continue
        source = rel.removeprefix("run/")
        expected = release["outputs_sha256"].get(source, alpha["outputs_sha256"].get(source, freeze["files"].get(source)))
        if expected and _sha(data) != expected:
            raise ValueError(f"Summary differs from the original scientific manifest: {source}")
    for dataset in DATASETS:
        protocol = _json(files[f"run/{dataset}/protocol.lock.json"])
        for name in ("statistics.json", "coverage.json"):
            if _sha(files[f"metadata/{dataset}/{name}"]) != protocol["data"]["metadata_files"][name]:
                raise ValueError(f"Metadata summary differs: {dataset}/{name}")


def verify_bundle(destination: Path) -> dict[str, Any]:
    """Verify a summary package on CPU without raw data, weights, Git, or torch."""
    destination = destination.resolve()
    manifest = _json((destination / "release_manifest.json").read_bytes())
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported summary manifest")
    files = {}
    for rel, expected in manifest["files"].items():
        data = _path(destination, rel).read_bytes()
        if _sha(data) != expected:
            raise ValueError(f"Published summary changed: {rel}")
        files[rel] = data
    _validate_release(files)
    return dict(status="PASS", version=manifest["version"], files=len(manifest["files"]),
                framing="F2", training=False, inference=False,
                scope="Published summaries and their provenance; not a replay of model inference")


def export_bundle(repo: Path, run: Path, early: Path, metadata: Path,
                  destination: Path) -> dict[str, Any]:
    """Copy verified summaries and aggregate existing ranks into publication tables."""
    repo, run, early, metadata, destination = [p.resolve() for p in (repo, run, early, metadata, destination)]
    for source in (run, early, metadata):
        if destination.is_relative_to(source) or source.is_relative_to(destination):
            raise ValueError("Publication output must be separate from every source archive")
    if destination.exists():
        raise FileExistsError("Publication output exists; verify it or choose a new destination")
    release_data = (run / "finalization/release_complete.json").read_bytes()
    release = _json(release_data)
    freeze_data = _original(run / "freeze.json", release["freeze_sha256"])
    freeze = _json(freeze_data)
    alpha_data = (run / "finalization/alpha_complete.json").read_bytes()
    alpha = _json(alpha_data)
    files = {"run/freeze.json": freeze_data,
             "run/finalization/release_complete.json": release_data.replace(b"\r\n", b"\n"),
             "run/finalization/alpha_complete.json": alpha_data.replace(b"\r\n", b"\n")}
    source_inputs: dict[str, str] = {}
    support_changes = {}
    for rel, expected in freeze["source_sha256"].items():
        path = _path(repo, rel)
        try:
            _original(path, expected)
        except ValueError:
            # Publication bumps only package metadata; all other source bytes stay locked.
            data = path.read_bytes().replace(b"\r\n", b"\n")
            previous = data.replace(b'__version__ = "1.0.0"', b'__version__ = "0.1.0"', 1)
            version_only = rel == "src/taimma/__init__.py" and _sha(previous) == expected
            archive_logging = rel == "src/taimma/experiments/logger.py" and (expected, _sha(data)) == LOGGER_RELEASE
            if not version_only and not archive_logging:
                raise
            support_changes[rel] = dict(original_sha256=expected, published_sha256=_sha(data),
                reason="Package version only: 0.1.0 to 1.0.0" if version_only else "Explicit Git-unavailable provenance for source archives")
    summaries = ["results.csv", "beauty/framing_decision.json", "beauty/selected_s2/e0_e1_CE.json",
                 "beauty/title_only_comparison.json"]
    summaries += [f"finalization/{name}" for name in (
        "cpu_analysis.json", "rq1_paired_ci.csv", "positive_labels.csv", "table5_validation_test.csv",
        "test_paired_ci.csv", "test_family_comparisons.json", "validation_alpha_summary.csv")]
    for dataset in DATASETS:
        summaries += [f"{dataset}/{name}" for name in (
            "protocol.lock.json", "group_analysis.json", "robustness.csv",
            "robustness_diagnostics.csv", "late_rank_summary.csv")]
    for rel in summaries:
        expected = release["outputs_sha256"].get(rel, alpha["outputs_sha256"].get(rel, freeze["files"].get(rel)))
        if expected is None:
            if rel not in ("finalization/cpu_analysis.json", "finalization/rq1_paired_ci.csv", "finalization/positive_labels.csv"):
                raise ValueError(f"Summary lacks original provenance: {rel}")
            files["run/" + rel] = _path(run, rel).read_bytes().replace(b"\r\n", b"\n")
        else:
            files["run/" + rel] = _original(_path(run, rel), expected)
    for dataset in DATASETS:
        protocol = _json(files[f"run/{dataset}/protocol.lock.json"])
        for name in ("statistics.json", "coverage.json"):
            expected = protocol["data"]["metadata_files"][name]
            files[f"metadata/{dataset}/{name}"] = _original(metadata / dataset / name, expected)
    files["tables/source_composition.csv"] = _composition(files["run/results.csv"])
    _check_cpu_tables(run, freeze, files, source_inputs)
    files["tables/correspondence.csv"] = _correspondence(run, early, source_inputs)
    files["table_inputs.json"] = _dump(dict(inputs_sha256=source_inputs,
        source_composition="run/results.csv", ci_scope="Unadjusted paired-user percentile intervals; matched seeds averaged within user"))
    manifest = dict(schema_version=1, version=VERSION, framing="F2",
        scope="Summary evidence only; raw reviews, metadata embeddings, per-user predictions and model weights are not included",
        original_freeze_tag=freeze["git_tag"], original_freeze_commit=freeze["git"]["commit"],
        frozen_source_sha256=freeze["source_sha256"],
        publication_support_changes=support_changes,
        files={rel: _sha(data) for rel, data in sorted(files.items())})
    # Validate everything before creating the destination; never overwrite a frozen run.
    _validate_release(files)
    destination.mkdir(parents=True)
    for rel, data in files.items():
        path = _path(destination, rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    (destination / "release_manifest.json").write_bytes(_dump(manifest))
    return verify_bundle(destination)
