"""CSV-only paper Figures 2 and 3. No model loading, training or inference."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
try:
    from taimma.paths import resolve_project_path
except ImportError:
    # Staging/standalone use requires explicit absolute input/output paths.
    def resolve_project_path(value: str | Path) -> Path:
        return Path(value).expanduser().resolve()

DATASETS = ("beauty", "sports", "toys")
SEEDS = (42, 43, 44)
SOURCES = ("C", "B", "T")
SUBSETS = ("ID", "C", "B", "T", "CB", "CT", "BT", "CBT")
RATES = (0., .2, .4, .6, .8)
LONG_SOURCE = {"C": "category", "B": "brand", "T": "title"}


def csv_rows(path: Path, required: set[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"Missing CSV columns {sorted(required)}: {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    return rows


def metric_mean(path: Path) -> tuple[tuple[tuple[int, int], ...], float]:
    rows = csv_rows(path, {"user_idx", "target_idx", "rank", "ndcg@10"})
    pairs, values = [], []
    for row in rows:
        user, target, rank = (int(row[k]) for k in ("user_idx", "target_idx", "rank"))
        value = float(row["ndcg@10"])
        expected = 1 / math.log2(rank + 1) if target > 0 and 1 <= rank <= 10 else 0.
        if user < 0 or target < 0 or rank < 1 or not math.isfinite(value) or abs(value - expected) > 1e-10:
            raise ValueError(f"Invalid rank/NDCG record: {path}, user={user}")
        pairs.append((user, target)); values.append(value)
    if len({u for u, _ in pairs}) != len(pairs):
        raise ValueError(f"Duplicate users: {path}")
    return tuple(sorted(pairs)), statistics.mean(values)


def rank_path(root: Path, dataset: str, job: str) -> Path:
    folder = root / dataset / "jobs" / job
    record = json.loads((folder / "completed.json").read_text(encoding="utf-8"))
    relative = Path(record["checkpoint"]).parent / "per_user_test.csv"
    path = (folder / relative).resolve()
    if not path.is_relative_to(folder.resolve()) or not path.is_file():
        raise ValueError(f"Missing/unsafe diagnostic test rank path: {path}")
    return path


def aggregate_effects(root: Path, inputs: set[Path]) -> list[dict]:
    output = []
    for dataset in DATASETS:
        values = {}; cohort = None
        for seed in SEEDS:
            for family, subset in [("id", "ID")] + [(f, s) for f in ("early", "late") for s in SUBSETS[1:]]:
                job = f"id_{seed}" if subset == "ID" else f"{family}_{subset}_{seed}"
                path = rank_path(root, dataset, job)
                pairs, mean = metric_mean(path); inputs.add(path)
                if cohort is not None and pairs != cohort:
                    raise ValueError(f"Unpaired user/target cohort: {path}")
                cohort = pairs
                values[family, subset, seed] = mean
        for family in ("early", "late"):
            for source in SOURCES:
                without = "".join(s for s in SOURCES if s != source)
                add = [values[family, source, s] - values["id", "ID", s] for s in SEEDS]
                drop = [values[family, "CBT", s] - values[family, without, s] for s in SEEDS]
                output.append(dict(dataset=dataset, mechanism=family, source=source,
                    add_mean=statistics.mean(add), add_sd=statistics.stdev(add),
                    drop_mean=statistics.mean(drop), drop_sd=statistics.stdev(drop), n_seeds=3))
    return output


def aggregate_robustness(root: Path, inputs: set[Path]) -> list[dict]:
    output = []
    for dataset in DATASETS:
        path = root / dataset / "robustness.csv"
        rows = csv_rows(path, {"seed", "model", "source", "missing_rate", "ndcg_10"})
        inputs.add(path); values = {}
        for row in rows:
            key = (int(row["seed"]), row["model"], row["source"], float(row["missing_rate"]))
            value = float(row["ndcg_10"])
            if key in values or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Duplicate/invalid robustness record: {path}, {key}")
            values[key] = value
        expected = {(s, model, source, rate) for s in SEEDS for model in ("early_CBT", "early_sd")
                    for source in LONG_SOURCE.values() for rate in RATES}
        if set(values) != expected:
            raise ValueError(f"Incomplete/extra robustness grid: {path}; expected 90 records")
        for model in ("early_CBT", "early_sd"):
            for source in LONG_SOURCE.values():
                for rate in RATES:
                    numbers = [values[s, model, source, rate] for s in SEEDS]
                    output.append(dict(dataset=dataset, model=model, source=source, missing_rate=rate,
                        mean=statistics.mean(numbers), sd=statistics.stdev(numbers), n_seeds=3))
    return output


def export(fig: plt.Figure, base: Path) -> None:
    # Apply the requested tight PDF crop consistently to all preview formats.
    fig.canvas.draw()
    for extension in ("pdf", "svg", "png"):
        fig.savefig(base.with_suffix("." + extension), format=extension, dpi=300,
                    facecolor="white", bbox_inches="tight", pad_inches=0.02)
    rectangles = [list(ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted()).bounds) for ax in fig.axes]
    base.with_suffix(".layout.json").write_text(json.dumps({"axes_bounds_inches": rectangles,
        "figure_inches": list(fig.get_size_inches()), "uncertainty": "sample SD across seeds 42,43,44; not CI"}, indent=2))
    plt.close(fig)


def plot_effects(rows: list[dict], base: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.1), sharey=True)
    fig.subplots_adjust(left=.10, right=.985, bottom=.21, top=.76, wspace=.16)
    styles = [("early", "add", -.24, "#236A8B", "o", "Early: stand-alone"),
              ("early", "drop", -.08, "#236A8B", "s", "Early: conditional"),
              ("late", "add", .08, "#A36A42", "o", "Late: stand-alone"),
              ("late", "drop", .24, "#A36A42", "s", "Late: conditional")]
    handles = []
    for index, (ax, dataset) in enumerate(zip(axes, DATASETS)):
        for family, effect, offset, color, marker, label in styles:
            selected = [next(r for r in rows if (r["dataset"], r["mechanism"], r["source"]) == (dataset, family, source)) for source in SOURCES]
            h = ax.errorbar(np.arange(3) + offset, [100*r[effect+"_mean"] for r in selected],
                yerr=[100*r[effect+"_sd"] for r in selected], fmt=marker, ms=4.5,
                color=color, mfc=color if effect == "add" else "white", capsize=2, lw=1., label=label)
            if index == 0: handles.append(h)
        ax.axhline(0, color="#535B60", lw=.8, zorder=0)
        ax.set_xticks(range(3), SOURCES); ax.set_xlim(-.5, 2.5)
        ax.set_title(f"({chr(97+index)}) {dataset.capitalize()}", fontsize=10)
        ax.grid(axis="y", alpha=.18); ax.set_axisbelow(True)
    axes[0].set_ylabel(r"$\Delta$ NDCG@10 ($\times100$)")
    fig.legend(handles, [s[-1] for s in styles], loc="upper center", ncol=2,
               bbox_to_anchor=(.5, .99), fontsize=8.5, columnspacing=1.8)
    export(fig, base)


def plot_robustness(rows: list[dict], base: Path) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(7.2, 5.7))
    fig.subplots_adjust(left=.105, right=.985, bottom=.095, top=.89, wspace=.38, hspace=.55)
    for i, dataset in enumerate(DATASETS):
        for j, source in enumerate(LONG_SOURCE.values()):
            ax = axes[i, j]
            for model, color, style, marker, label in [("early_CBT", "#A36A42", "--", "s", "Without source dropout"),
                    ("early_sd", "#236A8B", "-", "o", "With source dropout")]:
                selected = sorted([r for r in rows if (r["dataset"], r["source"], r["model"]) == (dataset, source, model)], key=lambda r:r["missing_rate"])
                x = np.array([r["missing_rate"]*100 for r in selected]); y = np.array([r["mean"] for r in selected]); sd = np.array([r["sd"] for r in selected])
                ax.plot(x, y, style, color=color, marker=marker, ms=3, lw=1.3, label=label)
                ax.fill_between(x, y-sd, y+sd, color=color, alpha=.14, linewidth=0)
            ax.set_title(f"({chr(97+i*3+j)}) {dataset.capitalize()} / {source.capitalize()}", fontsize=9)
            ax.set_xticks([0, 20, 40, 60, 80]); ax.grid(alpha=.18); ax.set_axisbelow(True)
            if j == 0: ax.set_ylabel("Test NDCG@10", fontsize=8.5)
            if i == 2: ax.set_xlabel("Additional missingness (%)", fontsize=8.5)
            ax.tick_params(labelsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(.5,.995), fontsize=9)
    export(fig, base)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default="runs/memo0711_v3")
    parser.add_argument("--output", default="results/paper_figures_memo0711_v3")
    args = parser.parse_args()
    root, output = resolve_project_path(args.run_root), resolve_project_path(args.output)
    if output == root or output.is_relative_to(root):
        raise ValueError("Keep generated figures outside the immutable run directory")
    inputs: set[Path] = set()
    # Validate all inputs before writing or drawing anything.
    effects = aggregate_effects(root, inputs); robustness = aggregate_robustness(root, inputs)
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":9, "pdf.fonttype":42,
        "svg.fonttype":"none", "axes.spines.top":False, "axes.spines.right":False, "legend.frameon":False})
    write_csv(output / "source_effects.csv", effects); write_csv(output / "robustness_levels.csv", robustness)
    plot_effects(effects, output / "source_effects")
    plot_robustness(robustness, output / "robustness_three_datasets")
    manifest = {"run_root":str(root), "research_training":False, "research_inference":False,
        "metric":"test NDCG@10, diagnostic B/robustness, not UASF Test", "seeds":list(SEEDS),
        "error_bars":"sample SD across paired seed effects (Figure 2); sample SD of seed metrics (Figure 3)",
        "input_csv_sha256":{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(inputs)},
        "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output / "figure_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Exported Figures 2 and 3 from {len(inputs)} input CSVs.")


if __name__ == "__main__":
    main()
