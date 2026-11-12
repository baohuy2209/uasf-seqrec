"""Editable vector draft of the implemented UASF architecture; no image model."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
try:
    from taimma.paths import resolve_project_path
except ImportError:
    def resolve_project_path(value: str | Path) -> Path:
        return Path(value).expanduser().resolve()


def draw(output: Path) -> None:
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":9,
        "pdf.fonttype":42, "svg.fonttype":"none", "mathtext.fontset":"dejavusans"})
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    fig.subplots_adjust(left=.01, right=.99, bottom=.01, top=.99)
    ax.set(xlim=(0,100), ylim=(0,100)); ax.axis("off")
    ink="#24323B"; blue="#236A8B"; pale="#EEF4F7"; grey="#F5F6F7"

    def box(x: float, y: float, w: float, h: float, text: str, accent: bool=False, size: float=9) -> None:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.35,rounding_size=0.8",
            linewidth=1.,edgecolor=blue if accent else "#7B878E",facecolor=pale if accent else grey,zorder=2))
        ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=size,color=ink,linespacing=1.45,zorder=3)

    def edge(points: list[tuple[float,float]], control: bool=False) -> None:
        for a,b in zip(points[:-2],points[1:-1]):
            ax.plot([a[0],b[0]],[a[1],b[1]],color=blue if control else ink,lw=1.1,
                    linestyle=":" if control else "-",zorder=1)
        ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle="-|>",mutation_scale=10,
            linewidth=1.1,color=blue if control else ink,linestyle=":" if control else "-",zorder=1))

    ax.text(2,98,"(a) Context-dependent scoring",weight="bold",fontsize=10,color=ink)
    box(2,82,21,12,"Observed history\n+ catalog metadata\nCategory / Brand / Title",size=8.7)
    box(29,82,31,12,"Frozen ungated reference passes\nFull-source and source-off\nHistory, state, agreement features",size=8.5)
    box(66,82,31,12,"Gate MLP (32 hidden units)\nSigmoid, temperature = 1\nInference threshold = 0.1",accent=True,size=8.7)
    edge([(23.4,88),(28.6,88)]);edge([(60.4,88),(65.6,88)])

    # The same context weights control both history and candidate metadata.
    edge([(81.5,81.6),(81.5,78.5),(23,78.5),(23,73.4)],control=True)
    edge([(81.5,78.5),(75,78.5),(75,73.4)],control=True)
    ax.text(45,79.2,"Shared source weights (Category, Brand, Title)",ha="center",fontsize=8.3,color=ink,
            bbox=dict(facecolor="white",edgecolor="none",pad=.4))
    box(2,55,42,18,"Weighted history representations\nID + available metadata + position\nFrozen causal SASRec\nContext state",size=9)
    box(53,55,44,18,"Weighted candidate representations\nID + available metadata\nItem-level availability masks\nSame frozen source encoders",size=9)
    # Input data also feed the weighted representations, independently of gate features.
    edge([(12,81.6),(12,73.4)])
    edge([(19,81.6),(19,75),(62,75),(62,73.4)])
    edge([(23,54.6),(23,47),(31.6,47)])
    edge([(75,54.6),(75,47),(68.4,47)])
    box(32,42,36,10,"Early-fusion scores\nContext-candidate dot product",size=9)
    box(2,26,22,10,"Independent ID model\nSame-seed Train-base\nFrozen fallback",size=8.7)
    box(32,26,36,10,"All weights zero? Use ID scores\nOtherwise use early-fusion scores",accent=True,size=9)
    edge([(24.4,31),(31.6,31)]);edge([(50,41.6),(50,36.4)])
    edge([(97.4,88),(99,88),(99,31),(68.4,31)],control=True)
    ax.text(83,32.5,"All-off decision",fontsize=8.5,ha="center",color=ink)
    edge([(50,25.6),(50,23)])
    ax.text(50,21.6,"Full-catalog ranking with input-history masking",ha="center",fontsize=9,color=ink)
    ax.plot([2,98],[18.7,18.7],color="#A5ADB3",lw=.7)
    ax.text(2,16.5,"(b) Training stages",weight="bold",fontsize=10,color=ink)
    box(2,2,28,11,"S1 | Train-base\nTrain early source-dropout\nand independent ID models",size=8.6)
    box(36,2,28,11,"S2 | Train-meta\nFreeze references; derive utility\nlabels and target-free features",size=8.4)
    box(70,2,28,11,"S3 | Train-meta\nTrain only the gate\nCE + utility BCE + sparsity",accent=True,size=8.5)
    edge([(30.4,7.5),(35.6,7.5)]);edge([(64.4,7.5),(69.6,7.5)])
    output.mkdir(parents=True,exist_ok=True)
    for ext in ("pdf","svg","png"):
        fig.savefig(output/f"uasf_architecture.{ext}",format=ext,dpi=300,
                    bbox_inches="tight",pad_inches=.02,facecolor="white")
        if ext == "svg":
            path=output/f"uasf_architecture.{ext}"
            path.write_text(re.sub(r"<!DOCTYPE[^>]*>","",path.read_text(encoding="utf-8")),encoding="utf-8")
    plt.close(fig)
    print("Exported editable Figure 1 draft in PDF, SVG and PNG.")


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",default="results/paper_figures_memo0711_v3")
    args=parser.parse_args();draw(resolve_project_path(args.output))


if __name__=="__main__": main()
