"""Small CSV-integrity tests; never imports research models."""
import csv
import importlib.util
from pathlib import Path
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "visualize_memo0711.py"
if not SCRIPT.exists(): SCRIPT = Path(__file__).with_name("visualize_memo0711.py")
spec = importlib.util.spec_from_file_location("paper_plots", SCRIPT)
plot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot)


def write(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w=csv.DictWriter(f, fieldnames=columns); w.writeheader(); w.writerows(rows)


def test_rank_metric_and_oov(tmp_path: Path) -> None:
    p=tmp_path/"rank.csv"
    write(p, ["user_idx","target_idx","rank","ndcg@10"], [dict(user_idx=0,target_idx=1,rank=1,**{"ndcg@10":1}),dict(user_idx=1,target_idx=0,rank=12,**{"ndcg@10":0})])
    keys, mean=plot.metric_mean(p)
    assert len(keys)==2 and mean==.5
    with p.open("a") as f: f.write("0,1,1,1\n")
    with pytest.raises(ValueError, match="Duplicate"): plot.metric_mean(p)


def test_wrong_ndcg_fails(tmp_path: Path) -> None:
    p=tmp_path/"rank.csv"
    write(p, ["user_idx","target_idx","rank","ndcg@10"], [dict(user_idx=0,target_idx=1,rank=2,**{"ndcg@10":1})])
    with pytest.raises(ValueError, match="Invalid"): plot.metric_mean(p)


def test_robustness_requires_complete_three_seed_grid(tmp_path: Path) -> None:
    for dataset in plot.DATASETS:
        write(tmp_path/dataset/"robustness.csv", ["seed","model","source","missing_rate","ndcg_10"],
            [dict(seed=42,model="early_CBT",source="brand",missing_rate=0,ndcg_10=.05)])
    with pytest.raises(ValueError, match="Incomplete"): plot.aggregate_robustness(tmp_path,set())
