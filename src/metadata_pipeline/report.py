"""Generate a complete Table 1; unavailable measurements never become zero."""
import argparse
import sys
from pathlib import Path
import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0,str(SRC_DIR))
from taimma.paths import resolve_project_path
from .contracts import SOURCES, read_json, require


def generate_table1(metadata_dir="artifacts/metadata", output_file="reports/table_1_metadata.csv",
                    datasets=("beauty","sports","toys")):
    rows = []
    for key in datasets:
        directory = resolve_project_path(metadata_dir)/key
        cov, stats = read_json(directory/"coverage.json"), read_json(directory/"statistics.json")
        require(stats["metadata_status"] == "available", f"{key}: metadata not measured")
        for scope, name in [("train_base_catalog","Train-base Catalog"),("all_unique_items_cohort","All Unique Items Cohort")]:
            section = cov[scope]
            total = section["total_items"]
            require(total > 0, f"{key}: empty coverage scope")
            require(total == stats["catalog_train_base_items" if scope == "train_base_catalog" else "original_items"],
                    f"{key}: statistics/coverage item mismatch")
            row = {"Dataset":cov["dataset"],"Scope":name,"Total Items":total,
                   **{k:stats[k] for k in ("users","original_items","catalog_train_base_items","interactions",
                       "sparsity","mean_sequence_length","category_vocab_train_base","brand_vocab_train_base","metadata_status")},
                   "OOV Items Count":cov["oov_items_count"]}
            for source in SOURCES:
                info = section[source]
                require(info["available"]+info["missing"] == total and 0 <= info["available"] <= total
                        and abs(info["coverage_pct"]-round(info["available"]/total*100,2)) < 1e-8,
                        f"{key}/{source}: invalid coverage")
                title = source.capitalize()
                row.update({f"{title} Avail":info["available"],f"{title} Missing":info["missing"],
                            f"{title} Pct":info["coverage_pct"]})
            rows.append(row)
    require(bool(rows), "No datasets selected")
    frame = pd.DataFrame(rows)
    display = frame[["Dataset","Scope","users","Total Items","interactions","sparsity",
                     "mean_sequence_length","Category Pct","Brand Pct","Title Pct"]]
    # Resolve formatting dependencies before writing either output.
    markdown = "# Table 1 Dataset statistics and metadata coverage\n\n"+display.to_markdown(index=False)+"\n"
    output = resolve_project_path(output_file)
    output.parent.mkdir(parents=True,exist_ok=True)
    frame.to_csv(output,index=False, lineterminator="\n")
    output.with_suffix(".md").write_text(markdown,encoding="utf-8")
    print(f"Table 1 saved: {output}")
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata_dir",default="artifacts/metadata")
    parser.add_argument("--output",default="reports/table_1_metadata.csv")
    args = parser.parse_args()
    generate_table1(args.metadata_dir,args.output)


if __name__ == "__main__":
    main()
