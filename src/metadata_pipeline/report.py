"""
report.py - Generate Table 1 with metadata coverage and counts across Beauty, Sports, and Toys.
Outputs both CSV and Markdown formats into reports/.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
import yaml

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from taimma.paths import resolve_project_path


def generate_table1(
    metadata_dir: str = "artifacts/metadata",
    output_file: str = "reports/table_1_metadata.csv",
    datasets: List[str] = ["beauty", "sports", "toys"],
) -> pd.DataFrame:
    meta_path = resolve_project_path(metadata_dir)
    out_path = resolve_project_path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for ds in datasets:
        cov_file = meta_path / ds.lower() / "coverage.json"
        if not cov_file.exists():
            print(f"[Warning] Coverage file missing for {ds}: {cov_file}")
            continue

        with open(cov_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        ds_name = data.get("dataset", ds.capitalize())

        # Coverage on Train-base catalog
        tb_cov = data.get("train_base_catalog", {})
        all_cov = data.get("all_unique_items_cohort", {})

        n_tb = tb_cov.get("total_items", 0)
        cat_tb = tb_cov.get("category", {})
        brd_tb = tb_cov.get("brand", {})
        ttl_tb = tb_cov.get("title", {})

        n_all = all_cov.get("total_items", 0)
        cat_all = all_cov.get("category", {})
        brd_all = all_cov.get("brand", {})
        ttl_all = all_cov.get("title", {})

        rows.append({
            "Dataset": ds_name,
            "Scope": "Train-base Catalog",
            "Total Items": n_tb,
            "Category (Avail / %)": f"{cat_tb.get('available', 0):,} ({cat_tb.get('coverage_pct', 0.0):.1f}%)",
            "Brand (Avail / %)": f"{brd_tb.get('available', 0):,} ({brd_tb.get('coverage_pct', 0.0):.1f}%)",
            "Title (Avail / %)": f"{ttl_tb.get('available', 0):,} ({ttl_tb.get('coverage_pct', 0.0):.1f}%)",
            "Category Avail": cat_tb.get("available", 0),
            "Category Missing": cat_tb.get("missing", 0),
            "Category Pct": cat_tb.get("coverage_pct", 0.0),
            "Brand Avail": brd_tb.get("available", 0),
            "Brand Missing": brd_tb.get("missing", 0),
            "Brand Pct": brd_tb.get("coverage_pct", 0.0),
            "Title Avail": ttl_tb.get("available", 0),
            "Title Missing": ttl_tb.get("missing", 0),
            "Title Pct": ttl_tb.get("coverage_pct", 0.0),
            "OOV Items Count": data.get("oov_items_count", 0),
        })

        rows.append({
            "Dataset": ds_name,
            "Scope": "All Unique Items Cohort",
            "Total Items": n_all,
            "Category (Avail / %)": f"{cat_all.get('available', 0):,} ({cat_all.get('coverage_pct', 0.0):.1f}%)",
            "Brand (Avail / %)": f"{brd_all.get('available', 0):,} ({brd_all.get('coverage_pct', 0.0):.1f}%)",
            "Title (Avail / %)": f"{ttl_all.get('available', 0):,} ({ttl_all.get('coverage_pct', 0.0):.1f}%)",
            "Category Avail": cat_all.get("available", 0),
            "Category Missing": cat_all.get("missing", 0),
            "Category Pct": cat_all.get("coverage_pct", 0.0),
            "Brand Avail": brd_all.get("available", 0),
            "Brand Missing": brd_all.get("missing", 0),
            "Brand Pct": brd_all.get("coverage_pct", 0.0),
            "Title Avail": ttl_all.get("available", 0),
            "Title Missing": ttl_all.get("missing", 0),
            "Title Pct": ttl_all.get("coverage_pct", 0.0),
            "OOV Items Count": data.get("oov_items_count", 0),
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Table 1 saved to CSV: {out_path}")

    # Also save markdown version
    md_path = out_path.with_suffix(".md")
    display_cols = ["Dataset", "Scope", "Total Items", "Category (Avail / %)", "Brand (Avail / %)", "Title (Avail / %)"]
    if not df.empty:
        md_text = df[display_cols].to_markdown(index=False)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# Table 1: Dataset Metadata Coverage Summary\n\n{md_text}\n")
        print(f"Table 1 saved to Markdown: {md_path}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Generate Table 1 Metadata Summary.")
    parser.add_argument("--metadata_dir", default="artifacts/metadata", help="Metadata artifacts directory")
    parser.add_argument("--output", default="reports/table_1_metadata.csv", help="Output file path")
    args = parser.parse_args()

    generate_table1(metadata_dir=args.metadata_dir, output_file=args.output)


if __name__ == "__main__":
    main()

