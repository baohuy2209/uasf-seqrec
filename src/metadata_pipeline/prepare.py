"""
prepare.py - Extract and build normalized metadata, item mapping, and title embeddings.
Supports both Beauty, Sports, and Toys.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import yaml

# Resolve paths
SRC_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from taimma.paths import resolve_project_path
from taimma.data.ingestion import Amazon2014Ingestor
from taimma.data.split import SequenceSplitter, compute_file_sha256


def is_valid_string(val: Any) -> bool:
    if val is None or pd.isna(val):
        return False
    if not isinstance(val, str):
        val = str(val)
    return bool(val.strip())


def normalize_string(val: Optional[Any]) -> Optional[str]:
    if not is_valid_string(val):
        return None
    return str(val).strip()


def get_title_embeddings(
    titles: List[Optional[str]],
    encoder_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 256,
    target_dim: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Generate normalized embeddings for non-empty titles using SentenceTransformers or deterministic projection.
    Empty/missing titles have a deterministic zero vector.
    """
    num_items = len(titles)
    embedding_info = {
        "encoder": encoder_name,
        "num_items": num_items,
    }

    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(encoder_name)
        dim = getattr(model, "get_embedding_dimension", getattr(model, "get_sentence_embedding_dimension", lambda: 384))()
        embedding_info["dim"] = dim
        embedding_info["mode"] = "sentence-transformers"

        embeddings = np.zeros((num_items, dim), dtype=np.float32)
        valid_indices = [idx for idx, t in enumerate(titles) if is_valid_string(t)]
        valid_texts = [str(titles[idx]).strip() for idx in valid_indices]

        if valid_texts:
            encoded_vecs = model.encode(
                valid_texts,
                batch_size=batch_size,
                show_progress_bar=True,
                normalize_embeddings=True,
            )
            for idx, vec in zip(valid_indices, encoded_vecs):
                embeddings[idx] = vec

        return embeddings, embedding_info
    except Exception as exc:
        print(f"[Warning] SentenceTransformer failed or offline ({exc}). Using deterministic reproducible hash projection.")
        dim = target_dim or 64
        embedding_info["dim"] = dim
        embedding_info["mode"] = "deterministic-hash-projection"

        embeddings = np.zeros((num_items, dim), dtype=np.float32)
        import hashlib
        for idx, t in enumerate(titles):
            if is_valid_string(t):
                # Deterministic random vector from title string SHA256
                seed_val = int(hashlib.sha256(str(t).strip().encode("utf-8")).hexdigest()[:8], 16)
                rng = np.random.RandomState(seed_val)
                vec = rng.randn(dim).astype(np.float32)
                norm = np.linalg.norm(vec)
                embeddings[idx] = vec / (norm + 1e-12)

        return embeddings, embedding_info


def process_dataset_metadata(
    dataset_name: str,
    raw_root: str,
    output_base_dir: str,
    config_file: str,
    dataset_slug: Optional[str] = None,
    encoder_name: str = "all-MiniLM-L6-v2",
    expected_stats: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    print(f"\n=======================================================")
    print(f"Processing Metadata for: {dataset_name}")
    print(f"=======================================================")

    folder_name = dataset_slug.lower() if dataset_slug else dataset_name.lower()
    raw_root_path = resolve_project_path(raw_root)
    output_dir = resolve_project_path(output_base_dir) / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(resolve_project_path(config_file), "r", encoding="utf-8") as f:
        ds_config = yaml.safe_load(f)

    category_key = ds_config.get("dataset_name", dataset_name).replace("_2014", "")
    ingestor = Amazon2014Ingestor(str(raw_root_path), dataset_key=category_key)
    reviews_path, meta_path = ingestor.resolve_paths()

    print(f"- Reviews file: {reviews_path}")
    print(f"- Metadata file: {meta_path}")

    # Build sequence splits to lock exact Train-base catalog
    min_interactions = ds_config.get("min_interactions", 5)
    max_seq_len = ds_config.get("max_seq_len", 50)
    sequences, raw_meta_dict = ingestor.build_interaction_sequences(min_interactions=min_interactions)

    splitter = SequenceSplitter(max_seq_len=max_seq_len, min_seq_len=min_interactions)
    split_data = splitter.split(sequences)

    # All unique raw items in the 5-core interaction sequences
    all_raw_items_set = set()
    for seq in sequences:
        all_raw_items_set.update(seq.items)
    sorted_all_raw_items = sorted(list(all_raw_items_set))

    # Train-base catalog items (model_item_id 1..N)
    # item2idx has 1-based indexing for train-base items
    train_base_catalog_size = split_data.num_items
    total_raw_items_count = len(sorted_all_raw_items)

    print(f"Interaction Users: {len(sequences)}")
    print(f"Total Unique Items (All Cohort): {total_raw_items_count}")
    print(f"Train-base Catalog Items: {train_base_catalog_size}")

    if expected_stats:
        if "items" in expected_stats:
            print(f"Expected items: {expected_stats['items']} | Actual: {total_raw_items_count} | Match: {expected_stats['items'] == total_raw_items_count}")
        if "catalog_train_base_items" in expected_stats:
            print(f"Expected train-base: {expected_stats['catalog_train_base_items']} | Actual: {train_base_catalog_size} | Match: {expected_stats['catalog_train_base_items'] == train_base_catalog_size}")

    # Build item_map.csv
    # model_item_id: 1..N for Train-base items, 0 for OOV / padding
    # For alignment in metadata tables:
    # 1. We provide train_base_catalog (index 1..N) as the primary model catalog table
    # 2. We also record in_train_base flag for any out-of-catalog items
    catalog_rows = []
    # Index 0 is reserved for padding in sequential recommendation models
    # Row i corresponds to model_item_id i (1..N)
    for model_id in range(1, train_base_catalog_size + 1):
        raw_asin = split_data.idx2item[model_id]
        meta = raw_meta_dict.get(raw_asin, {})
        category = meta.get("category")
        brand = meta.get("brand")
        title = meta.get("title")

        has_cat = bool(category is not None and str(category).strip())
        has_brd = bool(brand is not None and str(brand).strip())
        has_ttl = bool(title is not None and str(title).strip())

        catalog_rows.append({
            "model_item_id": model_id,
            "raw_item_id": raw_asin,
            "category": category,
            "brand": brand,
            "title": title,
            "has_category": has_cat,
            "has_brand": has_brd,
            "has_title": has_ttl,
            "in_train_base": True,
        })

    # OOV items (items present in Val/Test but not in Train-base)
    oov_asins = sorted([asin for asin in sorted_all_raw_items if asin not in split_data.item2idx])
    for oov_idx, raw_asin in enumerate(oov_asins):
        meta = raw_meta_dict.get(raw_asin, {})
        category = meta.get("category")
        brand = meta.get("brand")
        title = meta.get("title")

        has_cat = bool(category is not None and str(category).strip())
        has_brd = bool(brand is not None and str(brand).strip())
        has_ttl = bool(title is not None and str(title).strip())

        catalog_rows.append({
            "model_item_id": 0,  # 0 indicates out-of-catalog / padding in model ID space
            "raw_item_id": raw_asin,
            "category": category,
            "brand": brand,
            "title": title,
            "has_category": has_cat,
            "has_brand": has_brd,
            "has_title": has_ttl,
            "in_train_base": False,
        })

    df_full = pd.DataFrame(catalog_rows)
    df_train_base = df_full[df_full["in_train_base"]].copy().sort_values("model_item_id")

    # Build integer vocabulary mappings for Category and Brand strictly from Train-base
    # Index 0 is reserved for padding / missing / unknown
    # 1. Category Vocab
    unique_cats = sorted(list(set(c for c in df_train_base["category"].dropna() if str(c).strip())))
    cat2idx = {cat: idx + 1 for idx, cat in enumerate(unique_cats)}
    with open(output_dir / "category_vocab.json", "w", encoding="utf-8") as f:
        json.dump(cat2idx, f, indent=2, ensure_ascii=False)

    # 2. Brand Vocab
    unique_brands = sorted(list(set(b for b in df_train_base["brand"].dropna() if str(b).strip())))
    brand2idx = {brand: idx + 1 for idx, brand in enumerate(unique_brands)}
    with open(output_dir / "brand_vocab.json", "w", encoding="utf-8") as f:
        json.dump(brand2idx, f, indent=2, ensure_ascii=False)

    # Assign integer idx columns (0 if missing or not in train-base vocab)
    df_train_base["category_idx"] = df_train_base["category"].map(lambda c: cat2idx.get(c, 0)).astype(int)
    df_train_base["brand_idx"] = df_train_base["brand"].map(lambda b: brand2idx.get(b, 0)).astype(int)

    # 1. Save item_map.csv
    item_map_path = output_dir / "item_map.csv"
    df_full[["model_item_id", "raw_item_id", "in_train_base"]].to_csv(item_map_path, index=False)

    # 2. Save metadata.csv (model_item_id, category, brand, title, has_*, category_idx, brand_idx)
    metadata_csv_path = output_dir / "metadata.csv"
    df_train_base[[
        "model_item_id", "raw_item_id",
        "category", "brand", "title",
        "category_idx", "brand_idx",
        "has_category", "has_brand", "has_title"
    ]].to_csv(metadata_csv_path, index=False)

    # 3. Save availability_mask.csv
    avail_mask_path = output_dir / "availability_mask.csv"
    df_train_base[["model_item_id", "has_category", "has_brand", "has_title"]].to_csv(avail_mask_path, index=False)

    # 4. Generate & Save title_embeddings.npy
    # Order strictly matches model_item_id 1..N (shape: [num_items, dim])
    titles_list = df_train_base["title"].tolist()
    embeddings, embed_info = get_title_embeddings(titles_list, encoder_name=encoder_name)
    title_npy_path = output_dir / "title_embeddings.npy"
    np.save(title_npy_path, embeddings)

    # 5. Calculate Coverage metrics on BOTH:
    # A) Train-base catalog (1..N)
    # B) Total unique items across all cohorts (All 5-core items)
    def calc_coverage(df_subset, scope_name):
        n_total = len(df_subset)
        cat_count = int(df_subset["has_category"].sum())
        brd_count = int(df_subset["has_brand"].sum())
        ttl_count = int(df_subset["has_title"].sum())
        return {
            "scope": scope_name,
            "total_items": n_total,
            "category": {
                "available": cat_count,
                "missing": n_total - cat_count,
                "coverage_pct": round(cat_count / max(1, n_total) * 100.0, 2),
            },
            "brand": {
                "available": brd_count,
                "missing": n_total - brd_count,
                "coverage_pct": round(brd_count / max(1, n_total) * 100.0, 2),
            },
            "title": {
                "available": ttl_count,
                "missing": n_total - ttl_count,
                "coverage_pct": round(ttl_count / max(1, n_total) * 100.0, 2),
            },
        }

    cov_train_base = calc_coverage(df_train_base, "train_base_catalog")
    cov_all = calc_coverage(df_full, "all_unique_items_cohort")

    coverage_payload = {
        "dataset": dataset_name,
        "train_base_catalog": cov_train_base,
        "all_unique_items_cohort": cov_all,
        "oov_items_count": len(oov_asins),
    }

    coverage_json_path = output_dir / "coverage.json"
    with open(coverage_json_path, "w", encoding="utf-8") as f:
        json.dump(coverage_payload, f, indent=2)

    # 6. Save Manifest
    manifest = {
        "dataset": dataset_name,
        "reviews_file": reviews_path,
        "reviews_sha256": compute_file_sha256(reviews_path),
        "metadata_file": meta_path,
        "metadata_sha256": compute_file_sha256(meta_path) if meta_path else None,
        "num_catalog_items": train_base_catalog_size,
        "num_total_cohort_items": total_raw_items_count,
        "num_oov_items": len(oov_asins),
        "title_embedding": {
            "path": str(title_npy_path.name),
            "shape": list(embeddings.shape),
            "encoder": embed_info["encoder"],
            "dim": embed_info["dim"],
            "mode": embed_info["mode"],
        },
        "coverage_train_base": cov_train_base,
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Artifacts successfully written to: {output_dir}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Prepare metadata artifacts for Beauty, Sports, and Toys.")
    parser.add_argument("--config", default="configs/metadata.yaml", help="Metadata pipeline config path")
    parser.add_argument("--dataset", choices=["beauty", "sports", "toys", "all"], default="all", help="Target dataset")
    parser.add_argument("--raw_root", default="data/raw", help="Raw dataset root directory")
    parser.add_argument("--output_dir", default="artifacts/metadata", help="Output artifact directory")
    args = parser.parse_args()

    with open(resolve_project_path(args.config), "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    encoder_name = config.get("pipeline", {}).get("title_encoder", "all-MiniLM-L6-v2")
    datasets_cfg = config.get("datasets", {})

    target_keys = ["beauty", "sports", "toys"] if args.dataset == "all" else [args.dataset]

    for key in target_keys:
        d_cfg = datasets_cfg.get(key)
        if not d_cfg:
            print(f"Skipping unknown dataset key: {key}")
            continue
        process_dataset_metadata(
            dataset_name=d_cfg["display_name"],
            raw_root=args.raw_root,
            output_base_dir=args.output_dir,
            config_file=d_cfg["config_file"],
            dataset_slug=key,
            encoder_name=encoder_name,
            expected_stats=d_cfg.get("expected_stats"),
        )


if __name__ == "__main__":
    main()

