"""Prepare real Amazon 2014 metadata with Train-base-only vocabularies."""
import argparse
import json
import sys
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from taimma.paths import PROJECT_ROOT, resolve_project_path
from taimma.data.ingestion import Amazon2014Ingestor
from taimma.data.split import SequenceSplitter, compute_file_sha256
from taimma.experiments.reporting import dataset_statistics
from .contracts import SOURCES, coverage, require, write_json


def is_valid_string(value):
    return isinstance(value, str) and bool(value.strip())


def get_title_embeddings(titles, encoder_name="all-MiniLM-L6-v2", batch_size=256, target_dim=None):
    """Encoder failures are fatal; only missing titles receive zero vectors."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(encoder_name)
    dimension = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
    dim = dimension()
    require(isinstance(dim, int) and dim > 0, "Invalid encoder dimension")
    require(target_dim is None or target_dim == dim, "Encoder dimension differs from target_dim")
    embeddings = np.zeros((len(titles), dim), dtype=np.float32)
    valid = [i for i, title in enumerate(titles) if is_valid_string(title)]
    if valid:
        encoded = np.asarray(model.encode([titles[i].strip() for i in valid], batch_size=batch_size,
                             show_progress_bar=True, normalize_embeddings=True), dtype=np.float32)
        require(encoded.shape == (len(valid), dim) and np.isfinite(encoded).all(), "Invalid Title vectors")
        require(np.allclose(np.linalg.norm(encoded, axis=1), 1, atol=1e-4), "Title vectors must have unit norm")
        embeddings[valid] = encoded
    return embeddings, dict(encoder=encoder_name, mode="sentence-transformers", dim=dim)


def source_path(path):
    path = Path(path).resolve()
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def process_dataset_metadata(dataset_name, raw_root, output_base_dir, config_file,
                             dataset_slug=None, encoder_name="all-MiniLM-L6-v2",
                             expected_stats=None, overwrite=False):
    output_dir = resolve_project_path(output_base_dir) / (dataset_slug or dataset_name).lower()
    require(not output_dir.exists() or overwrite, f"Output exists: {output_dir}; use --overwrite explicitly")
    config = yaml.safe_load(resolve_project_path(config_file).read_text(encoding="utf-8"))
    key = config["dataset_name"].replace("_2014", "")
    ingestor = Amazon2014Ingestor(str(resolve_project_path(raw_root)), key)
    reviews_path, meta_path = ingestor.resolve_paths()
    if meta_path is None:
        raise FileNotFoundError(f"Metadata missing for {key}; download meta_{key}.json.gz first")
    print(f"Reading reviews/metadata for {key}", flush=True)
    sequences, metadata = ingestor.build_interaction_sequences(config.get("min_interactions", 5))
    require(bool(sequences) and bool(metadata), "Reviews and metadata must be nonempty")
    split = SequenceSplitter(config.get("max_seq_len", 50), config.get("min_interactions", 5)).split(sequences)
    all_items = sorted({item for seq in sequences for item in seq.items})
    actual = dict(users=len(sequences), items=len(all_items), catalog_train_base_items=split.num_items,
                  interactions=sum(len(seq.items) for seq in sequences))
    for name, expected in {**config.get("expected_stats", {}), **(expected_stats or {})}.items():
        require(name in actual and actual[name] == expected,
                f"Dataset statistic {name}: expected {expected}, actual {actual.get(name)}")
    require(split.num_items > 0, "Empty Train-base catalog")
    rows = []
    for asin in all_items:
        rec = metadata.get(asin, {})
        labels = rec.get("categories") or ([rec["category"]] if rec.get("category") else [])
        labels = list(dict.fromkeys(str(x).strip() for x in labels if str(x).strip()))
        brand, title = rec.get("brand") or "", rec.get("title") or ""
        rows.append(dict(model_item_id=split.item2idx.get(asin, 0), raw_item_id=asin,
                         in_train_base=asin in split.item2idx, categories=labels, brand=brand, title=title,
                         has_category=bool(labels), has_brand=is_valid_string(brand), has_title=is_valid_string(title)))
    full = pd.DataFrame(rows)
    catalog = full[full.in_train_base].copy().sort_values("model_item_id")
    cat_vocab = {v: i + 1 for i, v in enumerate(sorted({v for vs in catalog.categories for v in vs}))}
    brand_vocab = {v: i + 1 for i, v in enumerate(sorted({v for v in catalog.brand if v}))}
    catalog["category_indices"] = catalog.categories.map(lambda vs: json.dumps([cat_vocab[v] for v in vs]))
    catalog["brand_idx"] = catalog.brand.map(lambda v: brand_vocab.get(v, 0))
    catalog["categories"] = catalog.categories.map(lambda vs: json.dumps(vs, ensure_ascii=False))
    embeddings, embed_info = get_title_embeddings(catalog.title.tolist(), encoder_name=encoder_name)
    stats = dataset_statistics(sequences, split, metadata, True, config["dataset_name"], config.get("expected_stats"))
    stats.update(title_encoder=embed_info["encoder"], title_encoder_status="encoded")
    cov = dict(dataset=dataset_name, train_base_catalog=coverage(catalog),
               all_unique_items_cohort=coverage(full), oov_items_count=len(full) - len(catalog))
    manifest = dict(schema_version=2, dataset=dataset_name, dataset_key=key, category_policy="multilabel_without_root",
                    reviews_file=source_path(reviews_path), reviews_sha256=compute_file_sha256(reviews_path),
                    metadata_file=source_path(meta_path), metadata_sha256=compute_file_sha256(meta_path),
                    num_catalog_items=len(catalog), num_total_cohort_items=len(full), num_oov_items=len(full)-len(catalog),
                    title_embedding=dict(path="title_embeddings.npy", shape=list(embeddings.shape), **embed_info),
                    preprocessing_config=config, coverage_train_base=cov["train_base_catalog"])
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    # Failed inputs/encoding cannot overwrite a previous complete handoff.
    with tempfile.TemporaryDirectory(prefix=".metadata-", dir=output_dir.parent) as scratch:
        stage = Path(scratch) / "payload"
        stage.mkdir()
        full[["model_item_id", "raw_item_id", "in_train_base"]].to_csv(stage/"item_map.csv", index=False, lineterminator="\n")
        catalog.drop(columns="in_train_base").to_csv(stage/"metadata.csv", index=False, lineterminator="\n")
        cohort = full.copy()
        cohort["categories"] = cohort.categories.map(lambda vs: json.dumps(vs, ensure_ascii=False))
        cohort.to_csv(stage/"cohort_metadata.csv", index=False, lineterminator="\n")
        catalog[["model_item_id", *[f"has_{s}" for s in SOURCES]]].to_csv(stage/"availability_mask.csv", index=False, lineterminator="\n")
        np.save(stage/"title_embeddings.npy", embeddings)
        for name, data in [("category_vocab",cat_vocab),("brand_vocab",brand_vocab),("coverage",cov),("statistics",stats)]:
            write_json(stage/f"{name}.json", data)
        manifest["artifact_sha256"] = {p.name:compute_file_sha256(str(p)) for p in stage.iterdir()}
        write_json(stage/"manifest.json", manifest)
        if output_dir.exists():
            backup = Path(scratch)/"previous"
            output_dir.rename(backup)
            try:
                stage.rename(output_dir)
            except Exception:
                backup.rename(output_dir)
                raise
        else:
            stage.rename(output_dir)
    print(f"Metadata prepared: {output_dir}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/metadata.yaml")
    parser.add_argument("--dataset", choices=["beauty","sports","toys","all"], default="all")
    parser.add_argument("--raw_root", default="data/raw")
    parser.add_argument("--output_dir", default="artifacts/metadata")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(resolve_project_path(args.config).read_text(encoding="utf-8"))
    for key in (["beauty","sports","toys"] if args.dataset == "all" else [args.dataset]):
        ds = config["datasets"][key]
        process_dataset_metadata(ds["display_name"],args.raw_root,args.output_dir,ds["config_file"],key,
                                 config["pipeline"]["title_encoder"],ds.get("expected_stats"),args.overwrite)


if __name__ == "__main__":
    main()
