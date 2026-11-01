"""Strict artifact, perturbation, loader and optional raw-source verification."""
import argparse
import json
import re
import sys
from pathlib import Path
import numpy as np
import torch
import yaml

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from taimma.paths import resolve_project_path
from taimma.data.ingestion import Amazon2014Ingestor
from taimma.data.split import SequenceSample, SequenceSplitter, compute_file_sha256
from taimma.training.multimodal_trainer import MultimodalSequenceDataset
from .contracts import SOURCES, categories, check_availability, coverage, flags, read_json, read_table, require
from .loader import load_catalog


def is_valid_sha256(value):
    return isinstance(value, str) and re.fullmatch(r"[a-fA-F0-9]{64}", value) is not None


def validate_perturbations(directory, avail, config):
    root = directory/"perturbations"
    manifest = read_json(root/"perturbations_manifest.json")
    ratios, seeds, sources = config["missing_ratios"], config["seeds"], config["sources"]
    require(manifest["masks"]["ratios"] == ratios and manifest["masks"]["seeds"] == seeds
            and manifest["masks"]["sources"] == sources, "Perturbation config/manifest mismatch")
    require(manifest["permutations"]["seeds"] == seeds and manifest["permutations"]["sources"] == sources,
            "Permutation config/manifest mismatch")
    expected_masks, expected_perms = set(), set()
    for source in sources:
        original = flags(avail[f"has_{source}"])
        valid = np.flatnonzero(original)
        for seed in seeds:
            for ratio in ratios:
                name = f"mask_{source}_ratio_{ratio:.2f}_seed_{seed}.csv"
                expected_masks.add(name)
                table = read_table(root/"masks"/name)
                expected = original.copy()
                rng = np.random.RandomState(seed + int(ratio*1000))
                expected[rng.choice(valid, size=round(len(valid)*ratio), replace=False)] = False
                require(np.array_equal(table.model_item_id, avail.model_item_id), f"{name}: ID alignment")
                require(np.array_equal(flags(table.original_available), original), f"{name}: base mask")
                require(np.array_equal(flags(table.simulated_available), expected), f"{name}: seed/ratio/mask mismatch")
                require(np.array_equal(flags(table.is_dropped), original & ~expected), f"{name}: is_dropped mismatch")
            name = f"perm_{source}_seed_{seed}.csv"
            expected_perms.add(name)
            table = read_table(root/"permutations"/name)
            expected = avail.model_item_id.to_numpy().copy()
            expected[valid] = expected[valid[np.random.RandomState(seed).permutation(len(valid))]]
            require(np.array_equal(table.model_item_id, avail.model_item_id), f"{name}: ID alignment")
            require(np.array_equal(table.permuted_source_item_id, expected), f"{name}: permutation mismatch")
            require(np.array_equal(flags(table.is_available), original), f"{name}: availability mismatch")
            require((table.source == source).all() and (table.seed == seed).all(), f"{name}: source/seed mismatch")
    require({p.name for p in (root/"masks").glob("*.csv")} == expected_masks, "Missing/extra mask files")
    require({p.name for p in (root/"permutations").glob("*.csv")} == expected_perms, "Missing/extra permutations")
    for section, names, count_key in [("masks", expected_masks, "num_masks_generated"),
                                       ("permutations", expected_perms, "num_permutations_generated")]:
        records = manifest[section]["manifest"]
        require(len(records) == len(names) and {r["file"] for r in records} == names
                and manifest[section][count_key] == len(names), f"{section}: incomplete manifest")
    return len(expected_masks), len(expected_perms)


def check_loader(directory, avail, config):
    attrs, masks = load_catalog(directory)
    n = len(avail)
    # Sample actual enabled/disabled rows, and exercise each saved perturbation type.
    for source in SOURCES:
        original = flags(avail[f"has_{source}"])
        indices = list(np.flatnonzero(original)[:2]) + list(np.flatnonzero(~original)[:2])
        sample = SequenceSample(0, [0] + [int(i)+1 for i in indices], 1, len(indices))
        batch = MultimodalSequenceDataset([sample], attrs, masks)[0]
        expected = torch.zeros_like(batch["source_seqs"][source])
        for pos, i in enumerate(indices, 1):
            if original[i]:
                expected[pos] = attrs[source][i]
        require(torch.equal(batch["source_seqs"][source], expected), f"{source}: real loader mask failure")
        seed, ratio = config["seeds"][0], max(config["missing_ratios"])
        mask_path = directory/"perturbations/masks"/f"mask_{source}_ratio_{ratio:.2f}_seed_{seed}.csv"
        perm_path = directory/"perturbations/permutations"/f"perm_{source}_seed_{seed}.csv"
        pert_attrs, pert_masks = load_catalog(directory, {source:mask_path}, {source:perm_path})
        table = read_table(mask_path)
        require(torch.equal(pert_masks[source], torch.tensor(flags(table.simulated_available), dtype=torch.float32)),
                f"{source}: saved synthetic mask not applied")
        donor = read_table(perm_path).permuted_source_item_id.to_numpy(dtype=np.int64)-1
        require(torch.equal(pert_attrs[source], attrs[source][torch.from_numpy(donor)]),
                f"{source}: saved permutation not applied")
        dropped = np.flatnonzero(original & ~flags(table.simulated_available))[:2]
        sample = SequenceSample(0, [0]+[int(i)+1 for i in dropped], 1, len(dropped))
        batch = MultimodalSequenceDataset([sample], pert_attrs, pert_masks)[0]
        require(torch.count_nonzero(batch["source_seqs"][source]) == 0,
                f"{source}: dropped features leaked through sequence loader")


def validate_dataset_artifacts(dataset_dir, verify_raw_hash=False, raw_root=None,
                               artifacts_only=False, perturbation_config=None):
    directory = Path(dataset_dir)
    manifest = read_json(directory/"manifest.json")
    require(manifest["dataset_key"].lower().split("_")[0] == directory.name, "Dataset directory/key mismatch")
    require(manifest.get("schema_version") == 2, "Regenerate legacy metadata with schema_version=2")
    require(manifest["category_policy"] == "multilabel_without_root", "Unexpected Category policy")
    meta, avail = read_table(directory/"metadata.csv"), read_table(directory/"availability_mask.csv")
    mapping, cohort = read_table(directory/"item_map.csv"), read_table(directory/"cohort_metadata.csv")
    cov, stats = read_json(directory/"coverage.json"), read_json(directory/"statistics.json")
    n = len(meta)
    require(n > 0, "Empty catalog")
    check_availability(meta)
    check_availability(avail)
    require(len(avail) == n, "Availability row count mismatch")
    require(mapping.raw_item_id.is_unique and cohort.raw_item_id.is_unique, "Duplicate raw item IDs")
    require(mapping[["model_item_id","raw_item_id","in_train_base"]].equals(
            cohort[["model_item_id","raw_item_id","in_train_base"]]), "Cohort/mapping mismatch")
    train = mapping[flags(mapping.in_train_base)].sort_values("model_item_id")
    require(np.array_equal(train.model_item_id, meta.model_item_id)
            and np.array_equal(train.raw_item_id, meta.raw_item_id), "Catalog mapping mismatch")
    require((mapping.loc[~flags(mapping.in_train_base), "model_item_id"] == 0).all(), "OOV IDs must be zero")
    cat_vocab, brand_vocab = read_json(directory/"category_vocab.json"), read_json(directory/"brand_vocab.json")
    labels = [categories(v) for v in meta.categories]
    expected_cats = {v:i+1 for i,v in enumerate(sorted({v for vs in labels for v in vs}))}
    expected_brands = {v:i+1 for i,v in enumerate(sorted({v for v in meta.brand if v.strip()}))}
    require(cat_vocab == expected_cats and brand_vocab == expected_brands, "Vocabulary differs from Train-base labels")
    require([json.loads(v) for v in meta.category_indices] == [[cat_vocab[v] for v in vs] for vs in labels],
            "Category indices/vocabulary mismatch")
    require(meta.brand_idx.tolist() == [brand_vocab.get(v,0) for v in meta.brand], "Brand indices/vocabulary mismatch")
    for table in (meta, cohort):
        true_presence = dict(category=np.asarray([bool(categories(v)) for v in table.categories]),
                             brand=table.brand.str.strip().ne("").to_numpy(), title=table.title.str.strip().ne("").to_numpy())
        for source in SOURCES:
            require(np.array_equal(flags(table[f"has_{source}"]), true_presence[source]), f"{source}: flags/content mismatch")
    for source in SOURCES:
        require(np.array_equal(flags(avail[f"has_{source}"]), flags(meta[f"has_{source}"])), f"{source}: availability/metadata mismatch")
    cohort_train = cohort[flags(cohort.in_train_base)].sort_values("model_item_id")
    for col in ("categories","brand","title",*[f"has_{s}" for s in SOURCES]):
        require(np.array_equal(cohort_train[col], meta[col]), f"Cohort/catalog mismatch: {col}")
    for name, table in [("train_base_catalog", meta),("all_unique_items_cohort",cohort)]:
        actual = coverage(table)
        require(all(cov[name][key] == value for key,value in actual.items()), f"{name}: coverage mismatch")
    oov = len(mapping)-n
    require(cov["oov_items_count"] == oov and manifest["num_oov_items"] == oov, "OOV count mismatch")
    require(manifest["num_catalog_items"] == n and manifest["num_total_cohort_items"] == len(mapping), "Manifest counts mismatch")
    require(stats["original_items"] == len(mapping) and stats["catalog_train_base_items"] == n
            and stats["metadata_status"] == "available", "Statistics/catalog mismatch")
    embed_info = manifest["title_embedding"]
    require(embed_info["mode"] == "sentence-transformers" and bool(embed_info["encoder"]), "Real Title encoder required")
    emb = np.load(directory/"title_embeddings.npy", allow_pickle=False)
    require(emb.ndim == 2 and emb.shape[0] == n and emb.shape[1] > 0 and emb.dtype == np.float32
            and np.isfinite(emb).all(), "Invalid Title embedding array")
    require(embed_info["shape"] == list(emb.shape) and embed_info["dim"] == emb.shape[1]
            and embed_info["path"] == "title_embeddings.npy", "Embedding manifest mismatch")
    has_title = flags(meta.has_title)
    require((emb[~has_title] == 0).all() and np.allclose(np.linalg.norm(emb[has_title],axis=1),1,atol=1e-4),
            "Title missing placeholders/unit norms invalid")
    required_hashes = {"metadata.csv","cohort_metadata.csv","availability_mask.csv","item_map.csv",
                       "title_embeddings.npy","category_vocab.json","brand_vocab.json","coverage.json","statistics.json"}
    hashes = manifest["artifact_sha256"]
    require(set(hashes) == required_hashes, "Incomplete artifact checksums")
    for name in required_hashes:
        require(is_valid_sha256(hashes[name]) and compute_file_sha256(str(directory/name)) == hashes[name],
                f"Artifact checksum mismatch: {name}")
    for key in ("reviews", "metadata"):
        require(is_valid_sha256(manifest.get(f"{key}_sha256")) and bool(manifest.get(f"{key}_file")), f"Missing {key} provenance")
    provenance = "NOT_CHECKED_ARTIFACTS_ONLY"
    require(not (artifacts_only and verify_raw_hash), "--artifacts-only cannot verify raw hashes")
    if not artifacts_only:
        if raw_root is not None:
            ingestor = Amazon2014Ingestor(str(resolve_project_path(raw_root)), manifest["dataset_key"])
            reviews, metadata = ingestor.resolve_paths()
        else:
            reviews = str(resolve_project_path(manifest["reviews_file"]))
            metadata = str(resolve_project_path(manifest["metadata_file"]))
        require(Path(reviews).is_file() and metadata is not None and Path(metadata).is_file(),
                "Raw inputs missing; use --raw_root or explicitly request --artifacts-only")
        provenance = "FILES_PRESENT_HASH_NOT_CHECKED"
        if verify_raw_hash:
            for key,path in [("reviews",reviews),("metadata",metadata)]:
                require(compute_file_sha256(str(path)) == manifest[f"{key}_sha256"], f"Raw {key} SHA256 mismatch")
            ingestor = Amazon2014Ingestor()
            ingestor.resolve_paths = lambda: (reviews, metadata)
            sequences, records = ingestor.build_interaction_sequences(manifest["preprocessing_config"].get("min_interactions",5))
            split = SequenceSplitter(min_seq_len=manifest["preprocessing_config"].get("min_interactions",5)).split(sequences)
            require([split.idx2item[i] for i in range(1,split.num_items+1)] == meta.raw_item_id.tolist(), "Raw reviews/catalog mismatch")
            require({x for seq in sequences for x in seq.items} == set(mapping.raw_item_id), "Raw cohort mismatch")
            for row in cohort.itertuples():
                rec = records.get(row.raw_item_id,{})
                require(categories(row.categories) == (rec.get("categories") or [])
                        and row.brand == (rec.get("brand") or "") and row.title == (rec.get("title") or ""),
                        f"Raw metadata mismatch: {row.raw_item_id}")
            from taimma.experiments.reporting import dataset_statistics
            actual_stats = dataset_statistics(sequences, split, records, True, stats["dataset"])
            for name in ("users", "original_items", "catalog_train_base_items", "interactions",
                         "category_vocab_train_base", "brand_vocab_train_base"):
                require(stats[name] == actual_stats[name], f"Raw statistics mismatch: {name}")
            provenance = "VERIFIED_SHA256_AND_RAW_MAPPING"
    if perturbation_config is None:
        perturbation_config = yaml.safe_load(resolve_project_path("configs/metadata.yaml").read_text(encoding="utf-8"))["perturbations"]
    counts = validate_perturbations(directory, avail, perturbation_config)
    check_loader(directory, avail, perturbation_config)
    return dict(status="PASS", dataset=directory.name, items=n, oov_items=oov, embedding_shape=list(emb.shape),
                coverage=cov["train_base_catalog"], provenance=provenance, masks=counts[0], permutations=counts[1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata_dir", default="artifacts/metadata")
    parser.add_argument("--dataset", choices=["beauty","sports","toys","all"], default="all")
    parser.add_argument("--config", default="configs/metadata.yaml")
    parser.add_argument("--raw_root")
    parser.add_argument("--artifacts-only", action="store_true")
    parser.add_argument("--verify-raw-hash", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(resolve_project_path(args.config).read_text(encoding="utf-8"))["perturbations"]
    failed = False
    for key in (["beauty","sports","toys"] if args.dataset == "all" else [args.dataset]):
        try:
            result = validate_dataset_artifacts(resolve_project_path(args.metadata_dir)/key,
                        args.verify_raw_hash,args.raw_root,args.artifacts_only,config)
            print(json.dumps(result, ensure_ascii=False))
        except Exception as exc:
            print(f"[FAIL] {key}: {exc}", file=sys.stderr)
            failed = True
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
