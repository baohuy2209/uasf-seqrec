"""Load aligned catalog tensors, including inference-time perturbations and training contrasts."""
import json
from pathlib import Path
import numpy as np
import torch
from .contracts import SOURCES, check_availability, flags, read_json, read_table, require


def load_catalog(dataset_dir, mask_files=None, permutation_files=None, random_files=None):
    """Return attrs/masks in ID order; permutations or random contrasts apply to history and candidates.

    Values at model ID i live at row i-1. Padding ID zero is handled by the
    sequence loader and is never a row in these arrays.
    """
    directory = Path(dataset_dir)
    manifest = read_json(directory/"manifest.json")
    require(manifest.get("schema_version") == 2 and manifest["title_embedding"]["mode"] == "sentence-transformers",
            "Schema v2 with real Title embeddings required")
    meta, avail = read_table(directory/"metadata.csv"), read_table(directory/"availability_mask.csv")
    check_availability(meta)
    check_availability(avail)
    require(len(meta) == len(avail), "Metadata/mask row count mismatch")

    labels = [json.loads(value) for value in meta.category_indices]
    cat = np.zeros((len(meta), max(1, max(map(len, labels), default=0))), dtype=np.int64)
    for i, values in enumerate(labels):
        cat[i, :len(values)] = values
    attrs = dict(category=torch.from_numpy(cat), brand=torch.tensor(meta.brand_idx.to_numpy(dtype=np.int64)),
                 title=torch.from_numpy(np.load(directory/"title_embeddings.npy", allow_pickle=False)))
    masks = {}
    for source in SOURCES:
        original = flags(avail[f"has_{source}"])
        require(np.array_equal(original, flags(meta[f"has_{source}"])), f"{source} mask differs from metadata")

        # Shuffled / Permuted contrast
        if source in (permutation_files or {}):
            table = read_table(permutation_files[source])
            require(np.array_equal(table.model_item_id, meta.model_item_id), "Permutation ID alignment")
            donor = table.permuted_source_item_id.to_numpy()
            require(np.array_equal(np.sort(donor), meta.model_item_id), "Permutation must be bijective")
            donor = donor.astype(np.int64) - 1
            require(np.array_equal(original[donor], original), "Permutation changed natural missingness")
            require(np.array_equal(donor[~original], np.flatnonzero(~original)), "Missing donors must stay fixed")
            attrs[source] = attrs[source][torch.from_numpy(donor)]

        # Random contrast
        if source in (random_files or {}):
            rand_path = Path(random_files[source])
            require(rand_path.exists(), f"Random contrast file missing: {rand_path}")
            if source == "category":
                table = read_table(rand_path)
                require(np.array_equal(table.model_item_id, meta.model_item_id), "Random category ID alignment")
                rand_labels = [json.loads(value) for value in table.random_category_indices]
                rand_cat = np.zeros((len(meta), max(1, max(map(len, rand_labels), default=0))), dtype=np.int64)
                for i, values in enumerate(rand_labels):
                    rand_cat[i, :len(values)] = values
                require(np.all(rand_cat[~original] == 0), "Missing items must have zero category")
                attrs["category"] = torch.from_numpy(rand_cat)
            elif source == "brand":
                table = read_table(rand_path)
                require(np.array_equal(table.model_item_id, meta.model_item_id), "Random brand ID alignment")
                rand_brand = table.random_brand_idx.to_numpy(dtype=np.int64)
                require(np.all(rand_brand[~original] == 0), "Missing items must have zero brand")
                attrs["brand"] = torch.tensor(rand_brand)
            elif source == "title":
                rand_vecs = np.load(rand_path, allow_pickle=False)
                require(rand_vecs.shape == (len(meta), 384), "Random title shape mismatch")
                require(np.all(rand_vecs[~original] == 0), "Missing items must have zero title vector")
                attrs["title"] = torch.from_numpy(rand_vecs)

        active = original.copy()
        if source in (mask_files or {}):
            table = read_table(mask_files[source])
            require(np.array_equal(table.model_item_id, meta.model_item_id), "Synthetic mask ID alignment")
            require(np.array_equal(flags(table.original_available), original), "Synthetic mask base mismatch")
            active = flags(table.simulated_available)
            require(not np.any(active & ~original), "Synthetic mask resurrected missing metadata")
        masks[source] = torch.tensor(active, dtype=torch.float32)
    return attrs, masks
