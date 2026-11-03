"""
test_data_ingestion.py - Unit tests for Amazon 2014 data ingestion and path resolution.

Tests:
  - Resolution of on-disk Amazon 2014 raw directories and files.
  - Fail-fast with FileNotFoundError on non-existent paths.
  - Streaming SHA256 checksum calculation.
  - Processed data artifact serialization and manifest creation.
"""

import json
import os
import sys
import tempfile
import pytest

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TEST_DIR)
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from taimma.data.ingestion import Amazon2014Ingestor
from taimma.data.split import SequenceSample, SplitData, compute_file_sha256, save_processed_data


def test_resolve_paths_missing_directory_raises():
    """Missing dataset path must raise explicit FileNotFoundError."""
    ingestor = Amazon2014Ingestor(raw_data_dir="non_existent_dir_12345", dataset_key="Beauty")
    with pytest.raises(FileNotFoundError) as exc_info:
        ingestor.resolve_paths()
    assert "Missing raw reviews file" in str(exc_info.value)
    assert "Beauty" in str(exc_info.value)


def test_resolve_paths_real_beauty():
    """Verify that resolve_paths locates real on-disk Beauty_5.json."""
    raw_root = os.path.join(REPO_ROOT, "data", "raw")
    if not os.path.exists(raw_root):
        pytest.skip(f"Real data root not present: {raw_root}")

    ingestor = Amazon2014Ingestor(raw_data_dir=raw_root, dataset_key="Beauty")
    reviews_path, meta_path = ingestor.resolve_paths()

    assert os.path.isfile(reviews_path)
    assert reviews_path.endswith("Beauty_5.json")
    # Metadata is optional for ID-only runs and present after metadata preparation.
    if meta_path is not None:
        assert os.path.isfile(meta_path)


def test_resolve_paths_sports_and_toys():
    """Verify that resolve_paths locates real Sports and Outdoors and Toys and Games."""
    raw_root = os.path.join(REPO_ROOT, "data", "raw")
    if not os.path.exists(raw_root):
        pytest.skip(f"Real data root not present: {raw_root}")

    sports_ingestor = Amazon2014Ingestor(raw_data_dir=raw_root, dataset_key="Sports_and_Outdoors")
    sports_rev, _ = sports_ingestor.resolve_paths()
    assert os.path.isfile(sports_rev)
    assert "Sports_and_Outdoors_5.json" in sports_rev

    toys_ingestor = Amazon2014Ingestor(raw_data_dir=raw_root, dataset_key="Toys_and_Games")
    toys_rev, _ = toys_ingestor.resolve_paths()
    assert os.path.isfile(toys_rev)
    assert "Toys_and_Games_5.json" in toys_rev


def test_compute_file_sha256_and_save_processed():
    """Test streaming SHA256 and saving processed data manifest."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a dummy raw reviews file
        dummy_file = os.path.join(tmp_dir, "test_raw.json")
        with open(dummy_file, "w", encoding="utf-8") as f:
            f.write('{"reviewerID": "U1", "asin": "I1"}\n')

        sha = compute_file_sha256(dummy_file)
        assert len(sha) == 64  # SHA256 hex string

        # Mock SplitData
        split_data = SplitData(
            train_base_samples=[],
            train_meta_samples=[],
            val_samples=[],
            test_samples=[],
            item2idx={"I1": 1},
            idx2item={1: "I1"},
            user2idx={"U1": 0},
            num_items=1,
            num_users=1,
            oov_counts={"val": 0, "test": 0},
            seen_train_items_per_user={0: {1}},
        )

        out_dir = os.path.join(tmp_dir, "processed")
        paths = save_processed_data(
            processed_dir=out_dir,
            split_data=split_data,
            raw_reviews_path=dummy_file,
            config={"max_seq_len": 50},
        )

        assert os.path.isfile(paths["item2idx"])
        assert os.path.isfile(paths["user2idx"])
        assert os.path.isfile(paths["stats"])
        assert os.path.isfile(paths["manifest"])

        with open(paths["manifest"], "r", encoding="utf-8") as f:
            manifest = json.load(f)
        assert manifest["raw_reviews_file"] == dummy_file
        assert manifest["raw_reviews_sha256"] == sha
        assert manifest["stats"]["num_items"] == 1


@pytest.mark.parametrize("folder", ["", "Sports_and_Outdoors", "Sports and Outdoors"])
@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize("with_ratings", [False, True])
def test_resolve_dataset_layouts_and_parse_metadata(tmp_path, folder, compressed, with_ratings):
    import gzip

    directory = tmp_path / folder
    directory.mkdir(exist_ok=True)
    suffix = ".gz" if compressed else ""
    reviews = directory / ("Sports_and_Outdoors_5.json" + suffix)
    metadata = directory / ("meta_Sports_and_Outdoors.json" + suffix)
    ratings = directory / ("ratings_Sports_and_Outdoors.csv" + suffix)
    if with_ratings:
        ratings.write_bytes(b"optional ratings fixture")
    opener = gzip.open if compressed else open
    with opener(reviews, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(reviewerID="u", asin="a", unixReviewTime=1)) + "\n")
    with opener(metadata, "wt", encoding="utf-8") as handle:
        handle.write(repr(dict(asin="a", brand=" ACME  Corp ", categories=[["Sports", "Ball"]])) + "\n")
    ingestor = Amazon2014Ingestor(str(tmp_path), "Sports_and_Outdoors")
    assert ingestor.ratings_path is None
    assert ingestor.resolve_paths() == (str(reviews), str(metadata))
    assert ingestor.ratings_path == (str(ratings) if with_ratings else None)
    sequences, records = ingestor.build_interaction_sequences(min_interactions=1)
    assert sequences[0].items == ["a"]
    assert records["a"]["brand"] == "acme corp"
    assert records["a"]["categories"] == ["Ball"]
    assert ingestor.meta_path == str(metadata)
