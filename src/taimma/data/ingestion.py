"""
ingestion.py - Ingestion and parsing for Amazon 2014 datasets (Beauty, Sports, Toys).

Follows Strict Data-First:
  - Fails fast with FileNotFoundError if real dataset files are missing.
  - Safe parsing of 2014 metadata using ast.literal_eval / json.
  - Deterministic timestamp sorting with tie-breaking.
"""

import ast
import gzip
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from ..paths import resolve_project_path


@dataclass
class InteractionSequence:
    user_id: str
    items: List[str]
    timestamps: List[int]
    ratings: List[float] = field(default_factory=list)


class Amazon2014Ingestor:
    """Ingest and preprocess Amazon 2014 raw review and metadata dumps."""

    OFFICIAL_URLS = {
        "Beauty": {
            "reviews": "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Beauty_5.json.gz",
            "metadata": "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Beauty.json.gz",
        },
        "Sports_and_Outdoors": {
            "reviews": "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Sports_and_Outdoors_5.json.gz",
            "metadata": "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Sports_and_Outdoors.json.gz",
        },
        "Toys_and_Games": {
            "reviews": "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Toys_and_Games_5.json.gz",
            "metadata": "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Toys_and_Games.json.gz",
        },
    }

    def __init__(self, raw_data_dir: str = "data/raw", dataset_key: str = "Beauty"):
        self.raw_data_dir = str(resolve_project_path(raw_data_dir))
        self.dataset_key = dataset_key
        self.reviews_path: Optional[str] = None
        self.ratings_path: Optional[str] = None
        self.meta_path: Optional[str] = None

    def resolve_paths(self) -> Tuple[str, Optional[str]]:
        """
        Locate raw reviews, ratings, and metadata files using pathlib.Path.
        Searches:
          - Directly in raw_data_dir
          - In category subdirectory (e.g. raw_data_dir / 'Beauty')
        Relative paths are anchored to the standalone checkout; there is no fallback dataset.
        Returns (reviews_path, meta_path) where meta_path is None if metadata is absent.
        Also caches self.reviews_path, self.ratings_path, self.meta_path.
        Fails fast with explicit FileNotFoundError if reviews file is not found.
        """
        norm_key = self.dataset_key
        spaced_key = self.dataset_key.replace("_", " ")

        candidate_dirs: List[Path] = []
        raw_dir = Path(self.raw_data_dir)
        if raw_dir.is_dir():
            candidate_dirs.append(raw_dir)
            if (raw_dir / norm_key).is_dir():
                candidate_dirs.append(raw_dir / norm_key)
            if spaced_key != norm_key and (raw_dir / spaced_key).is_dir():
                candidate_dirs.append(raw_dir / spaced_key)

        reviews_candidates = [
            f"{norm_key}_5.json",
            f"reviews_{norm_key}_5.json",
            f"{norm_key}_5.json.gz",
            f"reviews_{norm_key}_5.json.gz",
            f"{spaced_key}_5.json",
            f"reviews_{spaced_key}_5.json",
            f"{spaced_key}_5.json.gz",
            f"reviews_{spaced_key}_5.json.gz",
        ]

        ratings_candidates = [
            f"ratings_{norm_key}.csv",
            f"ratings_{spaced_key}.csv",
            f"{norm_key}_ratings.csv",
            f"{spaced_key}_ratings.csv",
            f"ratings_{norm_key}.csv.gz",
            f"ratings_{spaced_key}.csv.gz",
            f"{norm_key}_ratings.csv.gz",
            f"{spaced_key}_ratings.csv.gz",
        ]

        meta_candidates = [
            f"meta_{norm_key}.json.gz",
            f"meta_{norm_key}.json",
            f"meta_{spaced_key}.json.gz",
            f"meta_{spaced_key}.json",
            f"{norm_key}_meta.json",
            f"{spaced_key}_meta.json",
        ]

        reviews_path: Optional[str] = None
        for d in candidate_dirs:
            for rc in reviews_candidates:
                candidate_path = d / rc
                if candidate_path.is_file() and candidate_path.stat().st_size > 0:
                    reviews_path = str(candidate_path.resolve())
                    break
            if reviews_path:
                break

        ratings_path: Optional[str] = None
        for d in candidate_dirs:
            for rtc in ratings_candidates:
                candidate_path = d / rtc
                if candidate_path.is_file() and candidate_path.stat().st_size > 0:
                    ratings_path = str(candidate_path.resolve())
                    break
            if ratings_path:
                break

        meta_path: Optional[str] = None
        for d in candidate_dirs:
            for mc in meta_candidates:
                candidate_path = d / mc
                if candidate_path.is_file() and candidate_path.stat().st_size > 0:
                    meta_path = str(candidate_path.resolve())
                    break
            if meta_path:
                break

        if not reviews_path:
            searched_locations = [str(d) for d in candidate_dirs]
            url_info = self.OFFICIAL_URLS.get(self.dataset_key, {}).get("reviews", "N/A")
            raise FileNotFoundError(
                f"[TAIMMA Data Error] Missing raw reviews file for dataset '{self.dataset_key}'.\n"
                f"Searched directories: {searched_locations}\n"
                f"Checked filename patterns: {reviews_candidates[:4]}\n"
                f"Official download URL: {url_info}\n"
                f"Action: Verify raw data directory or run 'python scripts/download_amazon2014.py --dataset {self.dataset_key}'."
            )

        self.reviews_path = reviews_path
        self.ratings_path = ratings_path
        self.meta_path = meta_path
        return reviews_path, meta_path

    def resolve_all_paths(self) -> Tuple[str, Optional[str], Optional[str]]:
        """Return (reviews_path, ratings_path, meta_path)."""
        self.resolve_paths()
        return self.reviews_path, self.ratings_path, self.meta_path

    def parse_ratings_csv(self, ratings_path: Optional[str] = None) -> Dict[str, List[Tuple[str, int, float]]]:
        """
        Parse ratings CSV file (format: reviewerID, asin, rating, unixReviewTime).
        Used when investigating or comparing the unpruned ratings dump against 5-core reviews.
        """
        import csv
        path_to_open = ratings_path or self.ratings_path
        if not path_to_open or not os.path.exists(path_to_open):
            return {}
        user_interactions: Dict[str, List[Tuple[str, int, float]]] = {}
        with open(path_to_open, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or len(row) < 4:
                    continue
                reviewer_id, asin, rating_str, ts_str = row[0], row[1], row[2], row[3]
                try:
                    rating = float(rating_str)
                    unix_time = int(ts_str)
                except ValueError:
                    continue
                if reviewer_id not in user_interactions:
                    user_interactions[reviewer_id] = []
                user_interactions[reviewer_id].append((asin, unix_time, rating))
        return user_interactions

    @staticmethod
    def _open_file(filepath: str):
        if filepath.endswith(".gz"):
            return gzip.open(filepath, "rt", encoding="utf-8")
        return open(filepath, "r", encoding="utf-8")

    def parse_reviews(self, reviews_path: str) -> Dict[str, List[Tuple[str, int, float]]]:
        """Parse review interactions and group by reviewerID."""
        user_interactions: Dict[str, List[Tuple[str, int, float]]] = {}
        with self._open_file(reviews_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except Exception:
                    try:
                        record = ast.literal_eval(line)
                    except (ValueError, SyntaxError) as exc:
                        raise ValueError(f'Invalid record in {getattr(f, "name", "input")}: {line[:80]}') from exc

                reviewer_id = record.get("reviewerID")
                asin = record.get("asin")
                unix_time = int(record.get("unixReviewTime", 0))
                rating = float(record.get("overall", 1.0))

                if reviewer_id and asin:
                    if reviewer_id not in user_interactions:
                        user_interactions[reviewer_id] = []
                    user_interactions[reviewer_id].append((asin, unix_time, rating))

        return user_interactions

    def parse_metadata(self, meta_path: Optional[str]) -> Dict[str, Dict[str, Optional[str]]]:
        """Parse item metadata safely with Category, Brand, and Title normalization."""
        if not meta_path or not os.path.exists(meta_path):
            return {}

        meta_dict: Dict[str, Dict[str, Optional[str]]] = {}
        with self._open_file(meta_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except Exception:
                    try:
                        record = ast.literal_eval(line)
                    except (ValueError, SyntaxError) as exc:
                        raise ValueError(f'Invalid record in {getattr(f, "name", "input")}: {line[:80]}') from exc

                asin = record.get("asin")
                if not asin:
                    continue

                # Brand normalization: case and whitespace normalized, empty and Unknown treated as missing
                raw_brand = record.get("brand")
                brand = None
                if raw_brand and isinstance(raw_brand, str):
                    clean_brand = " ".join(raw_brand.split())
                    if clean_brand and clean_brand.lower() not in {"unknown", ""}:
                        brand = clean_brand.casefold()

                # Title
                raw_title = record.get("title")
                title = str(raw_title).strip() if raw_title else None

                # Category: Remove root hierarchy level, preserve remaining nodes across applicable paths (multilabel)
                raw_categories = record.get("categories", [])
                multilabel_cats: List[str] = []
                if raw_categories and isinstance(raw_categories, list):
                    for path in raw_categories:
                        if isinstance(path, list):
                            clean_nodes = [str(c).strip() for c in path if str(c).strip()]
                            # Remove root hierarchy level (node 0)
                            sub_nodes = clean_nodes[1:] if len(clean_nodes) > 1 else []
                            for node in sub_nodes:
                                if node not in multilabel_cats:
                                    multilabel_cats.append(node)
                        elif isinstance(path, str) and path.strip():
                            c_str = path.strip()
                            if c_str not in multilabel_cats:
                                multilabel_cats.append(c_str)

                meta_dict[asin] = {
                    "brand": brand,
                    "title": title,
                    "category": multilabel_cats[-1] if multilabel_cats else None,
                    "categories": multilabel_cats,
                }

        return meta_dict

    def build_interaction_sequences(self, min_interactions: int = 5, include_metadata: bool = True) -> Tuple[List[InteractionSequence], Dict[str, Dict[str, Optional[str]]]]:
        """
        Build sorted interaction sequences with deterministic timestamp tie-breaking.
        Applies min_interactions (k-core) filter.
        """
        reviews_path, meta_path = self.resolve_paths()
        raw_users = self.parse_reviews(reviews_path)
        meta_dict = self.parse_metadata(meta_path) if include_metadata else {}

        sequences: List[InteractionSequence] = []
        for user_id, inters in raw_users.items():
            if len(inters) < min_interactions:
                continue

            # Deterministic sorting: primary key timestamp, secondary key asin
            sorted_inters = sorted(inters, key=lambda x: (x[1], x[0]))

            # Remove consecutive immediate duplicates if any
            dedup_items: List[str] = []
            dedup_timestamps: List[int] = []
            dedup_ratings: List[float] = []

            for asin, ts, rating in sorted_inters:
                if not dedup_items or dedup_items[-1] != asin:
                    dedup_items.append(asin)
                    dedup_timestamps.append(ts)
                    dedup_ratings.append(rating)

            if len(dedup_items) >= min_interactions:
                sequences.append(
                    InteractionSequence(
                        user_id=user_id,
                        items=dedup_items,
                        timestamps=dedup_timestamps,
                        ratings=dedup_ratings,
                    )
                )

        return sequences, meta_dict
