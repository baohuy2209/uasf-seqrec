"""
download_amazon2014.py - Acquisition script for official Amazon 2014 benchmark dumps.

Downloads 5-core reviews and metadata from the official Stanford SNAP / Julian McAuley repository.
"""

import argparse
import os
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from taimma.paths import resolve_project_path

DATASET_URLS = {
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


def download_file(url: str, dest_path: str):
    """Download file with progress indicator."""
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        print(f"File already exists: {dest_path} ({os.path.getsize(dest_path) / 1024 / 1024:.2f} MB)")
        return

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    print(f"Downloading from: {url}")
    print(f"Saving to:       {dest_path}")

    def progress_callback(blocks, block_size, total_size):
        downloaded = blocks * block_size
        if total_size > 0:
            percent = downloaded / total_size * 100
            sys.stdout.write(
                f"\rProgress: {percent:.1f}% ({downloaded / 1024 / 1024:.1f}/{total_size / 1024 / 1024:.1f} MB)"
            )
            sys.stdout.flush()

    urllib.request.urlretrieve(url, dest_path, reporthook=progress_callback)
    print("\nDownload complete!")


def main():
    parser = argparse.ArgumentParser(description="Download official Amazon 2014 raw benchmark data.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="Beauty",
        choices=["Beauty", "Sports_and_Outdoors", "Toys_and_Games"],
        help="Dataset category to download",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/raw",
        help="Target raw directory",
    )
    parser.add_argument(
        "--include_metadata",
        action="store_true",
        default=True,
        help="Download metadata file alongside 5-core reviews",
    )
    args = parser.parse_args()
    args.output_dir = str(resolve_project_path(args.output_dir))

    urls = DATASET_URLS[args.dataset]
    rev_dest = os.path.join(args.output_dir, f"reviews_{args.dataset}_5.json.gz")
    download_file(urls["reviews"], rev_dest)

    if args.include_metadata:
        meta_dest = os.path.join(args.output_dir, f"meta_{args.dataset}.json.gz")
        download_file(urls["metadata"], meta_dest)


if __name__ == "__main__":
    main()
