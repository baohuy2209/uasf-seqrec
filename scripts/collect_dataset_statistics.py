"""Collect Table 1 inputs from the actual three review files; never train models."""
import argparse
import json
from pathlib import Path
import sys

import pandas as pd
import yaml

TAIMMA_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TAIMMA_ROOT
sys.path.insert(0, str(TAIMMA_ROOT / 'src'))
from taimma.paths import resolve_project_path
from taimma.data.ingestion import Amazon2014Ingestor
from taimma.data.split import SequenceSplitter, compute_file_sha256
from taimma.experiments.reporting import dataset_statistics, write_json


def main():
    parser = argparse.ArgumentParser(description='Read-only Table 1 collection from actual Amazon 2014 inputs')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--raw_root', default=str(PROJECT_ROOT / 'data/raw'))
    args = parser.parse_args()
    args.raw_root = str(resolve_project_path(args.raw_root))
    destination = resolve_project_path(args.output_dir)
    if destination.exists():
        raise FileExistsError(f'Choose a new output directory: {destination}')
    # Validate all review paths before writing the output bundle.
    sources = [(key, filename) for key, filename in (
        ('Beauty', 'beauty_2014.yaml'), ('Sports_and_Outdoors', 'sports_and_outdoors_2014.yaml'),
        ('Toys_and_Games', 'toys_and_games_2014.yaml'))]
    for key, _ in sources:
        Amazon2014Ingestor(args.raw_root, key).resolve_paths()
    destination.mkdir(parents=True, exist_ok=False)
    rows = []
    for key, filename in sources:
        config = yaml.safe_load((TAIMMA_ROOT / 'configs/datasets' / filename).read_text(encoding='utf-8'))
        ingestor = Amazon2014Ingestor(args.raw_root, key)
        sequences, metadata = ingestor.build_interaction_sequences(config['min_interactions'])
        split = SequenceSplitter(config['max_seq_len'], config['min_interactions']).split(sequences)
        stats = dataset_statistics(sequences, split, metadata, ingestor.meta_path is not None,
                                   config['dataset_name'], config['expected_stats'])
        stats.update(reviews_file=ingestor.reviews_path, reviews_sha256=compute_file_sha256(ingestor.reviews_path),
                     metadata_file=ingestor.meta_path,
                     metadata_sha256=compute_file_sha256(ingestor.meta_path) if ingestor.meta_path else None,
                     preprocessing_config=config)
        write_json(destination / (key + '.json'), stats)
        row = {k: stats[k] for k in ('dataset', 'users', 'original_items', 'catalog_train_base_items', 'interactions',
                                    'sparsity', 'mean_sequence_length', 'metadata_status', 'category_vocab_train_base',
                                    'brand_vocab_train_base', 'reference_users_items_match')}
        row.update({'coverage_' + k: v for k, v in stats['metadata_coverage_original_items'].items()})
        rows.append(row)
    with (destination / 'table1.csv').open('x', encoding='utf-8', newline='') as f:
        pd.DataFrame(rows).to_csv(f, index=False)
    write_json(destination / 'manifest.json', {'data_mode': 'real', 'training_performed': False,
               'datasets': [row['dataset'] for row in rows],
               'missing_metadata_is_null_not_zero': True})
    print(f'Table 1 inputs saved to {destination.resolve()} (no model training)')


if __name__ == '__main__':
    main()
