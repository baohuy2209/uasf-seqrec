"""Compare saved checkpoint selection with exact validation histories."""
import argparse
import json
from pathlib import Path
import sys

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from taimma.data.split import compute_file_sha256


def audit(run_root, output):
    run_root = Path(run_root).resolve()
    progress = json.loads((run_root / 'progress.json').read_text())
    rows = []
    for key, entry in progress.items():
        if not isinstance(entry, dict) or 'checkpoint' not in entry or 'folder' not in entry:
            continue
        folder = Path(entry['folder'])
        history = json.loads((folder / 'history.json').read_text())
        checkpoint = torch.load(entry['checkpoint'], map_location='cpu', weights_only=True)
        values = history['val_ndcg_10']
        trained_best = max(values)
        trained_epoch = values.index(trained_best) + 1
        epoch = checkpoint['epoch']
        selected = checkpoint.get('val_ndcg_10', history['best_val_ndcg'])
        index = json.loads((folder / 'artifact_index.json').read_text())
        expected_sha = next(row['sha256'] for row in index['artifacts'] if row['path'] == 'checkpoint.pt')
        actual_sha = compute_file_sha256(entry['checkpoint'])
        checksum_ok = actual_sha == expected_sha and actual_sha == entry.get('checkpoint_sha256', actual_sha)
        initial_selected = epoch == 0
        scope = history.get('selection_scope', 'legacy_include_initial_validation' if key.startswith('late_') else 'trained_epochs_only')
        selection_ok = (epoch == history['best_epoch'] and
                        abs(selected - history['best_val_ndcg']) <= 1e-12 and
                        selected >= trained_best - 1e-12 and
                        ((initial_selected and scope == 'legacy_include_initial_validation') or epoch == trained_epoch))
        rows.append(dict(stage=key, checkpoint_epoch=epoch, history_best_epoch=history['best_epoch'],
                         selected_val_ndcg_10=selected, best_trained_epoch=trained_epoch,
                         best_trained_val_ndcg_10=trained_best, first_trained_val_ndcg_10=values[0],
                         last_trained_val_ndcg_10=values[-1], initial_checkpoint_selected=initial_selected,
                         selection_scope=scope,
                         checkpoint_sha256=actual_sha, checksum_verified=checksum_ok,
                         selection_verified=selection_ok, status='PASS' if selection_ok and checksum_ok else 'FAIL',
                         folder=str(folder)))
        del checkpoint
    if not rows:
        raise ValueError('No completed model checkpoints to audit')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / 'checkpoint_selection.csv', index=False)
    report = dict(run_root=str(run_root), status='PASS' if all(row['status'] == 'PASS' for row in rows) else 'FAIL',
                  models_checked=len(rows), epoch_zero_selected=sum(row['initial_checkpoint_selected'] for row in rows),
                  policy='Maximize full-precision validation NDCG@10; keep earliest tied epoch. '
                         'Use each run\'s recorded selection scope; historical late runs may include the initial checkpoint.', models=rows)
    (output / 'checkpoint_selection.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f"{report['status']}: {len(rows)} model checkpoints; {report['epoch_zero_selected']} selected epoch 0")
    print(output / 'checkpoint_selection.csv')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    raise SystemExit(0 if audit(args.run, args.output)['status'] == 'PASS' else 1)


if __name__ == '__main__':
    main()
