"""Late-only checkpoint rerun. Never executes early, controls, K2--K4 or S3."""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['plan', 'check-data', 'run', 'report'], default='plan')
    parser.add_argument('--datasets', nargs='+', choices=['beauty', 'sports', 'toys'], default=['beauty'])
    parser.add_argument('--config', default='configs/experiments/checkpoint2_0611.yaml')
    parser.add_argument('--output', default='runs/checkpoint2_late_20261106')
    parser.add_argument('--execute-training', action='store_true')
    parser.add_argument('--early-output', help='Read-only completed early output root, for --stage report')
    args = parser.parse_args()
    if args.stage == 'plan':
        print(json.dumps(dict(execution='DISABLED', datasets=args.datasets, output=args.output,
            per_dataset=dict(id_jobs=3, late_jobs=21, total_jobs=24), seeds=[42, 43, 44],
            training='Train-base; frozen ID; residual source dropout 0.2 per batch',
            selection='max full-precision validation NDCG@10; trained epochs only',
            resume='verified completed jobs only; interrupted jobs restart in a new attempt',
            scope='No early training, no controls, no K1-K4, no S2/E1/S3'), indent=2))
        return 0
    if args.stage == 'run' and not args.execute_training:
        parser.error('run requires --execute-training on the locked RTX4060 host')
    if args.early_output and args.stage != 'report':
        parser.error('--early-output is only supported for --stage report')
    import yaml
    from taimma.experiments.late_checkpoint import LateCheckpoint
    config = yaml.safe_load((ROOT / args.config).read_text(encoding='utf-8'))
    runner = LateCheckpoint(ROOT, config, args.output, execute_training=args.execute_training)
    for dataset in args.datasets:
        if args.stage == 'check-data':
            result = runner.check_data(dataset)
        elif args.stage == 'run':
            result = runner.run_late(dataset)
        else:
            # CPU tables from completed ranks: no model is initialized or scored.
            result = runner.summarize(dataset, args.early_output)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
