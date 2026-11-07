"""Only fill Table 5 from existing Beauty early-SD checkpoints. Never train."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/experiments/completion_0711.yaml')
    parser.add_argument('--early-root', default='runs/checkpoint2_full_20261106')
    parser.add_argument('--output', default='runs/completion_utility_table5_20261107')
    parser.add_argument('--publish-dir', default='results/utility_table5_20261107')
    parser.add_argument('--execute-inference', action='store_true', help='Without this flag, only check original checkpoint files')
    args = parser.parse_args()
    import yaml
    from taimma.experiments.table5_utility import Table5Utility, check_references, complete_table5
    early = (ROOT / args.early_root).resolve()
    evidence = check_references(early)
    print(json.dumps(dict(references=evidence, training='FORBIDDEN'), indent=2), flush=True)
    if not args.execute_inference:
        print('Check passed. Add --execute-inference to compute ONLY Table 5.'); return 0
    cfg = copy.deepcopy(yaml.safe_load((ROOT / args.config).read_text(encoding='utf-8')))
    cfg['datasets'] = ['beauty']
    cfg['input_roots']['early'] = str(early)
    run = Table5Utility(ROOT, cfg, args.output, execute_training=False, execute_inference=True)
    print('Table 5: early_sd_42/43 only; no grids, controls, K4, gates or training.', flush=True)
    print(complete_table5(run, early, ROOT / args.publish_dir))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
