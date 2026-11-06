"""Explicit, resumable execution for the supervisor's 03 November feedback."""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

STAGES=['plan','prepare','early-grid','controls','diagnostics','source-dropout',
        'groups','utility-alignment','choose','s2-e1','s3','transfer-late','transfer',
        'report','archive','all']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/experiments/checkpoint2_0611.yaml')
    parser.add_argument('--output',default=None)
    parser.add_argument('--datasets',nargs='+',choices=['beauty','sports','toys'],default=['beauty'])
    parser.add_argument('--stage',choices=STAGES,default='plan')
    parser.add_argument('--execute-training',action='store_true',help='Explicitly permit real model training on locked RTX4060')
    parser.add_argument('--execute-inference',action='store_true',help='Permit checkpoint inference; never enables training')
    parser.add_argument('--backbone',choices=['early_sd','late_joint'])
    parser.add_argument('--reason',default='')
    parser.add_argument('--archive-dir')
    args=parser.parse_args()
    import yaml
    cfg_path=(ROOT/args.config).resolve()
    config=yaml.safe_load(cfg_path.read_text(encoding='utf-8'))
    if args.stage=='plan':
        print(json.dumps(dict(execution='DISABLED',datasets=args.datasets,environment=config['environment'],
            stages=STAGES,plan=['prepare input checks and context-only grouping',
                'early grid: ID plus seven metadata subsets x three seeds',
                'training-time controls: three sources x two kinds x three seeds',
                'K1 CE, K2 held-out meta, K3 strong regularization, K4 joint fresh model',
                'early SD x three seeds; all eight same-model subset evaluations',
                'group selection on validation and descriptive test oracle',
                'preregistered CE/log-rank alignment and two-page mini report',
                'explicit backbone decision before selected-model S2/E1 and eligible S3',
                'Sports/Toys transfer in a second process with --stage transfer-late',
                'shared ZIP archive plus SHA256 manifests'],
            note='This command does not read raw data, initialize models, train, or run model inference.'),indent=2))
        return 0
    from taimma.experiments.checkpoint2 import Checkpoint2
    runner=Checkpoint2(ROOT,config,args.output or config['output'],
        execute_training=args.execute_training,execute_inference=args.execute_inference)
    if args.stage=='archive':
        if not args.archive_dir:parser.error('--archive-dir is required')
        print(runner.archive(args.archive_dir));return 0
    for dataset in args.datasets:
        if args.stage=='report':
            from taimma.experiments.mini_checkpoint_report import write_mini_report
            path=runner.output/dataset/'mini_checkpoint'
            print(write_mini_report(runner.mini_state(dataset),path));continue
        if args.stage=='choose':
            if not args.backbone or not args.reason:parser.error('choose requires --backbone and --reason')
            runner.lock_backbone(dataset,args.backbone,args.reason);continue
        if args.stage=='groups':runner.group_analysis(dataset);continue
        runner.prepare(dataset)
        if args.stage=='prepare':continue
        if args.stage in ['early-grid','all']:runner.early_grid(dataset)
        if args.stage in ['controls','all']:runner.controls(dataset)
        if args.stage in ['diagnostics','all']:
            runner.id_grid(dataset);runner.diagnostics(dataset)
        if args.stage in ['source-dropout','all']:runner.source_dropout(dataset)
        if args.stage in ['utility-alignment','all']:
            if runner.completed(dataset,'late_reference_42'):
                runner.alignment(dataset,['late_reference_42'],'late_alignment')
            elif args.stage=='utility-alignment':raise RuntimeError('Complete diagnostics first to obtain late_reference_42')
        if args.stage=='transfer-late':
            from taimma.training.checkpoint2_trainer import ModelSpec
            from taimma.experiments.checkpoint2_protocol import SEEDS,SUBSETS
            from taimma.experiments.checkpoint2 import subset_name
            runner.id_grid(dataset)
            for seed in SEEDS:
                for sources in SUBSETS[1:]:
                    runner.run_job(dataset,f'late_{subset_name(sources)}_{seed}',
                        ModelSpec('late',seed,sources=sources,source_dropout=.2,source_dropout_mode='batch'),id_job=f'id_{seed}')
        if args.stage=='transfer':runner.transfer(dataset)
        if args.stage=='all':runner.group_analysis(dataset)
        if args.stage in ['s2-e1','all'] and (runner.output/dataset/'backbone_decision.json').exists():runner.s2_e1(dataset)
        elif args.stage=='s2-e1':raise RuntimeError('Run choose with validation-based backbone and reason before S2/E1')
        if args.stage=='s3' and (runner.output/dataset/'selected_s2/e0_e1.json').exists():runner.s3(dataset)
        elif args.stage=='s3':raise RuntimeError('S3 requires completed eligible S2/E1')
        from taimma.experiments.mini_checkpoint_report import write_mini_report
        print(write_mini_report(runner.mini_state(dataset),runner.output/dataset/'mini_checkpoint'))
    return 0


if __name__=='__main__':raise SystemExit(main())
