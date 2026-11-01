"""Measure one real Beauty Train-base epoch with 1, 2 and 3 CUDA processes."""
import argparse
import multiprocessing as mp
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys
import time
import traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def worker(ready,results,start,dim,seed,batch_size,threads):
    try:
        import torch
        from torch.utils.data import DataLoader
        from taimma.data.ingestion import Amazon2014Ingestor
        from taimma.data.split import SequenceSplitter
        from taimma.models.sasrec import SASRecModel
        from taimma.evaluation.evaluator import IndependentEvaluator
        from taimma.training.trainer import SASRecTrainer,SequenceDataset,set_seed
        torch.set_num_threads(threads)
        set_seed(seed)
        seq,_=Amazon2014Ingestor(str(ROOT/'data/raw')).build_interaction_sequences(include_metadata=False)
        split=SequenceSplitter().split(seq)
        model=SASRecModel(split.num_items,embedding_dim=dim)
        trainer=SASRecTrainer(model,IndependentEvaluator(split.num_items),device='cuda',seed=seed)
        loader=DataLoader(SequenceDataset(split.train_base_samples),batch_size=batch_size,shuffle=True,
                          generator=torch.Generator().manual_seed(seed))
        ready.put(True)
        start.wait()
        torch.cuda.reset_peak_memory_stats()
        then=time.perf_counter()
        loss=trainer.train_epoch(loader)
        torch.cuda.synchronize()
        results.put(dict(samples=len(split.train_base_samples),train_sec=time.perf_counter()-then,loss=loss,
                         peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated()))
    except BaseException:
        ready.put(False)
        results.put(dict(error=traceback.format_exc()))


def main():
    from metadata_pipeline.contracts import write_json
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dim',type=int,default=64)
    parser.add_argument('--output',required=True)
    parser.add_argument('--batch_size',type=int,default=256)
    parser.add_argument('--threads',type=int,default=2)
    args=parser.parse_args()
    context=mp.get_context('spawn')
    rows=[]
    for count in [1,2,3]:
        ready,results=context.Queue(),context.Queue()
        start=context.Event()
        processes=[context.Process(target=worker,args=(ready,results,start,args.dim,42+i,args.batch_size,args.threads)) for i in range(count)]
        for p in processes:p.start()
        initialized=[ready.get(timeout=300) for _ in processes]
        begin=time.perf_counter()
        start.set()
        samples=[results.get(timeout=600) for _ in processes]
        for p in processes:p.join(timeout=30)
        elapsed=time.perf_counter()-begin
        errors=[r['error'] for r in samples if 'error' in r]
        row=dict(processes=count,elapsed_training_sec=elapsed,workers=samples,errors=errors,
                 examples_per_sec=sum(r.get('samples',0) for r in samples)/elapsed if not errors else 0)
        rows.append(row)
        print(row,flush=True)
        write_json(Path(args.output),dict(scope='one full real Beauty ID Train-base training epoch, excludes loading and validation',
                    dimension=args.dim,measurements=rows,best_workers=max(rows,key=lambda r:r['examples_per_sec'])['processes']))


if __name__=='__main__':main()
