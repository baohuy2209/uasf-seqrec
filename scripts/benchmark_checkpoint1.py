"""Measure one Beauty epoch up to the requested CUDA worker and RAM limits."""
import argparse
import multiprocessing as mp
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import queue
import sys
import time
import traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def worker(ready,results,start,dim,seed,batch_size,threads):
    initialized=False
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
        ready.put(dict(pid=os.getpid()))
        initialized=True
        start.wait()
        torch.cuda.reset_peak_memory_stats()
        then=time.perf_counter()
        loss=trainer.train_epoch(loader)
        torch.cuda.synchronize()
        import psutil
        memory=psutil.Process().memory_info()
        peak_rss=getattr(memory,'peak_wset',memory.rss)
        try:
            import resource
            peak_rss=max(peak_rss,resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform=='darwin' else 1024))
        except ImportError:
            pass
        results.put(dict(pid=os.getpid(),samples=len(split.train_base_samples),train_sec=time.perf_counter()-then,loss=loss,
                         peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_host_rss_bytes=peak_rss))
    except BaseException:
        error=dict(pid=os.getpid(),error=traceback.format_exc())
        if not initialized:
            ready.put(error)
        results.put(error)


def available_memory_bytes(cgroup=Path('/sys/fs/cgroup')):
    import psutil
    available=psutil.virtual_memory().available
    try:
        limit=(cgroup/'memory.max').read_text().strip()
        if limit != 'max':
            used=int((cgroup/'memory.current').read_text())
            stats=dict(line.split() for line in (cgroup/'memory.stat').read_text().splitlines())
            # Inactive file cache can be reclaimed without killing a process.
            available=min(available,max(0,int(limit)-used+int(stats.get('inactive_file',0))))
    except (OSError,ValueError):
        pass
    return available


def collect_messages(channel,processes,timeout,phase,parent_pid=None):
    deadline=time.monotonic()+timeout
    messages=[]
    while len(messages)<len(processes):
        if parent_pid is not None and os.getppid()!=parent_pid:
            raise RuntimeError('Training parent exited; stopping benchmark workers')
        remaining=deadline-time.monotonic()
        if remaining<=0:
            raise RuntimeError(f'Benchmark {phase} timed out after {timeout}s')
        try:
            message=channel.get(timeout=min(.2,remaining))
        except queue.Empty:
            failed=[(p.pid,p.exitcode) for p in processes if p.exitcode not in (None,0)]
            if failed:
                raise RuntimeError(f'Benchmark worker exited during {phase}: {failed}; check RAM/VRAM limits')
            if all(p.exitcode is not None for p in processes):
                raise RuntimeError(f'Benchmark workers exited without all {phase} results')
            continue
        if message.get('error'):
            raise RuntimeError(message['error'])
        messages.append(message)
    return messages


def measure(context,count,args):
    ready,results=context.Queue(),context.Queue()
    start=context.Event()
    processes=[context.Process(target=worker,args=(ready,results,start,args.dim,42+i,args.batch_size,args.threads))
               for i in range(count)]
    begin=time.perf_counter()
    try:
        for process in processes:process.start()
        collect_messages(ready,processes,300,'initialization',args.parent_pid)
        begin=time.perf_counter()
        start.set()
        samples=collect_messages(results,processes,600,'training',args.parent_pid)
        elapsed=time.perf_counter()-begin
        return dict(processes=count,status='COMPLETED',elapsed_training_sec=elapsed,workers=samples,errors=[],
                    examples_per_sec=sum(row['samples'] for row in samples)/elapsed)
    except RuntimeError as error:
        return dict(processes=count,status='FAILED',elapsed_training_sec=time.perf_counter()-begin,
                    workers=[],errors=[str(error)],examples_per_sec=0)
    finally:
        for process in processes:
            if process.pid is not None and process.is_alive():process.terminate()
        for process in processes:
            if process.pid is not None:
                process.join(timeout=5)
                if process.is_alive():
                    process.kill();process.join(timeout=5)
        for channel in [ready,results]:
            channel.close()
            channel.join_thread()


def benchmark(args,context=None):
    from metadata_pipeline.contracts import write_json
    context=context or mp.get_context('spawn')
    rows=[]
    worker_bytes=1024**3
    for count in range(1,args.workers+1):
        if args.parent_pid is not None and os.getppid()!=args.parent_pid:
            raise RuntimeError('Training parent exited; stopping benchmark')
        available=available_memory_bytes()
        required=int(worker_bytes*count*1.25)+512*1024**2
        if available<required:
            row=dict(processes=count,status='SKIPPED_INSUFFICIENT_RAM',workers=[],errors=[],
                     available_host_bytes=available,required_host_bytes=required,examples_per_sec=0)
        else:
            row=measure(context,count,args)
            if row['status']=='COMPLETED':
                worker_bytes=max(worker_bytes,max(sample['peak_host_rss_bytes'] for sample in row['workers']))
        rows.append(row)
        successful=[row for row in rows if row['status']=='COMPLETED']
        report=dict(scope='one full real Beauty ID Train-base epoch; excludes loading and validation',
                    dimension=args.dim,requested_max_workers=args.workers,measurements=rows,
                    best_workers=max(successful,key=lambda row:row['examples_per_sec'])['processes'] if successful else None)
        print(row,flush=True)
        output=Path(args.output)
        output.parent.mkdir(parents=True,exist_ok=True)
        temp=output.with_suffix('.tmp.json')
        write_json(temp,report);temp.replace(output)
        if row['status']!='COMPLETED':break
    if report['best_workers'] is None:
        raise RuntimeError('No safe successful CUDA benchmark; free RAM or run the driver with --workers 1 to skip it')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dim',type=int,default=64)
    parser.add_argument('--output',required=True)
    parser.add_argument('--batch_size',type=int,default=256)
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--workers',type=int,choices=[1,2,3],default=1)
    parser.add_argument('--parent-pid',type=int,help='Stop if the invoking training process exits')
    args=parser.parse_args()
    benchmark(args)


if __name__=='__main__':main()
