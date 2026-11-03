"""Worker limits, memory preflight and benchmark child failure regressions."""
import importlib.util
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import sys
import time
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def load_script(name):
    spec=importlib.util.spec_from_file_location('benchmark_test_'+name,ROOT/'scripts'/f'{name}.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('device,workers',[('cuda',1),('cpu',3)])
def test_sequential_driver_skips_benchmark(tmp_path,monkeypatch,device,workers):
    driver=load_script('run_checkpoint1')
    runner=driver.Checkpoint.__new__(driver.Checkpoint)
    runner.root=tmp_path
    runner.device=driver.torch.device(device)
    runner.args=SimpleNamespace(workers=workers)
    monkeypatch.setattr(driver.subprocess,'run',lambda *args,**kwargs:pytest.fail('Sequential execution must not spawn a benchmark'))
    runner.configure_workers()
    assert runner.args.workers==1
    report=json.loads((tmp_path/'throughput.json').read_text())
    assert report['measurements']==[]
    assert report['best_workers']==1


def test_driver_passes_worker_cap_and_parent_pid(tmp_path,monkeypatch):
    driver=load_script('run_checkpoint1')
    runner=driver.Checkpoint.__new__(driver.Checkpoint)
    runner.root=tmp_path
    runner.device=driver.torch.device('cuda')
    runner.backbone_dim=64
    runner.args=SimpleNamespace(workers=2,batch_size=256,threads=2)
    def benchmark(command,**kwargs):
        assert command[command.index('--workers')+1]=='2'
        assert command[command.index('--parent-pid')+1]==str(os.getpid())
        (tmp_path/'throughput.json').write_text(json.dumps(dict(measurements=[
            dict(processes=1,examples_per_sec=20,errors=[]),
            dict(processes=2,examples_per_sec=0,errors=['worker killed'])],best_workers=1)))
    monkeypatch.setattr(driver.subprocess,'run',benchmark)
    runner.configure_workers()
    assert runner.args.workers==1


def test_cached_benchmark_never_exceeds_requested_workers(tmp_path):
    driver=load_script('run_checkpoint1')
    runner=driver.Checkpoint.__new__(driver.Checkpoint)
    runner.root=tmp_path
    runner.device=driver.torch.device('cuda')
    runner.args=SimpleNamespace(workers=2)
    (tmp_path/'throughput.json').write_text(json.dumps(dict(best_workers=3,measurements=[
        dict(processes=i,examples_per_sec=i*10,errors=[]) for i in [1,2,3]])))
    runner.configure_workers()
    assert runner.args.workers==2


def exit_without_result():
    os._exit(9)


def failing_benchmark_worker(ready,results,start,dim,seed,batch_size,threads):
    if seed==42:
        os._exit(9)
    ready.put(dict(pid=os.getpid()))
    start.wait()


def test_failed_measurement_cleans_up_other_workers(monkeypatch):
    bench=load_script('benchmark_checkpoint1')
    monkeypatch.setattr(bench,'worker',failing_benchmark_worker)
    before={p.pid for p in mp.active_children()}
    args=SimpleNamespace(dim=64,batch_size=256,threads=2,parent_pid=None)
    result=bench.measure(mp.get_context('spawn'),2,args)
    assert result['status']=='FAILED'
    assert 'worker exited' in result['errors'][0]
    assert {p.pid for p in mp.active_children()}<=before


def test_dead_child_is_detected_without_waiting_for_queue_timeout():
    bench=load_script('benchmark_checkpoint1')
    context=mp.get_context('spawn')
    channel=context.Queue()
    child=context.Process(target=exit_without_result)
    begin=time.monotonic()
    child.start()
    try:
        with pytest.raises(RuntimeError,match='worker exited'):
            bench.collect_messages(channel,[child],20,'training')
        assert time.monotonic()-begin<10
    finally:
        if child.is_alive():child.terminate()
        child.join(timeout=5)
        channel.close();channel.join_thread()


def test_initialization_error_is_reported_without_starting_training():
    bench=load_script('benchmark_checkpoint1')
    channel=queue.Queue()
    channel.put(dict(error='CUDA out of memory'))
    with pytest.raises(RuntimeError,match='CUDA out of memory'):
        bench.collect_messages(channel,[SimpleNamespace(pid=1,exitcode=None)],20,'initialization')


def test_parent_death_stops_waiting(monkeypatch):
    bench=load_script('benchmark_checkpoint1')
    monkeypatch.setattr(bench.os,'getppid',lambda:123)
    with pytest.raises(RuntimeError,match='Training parent exited'):
        bench.collect_messages(queue.Queue(),[SimpleNamespace(pid=1,exitcode=None)],20,'training',parent_pid=456)


def test_cgroup_limit_controls_available_memory(tmp_path,monkeypatch):
    import psutil
    bench=load_script('benchmark_checkpoint1')
    monkeypatch.setattr(psutil,'virtual_memory',lambda:SimpleNamespace(available=10000))
    (tmp_path/'memory.max').write_text('1000')
    (tmp_path/'memory.current').write_text('900')
    (tmp_path/'memory.stat').write_text('inactive_file 50\n')
    assert bench.available_memory_bytes(tmp_path)==150


@pytest.mark.parametrize('failure',['ram','worker'])
def test_benchmark_falls_back_and_stops_launching_more_workers(tmp_path,monkeypatch,failure):
    bench=load_script('benchmark_checkpoint1')
    calls=[]
    def measure(context,count,args):
        calls.append(count)
        if count==2:
            return dict(processes=count,status='FAILED',errors=['killed'],workers=[],examples_per_sec=0)
        return dict(processes=count,status='COMPLETED',errors=[],examples_per_sec=50,
                    workers=[dict(peak_host_rss_bytes=1024**3)])
    def available():
        return 10*1024**3 if not calls or failure=='worker' else 1024**3
    monkeypatch.setattr(bench,'measure',measure)
    monkeypatch.setattr(bench,'available_memory_bytes',available)
    args=SimpleNamespace(dim=64,workers=3,parent_pid=None,output=str(tmp_path/'throughput.json'))
    report=bench.benchmark(args)
    assert report['best_workers']==1
    assert len(report['measurements'])==2
    assert calls==([1] if failure=='ram' else [1,2])
    assert report['measurements'][1]['status']==('SKIPPED_INSUFFICIENT_RAM' if failure=='ram' else 'FAILED')


def test_worker_limit_caps_benchmark_counts(tmp_path,monkeypatch):
    bench=load_script('benchmark_checkpoint1')
    calls=[]
    monkeypatch.setattr(bench,'available_memory_bytes',lambda:10*1024**3)
    def measure(context,count,args):
        calls.append(count)
        return dict(processes=count,status='COMPLETED',errors=[],examples_per_sec=count*50,
                    workers=[dict(peak_host_rss_bytes=1024**3)]*count)
    monkeypatch.setattr(bench,'measure',measure)
    report=bench.benchmark(SimpleNamespace(dim=64,workers=2,parent_pid=None,output=str(tmp_path/'throughput.json')))
    assert calls==[1,2]
    assert report['best_workers']==2
