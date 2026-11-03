import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0,str(ROOT/'src'))
from taimma.experiments.checkpoint_state import ID_SOURCES,reuse_completed_id_stages
from taimma.data.ingestion import InteractionSequence
from taimma.data.split import SequenceSplitter
from taimma.experiments.logger import ExperimentLogger


def dump(path,data):path.write_text(json.dumps(data),encoding='utf-8')


@pytest.fixture
def reuse_case(tmp_path):
    old,new=tmp_path/'previous',tmp_path/'new'
    old.mkdir();new.mkdir();folder=old/'id_run';folder.mkdir()
    checkpoint=folder/'checkpoint.pt';checkpoint.write_bytes(b'fixture checkpoint for hash verification only')
    per_user=folder/'per_user_test.csv';per_user.write_text('user_idx,recall@10,ndcg@10\n0,1,0.5\n1,0,0\n')
    dump(folder/'run_state.json',dict(execution_status='COMPLETED'))
    dump(folder/'artifact_index.json',dict(artifacts=[dict(path=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in [checkpoint,per_user]]))
    dump(folder/'config.json',dict(sources=[],mask_policy='input_history_strict'))
    source='class Checkpoint:\n    def model(self, split, seed, dim):\n        return dim\n'
    with zipfile.ZipFile(old/'source_snapshot.zip','w') as archive:archive.writestr('scripts/run_checkpoint1.py',source)
    lock=dict(seeds=[42,43,44],max_epochs=100,patience=10,learning_rate=.001,batch_size=256,
              main_mask_policy='input_history_strict',loo_reference_mask_policy='none',device='cuda',
              metadata_manifest_sha256='same-beauty-manifest',code_sha256={name:'same-id-source' for name in ID_SOURCES})
    dump(old/'protocol.lock.json',lock)
    dump(old/'progress.json',{'id_seed42':dict(folder=str(folder),checkpoint=str(checkpoint),seed=42,dim=64,sources=[]),
                             'late_CBT_seed42':dict(incomplete=True)})
    return old,new,lock,source,folder


def test_reuse_is_explicit_verified_and_excludes_residuals(reuse_case):
    old,new,lock,source,_=reuse_case
    current=copy.deepcopy(lock);current['metadata_manifest_sha256']={'beauty':lock['metadata_manifest_sha256']}
    result=reuse_completed_id_stages(old,new,current,source)
    assert set(result)=={'id_seed42'}
    assert result['id_seed42']['test_metrics_exact']==dict(recall_10=.5,ndcg_10=.25)
    assert (new/'reused_stages.json').is_file()
    assert 'checkpoint_sha256' not in json.loads((old/'progress.json').read_text())['id_seed42']


@pytest.mark.parametrize('corruption',['source','checkpoint','per_user','incomplete','policy','model'])
def test_reuse_rejects_changed_or_incomplete_evidence(reuse_case,corruption):
    old,new,lock,source,folder=reuse_case
    current=copy.deepcopy(lock)
    if corruption=='source':current['code_sha256'][ID_SOURCES[0]]='changed'
    elif corruption=='checkpoint':(folder/'checkpoint.pt').write_bytes(b'changed')
    elif corruption=='per_user':(folder/'per_user_test.csv').write_text('user_idx,recall@10,ndcg@10\n0,1,1\n')
    elif corruption=='incomplete':dump(folder/'run_state.json',dict(execution_status='RUNNING'))
    elif corruption=='policy':current['main_mask_policy']='none'
    elif corruption=='model':source=source.replace('return dim','return dim * 2')
    with pytest.raises(ValueError):reuse_completed_id_stages(old,new,current,source)


def test_real_driver_train_cache_and_resume_on_cpu_fixture(tmp_path):
    spec=importlib.util.spec_from_file_location('checkpoint_driver_test',ROOT/'scripts/run_checkpoint1.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    runner=module.Checkpoint.__new__(module.Checkpoint)
    runner.data_mode='synthetic_fixture'
    runner.args=SimpleNamespace(batch_size=4,epochs=1,patience=1,train_job=None)
    runner.device=torch.device('cpu');runner.root=tmp_path;runner.progress={};runner.backbone_dim=8
    sequences=[InteractionSequence(str(u),[str((u+i)%9) for i in range(8)],list(range(8))) for u in range(18)]
    runner.split=SequenceSplitter().split(sequences)
    n=runner.split.num_items
    runner.attrs={'category':torch.tensor([[i%3+1,(i+1)%3+1] for i in range(n)]),
                  'brand':torch.arange(n)%3+1,'title':torch.arange(n*6,dtype=torch.float32).reshape(n,6)/100}
    runner.masks={s:torch.tensor([float(i%3!=1) for i in range(n)]) for s in runner.attrs}
    runner.logger=ExperimentLogger(str(tmp_path/'runs'),str(tmp_path/'results.csv'))
    entry=runner.train('id_seed42',runner.split,42,dim=8)
    import pandas as pd
    for filename in ['epochs.csv', 'history.json', 'metrics.json']:
        assert (Path(entry['folder'])/filename).is_file()
    epoch_log=pd.read_csv(Path(entry['folder'])/'epochs.csv')
    assert epoch_log.epoch.tolist()==[0,1]
    assert epoch_log.loc[epoch_log.selected_checkpoint,'epoch'].tolist()==[1]
    metadata=json.loads((Path(entry['folder'])/'metrics.json').read_text())
    assert metadata['selection']['scope']=='trained_epochs_only'
    assert metadata['selection']['selected_epoch']==metadata['best_epoch']==1
    assert metadata['selection']['initial_validation'] is not None
    assert runner.train('id_seed42',runner.split,42,dim=8)==entry
    runner.id_scores(entry)
    assert (tmp_path/'score_cache_integrity/seed42.json').is_file()
    residual=runner.train('late_CBT_seed42',runner.split,42,dim=8,sources=('category','brand','title'),id_model=runner.load_id(entry))
    assert residual['sources']==['category','brand','title']
    assert json.loads((Path(residual['folder'])/'report_data.json').read_text())['data_mode']=='synthetic_fixture'
    model=runner.load_residual(residual)
    with torch.no_grad():scores=runner.predict_residual(model,runner.split.val_samples[:2])
    assert scores.shape==(2,n) and torch.isfinite(scores).all()
    cache=np.load(Path(entry['folder'])/'s_ID_test.npy',mmap_mode='r+')
    cache[0,0]+=10;cache.flush();del cache
    with pytest.raises(ValueError,match='cache scores'):
        runner.id_scores(entry)
