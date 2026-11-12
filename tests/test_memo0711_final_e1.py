"""CSV round-trip regression: preserve ties, strict metrics, and archive hashes."""
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

REPO=Path(__file__).resolve().parents[1]
if not (REPO/'src').exists():REPO=Path(r'D:\Github\uasf-seqrec')
sys.path.insert(0,str(REPO/'src'))
from taimma.experiments import memo0711_e1_evidence as legacy
try:
    from taimma.experiments.memo0711_final_e1 import inspect_e1_roundtrip
except ImportError:
    spec=importlib.util.spec_from_file_location('taimma.experiments.memo0711_final_e1',Path(__file__).with_name('memo0711_final_e1.py'))
    new=importlib.util.module_from_spec(spec);spec.loader.exec_module(new)
    inspect_e1_roundtrip=new.inspect_e1_roundtrip


def evidence(root: Path, wrong_metric: bool=False) -> dict:
    originals={}
    delta=np.array([.1,.10000000000000002,.2,.3,.4,-.1,-.2,-.3,-.4,-.5])
    prob=np.array([.15,.15000000000000002,.4,.5,.6,.3,.2,.1,.05,.01])
    label=(delta>0).astype(int)
    for metric in ('CE','LOG_RANK'):
        folder=root/'e1_oof_evidence'/metric;folder.mkdir(parents=True)
        report=root/'selected_s2'/f'e0_e1_{metric}.json';report.parent.mkdir(exist_ok=True)
        auc=float(roc_auc_score(label,prob))
        e1=dict(status='EVALUATED',oof_auc_lr=auc,oof_auc_tree=auc,baseline_len_auc=auc,
                oof_spearman_rho=float(spearmanr(prob,delta).statistic)+(.01 if wrong_metric else 0))
        report.write_text(json.dumps({'sources':{'category':{'e1':e1}}}))
        f=pd.DataFrame(dict(user_idx=np.arange(10),context_id=np.arange(10),fold=np.arange(10)%5,
            delta=delta,epsilon=0,label=label,source_name='category',probability_lr=prob,
            probability_tree=prob,probability_length=prob))
        p=folder/'paired_oof_predictions.csv';f.to_csv(p,index=False)
        summary=folder/'replay_summary.json';summary.write_text(json.dumps({'sources':[{'source':'category','status':'MATCHED_ORIGINAL_E1'}]}))
        manifest=dict(signature=dict(schema_version=1,sources={str(report):legacy.sha(report),str(Path(legacy.__file__)):legacy.sha(Path(legacy.__file__))}),
            files={p.name:legacy.sha(p),summary.name:legacy.sha(summary)})
        (folder/'manifest.json').write_text(json.dumps(manifest))
    for p in root.rglob('*'):
        if p.is_file():originals[p]=p.read_bytes()
    return originals


def test_round_trip_fixes_parser_only_and_preserves_archive(tmp_path: Path) -> None:
    originals=evidence(tmp_path)
    with pytest.raises(ValueError,match='oof_spearman_rho'):legacy.inspect_e1_evidence(tmp_path)
    assert inspect_e1_roundtrip(tmp_path)['status']=='PASS'
    assert all(p.read_bytes()==raw for p,raw in originals.items())


def test_round_trip_does_not_relax_metric_tolerance(tmp_path: Path) -> None:
    evidence(tmp_path,wrong_metric=True)
    with pytest.raises(ValueError,match='oof_spearman_rho'):inspect_e1_roundtrip(tmp_path)


def test_tampered_csv_is_still_rejected(tmp_path: Path) -> None:
    evidence(tmp_path)
    p=tmp_path/'e1_oof_evidence/CE/paired_oof_predictions.csv'
    p.write_bytes(p.read_bytes()+b'\n')
    with pytest.raises(ValueError,match='artifact changed'):inspect_e1_roundtrip(tmp_path)
