"""Verify experiment lineage and prepare shareable measured summaries."""
from pathlib import Path
import sys
import json
import hashlib
import numpy as np
import pandas as pd
import torch

import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run',default='runs/checkpoint1_20261101_batch')
args=parser.parse_args()
ROOT=Path(__file__).resolve().parents[1]
RUN=(ROOT/args.run).resolve()
progress=json.loads((RUN/'progress.json').read_text())
assert progress['checkpoint_experiments']=='COMPLETED'
def state(entry):return torch.load(entry['checkpoint'],map_location='cpu',weights_only=True)['model_state_dict']
ids={s:state(progress[f'id_seed{s}']) for s in [42,43,44]}
assert any(not torch.equal(ids[42][k],ids[43][k]) for k in ids[42]),'Reference ID backbones are identical'
checks=[]
for key,entry in progress.items():
    if not key.startswith('late_'):continue
    model=state(entry)
    for name,expected in ids[entry['seed']].items():
        assert torch.equal(model['id_model.'+name],expected),f'ID drift: {key}/{name}'
    checks.append(dict(run=key,frozen_id_matches=True,independent_reference_seed=entry['seed']))
pd.DataFrame(checks).to_csv(RUN/'frozen_id_integrity.csv',index=False)
registry=[]
for path in RUN.glob('results*.csv'):
    if path.name=='results_combined.csv':continue
    registry.append(pd.read_csv(path))
combined=pd.concat(registry,ignore_index=True).drop_duplicates('run_id').sort_values('timestamp')
combined.to_csv(RUN/'results_combined.csv',index=False)
combined.to_csv(ROOT/'reports/checkpoint1_results.csv',index=False)
id_time=np.mean([progress[f'id_seed{s}']['training_sec'] for s in [42,43,44]])/3600
late_time=np.mean([v['training_sec'] for k,v in progress.items() if k.startswith('late_')])/3600
early_time=progress['early_CBT_seed42']['training_sec']/3600
rows=[dict(block='ID Sports/Toys',remaining_jobs=6,estimated_gpu_job_hours=id_time*3*(1.8+.8),basis='Measured Beauty ID mean; Sports x1.8 and Toys x0.8'),
      dict(block='Late fusion Sports/Toys',remaining_jobs=42,estimated_gpu_job_hours=late_time*21*(1.8+.8),basis='Measured 21 Beauty late-fusion fits; Sports x1.8 and Toys x0.8'),
      dict(block='Early fusion remaining',remaining_jobs=62,estimated_gpu_job_hours=early_time*(21*(1+1.8+.8)-1),basis='One measured Beauty CBT run extrapolated to all combinations; provisional'),
      dict(block='Training controls',remaining_jobs=108,estimated_gpu_job_hours=(early_time+late_time)/2*108*1.2,basis='Unmeasured controls; cost estimated from early/late mean and dataset factors'),
      dict(block='UASF baseline ablation',remaining_jobs=None,estimated_gpu_job_hours=9.,basis='Original supervisor rough estimate; not measured')]
device=json.loads((RUN/'protocol.lock.json').read_text())['device']
if device!='cuda':
    for row in rows:
        row['estimated_gpu_job_hours']=None
        row['basis']='GPU cost not inferred from CPU-only execution'
pd.DataFrame(rows).to_csv(ROOT/'reports/checkpoint1_gpu_budget.csv',index=False)
json.dump(dict(remaining_estimated_gpu_job_hours=sum(r['estimated_gpu_job_hours'] for r in rows) if device=='cuda' else None,blocks=rows),
          (RUN/'budget_projection.json').open('w'),indent=2)
distribution=pd.read_csv(RUN/'delta_distribution.csv')
distribution.to_csv(ROOT/'reports/checkpoint1_delta_distribution.csv',index=False)
for name in ['diagnostic_B_per_seed.csv','diagnostic_B_summary.csv','robustness.csv','same_checkpoint_mask_sensitivity.csv',
             'baseline_and_shuffle.csv','frozen_id_integrity.csv']:
    (ROOT/'reports'/('checkpoint1_'+name)).write_bytes((RUN/name).read_bytes())
(ROOT/'reports/checkpoint1_e0_e1.json').write_bytes((RUN/'e0_e1.json').read_bytes())
print('Lineage verified, registries combined, measured tables and provisional budget exported')

source=Path(__file__)
(RUN/'analysis_source.py').write_bytes(source.read_bytes())
(RUN/'analysis_provenance.json').write_text(json.dumps(dict(script_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
       experiment_protocol_lock_sha256=hashlib.sha256((RUN/'protocol.lock.json').read_bytes()).hexdigest()),indent=2),encoding='utf-8')
