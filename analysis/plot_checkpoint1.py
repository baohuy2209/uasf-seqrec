"""Make publication-style figures from measured checkpoint outputs only."""
from pathlib import Path
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run',default='runs/checkpoint1_20261101_batch')
parser.add_argument('--output',default='.tmp/checkpoint1_figures')
args=parser.parse_args()
ROOT=Path(__file__).resolve().parents[1]
root=(ROOT/args.run).resolve()
out=(ROOT/args.output).resolve()
out.mkdir(parents=True,exist_ok=True)
frame=pd.read_csv(root/'robustness.csv')
data=frame[frame.control=='missing']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(6.7,2.05),layout='constrained')
for source,color,marker in [('category','#23518c','o'),('brand','#158174','s'),('title','#9252a0','^')]:
    values=data[data.source==source].groupby('ratio')['NDCG@10'].agg(['mean','std'])
    x=values.index.to_numpy()*100;y=values['mean'].to_numpy();sd=values['std'].fillna(0).to_numpy()
    ax.plot(x,y,label=source.title(),color=color,marker=marker,markersize=3,linewidth=1.4)
    ax.fill_between(x,y-sd,y+sd,color=color,alpha=.12,linewidth=0)
progress=json.loads((root/'progress.json').read_text())
baseline=sum(progress[f'id_seed{s}']['test']['NDCG@10'] for s in [42,43,44])/3
ax.axhline(baseline,color='#666666',linestyle='--',linewidth=1,label='ID only')
ax.set(xlabel='Additional missing metadata (%)',ylabel='Test NDCG@10',xticks=list(range(0,81,10)))
ax.grid(axis='y',alpha=.15);ax.legend(ncol=4,frameon=False,loc='upper right')
fig.savefig(out/'missingness.png',dpi=240)
plt.close(fig)
print(out/'missingness.png')
