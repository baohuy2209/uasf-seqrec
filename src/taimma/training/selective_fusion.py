"""S3 gates on a frozen S1, including differentiable early-fusion weights."""
from pathlib import Path
import copy
import numpy as np
import pandas as pd
import torch
from torch import nn
from ..models.source_dropout import SOURCES
from ..evaluation.evaluator import IndependentEvaluator
from ..experiments.reporting import save_checkpoint_atomic
from .checkpoint2_trainer import batches


class SelectiveGate(nn.Module):
    def __init__(self,input_dim,mode='contextual',regimes=None,hidden_dim=32,tau=1.,kappa=.1):
        super().__init__()
        if mode not in ['contextual','global','unsupervised']:raise ValueError('Unknown gate mode')
        if tau<=0 or not 0<=kappa<=1:raise ValueError('Invalid gate temperature/threshold')
        self.mode=mode;self.regimes=regimes or {};self.tau=tau;self.kappa=kappa
        self.mlp=nn.Sequential(nn.Linear(input_dim,hidden_dim),nn.ReLU(),nn.Linear(hidden_dim,3))
        self.global_logits=nn.Parameter(torch.zeros(3))
        self.group_logits=nn.Parameter(torch.zeros(10,3))

    def forward(self,features,groups,availability,threshold=False):
        b=features.size(0)
        if self.mode=='global':logits=self.global_logits.unsqueeze(0).expand(b,-1)
        else:
            logits=self.mlp(features)
            for i,s in enumerate(SOURCES):
                regime=self.regimes.get(s,'contextual' if self.mode=='unsupervised' else 'global')
                if regime=='group_level':
                    logits=torch.cat([logits[:,:i],self.group_logits[groups,i:i+1],logits[:,i+1:]],dim=1)
                elif regime=='global':
                    logits=torch.cat([logits[:,:i],self.global_logits[i].expand(b,1),logits[:,i+1:]],dim=1)
        beta=torch.sigmoid(logits/self.tau)
        alpha=beta*availability
        if threshold:alpha=alpha*(beta>=self.kappa)
        return alpha,beta


def feature_matrix(frame,columns,mean=None,scale=None):
    x=frame[columns].to_numpy(np.float64)
    if not np.isfinite(x).all():raise ValueError('Gate features must be finite')
    if mean is None:mean=x.mean(axis=0)
    if scale is None:scale=x.std(axis=0)
    scale=np.maximum(np.asarray(scale),1e-8)
    return ((x-mean)/scale).astype(np.float32),np.asarray(mean),scale


def gate_evaluate(adapter,gate,samples,features,columns,mean,scale,batch_size=256):
    adapter.model.eval();gate.eval()
    f=features.set_index('user_idx')
    if f.index.duplicated().any():raise ValueError('Duplicate feature user')
    ev=IndependentEvaluator(adapter.model.num_items,mask_policy='input_history_strict');records=[]
    with torch.no_grad():
        for batch in batches(samples,adapter.attrs,adapter.masks,batch_size):
            rows=f.loc[batch['user_indices']]
            x=feature_matrix(rows,columns,mean,scale)[0]
            available=np.column_stack([(rows[f'feat_{s}_coverage'].to_numpy()>0) for s in SOURCES]).astype(np.float32)
            alpha,_=gate(torch.tensor(x,device=adapter.device),
                torch.tensor(rows.group_idx.to_numpy(),device=adapter.device,dtype=torch.long),
                torch.tensor(available,device=adapter.device),threshold=True)
            scores=adapter.predict(batch['item_seqs'],alpha)
            result=ev.evaluate_ranks(scores,batch['targets'].tolist(),batch['user_indices'],histories=batch['item_seqs'].tolist())
            records.extend(result.per_user_records)
    return ev.result_from_records(records)


def train_gate(adapter,gate,train_samples,val_samples,train_features,val_features,
               labels,columns,folder,epochs=100,patience=20,batch_size=256,
               lr=.001,lambda_util=.5,lambda_sparse=.001,qualified_sources=()):
    """Only gate parameters train. Frozen S1 still differentiates wrt alpha.

    No no_grad around adapter.predict during S3: that would sever early input
    and candidate alpha gradients even though S1 parameters are frozen.
    """
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    for p in adapter.model.parameters():p.requires_grad=False
    adapter.model.eval();gate=gate.to(adapter.device)
    train_features=train_features.set_index('user_idx').loc[[s.user_idx for s in train_samples]].reset_index()
    x,mean,scale=feature_matrix(train_features,columns)
    feature_index={int(u):i for i,u in enumerate(train_features.user_idx)}
    y=labels.set_index('user_idx').loc[train_features.user_idx,list(SOURCES)].to_numpy(np.float32)
    if not np.isfinite(y).all() or np.any((y<0)|(y>1)):raise ValueError('Gate labels must be in [0,1]')
    availability=np.column_stack([(train_features[f'feat_{s}_coverage'].to_numpy()>0) for s in SOURCES]).astype(np.float32)
    supervised=np.asarray([s in qualified_sources for s in SOURCES],dtype=np.float32)
    optimizer=torch.optim.Adam(gate.parameters(),lr=lr)
    best=-1.;state=None;stale=0;history=[]
    loader=batches(train_samples,adapter.attrs,adapter.masks,batch_size,True,adapter.spec.seed)
    for epoch in range(1,epochs+1):
        gate.train();losses=[]
        for b in loader:
            idx=np.asarray([feature_index[int(u)] for u in b['user_indices']])
            xx=torch.tensor(x[idx],device=adapter.device)
            groups=torch.tensor(train_features.group_idx.to_numpy()[idx],device=adapter.device,dtype=torch.long)
            avail=torch.tensor(availability[idx],device=adapter.device)
            alpha,beta=gate(xx,groups,avail)
            optimizer.zero_grad(set_to_none=True)
            rec=nn.functional.cross_entropy(adapter.predict(b['item_seqs'],alpha),b['targets'].to(adapter.device)-1)
            mask=avail*torch.tensor(supervised,device=adapter.device)
            util=(nn.functional.binary_cross_entropy(beta,torch.tensor(y[idx],device=adapter.device),reduction='none')*mask).sum()/mask.sum().clamp(min=1.)
            loss=rec+lambda_util*util+lambda_sparse*alpha.mean()
            if not torch.isfinite(loss):raise ValueError('Nonfinite gate loss')
            loss.backward();optimizer.step();losses.append(float(loss.detach()))
        val=gate_evaluate(adapter,gate,val_samples,val_features,columns,mean,scale,batch_size)
        history.append(dict(epoch=epoch,loss=float(np.mean(losses)),val_ndcg_10=val.ndcg_10))
        if val.ndcg_10>best:
            best=val.ndcg_10;state=copy.deepcopy(gate.state_dict());stale=0
            save_checkpoint_atomic(folder/'gate.pt',dict(model_state_dict=state,epoch=epoch,val_ndcg_10=best,
                feature_columns=columns,feature_mean=mean.tolist(),feature_scale=scale.tolist()))
        else:stale+=1
        pd.DataFrame(history).to_csv(folder/'gate_epochs.csv',index=False,lineterminator='\n')
        if stale>=patience:break
    if state is None:raise RuntimeError('No gate epoch completed')
    gate.load_state_dict(state)
    return gate,mean,scale,history
