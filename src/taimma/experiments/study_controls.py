"""Versioned training controls matching the supervisor memo dated 20 October."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from .checkpoint2_protocol import sha, write_json
from metadata_pipeline.contracts import flags


def build_controls(metadata_dir, output_dir, seeds=(42,43,44), shuffled_source=None):
    meta=Path(metadata_dir);out=Path(output_dir)
    marker=out/'manifest.json'
    inputs={n:sha(meta/n) for n in ['metadata.csv','availability_mask.csv','title_embeddings.npy']}
    if marker.exists():
        m=json.loads(marker.read_text(encoding='utf-8'))
        if m['schema']!='uniform_gaussian_cov_v2' or m['input_sha256']!=inputs or m['seeds']!=list(seeds):
            raise ValueError('Control protocol changed; use a new completion output')
        for kind in ['random','shuffled']:
            for rec in m[kind].values():
                if sha(out/kind/rec['file'])!=rec['sha256']:raise ValueError('Control artifact changed')
        return m
    data=pd.read_csv(meta/'metadata.csv',keep_default_na=False)
    avail=pd.read_csv(meta/'availability_mask.csv',keep_default_na=False)
    if not np.array_equal(data.model_item_id,avail.model_item_id):raise ValueError('Control ID mismatch')
    titles=np.load(meta/'title_embeddings.npy',allow_pickle=False)
    cats=[json.loads(v) for v in data.category_indices]
    n=len(data);cat_vocab=max(v for row in cats for v in row);brand_vocab=int(data.brand_idx.max())
    masks={s:flags(avail['has_'+s]) for s in ['category','brand','title']}
    real=titles[masks['title']].astype(float)
    if real.shape[0]<2:raise ValueError('Need at least two real title vectors to estimate covariance')
    mean=real.mean(0);cov=np.atleast_2d(np.cov(real,rowvar=False))
    eig,vec=np.linalg.eigh(cov);root=vec*np.sqrt(np.maximum(eig,0.))
    m=dict(schema='uniform_gaussian_cov_v2',seeds=list(seeds),input_sha256=inputs,shuffled={},random={},
           title_distribution='Gaussian with empirical real-title mean and full covariance; no unit normalization',
           categorical_distribution='uniform vocabulary; distinct category labels preserve per-item cardinality')
    for kind in ['random','shuffled']:(out/kind).mkdir(parents=True,exist_ok=True)
    for seed in seeds:
        for si,s in enumerate(['category','brand','title']):
            rng=np.random.default_rng(np.random.SeedSequence([seed,si,2]));mask=masks[s]
            donor=np.arange(n);ix=np.flatnonzero(mask);donor[ix]=rng.permutation(ix)
            p=out/'shuffled'/f'perm_{s}_seed_{seed}.csv'
            old=Path(shuffled_source)/'shuffled'/p.name if shuffled_source else None
            if old is not None and old.exists():
                old_manifest=json.loads((Path(shuffled_source)/'manifest.json').read_text(encoding='utf-8'))
                if old_manifest['input_sha256']!=inputs:raise ValueError('Historical permutation metadata changed')
                records=[r for r in old_manifest['shuffled'].values() if r['file']==old.name]
                b=old.read_bytes()
                if len(records)!=1 or (sha(old)!=records[0]['sha256'] and hashlib.sha256(b.replace(b'\r\n',b'\n')).hexdigest()!=records[0]['sha256']):
                    raise ValueError('Historical permutation checksum changed')
                original=pd.read_csv(old)
                if not np.array_equal(original.model_item_id,data.model_item_id):raise ValueError('Historical permutation IDs changed')
                ix=original.permuted_source_item_id.to_numpy(int)-1
                if sorted(ix.tolist())!=list(range(n)) or not np.array_equal(mask[ix],mask) or not np.array_equal(ix[~mask],np.flatnonzero(~mask)):raise ValueError('Invalid historical permutation')
                p.write_bytes(old.read_bytes())
            else:
                pd.DataFrame(dict(model_item_id=data.model_item_id,permuted_source_item_id=data.model_item_id.to_numpy()[donor])).to_csv(p,index=False,lineterminator='\n')
            m['shuffled'][f'{s}_{seed}']=dict(file=p.name,sha256=sha(p))
            p=out/'random'/f'random_{s}_seed_{seed}.csv'
            if s=='category':
                values=[json.dumps(rng.choice(np.arange(1,cat_vocab+1),size=len(set(row)),replace=False).tolist()) if ok else '[]' for row,ok in zip(cats,mask)]
                pd.DataFrame(dict(model_item_id=data.model_item_id,random_category_indices=values)).to_csv(p,index=False,lineterminator='\n')
            elif s=='brand':
                values=np.zeros(n,dtype=int);values[mask]=rng.integers(1,brand_vocab+1,mask.sum())
                pd.DataFrame(dict(model_item_id=data.model_item_id,random_brand_idx=values)).to_csv(p,index=False,lineterminator='\n')
            else:
                p=p.with_suffix('.npy');values=np.zeros_like(titles)
                values[mask]=(rng.normal(size=(int(mask.sum()),len(mean)))@root.T+mean).astype(titles.dtype)
                np.save(p,values,allow_pickle=False)
            m['random'][f'{s}_{seed}']=dict(file=p.name,sha256=sha(p))
    write_json(marker,m);return m
