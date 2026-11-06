"""Independent artifact-level checks, without rerunning inference or selection."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parents[2];out=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
gm=pd.read_csv(out/'task1/group_metrics.csv')
sel=pd.read_csv(out/'task1/selected_configurations.csv')
per=pd.read_csv(out/'task1/summary_per_seed.csv')
groups=pd.read_csv(out/'task1/user_context_groups.csv')
assert not groups.duplicated(['split','user_idx']).any()
assert not groups.duplicated(['split','context_id']).any()
assert (groups.groupby('split').size()==22363).all()
assert (gm.groupby(['seed','split','subset']).n_users.sum()==22363).all()
assert gm.loc[gm.n_users==0,'ndcg_10'].isna().all()
for seed,g in per.groupby('seed'):
    rows=sel[sel.seed==seed]
    for method,col in [('group_validation_subset','chosen_by_validation'),('group_test_oracle','oracle_chosen_by_test')]:
        weighted=0
        for row in rows.itertuples():
            if row.test_users==0:continue
            subset=getattr(row,col)
            score=gm[(gm.seed==seed)&(gm.split=='test')&(gm.group_id==row.group_id)&(gm.subset==subset)].iloc[0]
            assert score.n_users==row.test_users
            weighted+=score.n_users*score.ndcg_10/22363
        assert abs(weighted-g[g.method==method].test_ndcg_10.iloc[0])<1e-14
summary=pd.read_csv(out/'task1/summary.csv').set_index('method')
for name,g in per.groupby('method'):
    assert abs(g.test_ndcg_10.mean()-summary.loc[name,'mean'])<1e-14
    assert abs(g.test_ndcg_10.std(ddof=1)-summary.loc[name,'std'])<1e-14
u=pd.read_csv(out/'task2/utility_supplement.csv')
v=pd.read_csv(out/'task2/validation_same_reference_ablation.csv')
assert len(u)==133440 and len(v)==22363*3*2
assert not u.duplicated(['reference_seed','source_name','context_id']).any()
assert not v.duplicated(['reference_seed','source_name','context_id']).any()
assert not u.isna().any().any() and not v.isna().any().any()
assert np.allclose(u.delta_ce,u.ce_without_m-u.ce_all,atol=1e-7)
assert np.allclose(u.delta_logrank,np.log2(1+u.rank_without_m)-np.log2(1+u.rank_all),atol=1e-12)
for frame in [u,v]:
    assert (frame[['rank_all','rank_without_m']]>=1).all().all()
    assert (frame[['rank_all','rank_without_m']]<=11999).all().all()
all_val=np.where(v.rank_all<=10,1/np.log2(1+v.rank_all),0)
without_val=np.where(v.rank_without_m<=10,1/np.log2(1+v.rank_without_m),0)
assert np.allclose(v.ndcg_all,all_val,atol=1e-15)
assert np.allclose(v.ndcg_without_m,without_val,atol=1e-15)
assert (v.loc[v.target_idx==0,['rank_all','rank_without_m']]==11999).all().all()
sign=pd.read_csv(out/'task2/sign_alignment.csv').set_index('source_name')
for name,g in v.groupby('source_name'):
    assert abs((g.ndcg_all-g.ndcg_without_m).mean()-sign.loc[name,'dval'])<1e-14
initial=(out/'initial_report.md').read_text()
assert initial in (R/'minh_report.md').read_text()
result=dict(status='PASS',checks=['unique contexts and full cohorts','weighted group reconstruction','sample SD',
                               'utility algebra','rank domain/OOV miss','NDCG rederived from same-reference ranks',
                               'Dval independent aggregation','initial readiness retained verbatim'],
            empty_groups={s:sorted(set(range(10))-set(g.group_id)) for s,g in groups.groupby('split')},
            group_minus_id=float(summary.loc['group_validation_subset','mean']-summary.loc['ID','mean']),
            oracle_minus_id=float(summary.loc['group_test_oracle','mean']-summary.loc['ID','mean']))
(out/'verification.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
