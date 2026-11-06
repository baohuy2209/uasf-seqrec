import pathlib,json,hashlib,sys,platform,datetime
sys.path.insert(0,str(pathlib.Path('src').resolve()))
import pandas as pd,numpy as np,torch
from taimma.data.ingestion import Amazon2014Ingestor
from taimma.data.split import SequenceSplitter
R=pathlib.Path('.').resolve(); root=R/'runs/checkpoint1_version1'
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
seq,_=Amazon2014Ingestor('data/raw','Beauty').build_interaction_sequences(include_metadata=False);sp=SequenceSplitter().split(seq)
a={'timestamp':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=7))).isoformat(),'environment':{'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'cuda':torch.cuda.is_available()},'all_result_sets':[str(p.relative_to(R)) for p in R.rglob('results.csv') if '.venv' not in p.parts],'runs':[]}
lock=json.loads((root/'protocol.lock.json').read_text());a['source_mismatch']=[p for p,h in lock['code_sha256'].items() if sha(R/p)!=h]
a['split']={'users':sp.num_users,'catalog':sp.num_items,'train_base_samples':len(sp.train_base_samples),'oov':sp.oov_counts}
pm=next((R/'data/processed').rglob('manifest.json'));md=json.loads(pm.read_text());a['raw_sha_matches_processed']=sha(md['raw_reviews_file'])==md['raw_reviews_sha256']
for name,mapping in [('item2idx',sp.item2idx),('user2idx',sp.user2idx)]:a[name+'_matches_processed']=json.loads((pm.parent/(name+'.json')).read_text())==mapping
reg=pd.read_csv(root/'results.csv')
for marker in sorted((root/'completed').glob('*.json')):
 e=json.loads(marker.read_text());p=pathlib.Path(e['folder']);c=json.loads((p/'config.json').read_text());state=json.loads((p/'run_state.json').read_text())
 rec={'key':marker.stem,'path':str(p.relative_to(R)),'seed':e['seed'],'sources':e.get('sources',[]),'config':c,'state':state['execution_status'],'git':json.loads((p/'run_manifest.json').read_text())['git_commit'],'checks':{}}
 idx=json.loads((p/'artifact_index.json').read_text())['artifacts'];rec['checks']['artifact_hash_failures']=[z['path'] for z in idx if not (p/z['path']).is_file() or sha(p/z['path'])!=z['sha256']]
 rec['checks']['checkpoint_hash']=sha(e['checkpoint'])==e['checkpoint_sha256']
 for split,samples in [('val',sp.val_samples),('test',sp.test_samples)]:
  f=pd.read_csv(p/f'per_user_{split}.csv')
  checks={'rows':len(f),'duplicates':int(f.user_idx.duplicated().sum()),'nulls':int(f.isna().sum().sum()),'rank_min':int(f['rank'].min()),'rank_max':int(f['rank'].max()),'integer_ranks':bool((f['rank']%1==0).all())}
  v=np.where(f['rank']<=10,1/np.log2(f['rank']+1),0);checks['ndcg_column_max_error']=float(np.max(abs(v-f['ndcg@10'])))
  row=reg[reg.run_id==p.name];checks['registry_ndcg_error']=float(abs(v.mean()-row.iloc[0][f'{split}_ndcg_10']))
  if c['protocol']=='four_segment':
   exp=pd.DataFrame([{'user_idx':s.user_idx,'target_idx':s.target} for s in samples]).set_index('user_idx');ff=f.set_index('user_idx')
   checks['user_target_match']=ff[['target_idx']].sort_index().equals(exp.sort_index())
   checks['oov_miss_valid']=bool((ff.loc[exp.target_idx==0,'rank']==sp.num_items+1).all())
  rec['checks'][split]=checks
 a['runs'].append(rec)
u=pd.read_csv(root/'utility.csv');man=json.loads((root/'utility.csv.manifest.json').read_text());co=pd.read_csv(root/'utility_cohort.csv')
a['utility']={'rows':len(u),'nulls':int(u.isna().sum().sum()),'duplicates':int(u.duplicated(['context_id','source_name','reference_seed']).sum()),'sha_valid':sha(root/'utility.csv')==man['file_sha256'],'ce_delta_max_error':float(abs(u.delta-(u.ce_without_m-u.ce_all)).max()),'cohorts':{},'manifest_metadata_valid':sha(man['dataset_manifest'])==man['dataset_manifest_sha256'],'cohort_manifest_valid':sha(root/'utility_cohort.json')==man['cohort_sha256']}
for (s,seed),g in u.groupby(['source_name','reference_seed']):a['utility']['cohorts'][f'{s}:{seed}']={'n':len(g),'same_users':set(g.user_idx)==set(co.loc[co.included,'user_idx'])}
a['utility']['context_ids_match_reconstruction']=all(hashlib.sha256(json.dumps([s.user_idx,s.history,s.target]).encode()).hexdigest()==q for s,q in zip([s for s in sp.train_meta_samples if s.target],u[(u.source_name=='category')&(u.reference_seed==42)].context_id))
a['reference_hashes']={seed:sha(e['path'])==e['sha256'] for seed,e in man['reference_checkpoints'].items()}
a['cache_files']=[{'path':str(p.relative_to(R)),'bytes':p.stat().st_size,'shape':list(np.load(p,mmap_mode='r').shape)} for p in root.rglob('s_ID*.npy')]
a['progress_completed']=json.loads((root/'progress.json').read_text()).get('checkpoint_experiments')
pathlib.Path('/tmp/minh_audit.json').write_text(json.dumps(a,indent=2))
print(json.dumps({k:v for k,v in a.items() if k not in ['runs','cache_files']},indent=2))
print('RUN CHECKS',[(r['key'],r['state'],r['checks']) for r in a['runs']])
