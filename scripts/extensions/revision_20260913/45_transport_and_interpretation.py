import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import importlib.util,json,hashlib,warnings
import numpy as np,pandas as pd,joblib
from sklearn.metrics import roc_auc_score
ROOT=Path(os.environ["ANALYSIS_ROOT"])
W=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('suite',W.parent.parent/'yue/work/42_replication_sensitivity_suite.py')
S=importlib.util.module_from_spec(spec);spec.loader.exec_module(S);S.SEED=20260914
P=ROOT/'data_private/revision_20260913';P.mkdir(exist_ok=True)
Q=W/'extension_results';Q.mkdir(exist_ok=True)
EXT=ROOT/'data_private/replication_extensions'
def write(n,x):(Q/n).write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def charls(s):
 d=pd.read_csv(ROOT/f'data_private/multoutcome/{S.MAP[s]}/validation.csv.gz',dtype={'person_id':str}).rename(columns=S.RENAME)
 old=pd.read_pickle(EXT/f'CHARLS_{s}_core_panel7_predictions.pkl');d=d.merge(old[['person_id','evaluable']],on='person_id',validate='one_to_one');return d
def frozen(model,d,schedule):
 grid=S.T.terms(pd.concat([d.assign(start=a,duration=b) for a,b in schedule],ignore_index=True))
 with warnings.catch_warnings(record=True) as caught:
  h=model.predict_proba(grid[model.feature_names_in_])[:,1]
 assert not caught,[str(x.message) for x in caught]
 risk=1-pd.DataFrame({'id':grid.person_id,'s':1-h}).groupby('id').s.prod()
 return d.person_id.map(risk).to_numpy()
def transport():
 rows=[];hashes={}
 elsa=pd.read_pickle(EXT/'elsa_w6_people.pkl')
 for s in S.MAP:
  for direction in ['CHARLS_to_ELSA','ELSA_to_CHARLS']:
   if direction=='CHARLS_to_ELSA':
    d=elsa.loc[elsa.outcome.eq(s)].copy();d['fasting_sample']=d.fasting_sample.map({1:1.,0:2.});schedule=[(0,2),(2,2),(4,2)]
    names=[f'CHARLS_{s}_core_reference',f'CHARLS_{s}_core_panel7']
   else:
    d=charls(s);d['fasting_sample']=d.fasting_sample.map({1:1.,2:0.});schedule=[(0,3),(3,2)]
    names=[f'ELSA_temporal_{s}_core_reference',f'ELSA_temporal_{s}_core_blood']
   pred=d[['person_id','event','evaluable']].copy()
   for kind,name in zip(['reference','blood'],names):
    path=EXT/(name+'.joblib');hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest();model=joblib.load(path)
    # Independently reproduce source study's saved target predictions before transfer.
    if direction=='CHARLS_to_ELSA':home=charls(s);old=pd.read_pickle(EXT/f'CHARLS_{s}_core_panel7_predictions.pkl');sc=[(0,3),(3,2)]
    else:
     dev=pd.read_pickle(EXT/'elsa_w4_people.pkl');home=elsa.loc[elsa.outcome.eq(s)&~elsa.person_id.isin(dev.person_id)].copy();old=pd.read_pickle(EXT/f'ELSA_{s}_core_temporal_panel7_predictions.pkl');sc=[(0,2),(2,2),(4,2)]
    check=pd.Series(frozen(model,home,sc),index=home.person_id);assert np.allclose(old.person_id.map(check),old[kind],atol=1e-12)
    pred[kind]=frozen(model,d,schedule)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==hashes[name]
   pred.to_pickle(P/f'{direction}_{s}.pkl');row=S.summarize(pred,direction,s,'core','frozen_panel7');rows.append(row)
   print(direction,s,round(row['delta_auroc'],4),flush=True)
 write('transport.json',rows);write('frozen_model_hashes.json',hashes)

def contrasts():
 rows=[]
 for cohort in ['CHARLS','ELSA']:
  for block in ['core','rich']:
   datasets=[]
   for s in S.MAP:
    d=pd.read_pickle(EXT/f'{cohort}_{s}_{block}_panel7_predictions.pkl');d=d.loc[d.evaluable].copy();d.person_id=d.person_id.astype(str);datasets.append((s,d))
   ids=sorted(set().union(*(set(d.person_id) for _,d in datasets)));index={p:i for i,p in enumerate(ids)};terms=[]
   for s,d in datasets:
    f=S.fast_metric(d.event.to_numpy(int),d.reference.to_numpy());g=S.fast_metric(d.event.to_numpy(int),d.blood.to_numpy());ii=np.array([index[p] for p in d.person_id]);terms.append((s,ii,f,g))
   def calc(w):
    w=np.asarray(w,dtype=float)
    ds=[g(w[ii])[0]-f(w[ii])[0] for s,ii,f,g in terms];return [float(np.mean(ds[:2])),float(np.mean(ds[2:])),float(np.mean(ds[:2])-np.mean(ds[2:]))]
   point=calc(np.ones(len(ids)));rng=np.random.default_rng(20260914);boot=np.array([calc(rng.multinomial(len(ids),np.full(len(ids),1/len(ids)))) for _ in range(2000)]);assert np.isfinite(boot).all()
   low,high=np.quantile(boot,[.025,.975],axis=0)
   row={'cohort':cohort,'block':block,'unique_n':len(ids),'direct_mean':point[0],'other_mean':point[1],'contrast':point[2],'direct_low':low[0],'direct_high':high[0],'other_low':low[1],'other_high':high[1],'low':low[2],'high':high[2]};rows.append(row);print('contrast',cohort,block,row['contrast'],flush=True)
 write('proximity_contrasts.json',rows)

def lipids():
 E=Path(os.environ["ELSA_DATA_DIR"]);C=Path(os.environ["CHARLS_DATA_DIR"])
 n=pd.read_stata(next(E.rglob('wave_6_elsa_nurse_data_v2.dta')),convert_categoricals=False).set_index('idauniq')
 # BNF codes may be strings; leading zero is clinically significant.
 drugcols=[c for c in n if c.startswith('DrC') and c[3:].isdigit()]
 stat=pd.Series(False,index=n.index);lip=stat.copy();unknown=pd.Series(False,index=n.index)
 for c in drugcols:
  j=int(c[3:]);use='MedBIA' if j==1 else f'MedBIA{j}';codes=n[c].astype(str).str.replace(r'\.0$','',regex=True).str.zfill(6)
  active=n[use].eq(1);valid=codes.str.match(r'^\d{6}$')&~codes.eq('000000')
  stat|=active&codes.eq('021201');lip|=active&codes.str.startswith('0212');unknown|=active&~valid
 # No medicine reported is an observed zero; an uncodeable medicine list remains uncertain.
 known=n.MedCNJD.eq(2)|(n.MedCNJD.eq(1)&~unknown)
 ep=pd.read_pickle(EXT/'elsa_w6_people.pkl');ep=ep.loc[ep.outcome.eq('hchole')&ep.evaluable].copy()
 ep=ep.join(pd.DataFrame({'statin':stat,'lipid_drug':lip,'drug_known':known}),on='person_id')
 old=pd.read_pickle(EXT/'ELSA_hchole_rich_panel7_predictions.pkl');old=old.loc[old.evaluable].copy();old=old.merge(ep[['person_id','statin','lipid_drug','drug_known']],on='person_id',validate='one_to_one')
 subsets=[]
 for label,mask in [('no_recorded_lipid_drug',old.drug_known&~old.lipid_drug),('recorded_lipid_drug',old.lipid_drug)]:
  x=old.loc[mask];assert x.event.sum()>0 and x.event.sum()<len(x);row=S.summarize(x,'ELSA','hchole','rich',label);subsets.append(row)
 # CHARLS item coverage within the actual disease-free evaluation risk set.
 raw=pd.read_stata(next(C.glob('2015/CHARLS2015r/Health_Status_and_Functioning.dta')),convert_categoricals=False)
 idc='ID' if 'ID' in raw else 'id';raw[idc]=raw[idc].astype(str)
 cp=charls('hchole');cp=cp.loc[cp.evaluable].copy();merged=cp[['person_id']].merge(raw[[idc]+[f'da010_2_s{i}' for i in range(1,5)]],left_on='person_id',right_on=idc,how='left',validate='one_to_one')
 assert merged[idc].notna().all(),'CHARLS ID linkage'
 meds={'elsa_evaluable_n':len(ep),'elsa_known_drug_status_n':int(ep.drug_known.sum()),'elsa_statin_n':int(ep.statin.sum()),'elsa_lipid_drug_n':int(ep.lipid_drug.sum()),'elsa_unknown_n':int((~ep.drug_known).sum()),'charls_evaluable_n':len(cp),'charls_any_treatment_item_observed':int(merged[[f'da010_2_s{i}' for i in range(1,5)]].notna().any(axis=1).sum()),'charls_recorded_western_medication_n':int(merged.da010_2_s2.eq(2).sum()),'bnf_statin_code':'021201','bnf_lipid_prefix':'0212'}
 write('medication_audit.json',meds);write('lipid_medication_subsets.json',subsets)
 # Exact additive decomposition of AUC increments, with independent cohort bootstrap.
 preds=[];desc=[]
 for cohort,data in [('CHARLS',cp),('ELSA',ep)]:
  pr=pd.read_pickle(EXT/f'{cohort}_hchole_rich_panel7_predictions.pkl');pr=pr.loc[pr.evaluable];preds.append(pr)
  row={'cohort':cohort,'n':len(pr),'events':int(pr.event.sum()),'event_pct':100*float(pr.event.mean())}
  for v in ['total_cholesterol','hdl_cholesterol','triglycerides']:
   vals=data[v].dropna();row[v+'_median']=float(vals.median());row[v+'_q1']=float(vals.quantile(.25));row[v+'_q3']=float(vals.quantile(.75))
  row.update(reference_auroc=roc_auc_score(pr.event,pr.reference),blood_auroc=roc_auc_score(pr.event,pr.blood));desc.append(row)
 def get_auc(d):
  y=d.event.to_numpy(int);f=S.fast_metric(y,d.reference.to_numpy());g=S.fast_metric(y,d.blood.to_numpy());return y,f,g
 fs=[get_auc(d) for d in preds];rng=np.random.default_rng(20260914);draw=[]
 for _ in range(2000):
  vals=[]
  for y,f,g in fs:
   p=np.flatnonzero(y==1);n0=np.flatnonzero(y==0);idx=np.r_[rng.choice(p,len(p),True),rng.choice(n0,len(n0),True)];w=np.bincount(idx,minlength=len(y)).astype(float);vals.append(g(w)[0]-f(w)[0])
  draw.append(vals[1]-vals[0])
 gap=(desc[1]['blood_auroc']-desc[1]['reference_auroc'])-(desc[0]['blood_auroc']-desc[0]['reference_auroc'])
 write('lipid_decomposition.json',{'cohorts':desc,'increment_gap':gap,'low':float(np.quantile(draw,.025)),'high':float(np.quantile(draw,.975)),'enhanced_difference':desc[1]['blood_auroc']-desc[0]['blood_auroc'],'reference_difference':desc[1]['reference_auroc']-desc[0]['reference_auroc']})
 print('medication',meds,flush=True)

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('task',choices=['transport','contrasts','lipids']);a=p.parse_args();globals()[a.task]()
