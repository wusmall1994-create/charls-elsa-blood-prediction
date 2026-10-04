import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import argparse,importlib.util,json,hashlib,time,warnings,sys
import numpy as np,pandas as pd,joblib,sklearn
from sklearn.model_selection import StratifiedKFold,GridSearchCV
from sklearn.metrics import roc_auc_score,brier_score_loss
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
W=Path(os.environ["ANALYSIS_ROOT"])/"extension_work";W.mkdir(parents=True,exist_ok=True);R=Path(os.environ["ANALYSIS_ROOT"]);P=R/'data_private';OLD=P/'replication_extensions';NEW=P/'priority_extension_20260922';NEW.mkdir(exist_ok=True);Q=W/'priority_results';Q.mkdir(exist_ok=True)
sp=importlib.util.spec_from_file_location('existing42',Path(__file__).resolve().parents[3] / 'scripts/core/42_replication_sensitivity_suite.py');X=importlib.util.module_from_spec(sp);sp.loader.exec_module(X);T=X.T
MODELS=['reference','simple','full'];SEED=20260922
def dump(path,x):
 path.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def read(cohort,outcome,phase):
 if cohort=='CHARLS':
  folder=X.MAP[outcome];d=pd.read_csv(P/'multoutcome'/folder/(phase+'.csv.gz'),dtype={'person_id':str}).rename(columns=X.RENAME)
  it=pd.read_csv(P/'multoutcome'/folder/(phase+'_intervals.csv.gz'),dtype={'person_id':str}).rename(columns={'interval_start_years':'start','interval_duration_years':'duration','event_this_interval':'label','source_wave':'wave'})
  d['evaluable']=d.event.eq(1)|d.person_id.isin(it.loc[it.wave.eq(5),'person_id'])
 else:
  w=4 if phase=='development' else 6;d=pd.read_pickle(OLD/f'elsa_w{w}_people.pkl');d=d.loc[d.outcome.eq(outcome)].copy();it=pd.read_pickle(OLD/f'elsa_w{w}_intervals.pkl');it=it.loc[it.outcome.eq(outcome)].copy()
 d.person_id=d.person_id.astype(str);it.person_id=it.person_id.astype(str)
 return d.reset_index(drop=True),it[['person_id','start','duration','label','wave']].reset_index(drop=True)
def masks(d,o):
 if o=='diabe':
  measured=d.fasting_sample.eq(1)&d.glucose.notna()&d.hba1c.notna();normal=measured&d.glucose.lt(126)&d.hba1c.lt(6.5);strict=measured&d.glucose.lt(100)&d.hba1c.lt(5.7)
 else:
  measured=d[['total_cholesterol','hdl_cholesterol','triglycerides']].notna().all(axis=1)&d.fasting_sample.notna()
  normal=measured&d.total_cholesterol.lt(190)&d.hdl_cholesterol.gt(40)&(d.total_cholesterol-d.hdl_cholesterol).lt(150)&d.triglycerides.lt(d.fasting_sample.map({1:150,0:175,2:175}));strict=normal
 return {'all':pd.Series(True,index=d.index),'measured':measured,'below_threshold':normal,'strict_descriptive':strict}
def subset(d,it,o,scenario):
 if scenario!='landmark':
  p=d.loc[masks(d,o)[scenario]].copy();return p.reset_index(drop=True),it.loc[it.person_id.isin(p.person_id)].copy()
 first=int(it.wave.min());second=first+1
 land=it.loc[it.wave.eq(first)&it.label.eq(0)].copy();land['landmark_time']=land.start+land.duration
 z=it.loc[it.wave.eq(second)].merge(land[['person_id','landmark_time']],on='person_id',validate='one_to_one')
 z['duration']=z.start+z.duration-z.landmark_time;z['start']=0.
 assert z.duration.gt(0).all() and z.person_id.is_unique
 p=d.loc[d.person_id.isin(z.person_id)].drop(columns=['event','evaluable']).merge(z[['person_id','label','landmark_time']],on='person_id',validate='one_to_one').rename(columns={'label':'event'})
 p['evaluable']=True
 return p.reset_index(drop=True),z[['person_id','start','duration','label','wave']].copy()
def features(cohort,o,model):
 cont,cat=X.features(cohort,'rich',o,'reference','panel7');cont=list(cont);cat=list(cat)
 if model!='reference':
  cont+=(['glucose'] if o=='diabe' else ['total_cholesterol','hdl_cholesterol','triglycerides']) if model=='simple' else list(X.B.BLOOD)
  cat+=['fasting_sample']
 return cont,cat
def rows_for(d,it):return T.terms(it.merge(d.drop(columns=['outcome'],errors='ignore'),on='person_id',validate='many_to_one').reset_index(drop=True))
def predict(est,test,cohort,scenario):
 schedule=[(0.,2.)] if scenario=='landmark' else ([(0.,3.),(3.,2.)] if cohort=='CHARLS' else [(0.,2.),(2.,2.),(4.,2.)])
 grid=T.terms(pd.concat([test.assign(start=a,duration=b) for a,b in schedule],ignore_index=True));h=est.predict_proba(grid)[:,1]
 surv=pd.DataFrame({'id':grid.person_id,'s':1-h}).groupby('id').s.prod();return test.person_id.map(1-surv).to_numpy()
def fit(train,it,test,c,o,m,scenario,key,seed,fixed=None):
 path=NEW/(key+'.joblib');meta=NEW/(key+'.json');cont,cat=features(c,o,m)
 signature=hashlib.sha256(pd.util.hash_pandas_object(train[['person_id','event']],index=False).values.tobytes()+pd.util.hash_pandas_object(it,index=False).values.tobytes()).hexdigest()
 if path.exists() and meta.exists():
  info=json.loads(meta.read_text());assert info['signature']==signature;est=joblib.load(path)
 else:
  rows=rows_for(train,it);splits=[]
  if fixed is None:
   assert train.event.value_counts().min()>=5
   for a,b in StratifiedKFold(5,shuffle=True,random_state=seed).split(train,train.event):
    ai=set(train.iloc[a].person_id);bi=set(train.iloc[b].person_id);assert not ai&bi
    splits.append((np.flatnonzero(rows.person_id.isin(ai)),np.flatnonzero(rows.person_id.isin(bi))))
   search=GridSearchCV(T.pipeline(cont,cat),{'model__C':[.01,.1,1,10,100]},scoring='neg_log_loss',cv=splits,n_jobs=1,error_score='raise')
  else:search=T.pipeline(cont,cat).set_params(model__C=fixed)
  with warnings.catch_warnings(record=True) as caught,threadpool_limits(limits=1):
   warnings.simplefilter('always');search.fit(rows[cont+cat],rows.label.astype(int))
  assert not any(issubclass(w.category,ConvergenceWarning) for w in caught),key
  est=search.best_estimator_ if fixed is None else search
  info={'signature':signature,'cohort':c,'outcome':o,'scenario':scenario,'model':m,'seed':seed,'C':float(est.named_steps['model'].C),'features':cont+cat,'train_n':len(train),'train_events':int(train.event.sum()),'rows':len(rows),'nonconvergence':False}
  joblib.dump(est,path,compress=3);dump(meta,info)
 return predict(est,test,c,scenario),info
def summarize(pred,c,o,scenario,repeat=None,bootstrap=True):
 e=pred.loc[pred.evaluable];assert e.person_id.is_unique;y=e.event.to_numpy(int);assert min(np.bincount(y))>0
 fs={m:X.fast_metric(y,e[m].to_numpy()) for m in MODELS};point={m:f(np.ones(len(y))) for m,f in fs.items()};deltas={}
 for a,b in [('simple','reference'),('full','reference'),('full','simple')]:
  v=point[a]-point[b];ci=np.full((2,3),np.nan)
  if bootstrap:
   rng=np.random.default_rng(X.SEED if scenario=='all' else SEED);pos=np.flatnonzero(y);neg=np.flatnonzero(y==0);draw=[]
   for _ in range(2000):
    ii=np.r_[rng.choice(pos,len(pos),True),rng.choice(neg,len(neg),True)];wt=np.bincount(ii,minlength=len(y)).astype(float);draw.append(fs[a](wt)-fs[b](wt))
   ci=np.quantile(draw,[.025,.975],axis=0)
  deltas[a+'_'+b]={metric: {'estimate':float(v[i]),**({'low':float(ci[0,i]),'high':float(ci[1,i])} if bootstrap else {})} for i,metric in enumerate(['auroc','brier','ap'])}
 return {'cohort':c,'outcome':o,'scenario':scenario,'repeat':repeat,'n':len(e),'events':int(y.sum()),'metrics':{m:dict(zip(['auroc','brier','ap'],map(float,v))) for m,v in point.items()},'deltas':deltas}
def run(c,o,stability=False):
 task=c+'_'+o+('_stability' if stability else '');output=Q/(task+'.json')
 if output.exists():print('cached',task,flush=True);return
 start=time.time();alltest,it=read(c,o,'validation');train,dit=read(c,o,'development') if c=='CHARLS' else (None,None);results=[];audit=[];tuning=[]
 if stability:
  if c=='ELSA':
   for seed in range(SEED,SEED+5):
    pred=alltest[['person_id','event','evaluable']].copy()
    for fold,(a,b) in enumerate(StratifiedKFold(5,shuffle=True,random_state=seed).split(alltest,alltest.event)):
     for m in MODELS:
      v,info=fit(alltest.iloc[a].reset_index(drop=True),it.loc[it.person_id.isin(alltest.iloc[a].person_id)],alltest.iloc[b],c,o,m,'all',f'{task}_{seed}_{fold}_{m}',seed+fold+1)
      pred.loc[b,m]=v;tuning.append(info)
    pred.to_pickle(NEW/f'{task}_{seed}_predictions.pkl');results.append(summarize(pred,c,o,'split_stability',seed,False));print(task,seed,'complete',flush=True)
  else:
   rng=np.random.default_rng(SEED)
   fixed={}
   for m in MODELS:
    f=(NEW/f'{c}_{o}_all_simple.json') if m=='simple' else OLD/('CHARLS_'+o+'_rich_'+('reference' if m=='reference' else 'panel7')+'_fit.json')
    fixed[m]=float(json.loads(f.read_text())['C'])
   for rep in range(50):
    sample=train.iloc[rng.integers(0,len(train),len(train))].copy().reset_index(drop=True);sample['original_id']=sample.person_id;sample.person_id=sample.person_id+'__copy'+sample.index.astype(str)
    bootit=dit.rename(columns={'person_id':'original_id'}).merge(sample[['person_id','original_id']],on='original_id').drop(columns='original_id');sample=sample.drop(columns='original_id');pred=alltest[['person_id','event','evaluable']].copy()
    for m in MODELS:
     pred[m],info=fit(sample,bootit,alltest,c,o,m,'all',f'{task}_{rep}_{m}',SEED,fixed[m]);tuning.append(info)
    results.append(summarize(pred,c,o,'development_refit',rep,False))
    if rep%10==9:print(task,rep+1,'complete',flush=True)
 else:
  for scenario in ['all','measured','below_threshold','landmark']:
   test,vit=subset(alltest,it,o,scenario);dev,di=subset(train,dit,o,scenario) if c=='CHARLS' else (None,None)
   for phase,d in [('evaluation',test)]+([('development',dev)] if c=='CHARLS' else []):
    e=d.loc[d.evaluable];z={'scenario':scenario,'phase':phase,'n':len(d),'events':int(d.event.sum()),'evaluable_n':len(e),'evaluable_events':int(e.event.sum()),'markers':{}}
    for col in ['age','glucose','hba1c','total_cholesterol','hdl_cholesterol','triglycerides']:
     vals=e[col].dropna();z['markers'][col]={'missing':int(e[col].isna().sum()),'median':float(vals.median()) if len(vals) else None,'q25':float(vals.quantile(.25)) if len(vals) else None,'q75':float(vals.quantile(.75)) if len(vals) else None}
    if scenario=='landmark':z['landmark_time']=list(map(float,d.landmark_time.quantile([0,.5,1])));z['interval_duration']=list(map(float,(vit if phase=='evaluation' else di).duration.quantile([0,.5,1])))
    audit.append(z)
   if int(test.loc[test.evaluable,'event'].sum())<30 or (c=='CHARLS' and dev.event.sum()<30):
    results.append({'cohort':c,'outcome':o,'scenario':scenario,'not_fitted':'fewer than 30 events'});continue
   pred=test[['person_id','event','evaluable']].copy()
   if scenario=='all':
    saved=pd.read_pickle(OLD/f'{c}_{o}_rich_panel7_predictions.pkl');saved.person_id=saved.person_id.astype(str)
    pred=pred.merge(saved[['person_id','reference','blood']],on='person_id',validate='one_to_one').rename(columns={'blood':'full'});assert len(pred)==len(test)
   if c=='CHARLS':
    assert not set(dev.person_id)&set(test.person_id)
    for m in (['simple'] if scenario=='all' else MODELS):
     v,info=fit(dev,di,test,c,o,m,scenario,f'{c}_{o}_{scenario}_{m}',X.SEED if scenario=='all' else SEED)
     pred[m]=pred.person_id.map(dict(zip(test.person_id,v)));tuning.append(info)
   else:
    if scenario=='all':
     folds=pd.read_pickle(P/f'elsa/{o}_outer_folds.pkl');folds.person_id=folds.person_id.astype(str);assign=test.person_id.map(folds.set_index('person_id').fold)
    else:
     assign=pd.Series(-1,index=test.index)
     for fold,(_,b) in enumerate(StratifiedKFold(5,shuffle=True,random_state=SEED).split(test,test.event)):assign.iloc[b]=fold
    for fold in range(5):
     a=test.loc[assign.ne(fold)].reset_index(drop=True);b=test.loc[assign.eq(fold)].reset_index(drop=True);assert not set(a.person_id)&set(b.person_id)
     for m in (['simple'] if scenario=='all' else MODELS):
      v,info=fit(a,vit.loc[vit.person_id.isin(a.person_id)],b,c,o,m,scenario,f'{c}_{o}_{scenario}_{fold}_{m}',T.SEED+fold+1 if scenario=='all' else SEED+fold+1);pred.loc[pred.person_id.isin(b.person_id),m]=pred.loc[pred.person_id.isin(b.person_id),'person_id'].map(dict(zip(b.person_id,v)));tuning.append(info)
   assert np.isfinite(pred[MODELS]).all().all();assert pred[MODELS].ge(0).all().all() and pred[MODELS].le(1).all().all()
   pred.to_pickle(NEW/f'{c}_{o}_{scenario}_predictions.pkl');results.append(summarize(pred,c,o,scenario));print(task,scenario,'n',results[-1]['n'],'events',results[-1]['events'],'full-simple',results[-1]['deltas']['full_simple']['auroc'],flush=True)
  strict=alltest.loc[masks(alltest,o)['strict_descriptive']&alltest.evaluable];audit.append({'scenario':'strict_descriptive','evaluable_n':len(strict),'evaluable_events':int(strict.event.sum())})
  if o=='diabe':
   base=pd.read_pickle(NEW/f'{c}_{o}_all_predictions.pkl');observed=set(alltest.loc[alltest.fasting_sample.eq(1)&alltest.glucose.notna(),'person_id'])
   for name,mask in [('glucose_observed',base.person_id.isin(observed)),('glucose_missing_or_unverified',~base.person_id.isin(observed))]:
    results.append(summarize(base.loc[mask],c,o,name))
 dump(output,{'results':results,'audit':audit,'tuning':tuning,'elapsed':time.time()-start,'versions':{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__},'contract_sha256':None})
 print('FINISHED',task,round(time.time()-start,1),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('cohort',choices=['CHARLS','ELSA']);ap.add_argument('outcome',choices=['diabe','hchole']);ap.add_argument('--stability',action='store_true');a=ap.parse_args();run(a.cohort,a.outcome,a.stability)
