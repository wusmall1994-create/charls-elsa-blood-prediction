"""Nested participant-level CV on checked ELSA cohorts; saves fold artifacts.

Primary core and richer-reference sensitivity are separate paired comparisons.
No tuning on held-out participants. No recalibration of their predictions.
"""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import time
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(os.environ["ANALYSIS_ROOT"])
PRIVATE=ROOT/'data_private'/'elsa'
QA=ROOT/'qa_logs'
SPEC=importlib.util.spec_from_file_location('cohort_builder',ROOT/'scripts/core/38_build_elsa_checked_cohorts.py')
B=importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(B)
SEED=20260912
TIME=['start','start_squared','duration','log_duration']
def terms(d):
    d=d.copy(); d['start_squared']=d.start**2; d['log_duration']=np.log(d.duration); return d
def pipeline(cont,cat):
    return Pipeline([('preprocess',ColumnTransformer([
        ('continuous',Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),('scale',StandardScaler())]),cont),
        ('categorical',Pipeline([('impute',SimpleImputer(strategy='most_frequent',add_indicator=True,keep_empty_features=True)),('encode',OneHotEncoder(drop='first',handle_unknown='ignore',sparse_output=False))]),cat)
    ])),('model',LogisticRegression(C=.1,solver='lbfgs',max_iter=3000))])
def metrics(y,p):
    z=np.log(np.clip(p,1e-8,1-1e-8)/(1-np.clip(p,1e-8,1-1e-8))).reshape(-1,1)
    cal=LogisticRegression(penalty=None,max_iter=3000).fit(z,y)
    return {'auroc':float(roc_auc_score(y,p)),'average_precision':float(average_precision_score(y,p)),'brier':float(brier_score_loss(y,p)),'mean_risk':float(np.mean(p)),'calibration_intercept':float(cal.intercept_[0]),'calibration_slope':float(cal.coef_[0,0])}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--blocks',nargs='+',default=['core','rich']); ap.add_argument('--outcomes',nargs='+',default=['diabe','hchole','hibpe','hearte','arthre','stroke','lunge']); args=ap.parse_args()
    qa=json.loads((QA/'elsa_cohort_quality.json').read_text(encoding='utf-8'))
    assert qa['training_gate']=='passed',qa['training_gate']
    for name,digest in qa['private_sha256'].items(): assert hashlib.sha256((PRIVATE/name).read_bytes()).hexdigest()==digest
    people=pd.read_pickle(PRIVATE/'checked_people.pkl'); intervals=pd.read_pickle(PRIVATE/'checked_intervals.pkl')
    results=[]; tuning=[]; starttime=time.time()
    status={'status':'running','seed':SEED,'started_utc':pd.Timestamp.now(tz='UTC').isoformat(),'outcomes':args.outcomes,'blocks':args.blocks,'completed':[],'cohort_hashes':qa['private_sha256']}
    (QA/'elsa_training_status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    for outcome in args.outcomes:
        p=people.loc[people.outcome.eq(outcome)].reset_index(drop=True)
        it=intervals.loc[intervals.outcome.eq(outcome)].copy()
        # One row per participant split, stratified by whether an event was observed.
        outer=list(StratifiedKFold(5,shuffle=True,random_state=SEED).split(p,p.event))
        assignment=p[['person_id','event','evaluable']].copy(); assignment['fold']=-1
        for fold,(_,te) in enumerate(outer): assignment.loc[te,'fold']=fold
        assignment.to_pickle(PRIVATE/f'{outcome}_outer_folds.pkl')
        for block in args.blocks:
            preds=p[['person_id','event','evaluable']].copy(); preds['fold']=assignment.fold; preds['reference']=np.nan; preds['blood']=np.nan
            for fold,(tr,te) in enumerate(outer):
                train=p.iloc[tr].reset_index(drop=True); test=p.iloc[te].reset_index(drop=True)
                assert not set(train.person_id)&set(test.person_id)
                trainrows=terms(it.loc[it.person_id.isin(train.person_id)].merge(train.drop(columns=['outcome']),on='person_id',validate='many_to_one').reset_index(drop=True))
                inner=[]
                for itr,ite in StratifiedKFold(5,shuffle=True,random_state=SEED+fold+1).split(train,train.event):
                    tri=set(train.iloc[itr].person_id); tei=set(train.iloc[ite].person_id)
                    assert not tri&tei
                    inner.append((np.flatnonzero(trainrows.person_id.isin(tri)),np.flatnonzero(trainrows.person_id.isin(tei))))
                grid=pd.concat([test.assign(start=s,duration=2.) for s in [0.,2.,4.]],ignore_index=True); grid=terms(grid)
                for model in ['reference','blood']:
                    cont=TIME+B.CORE_CONT+(B.RICH_CONT if block=='rich' else [])+(B.BLOOD if model=='blood' else [])
                    cat=[c for c in B.CORE_CAT if c!=outcome]+(B.RICH_CAT if block=='rich' else [])+(['fasting_sample'] if model=='blood' else [])
                    search=GridSearchCV(pipeline(cont,cat),{'model__C':[.01,.1,1.,10.,100.]},scoring='neg_log_loss',cv=inner,n_jobs=1,refit=True,error_score='raise')
                    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
                        warnings.simplefilter('always')
                        search.fit(trainrows[cont+cat],trainrows.label.astype(int))
                    convergence=[str(w.message) for w in caught if issubclass(w.category,ConvergenceWarning)]
                    if convergence: raise RuntimeError(f'Convergence failed: {outcome} {block} {model} {convergence[:1]}')
                    hazards=search.predict_proba(grid[cont+cat])[:,1]
                    survival=pd.DataFrame({'id':grid.person_id,'s':1-hazards}).groupby('id').s.prod()
                    pred=1-survival
                    preds.loc[te,model]=test.person_id.map(pred).to_numpy()
                    joblib.dump(search.best_estimator_,PRIVATE/f'{outcome}_{block}_{model}_fold{fold}.joblib',compress=3)
                    tuning.append({'outcome':outcome,'block':block,'model':model,'outer_fold':fold,'C':search.best_params_['model__C'],'inner_log_loss':float(-search.best_score_),'train_n':len(train),'test_n':len(test),'nonconvergence':False})
                print(f'{outcome} {block} fold {fold+1}/5 done',flush=True)
            assert preds[['reference','blood']].notna().all().all()
            assert preds[['reference','blood']].ge(0).all().all() and preds[['reference','blood']].le(1).all().all()
            preds.to_pickle(PRIVATE/f'{outcome}_{block}_oof.pkl')
            e=preds.loc[preds.evaluable]; y=e.event.to_numpy(); r=e.reference.to_numpy(); b=e.blood.to_numpy()
            mr=metrics(y,r); mb=metrics(y,b)
            rng=np.random.default_rng(SEED); pos=np.flatnonzero(y==1); neg=np.flatnonzero(y==0); diffs=[]
            for _ in range(2000):
                ii=np.r_[rng.choice(pos,len(pos),replace=True),rng.choice(neg,len(neg),replace=True)]
                diffs.append([roc_auc_score(y[ii],b[ii])-roc_auc_score(y[ii],r[ii]),np.mean((b[ii]-y[ii])**2-(r[ii]-y[ii])**2)])
            ci=np.quantile(diffs,[.025,.975],axis=0)
            row={'outcome':outcome,'block':block,'n':len(e),'events':int(y.sum()),'observed_risk':float(y.mean()),'reference':mr,'blood':mb,'delta_auroc':mb['auroc']-mr['auroc'],'delta_auroc_ci':ci[:,0].tolist(),'delta_brier':mb['brier']-mr['brier'],'delta_brier_ci':ci[:,1].tolist(),'evaluation':'pooled outer-fold predictions; nominal six-year risk; endpoint by wave9'}
            results.append(row); status['completed'].append(outcome+'_'+block)
            (QA/'elsa_nested_results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
            (QA/'elsa_nested_tuning.json').write_text(json.dumps(tuning,indent=2),encoding='utf-8')
            (QA/'elsa_training_status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
            print(f'COMPLETE {outcome} {block} delta AUROC {row["delta_auroc"]:.4f}',flush=True)
    status.update(status='completed',elapsed_seconds=time.time()-starttime,finished_utc=pd.Timestamp.now(tz='UTC').isoformat())
    (QA/'elsa_training_status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')

if __name__=='__main__': main()
