"""Paired participant bootstrap of glycemic additions and transported decision value.

Retunes every disease pipeline. Duplicate participant draws remain in one fold.
Only aggregate replicate metrics and tuning counts are exported.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[k]='1'
import argparse, importlib.util, json, hashlib, time, warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from scipy.special import logit, expit
from threadpoolctl import threadpool_limits

SEED=20261022
MODELS=['glycemic','full_hba1c','full']

def setup(args):
    global A,M,X,T,DATA
    A=args
    os.environ['ANALYSIS_ROOT']=str(args.data_private.parent)
    os.environ['ELSA_DATA_DIR']=str(args.data_private)
    spec=importlib.util.spec_from_file_location('comparators',args.code_dir/'scripts/analyses/panel_performance/71_comparator_transport.py')
    M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
    spec=importlib.util.spec_from_file_location('common',args.code_dir/'scripts/core/42_replication_sensitivity_suite.py')
    X=importlib.util.module_from_spec(spec);spec.loader.exec_module(X);T=X.T
    M.X=X;M.T=T;M.P=args.data_private;M.COMMON=args.common_panel_dir
    DATA={(c,phase):M.cohort(c,'diabe',phase) for c in ['CHARLS','ELSA'] for phase in ['development','validation']}
    for d,it in DATA.values():
        d['original_id']=d.person_id
        assert d.person_id.is_unique and set(it.person_id)<=set(d.person_id)

def bootstrap(d,it,rng):
    """Materialize participant multiplicities for preprocessing and fitting."""
    b=d.iloc[rng.integers(len(d),size=len(d))].copy().reset_index(drop=True)
    b['person_id']=['draw_'+str(i) for i in range(len(b))]
    bi=it.rename(columns={'person_id':'original_id'}).merge(b[['original_id','person_id']],on='original_id',validate='many_to_many').drop(columns='original_id')
    assert b.person_id.is_unique and len(b)==len(d)
    return b,bi

def splits(d,seed):
    u=d[['original_id','event']].drop_duplicates('original_id').reset_index(drop=True)
    for a,b in StratifiedKFold(5,shuffle=True,random_state=seed).split(u,u.event):
        train=set(u.iloc[a].original_id);test=set(u.iloc[b].original_id)
        assert not train&test
        yield np.flatnonzero(d.original_id.isin(train)),np.flatnonzero(d.original_id.isin(test))

def fit(d,it,c,block,model,seed):
    cont,cat=M.features(c,'diabe',model,block)
    rows=T.terms(it.merge(d.drop(columns=['outcome'],errors='ignore'),on='person_id',validate='many_to_one'))
    cv=[]
    for a,b in splits(d,seed):
        ai=np.flatnonzero(rows.person_id.isin(d.iloc[a].person_id));bi=np.flatnonzero(rows.person_id.isin(d.iloc[b].person_id))
        assert not set(rows.iloc[ai].original_id)&set(rows.iloc[bi].original_id)
        cv.append((ai,bi))
    search=GridSearchCV(T.pipeline(cont,cat),{'model__C':[.01,.1,1,10,100]},scoring='neg_log_loss',cv=cv,n_jobs=1,error_score='raise')
    with warnings.catch_warnings(record=True) as caught,threadpool_limits(limits=1):
        warnings.simplefilter('always');search.fit(rows[cont+cat],rows.label.astype(int))
    if any(issubclass(w.category,ConvergenceWarning) for w in caught):raise RuntimeError('Disease fit did not converge')
    return search.best_estimator_,float(search.best_params_['model__C'])

def metrics(y,p):
    return {'auroc':float(roc_auc_score(y,p)),'brier':float(brier_score_loss(y,p)),
            'nb10':float(np.mean((p>=.1)*(y-(1-y)/9))*100)}

def score(pred,label,stage,rep):
    y=pred.event.to_numpy(int);ms={m:metrics(y,pred[m].to_numpy()) for m in MODELS if m in pred}
    rows=[]
    for a,b in [('full_hba1c','glycemic'),('full','glycemic')]:
        if a not in ms:continue
        for metric in ['auroc','brier','nb10']:
            rows.append(dict(replicate=rep,setting=label,stage=stage,contrast=a+'_minus_'+b,metric=metric,
                             value=ms[a][metric]-ms[b][metric],n=len(y),events=int(y.sum())))
    return rows

def recalibrate(d,rep):
    y=d.event.to_numpy(int);res=d.copy();params=[]
    for k,(a,b) in enumerate(splits(d,SEED+rep+7000)):
        assert not set(d.iloc[a].original_id)&set(d.iloc[b].original_id)
        for m in MODELS:
            z=logit(np.clip(d[m].to_numpy(),1e-8,1-1e-8))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                f=LogisticRegression(penalty=None,max_iter=3000,tol=1e-10).fit(z[a,None],y[a])
            if any(issubclass(w.category,ConvergenceWarning) for w in caught):raise RuntimeError('Calibration did not converge')
            res.loc[b,m]=expit(f.intercept_[0]+f.coef_[0,0]*z[b])
            params.append(dict(fold=k,model=m,intercept=float(f.intercept_[0]),slope=float(f.coef_[0,0])))
    return res,params

def run(rep,args):
    setup(args);out=args.output_dir/f'replicate_{rep:03d}.json'
    if out.exists():return rep,'cached'
    start=time.time();rng=np.random.default_rng(SEED+rep);rows=[];tuning=[]
    # CHARLS: independent resampling of development and temporal evaluation participants.
    tr,ti=bootstrap(*DATA['CHARLS','development'],rng);te,_=bootstrap(*DATA['CHARLS','validation'],rng)
    assert not set(tr.original_id)&set(te.original_id)
    pred=te[['person_id','original_id','event','evaluable']].copy()
    for m in MODELS[:2]:
        f,c=fit(tr,ti,'CHARLS','rich',m,SEED+rep)
        pred[m]=M.predict(f,te,'CHARLS');tuning.append(dict(setting='CHARLS_rich',model=m,C=c))
    rows+=score(pred.loc[pred.evaluable].reset_index(drop=True),'CHARLS_nested','within',rep)
    # ELSA: bootstrap participants, repeat grouped nested five-fold CV and inner tuning.
    d,it=bootstrap(*DATA['ELSA','validation'],rng)
    pred=d[['person_id','original_id','event','evaluable']].copy()
    for m in MODELS[:2]:pred[m]=np.nan
    for k,(a,b) in enumerate(splits(d,SEED+rep+1000)):
        train=d.iloc[a].reset_index(drop=True);test=d.iloc[b].reset_index(drop=True)
        trainit=it.loc[it.person_id.isin(train.person_id)]
        for m in MODELS[:2]:
            f,c=fit(train,trainit,'ELSA','rich',m,SEED+rep+2000+k)
            pred.loc[b,m]=M.predict(f,test,'ELSA');tuning.append(dict(setting='ELSA_rich',fold=k,model=m,C=c))
    assert np.isfinite(pred[MODELS[:2]]).all().all()
    rows+=score(pred.loc[pred.evaluable].reset_index(drop=True),'ELSA_nested','within',rep)
    # Cross-cohort: independently resample source and target; freeze source fits.
    for source,target in [('CHARLS','ELSA'),('ELSA','CHARLS')]:
        tr,ti=bootstrap(*DATA[source,'development'],rng);te,_=bootstrap(*DATA[target,'validation'],rng)
        # Align target fasting to the unchanged source coding, as module 71.
        if source=='CHARLS':te['fasting_sample']=te.fasting_sample.map({1:1.,0:2.})
        else:
            assert set(tr.fasting_sample.dropna().unique())<={0.,1.}
            te['fasting_sample']=te.fasting_sample.map({1:1.,2:0.})
        pred=te[['person_id','original_id','event','evaluable']].copy()
        for m in MODELS:
            f,c=fit(tr,ti,source,'core',m,SEED+rep+3000)
            pred[m]=M.predict(f,te,target);tuning.append(dict(setting=source+'_to_'+target,model=m,C=c))
        pred=pred.loc[pred.evaluable].reset_index(drop=True)
        rows+=score(pred,source+'_to_'+target,'frozen',rep)
        updated,params=recalibrate(pred,rep)
        rows+=score(updated,source+'_to_'+target,'intercept_slope',rep)
    result=dict(replicate=rep,seed=SEED+rep,elapsed_seconds=time.time()-start,rows=rows,tuning=tuning,
                group_leakage_checks='passed',convergence='passed')
    tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(out)
    return rep,round(time.time()-start,1)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ['data-private','common-panel-dir','code-dir','output-dir']:ap.add_argument('--'+n,type=Path,required=True)
    ap.add_argument('--repetitions',type=int,default=100);ap.add_argument('--workers',type=int,default=6)
    args=ap.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    spec=dict(repetitions=args.repetitions,seed=SEED,penalty_grid=[.01,.1,1,10,100],inner_folds=5,outer_folds=5,
        resampling='ordinary paired participant bootstrap; independent source/development and target/evaluation draws; grouped duplicate identities',
        preprocessing='median/mode imputation with indicators, scaling and encoding re-estimated in every training fold',
        tuning='repeated in every refit using grouped participant inner folds',
        updating='intercept and slope re-estimated outside scored target folds in every source/target bootstrap',
        interpretation='exploratory refitting sensitivity; 2.5th–97.5th replicate percentiles are descriptive ranges, not high-precision confidence intervals',
        threshold=.1)
    (args.output_dir/'specification.json').write_text(json.dumps(spec,indent=2),encoding='utf-8')
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(run,r,args) for r in range(args.repetitions)]
        for future in as_completed(futures):print('completed',*future.result(),flush=True)
    data=[json.loads((args.output_dir/f'replicate_{r:03d}.json').read_text()) for r in range(args.repetitions)]
    df=pd.DataFrame([row for d in data for row in d['rows']]);df.to_csv(args.output_dir/'replicate_metrics.csv',index=False)
    summaries=[]
    for key,g in df.groupby(['setting','stage','contrast','metric'],sort=False):
        q=np.quantile(g.value,[.025,.5,.975]);summaries.append(dict(zip(['setting','stage','contrast','metric'],key),repeats=len(g),median=q[1],q025=q[0],q975=q[2],minimum=g.value.min(),maximum=g.value.max(),positive_fraction=float((g.value>0).mean())))
    pd.DataFrame(summaries).to_csv(args.output_dir/'summary.csv',index=False)
    pd.DataFrame([dict(replicate=d['replicate'],**t) for d in data for t in d['tuning']]).to_csv(args.output_dir/'tuning.csv',index=False)
    print('finished',len(data),'replicates',flush=True)

if __name__=='__main__':main()
