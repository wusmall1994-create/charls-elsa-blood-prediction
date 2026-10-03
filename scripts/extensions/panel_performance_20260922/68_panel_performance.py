import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import importlib.util,json,hashlib,warnings
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score,brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
W=Path(__file__).resolve().parent;Q=W/'panel_performance_results';Q.mkdir(exist_ok=True)
spec=importlib.util.spec_from_file_location('priority',W/'62_priority_analyses.py');A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A)
T=np.round(np.arange(.01,.30001,.005),6);odds=T/(1-T);models=A.MODELS
metrics=[];curves=[];summaries=[];audits=[]
for c in ['CHARLS','ELSA']:
 for o in ['diabe','hchole']:
  old=json.loads((W/f'priority_results/{c}_{o}.json').read_text())
  scenarios=['all','measured','below_threshold','landmark']+(['glucose_observed','glucose_missing_or_unverified'] if o=='diabe' else [])
  people,_=A.read(c,o,'validation')
  obs=set(people.loc[people.fasting_sample.eq(1)&people.glucose.notna(),'person_id'])
  for scenario in scenarios:
   path=A.NEW/f'{c}_{o}_{"all" if scenario.startswith("glucose_") else scenario}_predictions.pkl'
   d=pd.read_pickle(path);d=d.loc[d.evaluable].copy()
   if scenario.startswith('glucose_'):d=d.loc[d.person_id.isin(obs) if scenario=='glucose_observed' else ~d.person_id.isin(obs)]
   d=d.reset_index(drop=True);y=d.event.to_numpy(int);n=len(y);assert d.person_id.is_unique
   prior=next(r for r in old['results'] if r['scenario']==scenario);assert n==prior['n'] and int(y.sum())==prior['events']
   probs=d[models].to_numpy().T;assert np.isfinite(probs).all() and ((probs>=0)&(probs<=1)).all()
   for j,m in enumerate(models):
    p=probs[j];auc=roc_auc_score(y,p);brier=brier_score_loss(y,p)
    assert abs(auc-prior['metrics'][m]['auroc'])<1e-12 and abs(brier-prior['metrics'][m]['brier'])<1e-12
    z=np.log(np.clip(p,1e-8,1-1e-8)/(1-np.clip(p,1e-8,1-1e-8))).reshape(-1,1)
    with warnings.catch_warnings(record=True) as caught:
     warnings.simplefilter('always');cal=LogisticRegression(penalty=None,max_iter=3000,tol=1e-10).fit(z,y)
    assert not any(issubclass(w.category,ConvergenceWarning) for w in caught)
    metrics.append(dict(cohort=c,outcome=o,scenario=scenario,model=m,n=n,events=int(y.sum()),auroc=auc,brier=brier,mean_risk=float(p.mean()),observed_risk=float(y.mean()),calibration_intercept=float(cal.intercept_[0]),calibration_slope=float(cal.coef_[0,0])))
   bins=[np.searchsorted(T,p,side='right') for p in probs]
   def scores(w):
    result=[]
    for ix in bins:
     tp=np.cumsum(np.bincount(ix,weights=w*y,minlength=len(T)+1)[::-1])[::-1][1:]
     fp=np.cumsum(np.bincount(ix,weights=w*(1-y),minlength=len(T)+1)[::-1])[::-1][1:]
     result.append((tp-fp*odds)/w.sum())
    prev=np.dot(w,y)/w.sum();return np.array(result),prev-(1-prev)*odds
   point,allpoint=scores(np.ones(n));rng=np.random.default_rng(20260921);boot=[];allboot=[]
   for _ in range(2000):
    w=np.bincount(rng.integers(0,n,n),minlength=n);v,av=scores(w);boot.append(v);allboot.append(av)
   boot=np.array(boot);allboot=np.array(allboot)
   for j,m in enumerate(models):
    direct=np.array([((probs[j]>=t)*(y-(1-y)*t/(1-t))).mean() for t in T]);assert np.allclose(point[j],direct,atol=1e-14)
    lo,hi=np.quantile(boot[:,j,:],[.025,.975],axis=0)
    for i,t in enumerate(T):curves.append(dict(cohort=c,outcome=o,scenario=scenario,model=m,n=n,events=int(y.sum()),threshold=float(t),net_benefit=float(point[j,i]),low=float(lo[i]),high=float(hi[i]),assess_all=float(allpoint[i])))
   for t in [.05,.1,.2]:
    i=int(np.flatnonzero(np.isclose(T,t))[0]);r=dict(cohort=c,outcome=o,scenario=scenario,n=n,events=int(y.sum()),threshold=t,**{m:float(point[j,i]) for j,m in enumerate(models)},assess_all=float(allpoint[i]))
    for name,a,b in [('simple_reference',1,0),('full_reference',2,0),('full_simple',2,1),('full_default',2,None)]:
     value=point[a,i]-(point[b,i] if b is not None else max(allpoint[i],0))
     bs=boot[:,a,i]-(boot[:,b,i] if b is not None else np.maximum(allboot[:,i],0))
     r[name]={'estimate':float(value),'low':float(np.quantile(bs,.025)),'high':float(np.quantile(bs,.975))}
    summaries.append(r)
   audits.append(dict(cohort=c,outcome=o,scenario=scenario,n=n,events=int(y.sum()),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),auroc_brier_check=True,direct_net_benefit_check=True,calibration_converged=True))
   print(c,o,scenario,n,int(y.sum()),flush=True)
for name,value in [('metrics',metrics),('net_benefit_summary',summaries),('audit',audits)]:
 (Q/f'{name}.json').write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
pd.DataFrame(metrics).to_csv(Q/'metrics.csv',index=False);pd.DataFrame(curves).to_csv(Q/'curves.csv',index=False)
# Existing reference/full DCA results must remain identical, including the paired intervals.
existing=json.loads((W/'decision_curve_results/summary.json').read_text())
for r in summaries:
 if r['scenario']!='all':continue
 e=next(z for z in existing if z['family']=='common' and z['cohort']==r['cohort'] and z['outcome']==r['outcome'] and z['threshold']==r['threshold'])
 for a,b in [('reference','reference'),('full','blood'),('assess_all','assess_all')]:assert abs(r[a]-e[b])<1e-12
 for a,b in [('estimate','delta'),('low','delta_low'),('high','delta_high')]:assert abs(r['full_reference'][a]-e[b])<1e-12
(Q/'verification.json').write_text(json.dumps({'datasets':len(audits),'models':len(metrics),'existing_dca_reproduced':True,'contract_sha256':hashlib.sha256((W/'panel_performance_contract_20260922.md').read_bytes()).hexdigest()},indent=2))
print('PASS all saved metrics and existing decision curves reproduced',flush=True)
for r in summaries:
 if r['scenario']=='all' and r['threshold']==.1:print(r['cohort'],r['outcome'],'full-simple NB /100', {k:100*v for k,v in r['full_simple'].items()},flush=True)
