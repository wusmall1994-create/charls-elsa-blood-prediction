import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import numpy as np,pandas as pd,json,importlib.util
from scipy.special import expit,logit
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import brier_score_loss,roc_auc_score
W=Path(os.environ["ANALYSIS_ROOT"])/"extension_work";W.mkdir(parents=True,exist_ok=True);P=Path(os.environ["ANALYSIS_ROOT"]);Q=W/'competing_results';Q.mkdir(exist_ok=True)
out=P/'data_private/revision_competing';out.mkdir(exist_ok=True)
rows=[]
for file in (P/'data_private/revision_20260913').glob('*.pkl'):
 d=pd.read_pickle(file);d=d.loc[d.evaluable].reset_index(drop=True);y=d.event.to_numpy();folds=list(StratifiedKFold(5,shuffle=True,random_state=20260915).split(d,y));rng=np.random.default_rng(20260915)
 for block in ['reference','blood']:
  z=logit(np.clip(d[block].to_numpy(),1e-8,1-1e-8));pred={'unchanged':d[block].to_numpy().copy(),'intercept':np.zeros(len(d)),'intercept_slope':np.zeros(len(d))}
  for train,test in folds:
   fit=minimize_scalar(lambda a:np.mean(np.logaddexp(0,z[train]+a)-y[train]*(z[train]+a)),bounds=(-20,20),method='bounded');assert fit.success
   pred['intercept'][test]=expit(z[test]+fit.x)
   m=LogisticRegression(penalty=None,max_iter=3000).fit(z[train,None],y[train]);pred['intercept_slope'][test]=m.predict_proba(z[test,None])[:,1]
  for method,p in pred.items():
   cal=LogisticRegression(penalty=None,max_iter=3000).fit(logit(np.clip(p,1e-8,1-1e-8))[:,None],y)
   delta=(y-p)**2-(y-pred['unchanged'])**2;bs=[]
   for _ in range(2000):bs.append(float(delta[rng.integers(0,len(y),len(y))].mean()))
   rows.append({'transport':file.stem,'block':block,'method':method,'n':len(y),'events':int(y.sum()),'auroc':roc_auc_score(y,p),'brier':brier_score_loss(y,p),'brier_change':float(delta.mean()),'low':float(np.quantile(bs,.025)),'high':float(np.quantile(bs,.975)),'intercept':float(cal.intercept_[0]),'slope':float(cal.coef_[0,0])})
   d[block+'_'+method]=p
 d.to_pickle(out/(file.stem+'_recalibrated.pkl'));print(file.stem,flush=True)
(Q/'recalibration.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
