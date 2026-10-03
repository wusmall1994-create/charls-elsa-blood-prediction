import os
from pathlib import Path
import importlib.util,json,hashlib
import numpy as np,pandas as pd,joblib
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
W=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('priority',W/'62_priority_analyses.py');M=importlib.util.module_from_spec(sp);sp.loader.exec_module(M)
checks=[]
for c in ['CHARLS','ELSA']:
 for o in ['diabe','hchole']:
  f=M.Q/f'{c}_{o}.json';r=json.loads(f.read_text());p=pd.read_pickle(M.NEW/f'{c}_{o}_all_predictions.pkl')
  # Preserve the original common-panel bootstrap seed for identical baseline comparisons.
  r['results'][0]=M.summarize(p,c,o,'all');M.dump(f,r)
  for scenario in ['all','measured','below_threshold','landmark']:
   pred=pd.read_pickle(M.NEW/f'{c}_{o}_{scenario}_predictions.pkl');e=pred.loc[pred.evaluable];y=e.event.astype(int).to_numpy();row=next(x for x in r['results'] if x['scenario']==scenario)
   for model in M.MODELS:
    v=e[model].to_numpy();metrics=[roc_auc_score(y,v),brier_score_loss(y,v),average_precision_score(y,v)]
    assert np.allclose(metrics,list(row['metrics'][model].values()),atol=1e-12)
   assert np.isclose(row['deltas']['full_simple']['auroc']['estimate'],row['deltas']['full_reference']['auroc']['estimate']-row['deltas']['simple_reference']['auroc']['estimate'])
   d,it=M.read(c,o,'validation');sub,sit=M.subset(d,it,o,scenario)
   assert set(sub.person_id)==set(pred.person_id)
   if scenario=='landmark':
    first=it.wave.min();negative=set(it.loc[it.wave.eq(first)&it.label.eq(0),'person_id']);assert set(pred.person_id)<=negative
    assert sit.start.eq(0).all() and sit.wave.eq(first+1).all() and pred.evaluable.all()
    assert np.array_equal(pred.set_index('person_id').event.sort_index(),sit.set_index('person_id').label.sort_index())
   if scenario=='below_threshold':assert M.masks(sub,o)['below_threshold'].all()
   if scenario=='all':
    old=pd.read_pickle(M.OLD/f'{c}_{o}_rich_panel7_predictions.pkl');old.person_id=old.person_id.astype(str);z=pred.merge(old,on='person_id',suffixes=('','_old'))
    assert np.array_equal(z.reference,z.reference_old) and np.array_equal(z.full,z.blood)
   checks.append({'cohort':c,'outcome':o,'scenario':scenario,'n':len(e),'events':int(y.sum()),'independent_metrics':True,'cohort_and_time_checks':True,'sha256':hashlib.sha256((M.NEW/f'{c}_{o}_{scenario}_predictions.pkl').read_bytes()).hexdigest()})
  # Reproduce simple-model held-out predictions directly from stored artifacts.
  if c=='CHARLS':
   d,_=M.read(c,o,'validation');est=joblib.load(M.NEW/f'{c}_{o}_all_simple.joblib');prob=M.predict(est,d,c,'all');assert np.allclose(p.set_index('person_id').loc[d.person_id,'simple'],prob,atol=1e-14)
  else:
   d,_=M.read(c,o,'validation');folds=pd.read_pickle(M.P/f'elsa/{o}_outer_folds.pkl');folds.person_id=folds.person_id.astype(str)
   for k in range(5):
    test=d.loc[d.person_id.isin(folds.loc[folds.fold.eq(k),'person_id'])];est=joblib.load(M.NEW/f'{c}_{o}_all_{k}_simple.joblib');prob=M.predict(est,test,c,'all');assert np.allclose(p.set_index('person_id').loc[test.person_id,'simple'],prob,atol=1e-14)
M.dump(M.Q/'verification.json',{'checks':checks,'saved_simple_predictions_reproduced':True,'baseline_reference_and_full_predictions_unchanged':True,'note':'All-sample comparisons retain original seed 20260913 for compatibility; new sensitivity bootstrap seed 20260922.'})
print('VERIFIED',len(checks),'cohort/scenario combinations')
