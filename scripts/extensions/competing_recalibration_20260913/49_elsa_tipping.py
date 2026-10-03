import os
from pathlib import Path
import pandas as pd,numpy as np,json
from sklearn.metrics import roc_auc_score
W=Path(__file__).resolve().parent;P=Path(os.environ["ANALYSIS_ROOT"]);E=next(Path(os.environ["ELSA_DATA_DIR"]).rglob('stata13_se'));rows=[];details=[]
eol=pd.concat([pd.read_stata(E/f,columns=['idauniq','eidatey'],convert_categoricals=False) for f in ['elsa_endoflife_hcap2_w10.dta','wave_11_elsa_eol_eul.dta']]);known=set(eol.loc[eol.eidatey.between(2013,2018),'idauniq'])
for s in ['stroke','lunge']:
 d=pd.read_pickle(P/f'data_private/replication_extensions/ELSA_{s}_rich_panel7_predictions.pkl').reset_index(drop=True);y=d.event.to_numpy().copy();missing=~d.evaluable.to_numpy();forced=missing&d.person_id.isin(known).to_numpy();y[forced]=1;pool=np.flatnonzero(missing&~forced);score=(d.blood-d.reference).to_numpy();orders={'higher_blood_increment':pool[np.argsort(-score[pool])],'lower_blood_increment':pool[np.argsort(score[pool])],'random':np.random.default_rng(20260915).permutation(pool)}
 for name,order in orders.items():
  first=None
  for fraction in np.arange(0,1.00001,.01):
   yy=y.copy();n=round(fraction*len(pool));yy[order[:n]]=1;delta=roc_auc_score(yy,d.blood)-roc_auc_score(yy,d.reference)
   if first is None and delta>=0:first=float(fraction)
   if any(abs(fraction-v)<1e-8 for v in [0,.1,.25,.5,1]):rows.append(dict(outcome=s,allocation=name,fraction=float(fraction),n=len(d),original_evaluable=int(d.evaluable.sum()),unknown_n=len(pool),known_death_forced=int(forced.sum()),assigned_events=n,delta_auroc=delta))
  details.append(dict(outcome=s,allocation=name,first_nonnegative_fraction=first))
(W/'competing_results/elsa_tipping.json').write_text(json.dumps({'scenarios':rows,'thresholds':details,'definition':'Composite-outcome assignment stress test on saved disease predictions; unknown unassigned outcomes are assumed negative; observed terminal endpoints are fixed. Not sharp bounds or corrected disease cumulative incidence. EOL deaths through2018 force only originally unevaluable outcomes positive; 2019 death timing unresolved.'},indent=2),encoding='utf-8')
print(json.dumps(details))
