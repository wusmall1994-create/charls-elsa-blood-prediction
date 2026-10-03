import os
from pathlib import Path
import importlib.util,json,numpy as np,pandas as pd
W=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('c',W/'48_competing_models.py');C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)
report=[]
for s in C.outcomes:
 file=C.Q/f'{s}_report_first.json';d=json.loads(file.read_text());_,_,end,a=C.build(s,'validation','report_first');_,di,_,da=C.build(s,'development','report_first');_,dr,_,_=C.build(s,'development','death_first');assert di.equals(dr),'ordering sensitivity differs';assert a['same_wave_diagnosis_death']==0 and a['diagnosis_after_death_flag']==0
 for family in ['cause_specific','multinomial','composite']:
  p=pd.read_pickle(C.O/f'{s}_{family}_report_first.pkl');assert p.person_id.equals(end.person_id);assert np.isfinite(p[['reference','blood']]).all().all();assert p[['reference','blood']].ge(0).all().all() and p[['reference','blood']].le(1).all().all()
  if family=='cause_specific':
   row=C.summarize(end,{'reference':p.reference.to_numpy(),'blood':p.blood.to_numpy()},'net',s,'cause_specific_report_first');d['results']=[row if x['analysis']=='cause_specific_report_first' else x for x in d['results']]
  # Independent direct metric recomputation on saved predictions.
  kind={'cause_specific':'net','multinomial':'diagnosis','composite':'composite'}[family];y,w,_=C.weights(end,kind);ix=w>0
  from sklearn.metrics import roc_auc_score,brier_score_loss
  got=next(x for x in d['results'] if x['analysis']==family+'_report_first')
  for block in ['reference','blood']:
   assert abs(roc_auc_score(y[ix],p[block].to_numpy()[ix],sample_weight=w[ix])-got[block+'_auroc'])<1e-12
   assert abs(brier_score_loss(y[ix],p[block].to_numpy()[ix],sample_weight=w[ix])-got[block+'_brier'])<1e-12
  p['death_interval_entry']=end.death_interval_entry;p.to_pickle(C.O/f'{s}_{family}_report_first.pkl')
 file.write_text(json.dumps(d,indent=2),encoding='utf-8');report.append({'outcome':s,'verified':True,'order_invariant':True,'fits':len(d['tuning'])});print(s,flush=True)
(C.Q/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
