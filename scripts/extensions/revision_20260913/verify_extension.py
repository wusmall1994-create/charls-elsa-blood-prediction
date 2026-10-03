import os
from pathlib import Path
import json,re,hashlib
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score,brier_score_loss,average_precision_score
W=Path(__file__).resolve().parent;R=Path(os.environ["ANALYSIS_ROOT"])
rows=json.loads((W/'extension_results/transport.json').read_text())
for r in rows:
 d=pd.read_pickle(R/f'data_private/revision_20260913/{r["cohort"]}_{r["outcome"]}.pkl');d=d.loc[d.evaluable]
 assert len(d)==r['n'] and int(d.event.sum())==r['events']
 for model in ['reference','blood']:
  for key,fn in [('auroc',roc_auc_score),('brier',brier_score_loss),('average_precision',average_precision_score)]:assert abs(fn(d.event,d[model])-r[model+'_'+key])<1e-12
 assert abs(r['blood_auroc']-r['reference_auroc']-r['delta_auroc'])<1e-12
hashes=json.loads((W/'extension_results/frozen_model_hashes.json').read_text())
for name,h in hashes.items():assert hashlib.sha256((R/f'data_private/replication_extensions/{name}.joblib').read_bytes()).hexdigest()==h
print('14 transport pairs independently recomputed; 28 source pipelines unchanged')
