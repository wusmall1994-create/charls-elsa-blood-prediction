import os
from pathlib import Path
import json,importlib.util,hashlib
import numpy as np,pandas as pd,joblib
W=Path(__file__).resolve().parent;sp=importlib.util.spec_from_file_location('priority',W/'62_priority_analyses.py');M=importlib.util.module_from_spec(sp);sp.loader.exec_module(M)
def native(x):
 if isinstance(x,np.ndarray):return [native(v) for v in x]
 if isinstance(x,np.generic):return native(x.item())
 if isinstance(x,float) and not np.isfinite(x):return None
 if isinstance(x,dict):return {k:native(v) for k,v in x.items()}
 if isinstance(x,(tuple,list)):return [native(v) for v in x]
 return x
summ=[];audit=[];tune=[];stable=[];params=[]
for c in ['CHARLS','ELSA']:
 for o in ['diabe','hchole']:
  d=json.loads((M.Q/f'{c}_{o}.json').read_text());st=json.loads((M.Q/f'{c}_{o}_stability.json').read_text())
  tune+=d['tuning']+st['tuning']
  for r in d['results']:
   for comp,v in r['deltas'].items():
    for metric,z in v.items():summ.append(dict(cohort=c,outcome=o,scenario=r['scenario'],n=r['n'],events=r['events'],comparison=comp,metric=metric,**z))
  audit += [dict(cohort=c,outcome=o,**r) for r in d['audit']]
  for comp in ['simple_reference','full_reference','full_simple']:
   for metric in ['auroc','brier','ap']:
    vals=[r['deltas'][comp][metric]['estimate'] for r in st['results']]
    stable.append(dict(cohort=c,outcome=o,comparison=comp,metric=metric,repeats=len(vals),median=float(np.median(vals)),minimum=min(vals),maximum=max(vals),q025=float(np.quantile(vals,.025)),q975=float(np.quantile(vals,.975))))
  for fold in ([None] if c=='CHARLS' else range(5)):
   for model in M.MODELS:
    if model=='simple':f=M.NEW/(f'{c}_{o}_all_simple.joblib' if c=='CHARLS' else f'{c}_{o}_all_{fold}_simple.joblib')
    elif c=='CHARLS':f=M.OLD/f'CHARLS_{o}_rich_{"reference" if model=="reference" else "panel7"}.joblib'
    else:f=M.P/f'elsa/{o}_rich_{"reference" if model=="reference" else "blood"}_fold{fold}.joblib'
    est=joblib.load(f);pre=est.named_steps['preprocess'];cont=pre.named_transformers_['continuous'];cat=pre.named_transformers_['categorical']
    params.append(native(dict(cohort=c,outcome=o,fold=fold,model=model,source_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),C=est.named_steps['model'].C,intercept=est.named_steps['model'].intercept_,coefficients=est.named_steps['model'].coef_,transformed_feature_names=pre.get_feature_names_out(),continuous_columns=pre.transformers_[0][2],continuous_medians=cont.named_steps['impute'].statistics_,continuous_missing_indicators=cont.named_steps['impute'].indicator_.features_,scale_mean=cont.named_steps['scale'].mean_,scale_sd=cont.named_steps['scale'].scale_,categorical_columns=pre.transformers_[1][2],categorical_modes=cat.named_steps['impute'].statistics_,categorical_missing_indicators=cat.named_steps['impute'].indicator_.features_,categories=cat.named_steps['encode'].categories_,drop_indices=cat.named_steps['encode'].drop_idx_)))
pd.DataFrame(summ).to_csv(M.Q/'paired_results.csv',index=False);pd.DataFrame(stable).to_csv(M.Q/'refitting_stability.csv',index=False);pd.DataFrame(tune).drop(columns='signature').to_csv(M.Q/'tuning.csv',index=False)
M.dump(M.Q/'cohort_audit.json',audit);M.dump(M.Q/'baseline_model_parameters.json',params)
manifest={}
for c in ['CHARLS','ELSA']:
 for o in ['diabe','hchole']:
  if c=='CHARLS':
   for phase in ['development','validation']:
    for suffix in ['', '_intervals']:
     f=M.P/'multoutcome'/M.X.MAP[o]/(phase+suffix+'.csv.gz');manifest[str(f.relative_to(M.P))]=hashlib.sha256(f.read_bytes()).hexdigest()
  else:
   for k in ['people','intervals']:
    f=M.OLD/f'elsa_w6_{k}.pkl';manifest[str(f.relative_to(M.P))]=hashlib.sha256(f.read_bytes()).hexdigest()
M.dump(M.Q/'input_hashes.json',manifest)
(M.Q/'README.md').write_text('''# Baseline status and parsimonious panel extension

Recorded 22 September 2026. These aggregate files supplement the prior study; participant records, identifiers, fold membership and binary fitted models are not redistributed.

Run 61_feasibility.py to audit available samples. The dated analysis contract records the subsequent fitting rules. Run 62_priority_analyses.py CHARLS diabe (and the other three cohort/outcome combinations), then repeat with --stability. Run 63_priority_verify.py after all four main jobs, followed by 64_priority_exports.py. The verification script restores the original seed 20260913 for unchanged all-sample bootstrap comparisons; new sensitivity bootstraps use 20260922. This compatibility adjustment changes no predictions, samples or fitted models.

The scripts import project scripts 42_replication_sensitivity_suite.py, 39_train_elsa_nested_models.py and 38_build_elsa_checked_cohorts.py. Copies are included as dependencies. Set R/ROOT/P paths to the authorized local project as needed; original private checked cohorts and interval files are required. Upstream construction code is in the existing archive. The source-data hashes identify exact inputs. No original analysis cache is overwritten.

paired_results.csv contains all three pairwise comparisons, including unfavorable results. Baseline reference/full predictions are unchanged. cohort_audit.json includes population counts and measured distributions. tuning.csv records selected penalties and features without participant IDs. baseline_model_parameters.json contains the 36 original-population reference/simple/full pipelines as coefficient and preprocessing metadata, not participant-level predictions. These cover the two positive-reference outcomes and richer reference only, not all seven outcome models. Fixed nominal schedules and categorical handling are in the scripts. Current package versions are recorded in each job JSON.

refitting_stability.csv distinguishes five fully retuned ELSA nested splits from 50 CHARLS development participant bootstraps with fixed selected C. The ranges are descriptive refit spread, not sampling confidence intervals. Restricted and landmark results retain conditional bootstrap intervals; there is no claim that every sensitivity was subjected to full refitting uncertainty.

Threshold restrictions classify measured biochemical status, not confirmed absence of disease or treatment. Landmark analysis resets time, requires an observed negative first follow-up and uses the next observed scheduled wave only. CHARLS and ELSA retain different baseline-to-landmark gaps and different evaluation architectures.
''',encoding='utf-8')
print('Exported',len(summ),'metric comparisons;',len(params),'parameter sets;',len(tune),'new fits')
