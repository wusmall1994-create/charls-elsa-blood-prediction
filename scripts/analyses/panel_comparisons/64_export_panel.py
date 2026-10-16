import os
from pathlib import Path
import json,importlib.util,hashlib
import numpy as np,pandas as pd,joblib
W=Path(os.environ["ANALYSIS_ROOT"])/"analysis_work";W.mkdir(parents=True,exist_ok=True);sp=importlib.util.spec_from_file_location('panel',Path(__file__).resolve().parents[3] / 'scripts/analyses/panel_comparisons/62_panel_analyses.py');M=importlib.util.module_from_spec(sp);sp.loader.exec_module(M)
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
(M.Q/'README.md').write_text("""# Simplified panels, biochemical restrictions and landmark prediction

Run 61_feasibility.py to inspect eligible samples. Run 62_panel_analyses.py for CHARLS and ELSA with diabe and hchole, then repeat each job with --stability. Run 63_verify_panel.py and 64_export_panel.py after all four jobs complete.

These scripts use the shared core modules and private cohort files under ANALYSIS_ROOT. Full-sample bootstrap seed is 20260913; restriction and landmark seed is 20260922. Source hashes identify inputs. Statistical specifications and software versions are recorded with the results.

paired_results.csv contains three pairwise model comparisons. cohort_audit.json describes populations and measured distributions. tuning.csv records penalties and features. baseline_model_parameters.json contains preprocessing and coefficients for 36 full-population pipelines across two outcomes; it contains no individual predictions.

refitting_stability.csv summarizes five nested ELSA splits and 50 CHARLS development bootstraps with fixed selected penalties. These ranges describe refit variation, not confidence intervals. Restrictions classify measured biochemical status and do not establish absence of disease or treatment. Landmark prediction requires a negative first follow-up and evaluates the next scheduled wave.
""",encoding='utf-8')
print('Exported',len(summ),'metric comparisons;',len(params),'parameter sets;',len(tune),'model fits')
