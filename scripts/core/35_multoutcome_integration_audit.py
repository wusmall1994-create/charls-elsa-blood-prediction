import os
"""Audit frozen predictions and evaluate a common development-union exclusion.

No models are selected, updated or refitted here. All released files are aggregate.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss

ROOT = Path(os.environ["ANALYSIS_ROOT"])
QA = ROOT / 'qa_logs'
DATA = ROOT / 'data_private'

def main():
    predictions = pd.read_csv(DATA / 'multoutcome_primary_predictions.csv.gz', dtype={'person_id': str})
    results = pd.read_csv(QA / 'multoutcome_incremental_value_summary.csv')
    outcomes = results.outcome.tolist()
    development = {o: pd.read_csv(DATA / 'multoutcome' / o / 'development.csv.gz', dtype={'person_id': str}) for o in outcomes}
    union = set().union(*(set(d.person_id) for d in development.values()))
    rng = np.random.default_rng(20260910)
    sensitivity, characteristics, missing, audit = [], [], [], []
    for outcome in outcomes:
        d = development[outcome]
        v = pd.read_csv(DATA / 'multoutcome' / outcome / 'validation.csv.gz', dtype={'person_id': str})
        assert d.person_id.is_unique and v.person_id.is_unique
        assert not set(d.person_id) & set(v.person_id)
        p = predictions.loc[predictions.outcome.eq(outcome)]
        assert not p.duplicated(['person_id', 'model']).any()
        b = p.loc[p.model.eq('base_discrete_time')].set_index('person_id')
        e = p.loc[p.model.eq('blood_enhanced_discrete_time')].set_index('person_id').loc[b.index]
        assert b.index.equals(e.index)
        assert b[['event', 'evaluable_5y']].equals(e[['event', 'evaluable_5y']])
        selected = b.evaluable_5y.astype(bool)
        delta = roc_auc_score(b.loc[selected, 'event'], e.loc[selected, 'predicted_risk_5y']) - roc_auc_score(b.loc[selected, 'event'], b.loc[selected, 'predicted_risk_5y'])
        expected = results.loc[results.outcome.eq(outcome)].iloc[0]
        assert abs(delta - expected.delta_auc) < 1e-12
        audit.append({'outcome': outcome, 'within_outcome_overlap': 0, 'primary_evaluable': int(selected.sum()), 'recomputed_delta_auc': delta})
        restricted = selected & ~b.index.isin(union)
        y = b.loc[restricted, 'event'].to_numpy(dtype=int)
        pb = b.loc[restricted, 'predicted_risk_5y'].to_numpy()
        pe = e.loc[restricted, 'predicted_risk_5y'].to_numpy()
        pos, neg = np.flatnonzero(y), np.flatnonzero(y == 0)
        samples = []
        for _ in range(2000):
            idx = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
            samples.append(roc_auc_score(y[idx], pe[idx]) - roc_auc_score(y[idx], pb[idx]))
        low, high = np.quantile(samples, [.025, .975])
        sensitivity.append({'outcome': outcome, 'n': len(y), 'events': int(y.sum()), 'excluded_from_primary_evaluable': int(selected.sum())-len(y), 'base_auc': roc_auc_score(y,pb), 'blood_auc': roc_auc_score(y,pe), 'delta_auc': roc_auc_score(y,pe)-roc_auc_score(y,pb), 'delta_auc_ci_2_5': low, 'delta_auc_ci_97_5': high, 'delta_brier': brier_score_loss(y,pe)-brier_score_loss(y,pb)})
        for label, frame in [('development', d), ('evaluation', v)]:
            characteristics.append({'outcome': outcome, 'cohort': label, 'n': len(frame), 'age_median': frame.age.median(), 'age_q1': frame.age.quantile(.25), 'age_q3': frame.age.quantile(.75), 'sex_codes': json.dumps(frame.sex.value_counts(dropna=False).to_dict()), 'bmi_median': frame.bmi.median(), 'bmi_q1': frame.bmi.quantile(.25), 'bmi_q3': frame.bmi.quantile(.75)})
            for col in frame.columns:
                if col not in ['person_id','outcome','cohort','event','event_time_years','follow_up_years']:
                    missing.append({'outcome': outcome,'cohort': label,'predictor': col,'missing_n': int(frame[col].isna().sum()),'missing_pct': frame[col].isna().mean()*100})
        print(outcome, len(y), int(y.sum()), sensitivity[-1]['delta_auc'], flush=True)
    pd.DataFrame(sensitivity).to_csv(QA / 'multoutcome_common_union_sensitivity.csv', index=False)
    pd.DataFrame(characteristics).to_csv(QA / 'multoutcome_baseline_descriptives.csv', index=False)
    pd.DataFrame(missing).to_csv(QA / 'multoutcome_predictor_missingness.csv', index=False)
    pd.DataFrame(audit).to_csv(QA / 'multoutcome_prediction_integrity_audit.csv', index=False)
    (QA / 'multoutcome_integration_audit.json').write_text(json.dumps({'modeled_outcomes': len(outcomes), 'unique_development_union':len(union), 'all_within_outcome_overlap_checks_passed': True, 'all_primary_auc_recomputed':True, 'union_bootstrap_iterations':2000, 'seed':20260910}, indent=2), encoding='utf-8')

if __name__ == '__main__':
    main()
