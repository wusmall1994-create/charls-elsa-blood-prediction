"""Comparator adequacy, common-core frozen transport and target recalibration.

Private cohort files and fitted objects remain in the authorized data directory.
Only aggregate metrics, tuning metadata and file checksums are exported.
"""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
import argparse
import hashlib
import importlib.util
import json
import warnings
import platform
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.special import logit, expit
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

SEED = 20261017
MODELS = ['reference', 'simple', 'full', 'glycemic', 'full_hba1c']
THRESHOLDS = np.array([.05, .10, .20])


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def initialize(args):
    global X, T, P, COMMON, SAVED, PRIVATE, OUT
    P, COMMON, SAVED = args.data_private, args.common_panel_dir, args.panel_predictions_dir
    PRIVATE, OUT = args.output_private, args.output_dir
    PRIVATE.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('ANALYSIS_ROOT', str(P.parent))
    os.environ.setdefault('ELSA_DATA_DIR', str(P))  # No raw-release reads in this module.
    path = Path(__file__).resolve().parents[3] / 'scripts/core/42_replication_sensitivity_suite.py'
    spec = importlib.util.spec_from_file_location('common_panel_suite', path)
    X = importlib.util.module_from_spec(spec); spec.loader.exec_module(X); T = X.T
    inputs = {}
    for c in ['CHARLS', 'ELSA']:
        path = SAVED / f'{c}_diabe_all_predictions.pkl'
        inputs['panel_predictions/' + path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    for wave in [4, 6]:
        for kind in ['people', 'intervals']:
            path = COMMON / f'elsa_w{wave}_{kind}.pkl'
            inputs['common_panel/' + path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    for outcome in ['diabetes', 'dyslipidemia']:
        for phase in ['development', 'validation']:
            for suffix in ['.csv.gz', '_intervals.csv.gz']:
                path = P / 'multoutcome' / outcome / (phase + suffix)
                inputs[f'multoutcome/{outcome}/{path.name}'] = hashlib.sha256(path.read_bytes()).hexdigest()
    path = P / 'elsa/diabe_outer_folds.pkl'
    inputs['elsa/diabe_outer_folds.pkl'] = hashlib.sha256(path.read_bytes()).hexdigest()
    save(OUT / 'input_checksums.json', inputs)


def cohort(c, o, phase):
    if c == 'CHARLS':
        folder = P / 'multoutcome' / X.MAP[o]
        d = pd.read_csv(folder / (phase + '.csv.gz'), dtype={'person_id': str}).rename(columns=X.RENAME)
        it = pd.read_csv(folder / (phase + '_intervals.csv.gz'), dtype={'person_id': str}).rename(
            columns={'interval_start_years': 'start', 'interval_duration_years': 'duration',
                     'event_this_interval': 'label', 'source_wave': 'wave'})
        d['evaluable'] = d.event.eq(1) | d.person_id.isin(it.loc[it.wave.eq(5), 'person_id'])
    else:
        wave = 4 if phase == 'development' else 6
        d = pd.read_pickle(COMMON / f'elsa_w{wave}_people.pkl')
        it = pd.read_pickle(COMMON / f'elsa_w{wave}_intervals.pkl')
        d = d.loc[d.outcome.eq(o)].copy(); it = it.loc[it.outcome.eq(o)].copy()
    d.person_id = d.person_id.astype(str); it.person_id = it.person_id.astype(str)
    assert d.person_id.is_unique
    return d.reset_index(drop=True), it[['person_id', 'start', 'duration', 'label', 'wave']].reset_index(drop=True)


def features(c, o, model, block):
    cont, cat = X.features(c, block, o, 'reference', 'panel7')
    cont, cat = list(cont), list(cat)
    if model != 'reference':
        labs = (['glucose'] if o == 'diabe' else ['total_cholesterol', 'hdl_cholesterol', 'triglycerides']) if model == 'simple' else list(X.B.BLOOD)
        if model == 'glycemic': labs = ['glucose', 'hba1c']
        if model == 'full_hba1c': labs += ['hba1c']
        cont += labs; cat += ['fasting_sample']
    return cont, cat


def fit(d, it, c, o, model, block, key, seed):
    cont, cat = features(c, o, model, block)
    fields = [x for x in cont + cat if x not in T.TIME] + ['person_id', 'event']
    signature = hashlib.sha256(pd.util.hash_pandas_object(d[fields], index=False).values.tobytes()
                               + pd.util.hash_pandas_object(it, index=False).values.tobytes()
                               + repr((cont, cat, seed)).encode()).hexdigest()
    path = PRIVATE / (key + '.joblib'); meta = PRIVATE / (key + '.json')
    if path.exists() and meta.exists():
        info = json.loads(meta.read_text()); assert info['signature'] == signature
        return joblib.load(path), info
    rows = T.terms(it.merge(d.drop(columns=['outcome'], errors='ignore'), on='person_id', validate='many_to_one'))
    splits = []
    for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(d, d.event):
        splits.append((np.flatnonzero(rows.person_id.isin(d.iloc[a].person_id)),
                       np.flatnonzero(rows.person_id.isin(d.iloc[b].person_id))))
    search = GridSearchCV(T.pipeline(cont, cat), {'model__C': [.01, .1, 1, 10, 100]},
                          scoring='neg_log_loss', cv=splits, n_jobs=1, error_score='raise')
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
        warnings.simplefilter('always'); search.fit(rows[cont + cat], rows.label.astype(int))
    assert not any(issubclass(w.category, ConvergenceWarning) for w in caught), key
    est = search.best_estimator_
    info = dict(key=key, signature=signature, model=model, block=block, cohort=c, outcome=o,
                n=len(d), events=int(d.event.sum()), intervals=len(rows), seed=seed,
                C=float(search.best_params_['model__C']), features=cont + cat, nonconvergence=False)
    joblib.dump(est, path, compress=3); save(meta, info)
    print('fit', key, 'C', info['C'], flush=True)
    return est, info


def predict(est, d, c):
    schedule = [(0., 3.), (3., 2.)] if c == 'CHARLS' else [(0., 2.), (2., 2.), (4., 2.)]
    grid = T.terms(pd.concat([d.assign(start=a, duration=b) for a, b in schedule], ignore_index=True))
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Found unknown categories')
        h = est.predict_proba(grid[est.feature_names_in_])[:, 1]
    risk = 1 - pd.DataFrame({'id': grid.person_id, 's': 1 - h}).groupby('id').s.prod()
    return d.person_id.map(risk).to_numpy()


def hba1c():
    tuning, audit = [], []
    for c in ['CHARLS', 'ELSA']:
        d, it = cohort(c, 'diabe', 'validation')
        pred = pd.read_pickle(SAVED / f'{c}_diabe_all_predictions.pkl')
        pred.person_id = pred.person_id.astype(str)
        assert set(pred.person_id) == set(d.person_id)
        observed = d.fasting_sample.eq(1) & d.glucose.notna() & d.hba1c.notna()
        pred['both_measured'] = pred.person_id.isin(d.loc[observed, 'person_id'])
        audit.append(dict(cohort=c, n=len(d), evaluable=int(d.evaluable.sum()),
                          glucose_observed=int(d.glucose.notna().sum()), hba1c_observed=int(d.hba1c.notna().sum()),
                          both_measured_evaluable=int((observed & d.evaluable).sum())))
        if c == 'CHARLS':
            train, ti = cohort(c, 'diabe', 'development'); assert not set(train.person_id) & set(d.person_id)
            splits = [(train, ti, d, 20260913, 'temporal')]
        else:
            folds = pd.read_pickle(P / 'elsa/diabe_outer_folds.pkl'); folds.person_id = folds.person_id.astype(str)
            assignment = d.person_id.map(folds.set_index('person_id').fold); assert assignment.notna().all()
            splits = []
            for k in range(5):
                train = d.loc[assignment.ne(k)].reset_index(drop=True); test = d.loc[assignment.eq(k)].reset_index(drop=True)
                splits.append((train, it.loc[it.person_id.isin(train.person_id)], test, T.SEED + k + 1, str(k)))
        for train, ti, test, seed, label in splits:
            assert not set(train.person_id) & set(test.person_id)
            for model in ['glycemic', 'full_hba1c']:
                est, info = fit(train, ti, c, 'diabe', model, 'rich', f'{c}_rich_diabe_{label}_{model}', seed)
                values = pd.Series(predict(est, test, c), index=test.person_id)
                ids = pred.person_id.isin(test.person_id); pred.loc[ids, model] = pred.loc[ids, 'person_id'].map(values)
                tuning.append(info)
        assert np.isfinite(pred[MODELS]).all().all()
        pred.to_pickle(PRIVATE / f'hba1c_{c}.pkl')
    save(OUT / 'hba1c_tuning.json', tuning); save(OUT / 'hba1c_availability.json', audit)


def frozen():
    tuning, audit, hashes = [], [], {}
    for o in ['diabe', 'hchole']:
        for source in ['CHARLS', 'ELSA']:
            train, ti = cohort(source, o, 'development')
            models = MODELS if o == 'diabe' else MODELS[:3]
            fitted = {}
            for model in models:
                key = f'{source}_core_{o}_{model}'
                fitted[model], info = fit(train, ti, source, o, model, 'core', key, 20260913)
                tuning.append(info); hashes[key] = hashlib.sha256((PRIVATE / (key + '.joblib')).read_bytes()).hexdigest()
            for target in ['CHARLS', 'ELSA']:
                d, _ = cohort(target, o, 'validation')
                if source == target:
                    # Temporal evaluation excludes every participant in source development.
                    d = d.loc[~d.person_id.isin(train.person_id)].reset_index(drop=True)
                    assert not set(train.person_id) & set(d.person_id)
                elif source == 'CHARLS':
                    d['fasting_sample'] = d.fasting_sample.map({1: 1., 0: 2.})
                else:
                    d['fasting_sample'] = d.fasting_sample.map({1: 1., 2: 0.})
                pred = d[['person_id', 'event', 'evaluable']].copy()
                for model, est in fitted.items(): pred[model] = predict(est, d, target)
                assert np.isfinite(pred[models]).all().all()
                key = f'{source}_to_{target}_{o}'
                pred.to_pickle(PRIVATE / (key + '.pkl'))
                audit.append(dict(setting=key, n=len(pred), evaluable=int(pred.evaluable.sum()),
                                  events=int(pred.loc[pred.evaluable, 'event'].sum()), models=models,
                                  horizon_years=5 if target == 'CHARLS' else 6))
            for model in models:
                key = f'{source}_core_{o}_{model}'
                assert hashlib.sha256((PRIVATE / (key + '.joblib')).read_bytes()).hexdigest() == hashes[key]
    save(OUT / 'frozen_tuning.json', tuning); save(OUT / 'frozen_audit.json', audit)
    save(OUT / 'frozen_model_hashes.json', hashes)


def calibration(y, p):
    z = logit(np.clip(p, 1e-8, 1 - 1e-8))[:, None]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fit = LogisticRegression(penalty=None, max_iter=3000, tol=1e-10).fit(z, y)
    assert not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    return float(fit.intercept_[0]), float(fit.coef_[0, 0])


def recalibrate(d, models):
    y = d.event.to_numpy(int); result = {'unchanged': d[models].to_numpy(),
                                        'intercept': np.zeros((len(d), len(models))),
                                        'intercept_slope': np.zeros((len(d), len(models)))}
    fold_id = np.full(len(d), -1); params = []
    for k, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(d, y)):
        fold_id[b] = k
        for j, model in enumerate(models):
            z = logit(np.clip(d[model].to_numpy(), 1e-8, 1 - 1e-8))
            f = minimize_scalar(lambda alpha: np.mean(np.logaddexp(0, z[a] + alpha) - y[a] * (z[a] + alpha)),
                                bounds=(-20, 20), method='bounded'); assert f.success
            alpha, beta = calibration(y[a], d[model].to_numpy()[a])
            result['intercept'][b, j] = expit(z[b] + f.x)
            result['intercept_slope'][b, j] = expit(alpha + beta * z[b])
            params.append(dict(fold=k, model=model, train_n=len(a), test_n=len(b),
                               intercept_only=float(f.x), intercept=alpha, slope=beta))
    assert (fold_id >= 0).all()
    return result, fold_id, params


def evaluate_one(d, models, label, updating, repetitions):
    d = d.loc[d.evaluable].reset_index(drop=True); y = d.event.to_numpy(int)
    if updating:
        predictions, fold_id, params = recalibrate(d, models)
        save(OUT / (label + '_calibration_parameters.json'), params)
    else: predictions = {'unchanged': d[models].to_numpy()}; fold_id = np.full(len(d), -1)
    assert d.person_id.is_unique and min(np.bincount(y)) > 0
    pairs = [('simple', 'reference'), ('full', 'reference'), ('full', 'simple')]
    if 'glycemic' in models: pairs += [('glycemic', 'simple'), ('full', 'glycemic'), ('full_hba1c', 'glycemic')]
    rows, contrasts = [], []
    private = d[['person_id', 'event']].copy(); private['calibration_fold'] = fold_id
    for method, matrix in predictions.items():
        fs = [X.fast_metric(y, matrix[:, j]) for j in range(len(models))]
        points = np.array([f(np.ones(len(y))) for f in fs])
        terms = ((matrix[:, :, None] >= THRESHOLDS) * (y[:, None, None] -
                 (1 - y[:, None, None]) * THRESHOLDS / (1 - THRESHOLDS))) * 100
        nb = terms.mean(0); rng = np.random.default_rng(SEED)
        draws = np.zeros((repetitions, len(models), 6))
        for k in range(repetitions):
            weights = rng.multinomial(len(y), np.full(len(y), 1 / len(y))).astype(float)
            if np.dot(weights, y) == 0 or np.dot(weights, 1 - y) == 0: raise ValueError('Bootstrap lacks an outcome class')
            draws[k, :, :3] = [f(weights) for f in fs]
            draws[k, :, 3:] = np.einsum('i,ijk->jk', weights, terms) / len(y)
        allpoints = np.c_[points, nb]
        metric_names = ['auroc', 'brier', 'average_precision', 'nb_05_per100', 'nb_10_per100', 'nb_20_per100']
        for j, model in enumerate(models):
            alpha, beta = calibration(y, matrix[:, j])
            row = dict(setting=label, method=method, model=model, n=len(y), events=int(y.sum()),
                       mean_risk=float(matrix[:, j].mean()), calibration_intercept=alpha, calibration_slope=beta)
            row.update(dict(zip(metric_names, map(float, allpoints[j])))); rows.append(row)
            private[method + '_' + model] = matrix[:, j]
        for a, b in pairs:
            ia, ib = models.index(a), models.index(b)
            low, high = np.quantile(draws[:, ia] - draws[:, ib], [.025, .975], axis=0)
            for j, metric in enumerate(metric_names):
                contrasts.append(dict(setting=label, method=method, contrast=a + '_minus_' + b, metric=metric,
                                      estimate=float(allpoints[ia, j] - allpoints[ib, j]), low=float(low[j]), high=float(high[j])))
        # Default policies are calculated in the identical target evaluation sample.
        for j, threshold in enumerate(THRESHOLDS):
            default = np.maximum(0, (y.mean() - (1 - y.mean()) * threshold / (1 - threshold)) * 100)
            for row in rows:
                if row['method'] == method:
                    row[f'best_default_{int(threshold*100):02d}_per100'] = float(default)
    private.to_pickle(PRIVATE / (label + '_evaluated.pkl'))
    return rows, contrasts


def evaluate(repetitions):
    allrows, allcontrasts = [], []
    tasks = []
    for c in ['CHARLS', 'ELSA']:
        d = pd.read_pickle(PRIVATE / f'hba1c_{c}.pkl')
        tasks += [(d, MODELS, f'hba1c_{c}_all', False),
                  (d.loc[d.both_measured], MODELS, f'hba1c_{c}_measured', False)]
    for info in json.loads((OUT / 'frozen_audit.json').read_text()):
        tasks.append((pd.read_pickle(PRIVATE / (info['setting'] + '.pkl')), info['models'], info['setting'], True))
    for d, models, label, updating in tasks:
        cache = OUT / (label + '_metrics.json')
        signature = hashlib.sha256(pd.util.hash_pandas_object(d[['person_id', 'event', 'evaluable'] + models],
                                    index=False).values.tobytes() + repr((SEED, models, updating)).encode()).hexdigest()
        if cache.exists():
            data = json.loads(cache.read_text())
            valid = data.get('input_signature') == signature and data['bootstrap_repetitions'] == repetitions
        else:
            valid = False
        if not valid:
            rows, contrasts = evaluate_one(d, models, label, updating, repetitions)
            data = dict(input_signature=signature, bootstrap_repetitions=repetitions, rows=rows, contrasts=contrasts); save(cache, data)
        allrows += data['rows']; allcontrasts += data['contrasts']
        print('evaluated', label, flush=True)
    pd.DataFrame(allrows).to_csv(OUT / 'performance.csv', index=False)
    pd.DataFrame(allcontrasts).to_csv(OUT / 'contrasts.csv', index=False)
    save(OUT / 'analysis_specification.json', dict(seed=SEED, bootstrap_repetitions=repetitions,
         software=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                       scipy=scipy.__version__, sklearn=sklearn.__version__, joblib=joblib.__version__),
         bootstrap='paired unstratified participant percentile intervals conditional on fitted and cross-fitted predictions; no refitting',
         recalibration='five identical target folds across models; intercept-only and intercept-plus-slope fitted on other folds',
         comparisons='all exploratory; no multiplicity adjustment; 10% focus with 5% and 20% supporting',
         within_cohort='HbA1c comparisons use richer references and original temporal/nested splits; measured subgroup uses unchanged predictions',
         frozen='common-core predictors, source-only fitting, target-specific nominal horizon; same-cohort temporal targets exclude all source participants',
         limitations='target outcomes support recalibration; performance is not external validation of an updated model; diagnosis reports and incomplete observation remain'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['data-private', 'common-panel-dir', 'panel-predictions-dir', 'output-private', 'output-dir']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--task', choices=['hba1c', 'frozen', 'evaluate', 'all'], default='all')
    parser.add_argument('--bootstrap', type=int, default=2000)
    args = parser.parse_args(); initialize(args)
    if args.task in ['hba1c', 'all']: hba1c()
    if args.task in ['frozen', 'all']: frozen()
    if args.task in ['evaluate', 'all']: evaluate(args.bootstrap)
