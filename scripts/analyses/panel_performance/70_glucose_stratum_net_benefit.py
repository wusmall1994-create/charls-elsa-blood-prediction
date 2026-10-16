"""Exploratory ELSA diabetes contrast of net-benefit increments by glucose availability.

Resample participants independently within availability strata, preserving paired
model predictions and the joint threshold contributions. Predictions are fixed;
the percentile intervals exclude model-training and calibration uncertainty.
Only aggregate results and input checksums are exported.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def contributions(y, full, simple, thresholds):
    y = np.asarray(y, dtype=float)[:, None]
    t = np.asarray(thresholds, dtype=float)[None, :]
    action_difference = (np.asarray(full)[:, None] >= t).astype(int) - (
        np.asarray(simple)[:, None] >= t).astype(int)
    return action_difference * (y - (1 - y) * t / (1 - t))


def bootstrap_means(values, repetitions, rng):
    # Multinomial counts over identical contribution vectors are exactly the
    # ordinary participant bootstrap, without materializing repeated records.
    patterns, counts = np.unique(values, axis=0, return_counts=True)
    n = len(values)
    return rng.multinomial(n, counts / n, size=repetitions) @ patterns / n


def analyze(predictions, people, repetitions=20000, seed=20261016):
    thresholds = np.array([0.05, 0.10, 0.20])
    if repetitions < 1000:
        raise ValueError('At least 1000 bootstrap repetitions are required.')
    p = predictions.copy()
    if p['person_id'].isna().any() or people['person_id'].isna().any():
        raise ValueError('Missing participant identifiers.')
    if not p['evaluable'].isin([0, 1, False, True]).all():
        raise ValueError('Invalid evaluability flag.')
    p = p.loc[p['evaluable'].astype(bool)].copy()
    s = people.loc[people['outcome'].eq('diabe'),
                   ['person_id', 'fasting_sample', 'glucose']].copy()
    p['person_id'] = p['person_id'].astype(str)
    s['person_id'] = s['person_id'].astype(str)
    d = p.merge(s, on='person_id', how='left', validate='one_to_one', indicator=True)
    if not d['_merge'].eq('both').all():
        raise ValueError('Every evaluable prediction must match a diabetes participant.')
    if not d['event'].isin([0, 1]).all():
        raise ValueError('Nonbinary or missing outcome.')
    if not np.isfinite(d[['full', 'simple']].to_numpy()).all() or not (
            d[['full', 'simple']].ge(0) & d[['full', 'simple']].le(1)).all().all():
        raise ValueError('Predictions must be finite probabilities.')
    measured = d['fasting_sample'].eq(1) & d['glucose'].notna()
    rng = np.random.default_rng(seed)
    groups = {}
    draws = {}
    for label, mask in [('measured_fasting_glucose', measured),
                        ('missing_or_unverified_glucose', ~measured)]:
        g = d.loc[mask]
        if len(g) < 2:
            raise ValueError('Both availability strata must contain participants.')
        values = contributions(g['event'], g['full'], g['simple'], thresholds)
        groups[label] = {'n': len(g), 'events': int(g['event'].sum()),
                         'estimate': values.mean(axis=0) * 100,
                         'discordant_actions': ((g['full'].to_numpy()[:, None] >= thresholds)
                             != (g['simple'].to_numpy()[:, None] >= thresholds)).sum(axis=0)}
        draws[label] = bootstrap_means(values, repetitions, rng) * 100
    a, b = groups.values()
    contrast = b['estimate'] - a['estimate']
    contrast_draws = draws['missing_or_unverified_glucose'] - draws['measured_fasting_glucose']
    rows = []
    for j, t in enumerate(thresholds):
        row = {'threshold': float(t)}
        for label, g in groups.items():
            row[label + '_n'] = g['n']
            row[label + '_events'] = g['events']
            row[label + '_discordant_actions'] = int(g['discordant_actions'][j])
            row[label + '_increment_per_100'] = float(g['estimate'][j])
        row['contrast_per_100'] = float(contrast[j])
        row['ci_low_per_100'], row['ci_high_per_100'] = np.quantile(
            contrast_draws[:, j], [0.025, 0.975]).tolist()
        rows.append(row)
    return {'cohort': 'ELSA', 'outcome': 'diabe', 'bootstrap_repetitions': repetitions,
            'seed': seed, 'interval': 'pointwise percentile 95% confidence interval',
            'units': 'net benefit per 100 participants',
            'estimand': '(full - simple) missing/unverified minus (full - simple) measured fasting glucose',
            'resampling': 'participant bootstrap within availability strata; paired models; no event stratification',
            'scope': 'exploratory; fixed predictions; no model or calibration refitting; no multiplicity adjustment',
            'rows': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, required=True,
                        help='Trusted private ELSA_diabe_all_predictions.pkl')
    parser.add_argument('--people', type=Path, required=True,
                        help='Trusted private elsa_w6_people.pkl')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--n-bootstrap', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=20261016)
    args = parser.parse_args()
    result = analyze(pd.read_pickle(args.predictions), pd.read_pickle(args.people),
                     args.n_bootstrap, args.seed)
    result['input_sha256'] = {key: hashlib.sha256(path.read_bytes()).hexdigest()
                              for key, path in [('predictions', args.predictions), ('people', args.people)]}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'glucose_stratum_net_benefit.json').write_text(
        json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    pd.DataFrame(result['rows']).to_csv(args.output_dir / 'glucose_stratum_net_benefit.csv', index=False)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
