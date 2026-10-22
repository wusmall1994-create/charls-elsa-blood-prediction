"""Export aggregate diabetes decision curves from trusted private evaluated predictions."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluated-dir', type=Path, required=True,
                        help='Private directory containing both *_diabe_evaluated.pkl files')
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='Destination for aggregate curves and source checksums')
    args = parser.parse_args()
    rows, checksums = [], {}
    for source, target in [('CHARLS', 'ELSA'), ('ELSA', 'CHARLS')]:
        path = args.evaluated_dir / f'{source}_to_{target}_diabe_evaluated.pkl'
        checksums[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        data = pd.read_pickle(path)  # Only load trusted private analysis outputs.
        outcome = data.event.to_numpy(int)
        if not len(outcome) or not set(outcome).issubset({0, 1}):
            raise ValueError('Nonempty binary outcomes are required')
        for stage in ['unchanged', 'intercept_slope']:
            for threshold in np.arange(.05, .2001, .005):
                common = dict(direction=source + ' to ' + target, stage=stage,
                              threshold=round(threshold, 3), n=len(outcome),
                              events=int(outcome.sum()))
                for model in ['glycemic', 'full', 'full_hba1c']:
                    risk = data[stage + '_' + model].to_numpy()
                    if not np.isfinite(risk).all() or ((risk < 0) | (risk > 1)).any():
                        raise ValueError('Predictions must be finite probabilities')
                    benefit = np.mean((risk >= threshold) *
                                     (outcome - (1 - outcome) * threshold / (1 - threshold))) * 100
                    rows.append(dict(**common, model=model, net_benefit_per100=benefit))
                all_benefit = (outcome.mean() - (1 - outcome.mean()) * threshold / (1 - threshold)) * 100
                rows.append(dict(**common, model='assess_all', net_benefit_per100=all_benefit))
                rows.append(dict(**common, model='assess_none', net_benefit_per100=0.))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    columns = ['direction', 'stage', 'model', 'threshold', 'net_benefit_per100', 'n', 'events']
    pd.DataFrame(rows)[columns].to_csv(args.output_dir / 'decision_curve_source.csv', index=False)
    (args.output_dir / 'decision_curve_inputs.json').write_text(
        json.dumps(checksums, indent=2) + '\n', encoding='utf-8')
    print(f'Exported {len(rows)} aggregate curve records')


if __name__ == '__main__':
    main()
