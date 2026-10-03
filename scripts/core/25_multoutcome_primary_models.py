import os
"""Fit the locked base and blood-enhanced models across feasible CHARLS outcomes."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression


PROJECT = Path(os.environ["ANALYSIS_ROOT"])
DATA = PROJECT / "data_private" / "multoutcome"
QA = PROJECT / "qa_logs"

MODELED_OUTCOMES = [
    "hypertension",
    "dyslipidemia",
    "diabetes",
    "chronic_lung_disease",
    "liver_disease",
    "heart_disease",
    "stroke",
    "kidney_disease",
    "digestive_disease",
    "psychiatric_problem",
    "memory_related_disease",
    "arthritis_or_rheumatism",
    "asthma",
]


def load_model_module():
    path = PROJECT / "scripts" / "04_discrete_time_models.py"
    spec = spec_from_file_location("digestive_model_module", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import shared model functions from {path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODEL = load_model_module()


def main() -> None:
    results = []
    predictions = []
    logistic_grid = {"model__C": [0.01, 0.1, 1.0, 10.0, 100.0]}
    for outcome in MODELED_OUTCOMES:
        outcome_dir = DATA / outcome
        development = pd.read_csv(
            outcome_dir / "development.csv.gz", dtype={"person_id": str}
        )
        validation = pd.read_csv(
            outcome_dir / "validation.csv.gz", dtype={"person_id": str}
        )
        development_intervals = pd.read_csv(
            outcome_dir / "development_intervals.csv.gz", dtype={"person_id": str}
        )
        validation_intervals = pd.read_csv(
            outcome_dir / "validation_intervals.csv.gz", dtype={"person_id": str}
        )
        development_rows = MODEL.merge_intervals(development, development_intervals)
        validation_rows = MODEL.merge_intervals(validation, validation_intervals)
        specifications = [
            (
                "base_discrete_time",
                MODEL.BASE_CONTINUOUS,
                MODEL.BASE_CATEGORICAL,
            ),
            (
                "blood_enhanced_discrete_time",
                MODEL.BASE_CONTINUOUS + MODEL.PRIMARY_BLOOD_CONTINUOUS,
                MODEL.BASE_CATEGORICAL + MODEL.PRIMARY_BLOOD_CATEGORICAL,
            ),
        ]
        for name, continuous, categorical in specifications:
            result, prediction = MODEL.fit_model(
                name,
                development_rows,
                validation_rows,
                validation,
                validation_intervals,
                continuous,
                categorical,
                LogisticRegression(
                    penalty="l2", solver="lbfgs", max_iter=3000
                ),
                logistic_grid,
            )
            result["outcome"] = outcome
            prediction["outcome"] = outcome
            results.append(result)
            predictions.append(prediction)
            print(
                f"{outcome}: {name}; n={result['n_validation_5y_evaluable']}; "
                f"events={result['events_validation_5y']}; "
                f"AUROC={result['validation_5y_auc']:.4f}"
            )

    result_frame = pd.DataFrame(results)
    result_frame.to_csv(QA / "multoutcome_primary_model_results.csv", index=False)
    prediction_frame = pd.concat(predictions, ignore_index=True)
    prediction_frame.to_csv(
        PROJECT / "data_private" / "multoutcome_primary_predictions.csv.gz",
        index=False,
        compression="gzip",
    )
    print(f"\nResults: {QA / 'multoutcome_primary_model_results.csv'}")


if __name__ == "__main__":
    main()
