import os
"""Uniform nonlinear challenger for primary and proximal CHARLS outcomes."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier


PROJECT = Path(os.environ["ANALYSIS_ROOT"])
DATA = PROJECT / "data_private" / "multoutcome"
QA = PROJECT / "qa_logs"
OUTCOMES = [
    "hypertension",
    "dyslipidemia",
    "diabetes",
    "chronic_lung_disease",
    "heart_disease",
    "stroke",
    "kidney_disease",
    "digestive_disease",
    "arthritis_or_rheumatism",
]


def import_model_module():
    path = PROJECT / "scripts" / "04_discrete_time_models.py"
    spec = spec_from_file_location("digestive_model_module_nonlinear", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import model functions from {path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODEL = import_model_module()


def main() -> None:
    results = []
    predictions = []
    grid = {
        "model__learning_rate": [0.03, 0.08],
        "model__max_leaf_nodes": [7, 15],
        "model__l2_regularization": [0.1, 1.0],
    }
    for outcome in OUTCOMES:
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
        for name, continuous, categorical in (
            (
                "base_hist_gradient_boosting",
                MODEL.BASE_CONTINUOUS,
                MODEL.BASE_CATEGORICAL,
            ),
            (
                "blood_hist_gradient_boosting",
                MODEL.BASE_CONTINUOUS + MODEL.PRIMARY_BLOOD_CONTINUOUS,
                MODEL.BASE_CATEGORICAL + MODEL.PRIMARY_BLOOD_CATEGORICAL,
            ),
        ):
            result, prediction = MODEL.fit_model(
                name,
                development_rows,
                validation_rows,
                validation,
                validation_intervals,
                continuous,
                categorical,
                HistGradientBoostingClassifier(
                    max_iter=200,
                    min_samples_leaf=50,
                    early_stopping=False,
                    random_state=20260802,
                ),
                grid,
            )
            result["outcome"] = outcome
            prediction["outcome"] = outcome
            results.append(result)
            predictions.append(prediction)
            print(f"{outcome}: {name}; AUROC={result['validation_5y_auc']:.4f}")
    pd.DataFrame(results).to_csv(QA / "multoutcome_nonlinear_results.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(
        PROJECT / "data_private" / "multoutcome_nonlinear_predictions.csv.gz",
        index=False,
        compression="gzip",
    )


if __name__ == "__main__":
    main()
