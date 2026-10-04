import os
"""Full-assay sensitivity for diagnostically proximal comparator outcomes."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


PROJECT = Path(os.environ["ANALYSIS_ROOT"])
DATA = PROJECT / "data_private" / "multoutcome"
QA = PROJECT / "qa_logs"
OUTCOMES = ["dyslipidemia", "diabetes", "kidney_disease"]
N_BOOTSTRAP = 2000
SEED = 20260827


def import_model_module():
    path = Path(__file__).resolve().parents[2] / 'scripts/core/04_discrete_time_models.py'
    spec = spec_from_file_location("digestive_model_module_full_assay", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import model functions from {path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODEL = import_model_module()


def metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    brier = float(brier_score_loss(y, probability))
    return {
        "auc": float(roc_auc_score(y, probability)),
        "average_precision": float(average_precision_score(y, probability)),
        "brier": brier,
        "scaled_brier": float(1 - brier / (y.mean() * (1 - y.mean()))),
    }


def main() -> None:
    primary_predictions = pd.read_csv(
        PROJECT / "data_private" / "multoutcome_primary_predictions.csv.gz",
        dtype={"person_id": str},
    )
    model_results = []
    prediction_results = []
    summary_rows = []
    rng = np.random.default_rng(SEED)
    logistic_grid = {"model__C": [0.01, 0.1, 1.0, 10.0, 100.0]}

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
        result, prediction = MODEL.fit_model(
            "full_assay_sensitivity_discrete_time",
            development_rows,
            validation_rows,
            validation,
            validation_intervals,
            MODEL.BASE_CONTINUOUS + MODEL.SENSITIVITY_BLOOD_CONTINUOUS,
            MODEL.BASE_CATEGORICAL + MODEL.PRIMARY_BLOOD_CATEGORICAL,
            LogisticRegression(penalty="l2", solver="lbfgs", max_iter=3000),
            logistic_grid,
        )
        result["outcome"] = outcome
        prediction["outcome"] = outcome
        model_results.append(result)
        prediction_results.append(prediction)

        existing = primary_predictions.loc[
            primary_predictions["outcome"].eq(outcome)
            & primary_predictions["evaluable_5y"].astype(bool)
        ]
        full = prediction.loc[prediction["evaluable_5y"].astype(bool)]
        wide = existing.pivot(
            index=["person_id", "event"],
            columns="model",
            values="predicted_risk_5y",
        ).reset_index()
        wide = wide.merge(
            full[["person_id", "predicted_risk_5y"]].rename(
                columns={"predicted_risk_5y": "full_assay"}
            ),
            on="person_id",
            how="inner",
            validate="one_to_one",
        )
        y = wide["event"].astype(int).to_numpy()
        probabilities = {
            "base": wide["base_discrete_time"].to_numpy(),
            "primary_blood": wide["blood_enhanced_discrete_time"].to_numpy(),
            "full_assay": wide["full_assay"].to_numpy(),
        }
        points = {name: metrics(y, probability) for name, probability in probabilities.items()}
        event_index = np.flatnonzero(y == 1)
        nonevent_index = np.flatnonzero(y == 0)
        draws = {
            "full_minus_base": {metric: [] for metric in points["base"]},
            "full_minus_primary_blood": {metric: [] for metric in points["base"]},
        }
        for _ in range(N_BOOTSTRAP):
            sample = np.concatenate(
                [
                    rng.choice(event_index, size=len(event_index), replace=True),
                    rng.choice(nonevent_index, size=len(nonevent_index), replace=True),
                ]
            )
            sampled = {
                name: metrics(y[sample], probability[sample])
                for name, probability in probabilities.items()
            }
            for metric in points["base"]:
                draws["full_minus_base"][metric].append(
                    sampled["full_assay"][metric] - sampled["base"][metric]
                )
                draws["full_minus_primary_blood"][metric].append(
                    sampled["full_assay"][metric] - sampled["primary_blood"][metric]
                )
        for comparison, reference in (
            ("full_minus_base", "base"),
            ("full_minus_primary_blood", "primary_blood"),
        ):
            row = {"outcome": outcome, "comparison": comparison}
            for metric in points["base"]:
                values = np.asarray(draws[comparison][metric])
                row[f"delta_{metric}"] = points["full_assay"][metric] - points[reference][metric]
                row[f"delta_{metric}_ci_2_5"] = float(np.quantile(values, 0.025))
                row[f"delta_{metric}_ci_97_5"] = float(np.quantile(values, 0.975))
            summary_rows.append(row)
        print(
            f"{outcome}: full-assay AUROC={points['full_assay']['auc']:.4f}; "
            f"delta vs base={points['full_assay']['auc'] - points['base']['auc']:.4f}"
        )

    pd.DataFrame(model_results).to_csv(
        QA / "proximal_full_assay_model_results.csv", index=False
    )
    pd.concat(prediction_results, ignore_index=True).to_csv(
        PROJECT / "data_private" / "proximal_full_assay_predictions.csv.gz",
        index=False,
        compression="gzip",
    )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(QA / "proximal_full_assay_incremental_summary.csv", index=False)
    print("\n" + summary.to_string(index=False))


if __name__ == "__main__":
    main()
