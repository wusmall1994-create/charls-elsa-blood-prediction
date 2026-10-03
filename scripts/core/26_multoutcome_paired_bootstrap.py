import os
"""Paired bootstrap uncertainty for the CHARLS cross-outcome benchmark."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


PROJECT = Path(os.environ["ANALYSIS_ROOT"])
DATA = PROJECT / "data_private" / "multoutcome_primary_predictions.csv.gz"
QA = PROJECT / "qa_logs"
N_BOOTSTRAP = 2000
SEED = 20260826

OUTCOME_GROUP = {
    "hypertension": "primary_nonoverlap",
    "arthritis_or_rheumatism": "primary_nonoverlap",
    "heart_disease": "primary_nonoverlap",
    "digestive_disease": "primary_nonoverlap",
    "chronic_lung_disease": "primary_nonoverlap",
    "stroke": "primary_nonoverlap",
    "dyslipidemia": "direct_diagnostic_overlap",
    "diabetes": "direct_diagnostic_overlap",
    "kidney_disease": "partial_diagnostic_component",
    "liver_disease": "exploratory_lower_precision",
    "memory_related_disease": "exploratory_lower_precision",
    "asthma": "exploratory_lower_precision",
    "psychiatric_problem": "exploratory_lower_precision",
}


def metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    brier = float(brier_score_loss(y, probability))
    null_brier = float(y.mean() * (1 - y.mean()))
    return {
        "auc": float(roc_auc_score(y, probability)),
        "average_precision": float(average_precision_score(y, probability)),
        "brier": brier,
        "scaled_brier": float(1 - brier / null_brier),
    }


def calibration(y: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000).fit(logit, y)
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def interpretation(delta: float, lower: float, upper: float) -> str:
    if upper < 0.01:
        return "auc_gain_ge_0.01_excluded"
    if lower >= 0.01:
        return "auc_gain_ge_0.01_supported"
    if lower > 0:
        return "positive_auc_increment_but_0.01_not_established"
    return "auc_increment_inconclusive_for_0.01_margin"


def main() -> None:
    predictions = pd.read_csv(DATA, dtype={"person_id": str})
    predictions = predictions.loc[predictions["evaluable_5y"].astype(bool)]
    rng = np.random.default_rng(SEED)
    summary_rows = []
    bootstrap_rows = []

    for outcome, outcome_predictions in predictions.groupby("outcome", sort=True):
        wide = outcome_predictions.pivot(
            index=["person_id", "event"],
            columns="model",
            values="predicted_risk_5y",
        ).reset_index()
        y = wide["event"].astype(int).to_numpy()
        base = wide["base_discrete_time"].to_numpy()
        blood = wide["blood_enhanced_discrete_time"].to_numpy()
        point_base = metrics(y, base)
        point_blood = metrics(y, blood)
        base_intercept, base_slope = calibration(y, base)
        blood_intercept, blood_slope = calibration(y, blood)

        event_index = np.flatnonzero(y == 1)
        nonevent_index = np.flatnonzero(y == 0)
        differences = {metric: [] for metric in point_base}
        for iteration in range(N_BOOTSTRAP):
            sample = np.concatenate(
                [
                    rng.choice(event_index, size=len(event_index), replace=True),
                    rng.choice(nonevent_index, size=len(nonevent_index), replace=True),
                ]
            )
            sampled_y = y[sample]
            sampled_base = metrics(sampled_y, base[sample])
            sampled_blood = metrics(sampled_y, blood[sample])
            row = {"outcome": outcome, "iteration": iteration}
            for metric in differences:
                difference = sampled_blood[metric] - sampled_base[metric]
                differences[metric].append(difference)
                row[f"delta_{metric}"] = difference
            bootstrap_rows.append(row)

        outcome_row = {
            "outcome": outcome,
            "outcome_group": OUTCOME_GROUP[outcome],
            "n_evaluable": len(y),
            "events": int(y.sum()),
            "event_pct": float(y.mean() * 100),
            "base_calibration_intercept": base_intercept,
            "base_calibration_slope": base_slope,
            "blood_calibration_intercept": blood_intercept,
            "blood_calibration_slope": blood_slope,
        }
        for metric in point_base:
            values = np.asarray(differences[metric])
            outcome_row[f"base_{metric}"] = point_base[metric]
            outcome_row[f"blood_{metric}"] = point_blood[metric]
            outcome_row[f"delta_{metric}"] = point_blood[metric] - point_base[metric]
            outcome_row[f"delta_{metric}_ci_2_5"] = float(np.quantile(values, 0.025))
            outcome_row[f"delta_{metric}_ci_97_5"] = float(np.quantile(values, 0.975))
        outcome_row["auc_margin_interpretation"] = interpretation(
            outcome_row["delta_auc"],
            outcome_row["delta_auc_ci_2_5"],
            outcome_row["delta_auc_ci_97_5"],
        )
        summary_rows.append(outcome_row)

    summary = pd.DataFrame(summary_rows).sort_values("delta_auc", ascending=False)
    summary.to_csv(QA / "multoutcome_incremental_value_summary.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(
        PROJECT / "data_private" / "multoutcome_incremental_bootstrap.csv.gz",
        index=False,
        compression="gzip",
    )
    columns = [
        "outcome",
        "outcome_group",
        "n_evaluable",
        "events",
        "base_auc",
        "blood_auc",
        "delta_auc",
        "delta_auc_ci_2_5",
        "delta_auc_ci_97_5",
        "delta_average_precision",
        "delta_brier",
        "auc_margin_interpretation",
    ]
    print(summary[columns].to_string(index=False))


if __name__ == "__main__":
    main()
