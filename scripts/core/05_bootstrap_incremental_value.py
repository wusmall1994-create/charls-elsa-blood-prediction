import os
"""Paired participant bootstrap for independent five-year validation results."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


PROJECT = Path(os.environ["ANALYSIS_ROOT"])
DATA = PROJECT / "data_private"
QA = PROJECT / "qa_logs"
N_BOOTSTRAP = 2000
SEED = 20260802


def metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    brier = float(brier_score_loss(y, probability))
    null_brier = float(y.mean() * (1 - y.mean()))
    return {
        "auc": float(roc_auc_score(y, probability)),
        "average_precision": float(average_precision_score(y, probability)),
        "brier": brier,
        "scaled_brier": float(1 - brier / null_brier),
        "mean_risk": float(probability.mean()),
        "calibration_in_large_difference": float(probability.mean() - y.mean()),
    }


def calibration_model(y: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000).fit(logit, y)
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def main() -> None:
    predictions = pd.read_csv(
        DATA / "discrete_time_validation_predictions.csv.gz", dtype={"person_id": str}
    )
    predictions = predictions[predictions["evaluable_5y"].astype(bool)]
    wide = predictions.pivot(
        index=["person_id", "event"], columns="model", values="predicted_risk_5y"
    ).reset_index()
    model_names = [
        "base_discrete_time",
        "blood_enhanced_discrete_time",
        "base_hist_gradient_boosting",
        "blood_hist_gradient_boosting",
        "full_assay_sensitivity_discrete_time",
    ]
    y = wide["event"].astype(int).to_numpy()
    probabilities = {name: wide[name].to_numpy() for name in model_names}

    rng = np.random.default_rng(SEED)
    event_index = np.flatnonzero(y == 1)
    nonevent_index = np.flatnonzero(y == 0)
    bootstrap_rows = []
    for iteration in range(N_BOOTSTRAP):
        sample = np.concatenate(
            [
                rng.choice(event_index, size=len(event_index), replace=True),
                rng.choice(nonevent_index, size=len(nonevent_index), replace=True),
            ]
        )
        sampled_y = y[sample]
        row = {"iteration": iteration}
        for name in model_names:
            sampled_probability = probabilities[name][sample]
            for metric_name, value in metrics(sampled_y, sampled_probability).items():
                row[f"{name}__{metric_name}"] = value
            intercept, slope = calibration_model(sampled_y, sampled_probability)
            row[f"{name}__calibration_model_intercept"] = intercept
            row[f"{name}__calibration_model_slope"] = slope
        comparisons = {
            "blood_linear_minus_base_linear": (
                "blood_enhanced_discrete_time", "base_discrete_time"
            ),
            "base_nonlinear_minus_base_linear": (
                "base_hist_gradient_boosting", "base_discrete_time"
            ),
            "blood_nonlinear_minus_base_nonlinear": (
                "blood_hist_gradient_boosting", "base_hist_gradient_boosting"
            ),
            "blood_nonlinear_minus_base_linear": (
                "blood_hist_gradient_boosting", "base_discrete_time"
            ),
            "full_assay_minus_primary_blood": (
                "full_assay_sensitivity_discrete_time", "blood_enhanced_discrete_time"
            ),
        }
        for comparison, (candidate, reference) in comparisons.items():
            for metric_name in ("auc", "average_precision", "brier", "scaled_brier"):
                row[f"difference__{comparison}__{metric_name}"] = (
                    row[f"{candidate}__{metric_name}"] - row[f"{reference}__{metric_name}"]
                )
        bootstrap_rows.append(row)
    bootstrap = pd.DataFrame(bootstrap_rows)

    summary_rows = []
    for name in model_names:
        point = metrics(y, probabilities[name])
        intercept, slope = calibration_model(y, probabilities[name])
        for metric_name, point_value in point.items():
            values = bootstrap[f"{name}__{metric_name}"]
            summary_rows.append(
                {
                    "estimand": f"{name}__{metric_name}",
                    "estimate": point_value,
                    "ci_2_5": float(values.quantile(0.025)),
                    "ci_97_5": float(values.quantile(0.975)),
                }
            )
        for metric_name, point_value in (
            ("calibration_model_intercept", intercept),
            ("calibration_model_slope", slope),
        ):
            values = bootstrap[f"{name}__{metric_name}"]
            summary_rows.append(
                {
                    "estimand": f"{name}__{metric_name}",
                    "estimate": point_value,
                    "ci_2_5": float(values.quantile(0.025)),
                    "ci_97_5": float(values.quantile(0.975)),
                }
            )

    comparisons = {
        "blood_linear_minus_base_linear": ("blood_enhanced_discrete_time", "base_discrete_time"),
        "base_nonlinear_minus_base_linear": ("base_hist_gradient_boosting", "base_discrete_time"),
        "blood_nonlinear_minus_base_nonlinear": (
            "blood_hist_gradient_boosting", "base_hist_gradient_boosting"
        ),
        "blood_nonlinear_minus_base_linear": ("blood_hist_gradient_boosting", "base_discrete_time"),
        "full_assay_minus_primary_blood": (
            "full_assay_sensitivity_discrete_time", "blood_enhanced_discrete_time"
        ),
    }
    point_metrics = {name: metrics(y, probabilities[name]) for name in model_names}
    for comparison, (candidate, reference) in comparisons.items():
        for metric_name in ("auc", "average_precision", "brier", "scaled_brier"):
            values = bootstrap[f"difference__{comparison}__{metric_name}"]
            summary_rows.append(
                {
                    "estimand": f"difference__{comparison}__{metric_name}",
                    "estimate": point_metrics[candidate][metric_name] - point_metrics[reference][metric_name],
                    "ci_2_5": float(values.quantile(0.025)),
                    "ci_97_5": float(values.quantile(0.975)),
                }
            )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(QA / "discrete_time_bootstrap_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
