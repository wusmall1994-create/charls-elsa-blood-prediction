import os
"""IPCW five-year validation using all independently eligible 2015 participants."""

from pathlib import Path

import numpy as np
import pandas as pd
from sksurv.metrics import brier_score, cumulative_dynamic_auc
from sksurv.util import Surv


PROJECT = Path(os.environ["ANALYSIS_ROOT"])
DATA = PROJECT / "data_private"
QA = PROJECT / "qa_logs"
HORIZON = 4.5
N_BOOTSTRAP = 1000
SEED = 20260802


def survival_target(data: pd.DataFrame) -> np.ndarray:
    event = data["event"].astype(bool).to_numpy()
    time = np.where(
        event,
        data["event_time_years"].to_numpy(),
        data["follow_up_years"].to_numpy(),
    )
    return Surv.from_arrays(event=event, time=time)


def ipcw_metrics(y_train: np.ndarray, y_test: np.ndarray, risk: np.ndarray) -> tuple[float, float]:
    auc, _ = cumulative_dynamic_auc(y_train, y_test, risk, np.asarray([HORIZON]))
    _, brier = brier_score(y_train, y_test, (1 - risk).reshape(-1, 1), np.asarray([HORIZON]))
    return float(auc[0]), float(brier[0])


def main() -> None:
    development = pd.read_csv(DATA / "development_2011.csv.gz", dtype={"person_id": str})
    predictions = pd.read_csv(
        DATA / "discrete_time_validation_predictions.csv.gz", dtype={"person_id": str}
    )
    wide = predictions.pivot(
        index=["person_id", "event", "event_time_years", "follow_up_years"],
        columns="model",
        values="predicted_risk_4_5y",
    ).reset_index()
    models = [column for column in wide.columns if column not in {
        "person_id", "event", "event_time_years", "follow_up_years"
    }]
    y_train = survival_target(development)
    y_test = survival_target(wide)

    point = {}
    for model in models:
        point[model] = ipcw_metrics(y_train, y_test, wide[model].to_numpy())

    rng = np.random.default_rng(SEED)
    bootstrap_rows = []
    for iteration in range(N_BOOTSTRAP):
        sample = rng.choice(len(wide), size=len(wide), replace=True)
        sampled_y = y_test[sample]
        row = {"iteration": iteration}
        for model in models:
            auc, brier = ipcw_metrics(y_train, sampled_y, wide[model].to_numpy()[sample])
            row[f"{model}__auc"] = auc
            row[f"{model}__brier"] = brier
        bootstrap_rows.append(row)
    bootstrap = pd.DataFrame(bootstrap_rows)

    rows = []
    for model in models:
        for metric_index, metric in enumerate(("auc", "brier")):
            values = bootstrap[f"{model}__{metric}"]
            rows.append(
                {
                    "estimand": f"{model}__ipcw_4_5y_{metric}",
                    "estimate": point[model][metric_index],
                    "ci_2_5": float(values.quantile(0.025)),
                    "ci_97_5": float(values.quantile(0.975)),
                    "n_validation": len(wide),
                }
            )

    comparisons = {
        "blood_linear_minus_base_linear": ("blood_enhanced_discrete_time", "base_discrete_time"),
        "base_nonlinear_minus_base_linear": ("base_hist_gradient_boosting", "base_discrete_time"),
        "blood_nonlinear_minus_base_nonlinear": (
            "blood_hist_gradient_boosting", "base_hist_gradient_boosting"
        ),
    }
    for comparison, (candidate, reference) in comparisons.items():
        for metric_index, metric in enumerate(("auc", "brier")):
            values = bootstrap[f"{candidate}__{metric}"] - bootstrap[f"{reference}__{metric}"]
            rows.append(
                {
                    "estimand": f"difference__{comparison}__ipcw_4_5y_{metric}",
                    "estimate": point[candidate][metric_index] - point[reference][metric_index],
                    "ci_2_5": float(values.quantile(0.025)),
                    "ci_97_5": float(values.quantile(0.975)),
                    "n_validation": len(wide),
                }
            )
    output = pd.DataFrame(rows)
    output.to_csv(QA / "ipcw_validation_results.csv", index=False)
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
