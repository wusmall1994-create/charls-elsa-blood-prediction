import os
"""Paired uncertainty for nonlinear blood-minus-base increments."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


PROJECT = Path(os.environ["ANALYSIS_ROOT"])
DATA = PROJECT / "data_private" / "multoutcome_nonlinear_predictions.csv.gz"
QA = PROJECT / "qa_logs"
N_BOOTSTRAP = 2000
SEED = 20260829


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    brier = float(brier_score_loss(y, p))
    return {
        "auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "brier": brier,
        "scaled_brier": float(1 - brier / (y.mean() * (1 - y.mean()))),
    }


def main() -> None:
    data = pd.read_csv(DATA, dtype={"person_id": str})
    data = data.loc[data["evaluable_5y"].astype(bool)]
    rng = np.random.default_rng(SEED)
    rows = []
    for outcome, group in data.groupby("outcome", sort=True):
        wide = group.pivot(
            index=["person_id", "event"], columns="model", values="predicted_risk_5y"
        ).reset_index()
        y = wide["event"].astype(int).to_numpy()
        base = wide["base_hist_gradient_boosting"].to_numpy()
        blood = wide["blood_hist_gradient_boosting"].to_numpy()
        base_point = metrics(y, base)
        blood_point = metrics(y, blood)
        event_index = np.flatnonzero(y == 1)
        nonevent_index = np.flatnonzero(y == 0)
        draws = {metric: [] for metric in base_point}
        for _ in range(N_BOOTSTRAP):
            sample = np.concatenate(
                [
                    rng.choice(event_index, size=len(event_index), replace=True),
                    rng.choice(nonevent_index, size=len(nonevent_index), replace=True),
                ]
            )
            sampled_base = metrics(y[sample], base[sample])
            sampled_blood = metrics(y[sample], blood[sample])
            for metric in draws:
                draws[metric].append(sampled_blood[metric] - sampled_base[metric])
        row = {"outcome": outcome, "n_evaluable": len(y), "events": int(y.sum())}
        for metric in base_point:
            values = np.asarray(draws[metric])
            row[f"base_{metric}"] = base_point[metric]
            row[f"blood_{metric}"] = blood_point[metric]
            row[f"delta_{metric}"] = blood_point[metric] - base_point[metric]
            row[f"delta_{metric}_ci_2_5"] = float(np.quantile(values, 0.025))
            row[f"delta_{metric}_ci_97_5"] = float(np.quantile(values, 0.975))
        rows.append(row)
    results = pd.DataFrame(rows).sort_values("delta_auc", ascending=False)
    results.to_csv(QA / "multoutcome_nonlinear_incremental_summary.csv", index=False)
    print(
        results[
            [
                "outcome",
                "base_auc",
                "blood_auc",
                "delta_auc",
                "delta_auc_ci_2_5",
                "delta_auc_ci_97_5",
                "delta_average_precision",
                "delta_brier",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
