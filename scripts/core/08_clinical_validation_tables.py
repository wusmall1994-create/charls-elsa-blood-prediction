import os
"""Calibration, subgroup transport, and exploratory decision-curve tables."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score


PROJECT = Path(os.environ["ANALYSIS_ROOT"])
DATA = PROJECT / "data_private"
QA = PROJECT / "qa_logs"
SEED = 20260802
N_BOOTSTRAP = 1000
N_PAIRED_BOOTSTRAP = 2000
MODELS = ["base_discrete_time", "blood_enhanced_discrete_time"]


def bootstrap_metric(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    rng = np.random.default_rng(SEED)
    event_index = np.flatnonzero(y == 1)
    nonevent_index = np.flatnonzero(y == 0)
    auc_values, brier_values = [], []
    for _ in range(N_BOOTSTRAP):
        sample = np.concatenate(
            [
                rng.choice(event_index, len(event_index), replace=True),
                rng.choice(nonevent_index, len(nonevent_index), replace=True),
            ]
        )
        auc_values.append(roc_auc_score(y[sample], probability[sample]))
        brier_values.append(brier_score_loss(y[sample], probability[sample]))
    return {
        "auc": float(roc_auc_score(y, probability)),
        "auc_ci_2_5": float(np.quantile(auc_values, 0.025)),
        "auc_ci_97_5": float(np.quantile(auc_values, 0.975)),
        "brier": float(brier_score_loss(y, probability)),
        "brier_ci_2_5": float(np.quantile(brier_values, 0.025)),
        "brier_ci_97_5": float(np.quantile(brier_values, 0.975)),
    }


def bootstrap_auc_difference(
    y: np.ndarray, base_probability: np.ndarray, blood_probability: np.ndarray
) -> dict[str, float]:
    """Paired participant bootstrap for blood-minus-base AUROC differences."""
    rng = np.random.default_rng(SEED)
    event_index = np.flatnonzero(y == 1)
    nonevent_index = np.flatnonzero(y == 0)
    differences = []
    for _ in range(N_PAIRED_BOOTSTRAP):
        sample = np.concatenate(
            [
                rng.choice(event_index, len(event_index), replace=True),
                rng.choice(nonevent_index, len(nonevent_index), replace=True),
            ]
        )
        differences.append(
            roc_auc_score(y[sample], blood_probability[sample])
            - roc_auc_score(y[sample], base_probability[sample])
        )
    return {
        "auc_difference": float(
            roc_auc_score(y, blood_probability) - roc_auc_score(y, base_probability)
        ),
        "auc_difference_ci_2_5": float(np.quantile(differences, 0.025)),
        "auc_difference_ci_97_5": float(np.quantile(differences, 0.975)),
    }


def net_benefit(y: np.ndarray, probability: np.ndarray, threshold: float) -> float:
    positive = probability >= threshold
    tp = np.sum(positive & (y == 1))
    fp = np.sum(positive & (y == 0))
    return float(tp / len(y) - fp / len(y) * threshold / (1 - threshold))


def main() -> None:
    people = pd.read_csv(DATA / "validation_2015_independent.csv.gz", dtype={"person_id": str})
    predictions = pd.read_csv(
        DATA / "discrete_time_validation_predictions.csv.gz", dtype={"person_id": str}
    )
    predictions = predictions[
        predictions["evaluable_5y"].astype(bool) & predictions["model"].isin(MODELS)
    ]
    wide = predictions.pivot(
        index="person_id", columns="model", values="predicted_risk_5y"
    ).reset_index()
    data = people.merge(wide, on="person_id", how="inner", validate="one_to_one")
    data["age_group"] = pd.cut(
        data["age"], [44, 59, 74, np.inf], labels=["45-59", "60-74", ">=75"]
    )
    data["sex_group"] = data["sex"].map({1: "men", 2: "women"})
    data["hukou_group"] = data["rural_hukou"].map({0: "urban", 1: "rural"})

    subgroup_rows = []
    subgroup_incremental_rows = []
    groupings = {
        "overall": pd.Series("all", index=data.index),
        "age": data["age_group"],
        "sex": data["sex_group"],
        "hukou": data["hukou_group"],
    }
    for grouping, labels in groupings.items():
        for level in labels.dropna().unique():
            subset = data[labels.eq(level)]
            y = subset["event"].astype(int).to_numpy()
            if y.sum() < 10 or (len(y) - y.sum()) < 10:
                continue
            for model in MODELS:
                values = bootstrap_metric(y, subset[model].to_numpy())
                subgroup_rows.append(
                    {
                        "grouping": grouping,
                        "level": str(level),
                        "model": model,
                        "n": len(subset),
                        "events": int(y.sum()),
                        "observed_risk": float(y.mean()),
                        "mean_predicted_risk": float(subset[model].mean()),
                        **values,
                    }
                )
            paired_values = bootstrap_auc_difference(
                y,
                subset["base_discrete_time"].to_numpy(),
                subset["blood_enhanced_discrete_time"].to_numpy(),
            )
            subgroup_incremental_rows.append(
                {
                    "grouping": grouping,
                    "level": str(level),
                    "n": len(subset),
                    "events": int(y.sum()),
                    **paired_values,
                }
            )
    pd.DataFrame(subgroup_rows).to_csv(QA / "subgroup_validation_results.csv", index=False)
    pd.DataFrame(subgroup_incremental_rows).to_csv(
        QA / "subgroup_incremental_auc_results.csv", index=False
    )

    calibration_rows = []
    for model in MODELS:
        ranked = data.copy()
        ranked["risk_decile"] = pd.qcut(ranked[model], 10, labels=False, duplicates="drop") + 1
        for decile, subset in ranked.groupby("risk_decile", observed=True):
            calibration_rows.append(
                {
                    "model": model,
                    "risk_decile": int(decile),
                    "n": len(subset),
                    "events": int(subset["event"].sum()),
                    "mean_predicted_risk": float(subset[model].mean()),
                    "observed_risk": float(subset["event"].mean()),
                }
            )
    pd.DataFrame(calibration_rows).to_csv(QA / "calibration_deciles.csv", index=False)

    y = data["event"].astype(int).to_numpy()
    prevalence = float(y.mean())
    decision_rows = []
    for threshold in np.arange(0.05, 0.301, 0.01):
        treat_all = prevalence - (1 - prevalence) * threshold / (1 - threshold)
        decision_rows.extend(
            [
                {"strategy": "treat_none", "threshold": threshold, "net_benefit": 0.0},
                {"strategy": "treat_all", "threshold": threshold, "net_benefit": treat_all},
            ]
        )
        for model in MODELS:
            decision_rows.append(
                {
                    "strategy": model,
                    "threshold": threshold,
                    "net_benefit": net_benefit(y, data[model].to_numpy(), threshold),
                }
            )
    pd.DataFrame(decision_rows).to_csv(QA / "decision_curve_results.csv", index=False)
    print(pd.DataFrame(subgroup_rows).to_string(index=False))


if __name__ == "__main__":
    main()
