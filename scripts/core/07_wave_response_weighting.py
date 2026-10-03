import os
"""Wave-response weighting sensitivity for the nominal 2015--2020 endpoint."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT = Path(os.environ["ANALYSIS_ROOT"])
DATA = PROJECT / "data_private"
QA = PROJECT / "qa_logs"
SEED = 20260802
N_BOOTSTRAP = 1000

CONTINUOUS = ["age", "cesd10", "bmi", "grip_strength", "systolic_bp", "diastolic_bp"]
CATEGORICAL = [
    "sex", "education", "rural_hukou", "marital_status", "self_rated_health",
    "current_smoker", "alcohol_last_year", "adl_limitation", "hypertension",
    "diabetes", "heart_disease", "stroke", "lung_disease", "liver_disease",
    "kidney_disease", "arthritis", "asthma", "cancer",
]


def response_model() -> Pipeline:
    continuous = Pipeline(
        [("impute", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent", add_indicator=True)),
            ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)),
        ]
    )
    return Pipeline(
        [
            (
                "preprocess",
                ColumnTransformer(
                    [("continuous", continuous, CONTINUOUS), ("categorical", categorical, CATEGORICAL)],
                    sparse_threshold=0,
                ),
            ),
            ("model", LogisticRegression(C=0.1, max_iter=3000)),
        ]
    )


def weighted_metrics(y: np.ndarray, probability: np.ndarray, weight: np.ndarray) -> tuple[float, float]:
    return (
        float(roc_auc_score(y, probability, sample_weight=weight)),
        float(brier_score_loss(y, probability, sample_weight=weight)),
    )


def main() -> None:
    people = pd.read_csv(DATA / "validation_2015_independent.csv.gz", dtype={"person_id": str})
    intervals = pd.read_csv(
        DATA / "validation_2015_independent_intervals.csv.gz", dtype={"person_id": str}
    )
    predictions = pd.read_csv(
        DATA / "discrete_time_validation_predictions.csv.gz", dtype={"person_id": str}
    )
    wave4_events = set(
        intervals.loc[
            intervals["source_wave"].eq(4) & intervals["event_this_interval"].eq(1), "person_id"
        ]
    )
    wave5_observed = set(intervals.loc[intervals["source_wave"].eq(5), "person_id"])
    people["early_event"] = people["person_id"].isin(wave4_events)
    people["wave5_observed"] = people["person_id"].isin(wave5_observed)

    at_risk = people[~people["early_event"]].copy()
    model = response_model().fit(at_risk[CONTINUOUS + CATEGORICAL], at_risk["wave5_observed"])
    propensity = model.predict_proba(at_risk[CONTINUOUS + CATEGORICAL])[:, 1]
    response_auc = roc_auc_score(at_risk["wave5_observed"], propensity)
    at_risk["propensity"] = np.clip(propensity, 0.05, 0.99)
    stabilization = float(at_risk["wave5_observed"].mean())
    at_risk["weight"] = np.where(
        at_risk["wave5_observed"], stabilization / at_risk["propensity"], np.nan
    )

    early = people[people["early_event"]].copy()
    early["weight"] = 1.0
    evaluable = pd.concat(
        [early, at_risk[at_risk["wave5_observed"]]], ignore_index=True
    )
    lower, upper = evaluable["weight"].quantile([0.01, 0.99])
    evaluable["weight"] = evaluable["weight"].clip(lower=lower, upper=upper)

    prediction_wide = predictions.pivot(
        index="person_id", columns="model", values="predicted_risk_5y"
    )
    evaluable = evaluable.join(prediction_wide, on="person_id", validate="one_to_one")
    model_names = list(prediction_wide.columns)
    y = evaluable["event"].astype(int).to_numpy()
    weight = evaluable["weight"].to_numpy()
    point = {
        name: weighted_metrics(y, evaluable[name].to_numpy(), weight) for name in model_names
    }

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
        row = {"iteration": iteration}
        for name in model_names:
            auc, brier = weighted_metrics(
                y[sample], evaluable[name].to_numpy()[sample], weight[sample]
            )
            row[f"{name}__auc"] = auc
            row[f"{name}__brier"] = brier
        bootstrap_rows.append(row)
    bootstrap = pd.DataFrame(bootstrap_rows)

    rows = []
    for name in model_names:
        for metric_index, metric in enumerate(("auc", "brier")):
            values = bootstrap[f"{name}__{metric}"]
            rows.append(
                {
                    "estimand": f"{name}__wave_response_weighted_{metric}",
                    "estimate": point[name][metric_index],
                    "ci_2_5": float(values.quantile(0.025)),
                    "ci_97_5": float(values.quantile(0.975)),
                    "n_evaluable": len(evaluable),
                    "effective_sample_size": float(weight.sum() ** 2 / np.square(weight).sum()),
                    "response_model_auc": float(response_auc),
                }
            )
    output = pd.DataFrame(rows)
    output.to_csv(QA / "wave_response_weighted_results.csv", index=False)
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
