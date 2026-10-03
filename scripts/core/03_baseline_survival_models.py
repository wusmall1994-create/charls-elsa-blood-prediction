import os
"""Fit ridge-Cox baseline and blood-enhanced models with independent validation.

This is an initial benchmark, not the final tuned model. It quantifies whether blood
assays add signal before nonlinear models or SHAP are considered.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import brier_score, concordance_index_censored, cumulative_dynamic_auc
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.util import Surv


PROJECT = Path(os.environ["ANALYSIS_ROOT"])
DATA = PROJECT / "data_private"
QA = PROJECT / "qa_logs"

BASE_CONTINUOUS = [
    "age", "cesd10", "bmi", "grip_strength", "systolic_bp", "diastolic_bp",
]
BASE_CATEGORICAL = [
    "sex", "education", "rural_hukou", "marital_status", "self_rated_health",
    "current_smoker", "alcohol_last_year", "adl_limitation", "hypertension",
    "diabetes", "heart_disease", "stroke", "lung_disease", "liver_disease",
    "kidney_disease", "arthritis", "asthma", "cancer",
]
PRIMARY_BLOOD_CONTINUOUS = [
    "bun", "glucose", "creatinine", "total_cholesterol", "triglycerides",
    "hdl_cholesterol", "ldl_cholesterol", "crp", "uric_acid", "wbc",
    "hemoglobin", "hematocrit", "mcv", "platelets",
]
PRIMARY_BLOOD_CATEGORICAL = ["fasting_sample"]
EVALUATION_TIMES = np.asarray([3.0, 4.5])


def survival_target(data: pd.DataFrame) -> np.ndarray:
    event = data["event"].astype(bool).to_numpy()
    duration = np.where(
        event,
        data["event_time_years"].to_numpy(),
        data["follow_up_years"].to_numpy(),
    )
    return Surv.from_arrays(event=event, time=duration)


def preprocessing(continuous: list[str], categorical: list[str]) -> ColumnTransformer:
    continuous_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent", add_indicator=True)),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        [("continuous", continuous_pipe, continuous), ("categorical", categorical_pipe, categorical)],
        remainder="drop",
        sparse_threshold=0,
    )


def observed_km_risk(y: np.ndarray, time: float) -> float:
    km_time, km_survival = kaplan_meier_estimator(y["event"], y["time"])
    before = np.flatnonzero(km_time <= time)
    survival = km_survival[before[-1]] if len(before) else 1.0
    return float(1 - survival)


def fit_and_evaluate(
    name: str,
    development: pd.DataFrame,
    validation: pd.DataFrame,
    continuous: list[str],
    categorical: list[str],
) -> list[dict[str, float | str]]:
    features = continuous + categorical
    x_dev = development[features]
    x_val = validation[features]
    y_dev = survival_target(development)
    y_val = survival_target(validation)

    estimator = Pipeline(
        [
            ("preprocess", preprocessing(continuous, categorical)),
            ("model", CoxPHSurvivalAnalysis()),
        ]
    )
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=20260802)
    splits = list(splitter.split(x_dev, y_dev["event"]))
    search = GridSearchCV(
        estimator,
        param_grid={"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
        cv=splits,
        scoring=None,
        n_jobs=-1,
        refit=True,
    )
    search.fit(x_dev, y_dev)
    best = search.best_estimator_
    risk = best.predict(x_val)
    c_index = concordance_index_censored(y_val["event"], y_val["time"], risk)[0]
    auc, _ = cumulative_dynamic_auc(y_dev, y_val, risk, EVALUATION_TIMES)

    transformed = best.named_steps["preprocess"].transform(x_val)
    functions = best.named_steps["model"].predict_survival_function(transformed)
    survival_prob = np.asarray([[fn(time) for time in EVALUATION_TIMES] for fn in functions])
    _, brier = brier_score(y_dev, y_val, survival_prob, EVALUATION_TIMES)

    rows = []
    for index, time in enumerate(EVALUATION_TIMES):
        predicted_risk = float(np.mean(1 - survival_prob[:, index]))
        observed_risk = observed_km_risk(y_val, float(time))
        rows.append(
            {
                "model": name,
                "n_development": len(development),
                "events_development": int(development["event"].sum()),
                "n_validation": len(validation),
                "events_validation": int(validation["event"].sum()),
                "best_alpha": float(search.best_params_["model__alpha"]),
                "cv_c_index": float(search.best_score_),
                "validation_c_index": float(c_index),
                "time_years": float(time),
                "validation_auc": float(auc[index]),
                "validation_brier": float(brier[index]),
                "mean_predicted_risk": predicted_risk,
                "km_observed_risk": observed_risk,
                "calibration_in_large_difference": predicted_risk - observed_risk,
            }
        )
    return rows


def main() -> None:
    development = pd.read_csv(DATA / "development_2011.csv.gz")
    validation = pd.read_csv(DATA / "validation_2015_independent.csv.gz")
    results = []
    results.extend(
        fit_and_evaluate(
            "base_ridge_cox",
            development,
            validation,
            BASE_CONTINUOUS,
            BASE_CATEGORICAL,
        )
    )
    results.extend(
        fit_and_evaluate(
            "blood_enhanced_ridge_cox",
            development,
            validation,
            BASE_CONTINUOUS + PRIMARY_BLOOD_CONTINUOUS,
            BASE_CATEGORICAL + PRIMARY_BLOOD_CATEGORICAL,
        )
    )
    output = pd.DataFrame(results)
    output.to_csv(QA / "baseline_model_results.csv", index=False)
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
