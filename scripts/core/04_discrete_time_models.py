import os
"""Fit interval-aligned pooled-logistic models with participant-grouped resampling.

The primary external endpoint is cumulative risk through the nominal 2015--2020
schedule. Participants are evaluable if they experienced an event before 2020 or
were observed disease-free in the 2020 wave; earlier disease-free censoring is not
silently treated as a non-event.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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
SENSITIVITY_BLOOD_CONTINUOUS = PRIMARY_BLOOD_CONTINUOUS + ["hba1c", "cystatin_c"]
TIME_CONTINUOUS = [
    "interval_start_years", "interval_start_squared", "interval_duration_years",
    "log_interval_duration",
]


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


def add_time_terms(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result["interval_start_squared"] = result["interval_start_years"] ** 2
    result["log_interval_duration"] = np.log(result["interval_duration_years"])
    return result


def merge_intervals(participants: pd.DataFrame, intervals: pd.DataFrame) -> pd.DataFrame:
    merged = intervals.merge(participants, on="person_id", how="left", validate="many_to_one")
    if merged["age"].isna().all():
        raise ValueError("Participant features failed to merge; check person_id types.")
    return add_time_terms(merged)


def nominal_grid(
    participants: pd.DataFrame, schedule: tuple[tuple[float, float], ...]
) -> pd.DataFrame:
    pieces = []
    for number, (start, duration) in enumerate(schedule, start=1):
        part = participants.copy()
        part["interval_number"] = number
        part["interval_start_years"] = start
        part["interval_duration_years"] = duration
        pieces.append(part)
    return add_time_terms(pd.concat(pieces, ignore_index=True))


def cumulative_risk(prediction_rows: pd.DataFrame, hazards: np.ndarray) -> pd.Series:
    frame = prediction_rows[["person_id", "interval_number"]].copy()
    frame["survival"] = 1 - np.clip(hazards, 1e-8, 1 - 1e-8)
    return 1 - frame.sort_values(["person_id", "interval_number"]).groupby("person_id")["survival"].prod()


def fit_model(
    name: str,
    development_rows: pd.DataFrame,
    validation_rows: pd.DataFrame,
    validation_people: pd.DataFrame,
    validation_intervals: pd.DataFrame,
    continuous: list[str],
    categorical: list[str],
    classifier,
    parameter_grid: dict[str, list],
) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    features = TIME_CONTINUOUS + continuous + categorical
    estimator = Pipeline(
        [
            ("preprocess", preprocessing(TIME_CONTINUOUS + continuous, categorical)),
            ("model", classifier),
        ]
    )
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20260802)
    search = GridSearchCV(
        estimator,
        param_grid=parameter_grid,
        scoring="neg_log_loss",
        cv=splitter,
        n_jobs=-1,
        refit=True,
    )
    y_dev = development_rows["event_this_interval"].astype(int)
    search.fit(
        development_rows[features],
        y_dev,
        groups=development_rows["person_id"],
    )
    model = search.best_estimator_

    interval_probability = model.predict_proba(validation_rows[features])[:, 1]
    y_interval = validation_rows["event_this_interval"].astype(int).to_numpy()

    grid_5y = nominal_grid(validation_people, ((0.0, 3.0), (3.0, 2.0)))
    hazard_5y = model.predict_proba(grid_5y[features])[:, 1]
    risk_5y = cumulative_risk(grid_5y, hazard_5y)
    grid_4_5y = nominal_grid(validation_people, ((0.0, 3.0), (3.0, 1.5)))
    hazard_4_5y = model.predict_proba(grid_4_5y[features])[:, 1]
    risk_4_5y = cumulative_risk(grid_4_5y, hazard_4_5y)

    reached_2020 = set(
        validation_intervals.loc[validation_intervals["source_wave"].eq(5), "person_id"]
    )
    events = set(validation_people.loc[validation_people["event"].eq(1), "person_id"])
    evaluable_ids = reached_2020 | events
    evaluation = validation_people[
        ["person_id", "event", "event_time_years", "follow_up_years"]
    ].copy()
    evaluation["predicted_risk_5y"] = evaluation["person_id"].map(risk_5y)
    evaluation["predicted_risk_4_5y"] = evaluation["person_id"].map(risk_4_5y)
    evaluation["evaluable_5y"] = evaluation["person_id"].isin(evaluable_ids)
    complete = evaluation[evaluation["evaluable_5y"]]
    y_5y = complete["event"].astype(int).to_numpy()
    p_5y = complete["predicted_risk_5y"].to_numpy()

    result = {
        "model": name,
        "best_parameters": ";".join(
            f"{key.replace('model__', '')}={value}" for key, value in search.best_params_.items()
        ),
        "cv_interval_log_loss": float(-search.best_score_),
        "validation_interval_auc": float(roc_auc_score(y_interval, interval_probability)),
        "validation_interval_brier": float(brier_score_loss(y_interval, interval_probability)),
        "validation_interval_log_loss": float(log_loss(y_interval, interval_probability)),
        "n_validation_5y_evaluable": int(len(complete)),
        "events_validation_5y": int(y_5y.sum()),
        "validation_5y_auc": float(roc_auc_score(y_5y, p_5y)),
        "validation_5y_average_precision": float(average_precision_score(y_5y, p_5y)),
        "validation_5y_brier": float(brier_score_loss(y_5y, p_5y)),
        "validation_5y_scaled_brier": float(
            1 - brier_score_loss(y_5y, p_5y) / (y_5y.mean() * (1 - y_5y.mean()))
        ),
        "mean_predicted_risk_5y": float(p_5y.mean()),
        "observed_event_proportion_5y": float(y_5y.mean()),
        "calibration_in_large_difference_5y": float(p_5y.mean() - y_5y.mean()),
    }
    evaluation["model"] = name
    return result, evaluation


def main() -> None:
    development = pd.read_csv(DATA / "development_2011.csv.gz", dtype={"person_id": str})
    validation = pd.read_csv(
        DATA / "validation_2015_independent.csv.gz", dtype={"person_id": str}
    )
    development_intervals = pd.read_csv(
        DATA / "development_2011_intervals.csv.gz", dtype={"person_id": str}
    )
    validation_intervals = pd.read_csv(
        DATA / "validation_2015_independent_intervals.csv.gz", dtype={"person_id": str}
    )
    development_rows = merge_intervals(development, development_intervals)
    validation_rows = merge_intervals(validation, validation_intervals)

    results = []
    predictions = []
    logistic_grid = {"model__C": [0.01, 0.1, 1.0, 10.0, 100.0]}
    nonlinear_grid = {
        "model__learning_rate": [0.03, 0.08],
        "model__max_leaf_nodes": [7, 15],
        "model__l2_regularization": [0.1, 1.0],
    }
    specifications = [
        (
            "base_discrete_time",
            BASE_CONTINUOUS,
            BASE_CATEGORICAL,
            LogisticRegression(penalty="l2", solver="lbfgs", max_iter=3000),
            logistic_grid,
        ),
        (
            "blood_enhanced_discrete_time",
            BASE_CONTINUOUS + PRIMARY_BLOOD_CONTINUOUS,
            BASE_CATEGORICAL + PRIMARY_BLOOD_CATEGORICAL,
            LogisticRegression(penalty="l2", solver="lbfgs", max_iter=3000),
            logistic_grid,
        ),
        (
            "base_hist_gradient_boosting",
            BASE_CONTINUOUS,
            BASE_CATEGORICAL,
            HistGradientBoostingClassifier(
                max_iter=200, min_samples_leaf=50, early_stopping=False, random_state=20260802
            ),
            nonlinear_grid,
        ),
        (
            "blood_hist_gradient_boosting",
            BASE_CONTINUOUS + PRIMARY_BLOOD_CONTINUOUS,
            BASE_CATEGORICAL + PRIMARY_BLOOD_CATEGORICAL,
            HistGradientBoostingClassifier(
                max_iter=200, min_samples_leaf=50, early_stopping=False, random_state=20260802
            ),
            nonlinear_grid,
        ),
        (
            "full_assay_sensitivity_discrete_time",
            BASE_CONTINUOUS + SENSITIVITY_BLOOD_CONTINUOUS,
            BASE_CATEGORICAL + PRIMARY_BLOOD_CATEGORICAL,
            LogisticRegression(penalty="l2", solver="lbfgs", max_iter=3000),
            logistic_grid,
        ),
    ]
    for name, continuous, categorical, classifier, grid_parameters in specifications:
        result, prediction = fit_model(
            name,
            development_rows,
            validation_rows,
            validation,
            validation_intervals,
            continuous,
            categorical,
            classifier,
            grid_parameters,
        )
        results.append(result)
        predictions.append(prediction)

    result_frame = pd.DataFrame(results)
    result_frame.to_csv(QA / "discrete_time_model_results.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(
        DATA / "discrete_time_validation_predictions.csv.gz", index=False, compression="gzip"
    )
    print(result_frame.to_string(index=False))


if __name__ == "__main__":
    main()
