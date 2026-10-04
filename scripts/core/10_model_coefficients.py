import os
"""Refit prespecified linear models and export penalized coefficient tables."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


PROJECT = Path(os.environ["ANALYSIS_ROOT"])
DATA = PROJECT / "data_private"
QA = PROJECT / "qa_logs"


def load_modeling_module():
    spec = spec_from_file_location("discrete_modeling", Path(__file__).resolve().parents[2] / 'scripts/core/04_discrete_time_models.py')
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def fit_export(
    name: str,
    rows: pd.DataFrame,
    continuous: list[str],
    categorical: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    modeling = load_modeling_module()
    features = modeling.TIME_CONTINUOUS + continuous + categorical
    estimator = Pipeline(
        [
            ("preprocess", modeling.preprocessing(modeling.TIME_CONTINUOUS + continuous, categorical)),
            ("model", LogisticRegression(C=0.1, penalty="l2", solver="lbfgs", max_iter=3000)),
        ]
    ).fit(rows[features], rows["event_this_interval"].astype(int))
    names = estimator.named_steps["preprocess"].get_feature_names_out()
    coefficients = estimator.named_steps["model"].coef_[0]
    output = pd.DataFrame(
        {
            "model": name,
            "transformed_feature": names,
            "penalized_log_odds_coefficient": coefficients,
            "penalized_odds_ratio": np.exp(coefficients),
            "absolute_coefficient": np.abs(coefficients),
        }
    )
    intercept = float(estimator.named_steps["model"].intercept_[0])
    output = pd.concat(
        [
            pd.DataFrame(
                {
                    "model": [name],
                    "transformed_feature": ["model_intercept"],
                    "penalized_log_odds_coefficient": [intercept],
                    "penalized_odds_ratio": [np.exp(intercept)],
                    "absolute_coefficient": [abs(intercept)],
                }
            ),
            output,
        ],
        ignore_index=True,
    ).sort_values("absolute_coefficient", ascending=False)

    fitted_preprocessor = estimator.named_steps["preprocess"]
    continuous_names = modeling.TIME_CONTINUOUS + continuous
    continuous_pipe = fitted_preprocessor.named_transformers_["continuous"]
    imputer = continuous_pipe.named_steps["impute"]
    scaler = continuous_pipe.named_steps["scale"]
    transformed_continuous = continuous_pipe.get_feature_names_out(continuous_names)
    imputation = dict(zip(continuous_names, imputer.statistics_))
    parameter_rows = []
    for transformed_name, mean, scale in zip(
        transformed_continuous, scaler.mean_, scaler.scale_
    ):
        parameter_rows.append(
            {
                "model": name,
                "predictor_type": "continuous",
                "feature": transformed_name,
                "imputation_value": imputation.get(transformed_name, np.nan),
                "standardization_mean": mean,
                "standardization_scale": scale,
                "encoded_categories": "",
            }
        )
    categorical_pipe = fitted_preprocessor.named_transformers_["categorical"]
    categorical_imputer = categorical_pipe.named_steps["impute"]
    encoder = categorical_pipe.named_steps["onehot"]
    for feature, impute_value, categories in zip(
        categorical, categorical_imputer.statistics_, encoder.categories_
    ):
        parameter_rows.append(
            {
                "model": name,
                "predictor_type": "categorical",
                "feature": feature,
                "imputation_value": impute_value,
                "standardization_mean": np.nan,
                "standardization_scale": np.nan,
                "encoded_categories": ";".join(map(str, categories)),
            }
        )
    return output, pd.DataFrame(parameter_rows)


def main() -> None:
    modeling = load_modeling_module()
    people = pd.read_csv(DATA / "development_2011.csv.gz", dtype={"person_id": str})
    intervals = pd.read_csv(
        DATA / "development_2011_intervals.csv.gz", dtype={"person_id": str}
    )
    rows = modeling.merge_intervals(people, intervals)
    fitted = [
        fit_export("base_discrete_time", rows, modeling.BASE_CONTINUOUS, modeling.BASE_CATEGORICAL),
        fit_export(
            "blood_enhanced_discrete_time",
            rows,
            modeling.BASE_CONTINUOUS + modeling.PRIMARY_BLOOD_CONTINUOUS,
            modeling.BASE_CATEGORICAL + modeling.PRIMARY_BLOOD_CATEGORICAL,
        ),
    ]
    output = pd.concat([item[0] for item in fitted], ignore_index=True)
    preprocessing_parameters = pd.concat([item[1] for item in fitted], ignore_index=True)
    output.to_csv(QA / "penalized_model_coefficients.csv", index=False)
    preprocessing_parameters.to_csv(QA / "preprocessing_parameters.csv", index=False)
    for model, table in output.groupby("model", sort=False):
        print("\n", model)
        print(table.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
