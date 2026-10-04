import os
"""Sensitivity analysis requiring a later positive report to confirm an incident event."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


PROJECT = Path(os.environ["ANALYSIS_ROOT"])
ROOT = Path(os.environ["CHARLS_DATA_DIR"])
DATA = PROJECT / "data_private"
QA = PROJECT / "qa_logs"


def load_modeling_module():
    spec = spec_from_file_location("discrete_modeling", Path(__file__).resolve().parents[2] / 'scripts/core/04_discrete_time_models.py')
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def status_table() -> pd.DataFrame:
    harmonized = pd.read_stata(
        ROOT / "Harmonized CHARLS" / "H_CHARLS_D_Data" / "H_CHARLS_D_Data.dta",
        columns=["ID", "r2digeste", "r3digeste", "r4digeste"],
        convert_categoricals=False,
    )
    health = pd.read_stata(
        ROOT / "2020" / "CHARLS2020r" / "Health_Status_and_Functioning.dta",
        columns=["ID", "zdisease_10_", "da002_10_", "da003_10_"],
        convert_categoricals=False,
    )
    prior_positive = health["zdisease_10_"].eq(1)
    prior_disputed = health["da002_10_"].eq(99)
    current_positive = health["da003_10_"].eq(1)
    current_negative = health["da003_10_"].eq(2)
    health["digest2020"] = np.where(
        (prior_positive & ~prior_disputed) | current_positive,
        1,
        np.where((prior_positive & prior_disputed) | current_negative, 0, np.nan),
    )
    return harmonized.merge(health[["ID", "digest2020"]], on="ID", how="left")


def confirmation_map(statuses: pd.DataFrame, ids: set[str], waves: list[tuple[int, str]]) -> dict[str, tuple[int | None, bool]]:
    selected = statuses[statuses["ID"].astype(str).isin(ids)]
    result = {}
    for _, row in selected.iterrows():
        observed = [(wave, int(row[column])) for wave, column in waves if pd.notna(row[column])]
        positive_positions = [index for index, (_, value) in enumerate(observed) if value == 1]
        if not positive_positions:
            result[str(row["ID"])] = (None, False)
            continue
        first_position = positive_positions[0]
        first_wave = observed[first_position][0]
        confirmed = any(value == 1 for _, value in observed[first_position + 1 :])
        result[str(row["ID"])] = (first_wave, confirmed)
    return result


def apply_confirmation(
    people: pd.DataFrame,
    intervals: pd.DataFrame,
    mapping: dict[str, tuple[int | None, bool]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    kept = []
    for person_id, group in intervals.groupby("person_id", sort=False):
        first_wave, confirmed = mapping.get(str(person_id), (None, False))
        group = group.sort_values("interval_end_years").copy()
        if first_wave is None:
            adjusted = group.copy()
            adjusted["event_this_interval"] = 0
        elif confirmed:
            adjusted = group[group["source_wave"].le(first_wave)].copy()
            adjusted["event_this_interval"] = adjusted["source_wave"].eq(first_wave).astype(int)
        else:
            adjusted = group[group["source_wave"].lt(first_wave)].copy()
            adjusted["event_this_interval"] = 0
        if not adjusted.empty:
            kept.append(adjusted)
    adjusted_intervals = pd.concat(kept, ignore_index=True)
    aggregate = adjusted_intervals.groupby("person_id").agg(
        event=("event_this_interval", "max"),
        follow_up_years=("interval_end_years", "max"),
    )
    event_time = (
        adjusted_intervals[adjusted_intervals["event_this_interval"].eq(1)]
        .set_index("person_id")["interval_end_years"]
    )
    adjusted_people = people[people["person_id"].isin(aggregate.index)].copy()
    adjusted_people["event"] = adjusted_people["person_id"].map(aggregate["event"]).astype(int)
    adjusted_people["follow_up_years"] = adjusted_people["person_id"].map(
        aggregate["follow_up_years"]
    )
    adjusted_people["event_time_years"] = adjusted_people["person_id"].map(event_time)
    return adjusted_people, adjusted_intervals


def main() -> None:
    modeling = load_modeling_module()
    statuses = status_table()
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
    development_map = confirmation_map(
        statuses,
        set(development["person_id"]),
        [(2, "r2digeste"), (3, "r3digeste"), (4, "r4digeste"), (5, "digest2020")],
    )
    validation_map = confirmation_map(
        statuses,
        set(validation["person_id"]),
        [(4, "r4digeste"), (5, "digest2020")],
    )
    development, development_intervals = apply_confirmation(
        development, development_intervals, development_map
    )
    validation, validation_intervals = apply_confirmation(
        validation, validation_intervals, validation_map
    )
    development_rows = modeling.merge_intervals(development, development_intervals)
    validation_rows = modeling.merge_intervals(validation, validation_intervals)

    results, predictions = [], []
    specifications = [
        ("confirmed_base_discrete_time", modeling.BASE_CONTINUOUS, modeling.BASE_CATEGORICAL),
        (
            "confirmed_blood_discrete_time",
            modeling.BASE_CONTINUOUS + modeling.PRIMARY_BLOOD_CONTINUOUS,
            modeling.BASE_CATEGORICAL + modeling.PRIMARY_BLOOD_CATEGORICAL,
        ),
    ]
    for name, continuous, categorical in specifications:
        result, prediction = modeling.fit_model(
            name,
            development_rows,
            validation_rows,
            validation,
            validation_intervals,
            continuous,
            categorical,
            LogisticRegression(penalty="l2", solver="lbfgs", max_iter=3000),
            {"model__C": [0.01, 0.1, 1.0, 10.0, 100.0]},
        )
        results.append(result)
        predictions.append(prediction)
    output = pd.DataFrame(results)
    output["n_development_confirmed"] = len(development)
    output["events_development_confirmed"] = int(development["event"].sum())
    output["n_validation_confirmed"] = len(validation)
    output["events_validation_confirmed"] = int(validation["event"].sum())
    output.to_csv(QA / "confirmed_outcome_results.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(
        DATA / "confirmed_outcome_predictions.csv.gz", index=False, compression="gzip"
    )
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
