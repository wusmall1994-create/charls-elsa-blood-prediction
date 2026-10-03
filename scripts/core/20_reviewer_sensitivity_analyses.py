"""Aggregate audits requested during scientific review.

Outputs contain no participant identifiers. The script quantifies selection into the
non-overlapping 2015 evaluation cohort, identifies deaths recorded after each cohort
baseline, and evaluates the primary models after adding known deaths as observed
competing non-events at the nominal five-year endpoint.
"""

from __future__ import annotations
import os

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


PROJECT = Path(os.environ["ANALYSIS_ROOT"])
RAW = Path(os.environ["CHARLS_DATA_DIR"])
DATA = PROJECT / "data_private"
QA = PROJECT / "qa_logs"
SEED = 20260805
N_BOOTSTRAP = 2000


def load_cohort_module():
    spec = spec_from_file_location("cohort_builder", PROJECT / "scripts" / "02_build_model_cohorts.py")
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def standardized_difference(a: pd.Series, b: pd.Series) -> float:
    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else np.nan


def binary_standardized_difference(a: pd.Series, b: pd.Series) -> float:
    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()
    pa, pb = float(a.mean()), float(b.mean())
    pooled = np.sqrt((pa * (1 - pa) + pb * (1 - pb)) / 2)
    return float((pa - pb) / pooled) if pooled > 0 else np.nan


def continuous_display(series: pd.Series) -> str:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return f"{values.median():.1f} ({values.quantile(0.25):.1f}-{values.quantile(0.75):.1f})"


def binary_display(series: pd.Series) -> str:
    values = pd.to_numeric(series, errors="coerce").dropna()
    count = int(values.sum())
    return f"{count:,} ({100 * values.mean():.1f}%)"


def build_selection_audit() -> None:
    cohort = load_cohort_module()
    core = cohort.load_core()
    development = pd.read_csv(DATA / "development_2011.csv.gz", dtype={"person_id": str})
    final_validation = pd.read_csv(
        DATA / "validation_2015_independent.csv.gz", dtype={"person_id": str}
    )
    development_ids = set(development["person_id"])

    blood15 = pd.read_stata(cohort.BLOOD_2015, convert_categoricals=False)
    potential = core[
        core["inw3"].eq(1) & core["r3agey"].ge(45) & core["r3digeste"].eq(0)
    ].merge(
        blood15[["ID", *cohort.LAB_2015.values()]],
        on="ID",
        how="inner",
        validate="one_to_one",
    )
    potential = cohort.add_outcome(potential, baseline_wave=3)
    potential = potential[
        potential["follow_up_years"].notna() & potential["follow_up_years"].gt(0)
    ].copy()
    potential["person_id"] = potential["ID"].astype(str)
    potential["selection_group"] = np.where(
        potential["person_id"].isin(development_ids),
        "overlap_excluded",
        "nonoverlap_included",
    )

    counts = potential["selection_group"].value_counts()
    assert len(potential) == 7119
    assert int(counts["overlap_excluded"]) == 4614
    assert int(counts["nonoverlap_included"]) == len(final_validation) == 2505

    blood11 = pd.read_stata(cohort.BLOOD_2011, columns=["ID"], convert_categoricals=False)
    old_id = blood11["ID"].astype(str)
    blood11_ids = set(old_id.str[:9] + "0" + old_id.str[9:])
    core_index = core.set_index(core["ID"].astype(str), drop=False)

    categories = []
    for person_id in final_validation["person_id"]:
        row = core_index.loc[person_id]
        if not row["inw1"] == 1:
            category = "Not interviewed in 2011"
        elif pd.isna(row["r1agey"]) or row["r1agey"] < 45:
            category = "Not age-eligible in 2011"
        elif not row["r1digeste"] == 0:
            category = "Not disease-free or outcome missing in 2011"
        elif person_id not in blood11_ids:
            category = "No 2011 blood-file match"
        else:
            category = "No valid post-2011 outcome"
        categories.append(category)
    source = pd.Series(categories, name="source_category").value_counts().rename_axis(
        "source_category"
    ).reset_index(name="n")
    source["percent"] = 100 * source["n"] / len(final_validation)
    source.to_csv(QA / "evaluation_source_composition.csv", index=False)

    features = cohort.canonicalize_features(potential, wave=3)
    for canonical, raw_name in cohort.LAB_2015.items():
        features[canonical] = potential[raw_name].to_numpy()
    features["selection_group"] = potential["selection_group"].to_numpy()
    features["women"] = pd.to_numeric(features["sex"], errors="coerce").eq(2).astype(float)

    specifications = [
        ("Age, years", "age", "continuous"),
        ("Women", "women", "binary"),
        ("Rural hukou", "rural_hukou", "binary"),
        ("Current smoker", "current_smoker", "binary"),
        ("Hypertension", "hypertension", "binary"),
        ("Diabetes", "diabetes", "binary"),
        ("Heart disease", "heart_disease", "binary"),
        ("Liver disease", "liver_disease", "binary"),
        ("Arthritis", "arthritis", "binary"),
        ("Body mass index, kg/m2", "bmi", "continuous"),
        ("Systolic blood pressure, mmHg", "systolic_bp", "continuous"),
        ("Glucose, mg/dL", "glucose", "continuous"),
        ("LDL cholesterol, mg/dL", "ldl_cholesterol", "continuous"),
        ("Hemoglobin, g/dL", "hemoglobin", "continuous"),
    ]
    included = features[features["selection_group"].eq("nonoverlap_included")]
    excluded = features[features["selection_group"].eq("overlap_excluded")]
    rows = []
    for label, variable, kind in specifications:
        formatter = continuous_display if kind == "continuous" else binary_display
        smd = standardized_difference if kind == "continuous" else binary_standardized_difference
        rows.append(
            {
                "characteristic": label,
                "all_potentially_eligible_2015": formatter(features[variable]),
                "overlap_excluded": formatter(excluded[variable]),
                "nonoverlap_included": formatter(included[variable]),
                "standardized_difference_included_minus_excluded": smd(
                    included[variable], excluded[variable]
                ),
            }
        )
    pd.DataFrame(rows).to_csv(QA / "evaluation_selection_comparison.csv", index=False)


def death_ids() -> dict[int, set[str]]:
    exit13 = pd.read_stata(
        RAW / "2013" / "CHARLS2013_Dataset" / "Exit_Interview.dta",
        columns=["ID"],
        convert_categoricals=False,
    )
    result = {2013: set(exit13["ID"].astype(str))}
    for year in (2015, 2018, 2020):
        folder = "CHARLS2015r" if year == 2015 else "CHARLS2018r" if year == 2018 else "CHARLS2020r"
        sample = pd.read_stata(
            RAW / str(year) / folder / "Sample_Infor.dta",
            columns=["ID", "died"],
            convert_categoricals=False,
        )
        result[year] = set(sample.loc[sample["died"].eq(1), "ID"].astype(str))
    return result


def prediction_metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    brier = float(brier_score_loss(y, probability))
    null_brier = float(y.mean() * (1 - y.mean()))
    return {
        "auc": float(roc_auc_score(y, probability)),
        "average_precision": float(average_precision_score(y, probability)),
        "brier": brier,
        "scaled_brier": float(1 - brier / null_brier),
        "mean_predicted_risk": float(probability.mean()),
        "observed_event_proportion": float(y.mean()),
    }


def build_mortality_audit() -> None:
    deaths_by_wave = death_ids()
    development = pd.read_csv(DATA / "development_2011.csv.gz", dtype={"person_id": str})
    validation = pd.read_csv(
        DATA / "validation_2015_independent.csv.gz", dtype={"person_id": str}
    )
    cohort_specs = {
        "development_2011": (development, (2013, 2015, 2018, 2020)),
        "evaluation_2015": (validation, (2018, 2020)),
    }
    summary = []
    for cohort_name, (people, waves) in cohort_specs.items():
        all_deaths = set().union(*(deaths_by_wave[wave] for wave in waves))
        cohort_ids = set(people["person_id"])
        identified = cohort_ids & all_deaths
        events_among_deaths = int(
            people.loc[people["person_id"].isin(identified), "event"].sum()
        )
        summary.append(
            {
                "cohort": cohort_name,
                "n": len(people),
                "identified_deaths_after_baseline": len(identified),
                "digestive_first_reports_among_identified_deaths": events_among_deaths,
                "identified_deaths_without_prior_digestive_report": len(identified) - events_among_deaths,
            }
        )
        for wave in waves:
            summary.append(
                {
                    "cohort": f"{cohort_name}_death_record_{wave}",
                    "n": len(people),
                    "identified_deaths_after_baseline": len(cohort_ids & deaths_by_wave[wave]),
                    "digestive_first_reports_among_identified_deaths": np.nan,
                    "identified_deaths_without_prior_digestive_report": np.nan,
                }
            )
    pd.DataFrame(summary).to_csv(QA / "mortality_status_summary.csv", index=False)

    validation_deaths = set(validation["person_id"]) & (
        deaths_by_wave[2018] | deaths_by_wave[2020]
    )
    predictions = pd.read_csv(
        DATA / "discrete_time_validation_predictions.csv.gz", dtype={"person_id": str}
    )
    predictions = predictions[
        predictions["model"].isin(["base_discrete_time", "blood_enhanced_discrete_time"])
    ].copy()
    predictions["known_death"] = predictions["person_id"].isin(validation_deaths)
    predictions["mortality_evaluable_5y"] = (
        predictions["evaluable_5y"].astype(bool) | predictions["known_death"]
    )
    predictions = predictions[predictions["mortality_evaluable_5y"]]
    wide = predictions.pivot(
        index=["person_id", "event"], columns="model", values="predicted_risk_5y"
    ).reset_index()
    y = wide["event"].astype(int).to_numpy()
    models = ["base_discrete_time", "blood_enhanced_discrete_time"]
    probabilities = {name: wide[name].to_numpy() for name in models}

    rng = np.random.default_rng(SEED)
    event_index = np.flatnonzero(y == 1)
    nonevent_index = np.flatnonzero(y == 0)
    boot = {f"{model}__{metric}": [] for model in models for metric in ("auc", "average_precision", "brier", "scaled_brier")}
    boot.update({f"difference__{metric}": [] for metric in ("auc", "average_precision", "brier", "scaled_brier")})
    for _ in range(N_BOOTSTRAP):
        sample = np.concatenate(
            [
                rng.choice(event_index, len(event_index), replace=True),
                rng.choice(nonevent_index, len(nonevent_index), replace=True),
            ]
        )
        sampled = {
            model: prediction_metrics(y[sample], probabilities[model][sample]) for model in models
        }
        for model in models:
            for metric in ("auc", "average_precision", "brier", "scaled_brier"):
                boot[f"{model}__{metric}"].append(sampled[model][metric])
        for metric in ("auc", "average_precision", "brier", "scaled_brier"):
            boot[f"difference__{metric}"].append(
                sampled["blood_enhanced_discrete_time"][metric]
                - sampled["base_discrete_time"][metric]
            )

    point = {model: prediction_metrics(y, probabilities[model]) for model in models}
    rows = []
    for model in models:
        row = {
            "analysis": "known_deaths_added_as_competing_nonevents",
            "model": model,
            "n": len(wide),
            "events": int(y.sum()),
            "known_deaths_added": int(
                len(validation_deaths - set(
                    predictions.loc[predictions["evaluable_5y"].astype(bool), "person_id"]
                ))
            ),
        }
        for metric, value in point[model].items():
            row[metric] = value
            if metric in ("auc", "average_precision", "brier", "scaled_brier"):
                values = np.asarray(boot[f"{model}__{metric}"])
                row[f"{metric}_ci_2_5"] = float(np.quantile(values, 0.025))
                row[f"{metric}_ci_97_5"] = float(np.quantile(values, 0.975))
        rows.append(row)
    difference = {
        "analysis": "known_deaths_added_as_competing_nonevents",
        "model": "blood_minus_base",
        "n": len(wide),
        "events": int(y.sum()),
        "known_deaths_added": rows[0]["known_deaths_added"],
    }
    for metric in ("auc", "average_precision", "brier", "scaled_brier"):
        difference[metric] = (
            point["blood_enhanced_discrete_time"][metric] - point["base_discrete_time"][metric]
        )
        values = np.asarray(boot[f"difference__{metric}"])
        difference[f"{metric}_ci_2_5"] = float(np.quantile(values, 0.025))
        difference[f"{metric}_ci_97_5"] = float(np.quantile(values, 0.975))
    rows.append(difference)
    pd.DataFrame(rows).to_csv(QA / "mortality_competing_event_results.csv", index=False)


def main() -> None:
    build_selection_audit()
    build_mortality_audit()
    print(pd.read_csv(QA / "evaluation_source_composition.csv").to_string(index=False))
    print(pd.read_csv(QA / "mortality_status_summary.csv").to_string(index=False))
    print(pd.read_csv(QA / "mortality_competing_event_results.csv").to_string(index=False))


if __name__ == "__main__":
    main()
