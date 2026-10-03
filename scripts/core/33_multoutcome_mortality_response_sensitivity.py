import os
"""Mortality and wave-5 response sensitivities for principal CHARLS outcomes."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


PROJECT = Path(os.environ["ANALYSIS_ROOT"])
RAW = Path(os.environ["CHARLS_DATA_DIR"])
DATA = PROJECT / "data_private" / "multoutcome"
PREDICTIONS = PROJECT / "data_private" / "multoutcome_primary_predictions.csv.gz"
QA = PROJECT / "qa_logs"
N_BOOTSTRAP = 2000
SEED = 20260901

OUTCOMES = [
    "hypertension",
    "dyslipidemia",
    "diabetes",
    "chronic_lung_disease",
    "heart_disease",
    "stroke",
    "kidney_disease",
    "digestive_disease",
    "arthritis_or_rheumatism",
]


def import_response_module():
    path = PROJECT / "scripts" / "07_wave_response_weighting.py"
    spec = spec_from_file_location("digestive_response_module", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import response model from {path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RESPONSE = import_response_module()


def import_cohort_module():
    path = PROJECT / "scripts" / "24_build_multoutcome_cohorts.py"
    spec = spec_from_file_location("multoutcome_cohort_module_for_response", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import cohort functions from {path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


COHORT = import_cohort_module()


def death_ids() -> dict[int, set[str]]:
    exit13 = pd.read_stata(
        RAW / "2013" / "CHARLS2013_Dataset" / "Exit_Interview.dta",
        columns=["ID"],
        convert_categoricals=False,
    )
    result = {2013: set(exit13["ID"].astype(str))}
    for year in (2015, 2018, 2020):
        folder = (
            "CHARLS2015r" if year == 2015 else "CHARLS2018r" if year == 2018 else "CHARLS2020r"
        )
        sample = pd.read_stata(
            RAW / str(year) / folder / "Sample_Infor.dta",
            columns=["ID", "died"],
            convert_categoricals=False,
        )
        result[year] = set(sample.loc[sample["died"].eq(1), "ID"].astype(str))
    return result


def metrics(
    y: np.ndarray, probability: np.ndarray, weight: np.ndarray | None = None
) -> dict[str, float]:
    brier = float(brier_score_loss(y, probability, sample_weight=weight))
    if weight is None:
        prevalence = float(y.mean())
    else:
        prevalence = float(np.average(y, weights=weight))
    null_brier = prevalence * (1 - prevalence)
    return {
        "auc": float(roc_auc_score(y, probability, sample_weight=weight)),
        "average_precision": float(
            average_precision_score(y, probability, sample_weight=weight)
        ),
        "brier": brier,
        "scaled_brier": float(1 - brier / null_brier),
    }


def paired_bootstrap(
    y: np.ndarray,
    base: np.ndarray,
    blood: np.ndarray,
    rng: np.random.Generator,
    weight: np.ndarray | None = None,
) -> dict[str, dict[str, float]]:
    point_base = metrics(y, base, weight)
    point_blood = metrics(y, blood, weight)
    event_index = np.flatnonzero(y == 1)
    nonevent_index = np.flatnonzero(y == 0)
    draws = {metric: [] for metric in point_base}
    for _ in range(N_BOOTSTRAP):
        sample = np.concatenate(
            [
                rng.choice(event_index, size=len(event_index), replace=True),
                rng.choice(nonevent_index, size=len(nonevent_index), replace=True),
            ]
        )
        sampled_weight = weight[sample] if weight is not None else None
        sampled_base = metrics(y[sample], base[sample], sampled_weight)
        sampled_blood = metrics(y[sample], blood[sample], sampled_weight)
        for metric in draws:
            draws[metric].append(sampled_blood[metric] - sampled_base[metric])
    result = {"base": point_base, "blood": point_blood, "difference": {}}
    for metric, values in draws.items():
        array = np.asarray(values)
        result["difference"][metric] = point_blood[metric] - point_base[metric]
        result["difference"][f"{metric}_ci_2_5"] = float(np.quantile(array, 0.025))
        result["difference"][f"{metric}_ci_97_5"] = float(np.quantile(array, 0.975))
    return result


def wide_predictions(predictions: pd.DataFrame, outcome: str) -> pd.DataFrame:
    selected = predictions.loc[
        predictions["outcome"].eq(outcome)
        & predictions["model"].isin(
            ["base_discrete_time", "blood_enhanced_discrete_time"]
        )
    ]
    wide = selected.pivot(
        index=["person_id", "event", "evaluable_5y"],
        columns="model",
        values="predicted_risk_5y",
    ).reset_index()
    return wide


def mortality_sensitivity(
    predictions: pd.DataFrame,
    deaths: dict[int, set[str]],
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    status_rows = []
    performance_rows = []
    for outcome in OUTCOMES:
        development = pd.read_csv(
            DATA / outcome / "development.csv.gz", dtype={"person_id": str}
        )
        validation = pd.read_csv(
            DATA / outcome / "validation.csv.gz", dtype={"person_id": str}
        )
        for cohort, people, waves in (
            ("development_2011", development, (2013, 2015, 2018, 2020)),
            ("evaluation_2015", validation, (2018, 2020)),
        ):
            known_deaths = set().union(*(deaths[wave] for wave in waves))
            identified = set(people["person_id"]) & known_deaths
            events_among_deaths = int(
                people.loc[people["person_id"].isin(identified), "event"].sum()
            )
            status_rows.append(
                {
                    "outcome": outcome,
                    "cohort": cohort,
                    "n": len(people),
                    "identified_deaths_after_baseline": len(identified),
                    "events_among_identified_deaths": events_among_deaths,
                    "deaths_without_prior_reported_event": len(identified)
                    - events_among_deaths,
                }
            )

        validation_deaths = set(validation["person_id"]) & (deaths[2018] | deaths[2020])
        wide = wide_predictions(predictions, outcome)
        already_evaluable = set(
            wide.loc[wide["evaluable_5y"].astype(bool), "person_id"]
        )
        wide["known_death"] = wide["person_id"].isin(validation_deaths)
        analysis = wide.loc[
            wide["evaluable_5y"].astype(bool) | wide["known_death"]
        ].copy()
        y = analysis["event"].astype(int).to_numpy()
        base = analysis["base_discrete_time"].to_numpy()
        blood = analysis["blood_enhanced_discrete_time"].to_numpy()
        result = paired_bootstrap(y, base, blood, rng)
        row = {
            "outcome": outcome,
            "analysis": "known_deaths_added_as_competing_nonevents",
            "n": len(analysis),
            "events": int(y.sum()),
            "known_deaths_total": len(validation_deaths),
            "known_deaths_added": len(validation_deaths - already_evaluable),
        }
        for model in ("base", "blood"):
            for metric, value in result[model].items():
                row[f"{model}_{metric}"] = value
        for metric, value in result["difference"].items():
            row[f"delta_{metric}"] = value
        performance_rows.append(row)
    return pd.DataFrame(status_rows), pd.DataFrame(performance_rows)


def response_weighting(
    predictions: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    rows = []
    core = COHORT.load_core()
    blood_file_ids = pd.read_stata(
        COHORT.BLOOD_2015, columns=["ID"], convert_categoricals=False
    )
    for outcome in OUTCOMES:
        suffix = COHORT.OUTCOMES[outcome][0]
        people = pd.read_csv(
            DATA / outcome / "validation.csv.gz", dtype={"person_id": str}
        )
        development = pd.read_csv(
            DATA / outcome / "development.csv.gz", dtype={"person_id": str}
        )
        intervals = pd.read_csv(
            DATA / outcome / "validation_intervals.csv.gz", dtype={"person_id": str}
        )

        # Stage 1: probability of having any valid post-2015 outcome observation.
        candidates = core.loc[
            core["inw3"].eq(1)
            & core["r3agey"].ge(45)
            & core[f"r3{suffix}"].eq(0)
        ].merge(blood_file_ids, on="ID", how="inner", validate="one_to_one")
        candidates = candidates.loc[
            ~candidates["ID"].astype(str).isin(set(development["person_id"]))
        ].copy()
        candidate_features = COHORT.participant_features(candidates, wave=3)
        candidate_features["any_followup"] = (
            candidates[f"r4{suffix}"].notna()
            | candidates[f"status2020_{outcome}"].notna()
        ).to_numpy()
        if int(candidate_features["any_followup"].sum()) != len(people):
            raise ValueError(f"Any-follow-up count mismatch for {outcome}")
        stage1_model = RESPONSE.response_model().fit(
            candidate_features[RESPONSE.CONTINUOUS + RESPONSE.CATEGORICAL],
            candidate_features["any_followup"],
        )
        stage1_propensity = stage1_model.predict_proba(
            candidate_features[RESPONSE.CONTINUOUS + RESPONSE.CATEGORICAL]
        )[:, 1]
        stage1_auc = float(
            roc_auc_score(candidate_features["any_followup"], stage1_propensity)
        )
        candidate_features["stage1_propensity"] = np.clip(
            stage1_propensity, 0.05, 0.99
        )
        stage1_stabilization = float(candidate_features["any_followup"].mean())
        stage1_weights = candidate_features.loc[
            candidate_features["any_followup"], ["person_id", "stage1_propensity"]
        ].copy()
        stage1_weights["stage1_weight"] = (
            stage1_stabilization / stage1_weights["stage1_propensity"]
        )
        people = people.merge(
            stage1_weights[["person_id", "stage1_weight"]],
            on="person_id",
            how="left",
            validate="one_to_one",
        )
        if people["stage1_weight"].isna().any():
            raise ValueError(f"Missing stage-1 response weight for {outcome}")

        # Stage 2: probability of a valid 2020 status among those still at risk.
        wave4_events = set(
            intervals.loc[
                intervals["source_wave"].eq(4)
                & intervals["event_this_interval"].eq(1),
                "person_id",
            ]
        )
        wave5_observed = set(
            intervals.loc[intervals["source_wave"].eq(5), "person_id"]
        )
        people["early_event"] = people["person_id"].isin(wave4_events)
        people["wave5_observed"] = people["person_id"].isin(wave5_observed)
        at_risk = people.loc[~people["early_event"]].copy()
        stage2_model = RESPONSE.response_model().fit(
            at_risk[RESPONSE.CONTINUOUS + RESPONSE.CATEGORICAL],
            at_risk["wave5_observed"],
        )
        propensity = stage2_model.predict_proba(
            at_risk[RESPONSE.CONTINUOUS + RESPONSE.CATEGORICAL]
        )[:, 1]
        stage2_auc = float(roc_auc_score(at_risk["wave5_observed"], propensity))
        at_risk["propensity"] = np.clip(propensity, 0.05, 0.99)
        stabilization = float(at_risk["wave5_observed"].mean())
        at_risk["stage2_weight"] = np.where(
            at_risk["wave5_observed"],
            stabilization / at_risk["propensity"],
            np.nan,
        )
        at_risk["weight"] = at_risk["stage1_weight"] * at_risk["stage2_weight"]
        early = people.loc[people["early_event"]].copy()
        early["weight"] = early["stage1_weight"]
        evaluable = pd.concat(
            [early, at_risk.loc[at_risk["wave5_observed"]]], ignore_index=True
        )
        lower, upper = evaluable["weight"].quantile([0.01, 0.99])
        evaluable["weight"] = evaluable["weight"].clip(lower=lower, upper=upper)

        wide = wide_predictions(predictions, outcome).drop(columns=["event", "evaluable_5y"])
        evaluable = evaluable.merge(
            wide, on="person_id", how="left", validate="one_to_one"
        )
        y = evaluable["event"].astype(int).to_numpy()
        base = evaluable["base_discrete_time"].to_numpy()
        blood_probability = evaluable["blood_enhanced_discrete_time"].to_numpy()
        weight = evaluable["weight"].to_numpy()
        result = paired_bootstrap(y, base, blood_probability, rng, weight=weight)
        row = {
            "outcome": outcome,
            "analysis": "two_stage_followup_and_wave5_response_weighted_fixed_weights",
            "baseline_candidate_n": len(candidate_features),
            "any_followup_n": int(candidate_features["any_followup"].sum()),
            "no_followup_n": int((~candidate_features["any_followup"]).sum()),
            "any_followup_response_pct": float(
                candidate_features["any_followup"].mean() * 100
            ),
            "any_followup_response_model_auc": stage1_auc,
            "validation_cohort_n": len(people),
            "early_events_n": int(people["early_event"].sum()),
            "at_risk_after_wave4_n": len(at_risk),
            "wave5_observed_among_at_risk_n": int(at_risk["wave5_observed"].sum()),
            "wave5_response_pct": float(at_risk["wave5_observed"].mean() * 100),
            "weighted_evaluable_n": len(evaluable),
            "events": int(y.sum()),
            "effective_sample_size": float(weight.sum() ** 2 / np.square(weight).sum()),
            "wave5_response_model_auc": stage2_auc,
            "weight_p1": float(evaluable["weight"].min()),
            "weight_p99": float(evaluable["weight"].max()),
        }
        for model in ("base", "blood"):
            for metric, value in result[model].items():
                row[f"{model}_{metric}"] = value
        for metric, value in result["difference"].items():
            row[f"delta_{metric}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    predictions = pd.read_csv(PREDICTIONS, dtype={"person_id": str})
    deaths = death_ids()
    rng = np.random.default_rng(SEED)
    mortality_status, mortality_performance = mortality_sensitivity(
        predictions, deaths, rng
    )
    response = response_weighting(predictions, rng)
    mortality_status.to_csv(
        QA / "multoutcome_mortality_status_summary.csv", index=False
    )
    mortality_performance.to_csv(
        QA / "multoutcome_mortality_competing_nonevent_summary.csv", index=False
    )
    response.to_csv(QA / "multoutcome_two_stage_response_weighted_summary.csv", index=False)
    display = [
        "outcome",
        "n",
        "events",
        "known_deaths_added",
        "delta_auc",
        "delta_auc_ci_2_5",
        "delta_auc_ci_97_5",
        "delta_brier",
    ]
    print("Mortality sensitivity")
    print(mortality_performance[display].to_string(index=False))
    response_display = [
        "outcome",
        "wave5_response_pct",
        "effective_sample_size",
        "any_followup_response_pct",
        "any_followup_response_model_auc",
        "wave5_response_model_auc",
        "delta_auc",
        "delta_auc_ci_2_5",
        "delta_auc_ci_97_5",
        "delta_brier",
    ]
    print("\nTwo-stage follow-up and wave-5 response weighting")
    print(response[response_display].to_string(index=False))


if __name__ == "__main__":
    main()
