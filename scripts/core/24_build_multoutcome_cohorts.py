import os
"""Build private outcome-specific CHARLS cohorts for the locked benchmark."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(os.environ["ANALYSIS_ROOT"])
ROOT = Path(os.environ["CHARLS_DATA_DIR"])
HARMONIZED = ROOT / "Harmonized CHARLS" / "H_CHARLS_D_Data" / "H_CHARLS_D_Data.dta"
BLOOD_2011 = ROOT / "2011" / "Blood_20140429" / "Blood_20140429.dta"
BLOOD_2015 = ROOT / "2015" / "Blood" / "Blood.dta"
HEALTH_2020 = ROOT / "2020" / "CHARLS2020r" / "Health_Status_and_Functioning.dta"
SAMPLE_2020 = ROOT / "2020" / "CHARLS2020r" / "Sample_Infor.dta"
PRIVATE_OUT = PROJECT / "data_private" / "multoutcome"
QA_OUT = PROJECT / "qa_logs"


def load_existing_cohort_module():
    path = Path(__file__).resolve().parents[2] / 'scripts/core/02_build_model_cohorts.py'
    spec = spec_from_file_location("digestive_cohort_module", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import shared cohort definitions from {path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SHARED = load_existing_cohort_module()

OUTCOMES = {
    "hypertension": ("hibpe", 1),
    "dyslipidemia": ("dyslipe", 2),
    "diabetes": ("diabe", 3),
    "cancer": ("cancre", 4),
    "chronic_lung_disease": ("lunge", 5),
    "liver_disease": ("livere", 6),
    "heart_disease": ("hearte", 7),
    "stroke": ("stroke", 8),
    "kidney_disease": ("kidneye", 9),
    "digestive_disease": ("digeste", 10),
    "psychiatric_problem": ("psyche", 11),
    "memory_related_disease": ("memrye", 12),
    "arthritis_or_rheumatism": ("arthre", 14),
    "asthma": ("asthmae", 15),
}


def recode_2020(raw: pd.DataFrame, index: int) -> pd.Series:
    prior_positive = raw[f"zdisease_{index}_"].eq(1)
    prior_disputed = raw[f"da002_{index}_"].eq(99)
    current_positive = raw[f"da003_{index}_"].eq(1)
    current_negative = raw[f"da003_{index}_"].eq(2)
    return pd.Series(
        np.where(
            (prior_positive & ~prior_disputed) | current_positive,
            1,
            np.where((prior_positive & prior_disputed) | current_negative, 0, np.nan),
        ),
        index=raw.index,
        dtype="float64",
    )


def load_core() -> pd.DataFrame:
    columns = ["ID", *SHARED.STATIC.values(), "inw1", "inw3"]
    for wave in (1, 2, 3, 4):
        columns.extend([f"r{wave}iwy", f"r{wave}iwm"])
        columns.extend(f"r{wave}{suffix}" for suffix, _ in OUTCOMES.values())
    for wave in (1, 3):
        columns.extend(f"r{wave}{suffix}" for suffix in SHARED.WAVE_FEATURES.values())
    core = pd.read_stata(
        HARMONIZED,
        columns=list(dict.fromkeys(columns)),
        convert_categoricals=False,
    )

    raw_columns = ["ID"]
    for _, index in OUTCOMES.values():
        raw_columns.extend(
            [f"zdisease_{index}_", f"da002_{index}_", f"da003_{index}_"]
        )
    health = pd.read_stata(
        HEALTH_2020,
        columns=list(dict.fromkeys(raw_columns)),
        convert_categoricals=False,
    )
    sample = pd.read_stata(
        SAMPLE_2020, columns=["ID", "iyear", "imonth"], convert_categoricals=False
    )
    wave5 = health.merge(sample, on="ID", how="left", validate="one_to_one")
    for name, (_, index) in OUTCOMES.items():
        wave5[f"status2020_{name}"] = recode_2020(wave5, index)
    keep = ["ID", "iyear", "imonth", *[f"status2020_{name}" for name in OUTCOMES]]
    return core.merge(wave5[keep], on="ID", how="left", validate="one_to_one")


def add_outcome(
    data: pd.DataFrame, name: str, suffix: str, baseline_wave: int
) -> pd.DataFrame:
    result = data.copy()
    baseline_nominal = 2011 if baseline_wave == 1 else 2015
    base_month = SHARED.month_index(
        result[f"r{baseline_wave}iwy"],
        result[f"r{baseline_wave}iwm"],
        baseline_nominal,
    )
    follow_waves = (
        [(2, 2013), (3, 2015), (4, 2018), (5, 2020)]
        if baseline_wave == 1
        else [(4, 2018), (5, 2020)]
    )
    events = []
    event_times = []
    follow_times = []
    for row_index, participant in result.iterrows():
        observed = []
        for wave, nominal in follow_waves:
            if wave == 5:
                status = participant[f"status2020_{name}"]
                date = SHARED.month_index(
                    pd.Series([participant["iyear"]]),
                    pd.Series([participant["imonth"]]),
                    nominal,
                ).iloc[0]
            else:
                status = participant[f"r{wave}{suffix}"]
                date = SHARED.month_index(
                    pd.Series([participant[f"r{wave}iwy"]]),
                    pd.Series([participant[f"r{wave}iwm"]]),
                    nominal,
                ).iloc[0]
            if pd.notna(status) and pd.notna(date):
                observed.append((float(date), int(status)))
        observed.sort(key=lambda item: item[0])
        positives = [date for date, status in observed if status == 1]
        if positives:
            end = positives[0]
            event = 1
            event_time = (end - float(base_month.loc[row_index])) / 12
        else:
            end = observed[-1][0] if observed else np.nan
            event = 0
            event_time = np.nan
        follow_time = (
            (end - float(base_month.loc[row_index])) / 12 if pd.notna(end) else np.nan
        )
        events.append(event)
        event_times.append(event_time)
        follow_times.append(follow_time)
    result["event"] = events
    result["event_time_years"] = event_times
    result["follow_up_years"] = follow_times
    return result


def participant_features(data: pd.DataFrame, wave: int) -> pd.DataFrame:
    result = SHARED.canonicalize_features(data, wave)
    bmi = pd.to_numeric(result["bmi"], errors="coerce")
    result["bmi"] = bmi.where(bmi.between(10, 80))
    return result


def add_labs(
    result: pd.DataFrame, eligible: pd.DataFrame, lab_mapping: dict[str, str]
) -> pd.DataFrame:
    for canonical, source in lab_mapping.items():
        result[canonical] = eligible[source].to_numpy()
    result["triglycerides"] = pd.to_numeric(
        result["triglycerides"], errors="coerce"
    ).clip(upper=500)
    return result


def build_development(core: pd.DataFrame, name: str, suffix: str) -> pd.DataFrame:
    blood = pd.read_stata(BLOOD_2011, convert_categoricals=False)
    old_id = blood["ID"].astype(str)
    blood["ID12"] = old_id.str[:9] + "0" + old_id.str[9:]
    eligible = core.loc[
        core["inw1"].eq(1) & core["r1agey"].ge(45) & core[f"r1{suffix}"].eq(0)
    ].merge(
        blood[["ID12", *SHARED.LAB_2011.values()]],
        left_on="ID",
        right_on="ID12",
        how="inner",
        validate="one_to_one",
    )
    eligible = add_outcome(eligible, name, suffix, baseline_wave=1)
    eligible = eligible.loc[
        eligible["follow_up_years"].notna() & eligible["follow_up_years"].gt(0)
    ]
    result = add_labs(
        participant_features(eligible, wave=1), eligible, SHARED.LAB_2011
    )
    result["event"] = eligible["event"].to_numpy()
    result["event_time_years"] = eligible["event_time_years"].to_numpy()
    result["follow_up_years"] = eligible["follow_up_years"].to_numpy()
    result["cohort"] = "development_2011"
    result["outcome"] = name
    return result


def build_validation(
    core: pd.DataFrame, name: str, suffix: str, development_ids: set[str]
) -> pd.DataFrame:
    blood = pd.read_stata(BLOOD_2015, convert_categoricals=False)
    eligible = core.loc[
        core["inw3"].eq(1) & core["r3agey"].ge(45) & core[f"r3{suffix}"].eq(0)
    ].merge(
        blood[["ID", *SHARED.LAB_2015.values()]],
        on="ID",
        how="inner",
        validate="one_to_one",
    )
    eligible = eligible.loc[~eligible["ID"].astype(str).isin(development_ids)]
    eligible = add_outcome(eligible, name, suffix, baseline_wave=3)
    eligible = eligible.loc[
        eligible["follow_up_years"].notna() & eligible["follow_up_years"].gt(0)
    ]
    result = add_labs(participant_features(eligible, wave=3), eligible, SHARED.LAB_2015)
    result["event"] = eligible["event"].to_numpy()
    result["event_time_years"] = eligible["event_time_years"].to_numpy()
    result["follow_up_years"] = eligible["follow_up_years"].to_numpy()
    result["cohort"] = "validation_2015_nonoverlap"
    result["outcome"] = name
    return result


def build_intervals(
    core: pd.DataFrame,
    participant_ids: set[str],
    name: str,
    suffix: str,
    baseline_wave: int,
    cohort: str,
) -> pd.DataFrame:
    baseline_nominal = 2011 if baseline_wave == 1 else 2015
    follow_waves = (
        [(2, 2013), (3, 2015), (4, 2018), (5, 2020)]
        if baseline_wave == 1
        else [(4, 2018), (5, 2020)]
    )
    selected = core.loc[core["ID"].astype(str).isin(participant_ids)].copy()
    base_month = SHARED.month_index(
        selected[f"r{baseline_wave}iwy"],
        selected[f"r{baseline_wave}iwm"],
        baseline_nominal,
    )
    rows = []
    for row_index, participant in selected.iterrows():
        origin = float(base_month.loc[row_index])
        previous = origin
        interval_number = 0
        for wave, nominal in follow_waves:
            if wave == 5:
                status = participant[f"status2020_{name}"]
                date = SHARED.month_index(
                    pd.Series([participant["iyear"]]),
                    pd.Series([participant["imonth"]]),
                    nominal,
                ).iloc[0]
            else:
                status = participant[f"r{wave}{suffix}"]
                date = SHARED.month_index(
                    pd.Series([participant[f"r{wave}iwy"]]),
                    pd.Series([participant[f"r{wave}iwm"]]),
                    nominal,
                ).iloc[0]
            if pd.isna(status) or pd.isna(date) or float(date) <= previous:
                continue
            interval_number += 1
            end = float(date)
            event = int(status == 1)
            rows.append(
                {
                    "person_id": str(participant["ID"]),
                    "outcome": name,
                    "cohort": cohort,
                    "interval_number": interval_number,
                    "source_wave": wave,
                    "interval_start_years": (previous - origin) / 12,
                    "interval_end_years": (end - origin) / 12,
                    "interval_duration_years": (end - previous) / 12,
                    "event_this_interval": event,
                }
            )
            previous = end
            if event:
                break
    return pd.DataFrame(rows)


def main() -> None:
    PRIVATE_OUT.mkdir(parents=True, exist_ok=True)
    QA_OUT.mkdir(parents=True, exist_ok=True)
    core = load_core()
    summaries = []
    for name, (suffix, _) in OUTCOMES.items():
        outcome_dir = PRIVATE_OUT / name
        outcome_dir.mkdir(parents=True, exist_ok=True)
        development = build_development(core, name, suffix)
        validation = build_validation(
            core, name, suffix, set(development["person_id"].astype(str))
        )
        development_intervals = build_intervals(
            core,
            set(development["person_id"].astype(str)),
            name,
            suffix,
            1,
            "development_2011",
        )
        validation_intervals = build_intervals(
            core,
            set(validation["person_id"].astype(str)),
            name,
            suffix,
            3,
            "validation_2015_nonoverlap",
        )
        development.to_csv(
            outcome_dir / "development.csv.gz", index=False, compression="gzip"
        )
        validation.to_csv(
            outcome_dir / "validation.csv.gz", index=False, compression="gzip"
        )
        development_intervals.to_csv(
            outcome_dir / "development_intervals.csv.gz",
            index=False,
            compression="gzip",
        )
        validation_intervals.to_csv(
            outcome_dir / "validation_intervals.csv.gz",
            index=False,
            compression="gzip",
        )
        summaries.append(
            {
                "outcome": name,
                "development_n": len(development),
                "development_events": int(development["event"].sum()),
                "development_interval_rows": len(development_intervals),
                "evaluation_n": len(validation),
                "evaluation_events": int(validation["event"].sum()),
                "evaluation_interval_rows": len(validation_intervals),
            }
        )
    summary = pd.DataFrame(summaries)
    summary.to_csv(QA_OUT / "multoutcome_cohort_build_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
