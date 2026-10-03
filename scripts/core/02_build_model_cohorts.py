import os
"""Build private model-ready cohorts and an aggregate cross-wave transport audit.

This script writes derived participant-level files under data_private/. They retain the
CHARLS pseudonymous ID for reproducible cohort exclusion and must not be publicly shared.
"""

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

PRIVATE_OUT = PROJECT / "data_private"
QA_OUT = PROJECT / "qa_logs"

STATIC = {
    "sex": "ragender",
    "education": "raeduc_c",
}

WAVE_FEATURES = {
    "age": "agey",
    "rural_hukou": "rural2",
    "marital_status": "mstat",
    "self_rated_health": "shlt",
    "cesd10": "cesd10",
    "bmi": "mbmi",
    "grip_strength": "gripsum",
    "systolic_bp": "systo",
    "diastolic_bp": "diasto",
    "current_smoker": "smoken",
    "alcohol_last_year": "drinkl",
    "adl_limitation": "adlab_c",
    "hypertension": "hibpe",
    "diabetes": "diabe",
    "heart_disease": "hearte",
    "stroke": "stroke",
    "lung_disease": "lunge",
    "liver_disease": "livere",
    "kidney_disease": "kidneye",
    "arthritis": "arthre",
    "asthma": "asthmae",
    "cancer": "cancre",
}

LAB_2011 = {
    "fasting_sample": "qc1_va003",
    "bun": "newbun",
    "glucose": "newglu",
    "creatinine": "newcrea",
    "total_cholesterol": "newcho",
    "triglycerides": "newtg",
    "hdl_cholesterol": "newhdl",
    "ldl_cholesterol": "newldl",
    "crp": "newcrp",
    "hba1c": "newhba1c",
    "uric_acid": "newua",
    "wbc": "qc1_vb002",
    "hemoglobin": "qc1_vb004",
    "hematocrit": "qc1_vb005",
    "mcv": "qc1_vb006",
    "platelets": "qc1_vb009",
    "cystatin_c": "cystatinc",
}

LAB_2015 = {
    "fasting_sample": "bl_fasting",
    "bun": "bl_bun",
    "glucose": "bl_glu",
    "creatinine": "bl_crea",
    "total_cholesterol": "bl_cho",
    "triglycerides": "bl_tg",
    "hdl_cholesterol": "bl_hdl",
    "ldl_cholesterol": "bl_ldl",
    "crp": "bl_crp",
    "hba1c": "bl_hbalc",
    "uric_acid": "bl_ua",
    "wbc": "bl_wbc",
    "hemoglobin": "bl_hgb",
    "hematocrit": "bl_hct",
    "mcv": "bl_mcv",
    "platelets": "bl_plt",
    "cystatin_c": "bl_cysc",
}

LAB_UNITS = {
    "fasting_sample": "binary",
    "bun": "mg/dL",
    "glucose": "mg/dL",
    "creatinine": "mg/dL",
    "total_cholesterol": "mg/dL",
    "triglycerides": "mg/dL",
    "hdl_cholesterol": "mg/dL",
    "ldl_cholesterol": "mg/dL",
    "crp": "mg/L",
    "hba1c": "%",
    "uric_acid": "mg/dL",
    "wbc": "10^3/uL",
    "hemoglobin": "g/dL",
    "hematocrit": "%",
    "mcv": "fL",
    "platelets": "10^9/L",
    "cystatin_c": "mg/L",
}


def month_index(year: pd.Series, month: pd.Series, nominal_year: int) -> pd.Series:
    y = pd.to_numeric(year, errors="coerce").fillna(nominal_year)
    m = pd.to_numeric(month, errors="coerce").where(lambda x: x.between(1, 12), 7).fillna(7)
    return y * 12 + m - 1


def load_core() -> pd.DataFrame:
    columns = ["ID", *STATIC.values()]
    for wave in (1, 2, 3, 4):
        columns.extend([f"r{wave}digeste", f"r{wave}iwy", f"r{wave}iwm"])
    columns.extend(["inw1", "inw3"])
    for wave in (1, 3):
        columns.extend(f"r{wave}{suffix}" for suffix in WAVE_FEATURES.values())
    columns = list(dict.fromkeys(columns))
    core = pd.read_stata(HARMONIZED, columns=columns, convert_categoricals=False)

    health20 = pd.read_stata(
        HEALTH_2020,
        columns=["ID", "zdisease_10_", "da002_10_", "da003_10_"],
        convert_categoricals=False,
    )
    sample20 = pd.read_stata(
        SAMPLE_2020, columns=["ID", "iyear", "imonth"], convert_categoricals=False
    )
    wave5 = health20.merge(sample20, on="ID", how="left", validate="one_to_one")
    prior_positive = wave5["zdisease_10_"].eq(1)
    prior_disputed = wave5["da002_10_"].eq(99)
    current_positive = wave5["da003_10_"].eq(1)
    current_negative = wave5["da003_10_"].eq(2)
    wave5["digest2020"] = np.where(
        (prior_positive & ~prior_disputed) | current_positive,
        1,
        np.where((prior_positive & prior_disputed) | current_negative, 0, np.nan),
    )
    return core.merge(
        wave5[["ID", "digest2020", "iyear", "imonth"]],
        on="ID",
        how="left",
        validate="one_to_one",
    )


def add_outcome(data: pd.DataFrame, baseline_wave: int) -> pd.DataFrame:
    result = data.copy()
    baseline_nominal = 2011 if baseline_wave == 1 else 2015
    base_month = month_index(
        result[f"r{baseline_wave}iwy"], result[f"r{baseline_wave}iwm"], baseline_nominal
    )
    follow_waves = (
        [(2, 2013), (3, 2015), (4, 2018), (5, 2020)]
        if baseline_wave == 1
        else [(4, 2018), (5, 2020)]
    )
    status_columns = []
    date_columns = []
    for wave, nominal in follow_waves:
        if wave == 5:
            status = result["digest2020"]
            date = month_index(result["iyear"], result["imonth"], nominal)
        else:
            status = result[f"r{wave}digeste"]
            date = month_index(result[f"r{wave}iwy"], result[f"r{wave}iwm"], nominal)
        status_name = f"_status_{wave}"
        date_name = f"_date_{wave}"
        result[status_name] = status
        result[date_name] = date
        status_columns.append(status_name)
        date_columns.append(date_name)

    event_times = []
    follow_times = []
    events = []
    for _, row in result.iterrows():
        observed = []
        for status_name, date_name in zip(status_columns, date_columns):
            status = row[status_name]
            if pd.notna(status):
                observed.append((row[date_name], int(status)))
        observed.sort(key=lambda item: item[0])
        positives = [date for date, status in observed if status == 1]
        if positives:
            end = positives[0]
            event = 1
            event_time = end
        else:
            end = observed[-1][0] if observed else np.nan
            event = 0
            event_time = np.nan
        duration = (end - base_month.loc[row.name]) / 12 if pd.notna(end) else np.nan
        event_times.append((event_time - base_month.loc[row.name]) / 12 if event else np.nan)
        follow_times.append(duration)
        events.append(event)
    result["event"] = events
    result["event_time_years"] = event_times
    result["follow_up_years"] = follow_times
    return result.drop(columns=status_columns + date_columns)


def canonicalize_features(data: pd.DataFrame, wave: int) -> pd.DataFrame:
    result = pd.DataFrame({"person_id": data["ID"].astype(str)})
    for canonical, source in STATIC.items():
        result[canonical] = data[source]
    for canonical, suffix in WAVE_FEATURES.items():
        result[canonical] = data[f"r{wave}{suffix}"]
    return result


def build_interval_rows(
    core: pd.DataFrame, participant_ids: set[str], baseline_wave: int, cohort: str
) -> pd.DataFrame:
    """Create one row per actually observed follow-up interval, stopping at first event."""
    baseline_nominal = 2011 if baseline_wave == 1 else 2015
    follow_waves = (
        [(2, 2013), (3, 2015), (4, 2018), (5, 2020)]
        if baseline_wave == 1
        else [(4, 2018), (5, 2020)]
    )
    selected = core[core["ID"].astype(str).isin(participant_ids)].copy()
    base_month = month_index(
        selected[f"r{baseline_wave}iwy"],
        selected[f"r{baseline_wave}iwm"],
        baseline_nominal,
    )
    rows = []
    for index, participant in selected.iterrows():
        origin = float(base_month.loc[index])
        previous = origin
        interval_number = 0
        for wave, nominal in follow_waves:
            if wave == 5:
                status = participant["digest2020"]
                date = month_index(
                    pd.Series([participant["iyear"]]),
                    pd.Series([participant["imonth"]]),
                    nominal,
                ).iloc[0]
            else:
                status = participant[f"r{wave}digeste"]
                date = month_index(
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


def build_development(core: pd.DataFrame) -> pd.DataFrame:
    blood = pd.read_stata(BLOOD_2011, convert_categoricals=False)
    old_id = blood["ID"].astype(str)
    blood["ID12"] = old_id.str[:9] + "0" + old_id.str[9:]
    eligible = core[
        core["inw1"].eq(1) & core["r1agey"].ge(45) & core["r1digeste"].eq(0)
    ].merge(
        blood[["ID12", *LAB_2011.values()]],
        left_on="ID",
        right_on="ID12",
        how="inner",
        validate="one_to_one",
    )
    eligible = add_outcome(eligible, baseline_wave=1)
    eligible = eligible[eligible["follow_up_years"].notna() & eligible["follow_up_years"].gt(0)]
    result = canonicalize_features(eligible, wave=1)
    bmi = pd.to_numeric(result["bmi"], errors="coerce")
    result["bmi"] = bmi.where(bmi.between(10, 80))
    for canonical, source in LAB_2011.items():
        result[canonical] = eligible[source].to_numpy()
    result["triglycerides"] = pd.to_numeric(
        result["triglycerides"], errors="coerce"
    ).clip(upper=500)
    result["event"] = eligible["event"].to_numpy()
    result["event_time_years"] = eligible["event_time_years"].to_numpy()
    result["follow_up_years"] = eligible["follow_up_years"].to_numpy()
    result["cohort"] = "development_2011"
    return result


def build_validation(core: pd.DataFrame, development_ids: set[str]) -> pd.DataFrame:
    blood = pd.read_stata(BLOOD_2015, convert_categoricals=False)
    eligible = core[
        core["inw3"].eq(1) & core["r3agey"].ge(45) & core["r3digeste"].eq(0)
    ].merge(
        blood[["ID", *LAB_2015.values()]], on="ID", how="inner", validate="one_to_one"
    )
    eligible = eligible[~eligible["ID"].astype(str).isin(development_ids)]
    eligible = add_outcome(eligible, baseline_wave=3)
    eligible = eligible[eligible["follow_up_years"].notna() & eligible["follow_up_years"].gt(0)]
    result = canonicalize_features(eligible, wave=3)
    bmi = pd.to_numeric(result["bmi"], errors="coerce")
    result["bmi"] = bmi.where(bmi.between(10, 80))
    for canonical, source in LAB_2015.items():
        result[canonical] = eligible[source].to_numpy()
    result["triglycerides"] = pd.to_numeric(
        result["triglycerides"], errors="coerce"
    ).clip(upper=500)
    result["event"] = eligible["event"].to_numpy()
    result["event_time_years"] = eligible["event_time_years"].to_numpy()
    result["follow_up_years"] = eligible["follow_up_years"].to_numpy()
    result["cohort"] = "validation_2015_independent"
    return result


def standardized_difference(a: pd.Series, b: pd.Series) -> float:
    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else np.nan


def transport_audit(development: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
    features = [
        *STATIC.keys(), *WAVE_FEATURES.keys(), *LAB_2011.keys(),
    ]
    rows = []
    for feature in dict.fromkeys(features):
        dev = pd.to_numeric(development[feature], errors="coerce")
        val = pd.to_numeric(validation[feature], errors="coerce")
        rows.append(
            {
                "feature": feature,
                "unit": LAB_UNITS.get(feature, "coded_or_native"),
                "development_n": int(dev.notna().sum()),
                "validation_n": int(val.notna().sum()),
                "development_missing_pct": float(dev.isna().mean() * 100),
                "validation_missing_pct": float(val.isna().mean() * 100),
                "development_median": float(dev.median()) if dev.notna().any() else np.nan,
                "validation_median": float(val.median()) if val.notna().any() else np.nan,
                "development_q1": float(dev.quantile(0.25)) if dev.notna().any() else np.nan,
                "development_q3": float(dev.quantile(0.75)) if dev.notna().any() else np.nan,
                "validation_q1": float(val.quantile(0.25)) if val.notna().any() else np.nan,
                "validation_q3": float(val.quantile(0.75)) if val.notna().any() else np.nan,
                "standardized_mean_difference": standardized_difference(dev, val),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    PRIVATE_OUT.mkdir(parents=True, exist_ok=True)
    QA_OUT.mkdir(parents=True, exist_ok=True)
    core = load_core()
    development = build_development(core)
    validation = build_validation(core, set(development["person_id"]))

    development_intervals = build_interval_rows(
        core, set(development["person_id"]), 1, "development_2011"
    )
    validation_intervals = build_interval_rows(
        core, set(validation["person_id"]), 3, "validation_2015_independent"
    )

    development.to_csv(PRIVATE_OUT / "development_2011.csv.gz", index=False, compression="gzip")
    validation.to_csv(
        PRIVATE_OUT / "validation_2015_independent.csv.gz", index=False, compression="gzip"
    )
    development_intervals.to_csv(
        PRIVATE_OUT / "development_2011_intervals.csv.gz", index=False, compression="gzip"
    )
    validation_intervals.to_csv(
        PRIVATE_OUT / "validation_2015_independent_intervals.csv.gz",
        index=False,
        compression="gzip",
    )
    audit = transport_audit(development, validation)
    audit.to_csv(QA_OUT / "predictor_transport_audit.csv", index=False)
    summary = pd.DataFrame(
        [
            {
                "cohort": "development_2011",
                "n": len(development),
                "events": int(development["event"].sum()),
                "event_pct": float(development["event"].mean() * 100),
            },
            {
                "cohort": "validation_2015_independent",
                "n": len(validation),
                "events": int(validation["event"].sum()),
                "event_pct": float(validation["event"].mean() * 100),
            },
        ]
    )
    summary.to_csv(QA_OUT / "cohort_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(
        "Interval rows:",
        len(development_intervals),
        "development;",
        len(validation_intervals),
        "validation",
    )
    print(f"Transport audit: {QA_OUT / 'predictor_transport_audit.csv'}")


if __name__ == "__main__":
    main()
