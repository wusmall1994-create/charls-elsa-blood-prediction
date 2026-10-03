import os
"""Create aggregate cohort-flow and baseline-characteristic tables for the manuscript."""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(os.environ["ANALYSIS_ROOT"])
ROOT = Path(os.environ["CHARLS_DATA_DIR"])
DATA = PROJECT / "data_private"
QA = PROJECT / "qa_logs"
HARMONIZED = ROOT / "Harmonized CHARLS" / "H_CHARLS_D_Data" / "H_CHARLS_D_Data.dta"
BLOOD_2011 = ROOT / "2011" / "Blood_20140429" / "Blood_20140429.dta"
BLOOD_2015 = ROOT / "2015" / "Blood" / "Blood.dta"
HEALTH_2020 = ROOT / "2020" / "CHARLS2020r" / "Health_Status_and_Functioning.dta"

CONTINUOUS = [
    "age", "cesd10", "bmi", "grip_strength", "systolic_bp", "diastolic_bp",
    "bun", "glucose", "creatinine", "total_cholesterol", "triglycerides",
    "hdl_cholesterol", "ldl_cholesterol", "crp", "uric_acid", "wbc",
    "hemoglobin", "hematocrit", "mcv", "platelets", "hba1c", "cystatin_c",
]
CATEGORICAL = [
    "sex", "education", "rural_hukou", "marital_status", "self_rated_health",
    "current_smoker", "alcohol_last_year", "adl_limitation", "hypertension",
    "diabetes", "heart_disease", "stroke", "lung_disease", "liver_disease",
    "kidney_disease", "arthritis", "asthma", "cancer", "fasting_sample",
]


def smd_continuous(a: pd.Series, b: pd.Series) -> float:
    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((b.mean() - a.mean()) / pooled) if pooled > 0 else np.nan


def smd_binary(p_a: float, p_b: float) -> float:
    pooled = np.sqrt((p_a * (1 - p_a) + p_b * (1 - p_b)) / 2)
    return float((p_b - p_a) / pooled) if pooled > 0 else np.nan


def cohort_flow(development_ids: set[str]) -> pd.DataFrame:
    core = pd.read_stata(
        HARMONIZED,
        columns=[
            "ID", "inw1", "inw3", "r1agey", "r3agey", "r1digeste", "r2digeste",
            "r3digeste", "r4digeste",
        ],
        convert_categoricals=False,
    )
    blood11 = pd.read_stata(BLOOD_2011, columns=["ID"], convert_categoricals=False)
    old = blood11["ID"].astype(str)
    blood11_ids = set(old.str[:9] + "0" + old.str[9:])
    blood15_ids = set(
        pd.read_stata(BLOOD_2015, columns=["ID"], convert_categoricals=False)["ID"].astype(str)
    )
    dev_age = core[core["inw1"].eq(1) & core["r1agey"].ge(45)]
    dev_risk = dev_age[dev_age["r1digeste"].eq(0)]
    dev_blood = dev_risk[dev_risk["ID"].astype(str).isin(blood11_ids)]

    val_age = core[core["inw3"].eq(1) & core["r3agey"].ge(45)]
    val_risk = val_age[val_age["r3digeste"].eq(0)]
    val_blood = val_risk[val_risk["ID"].astype(str).isin(blood15_ids)]
    health = pd.read_stata(
        HEALTH_2020,
        columns=["ID", "zdisease_10_", "da002_10_", "da003_10_"],
        convert_categoricals=False,
    )
    prior = health["zdisease_10_"].eq(1)
    disputed = health["da002_10_"].eq(99)
    health["digest2020"] = np.where(
        (prior & ~disputed) | health["da003_10_"].eq(1),
        1,
        np.where((prior & disputed) | health["da003_10_"].eq(2), 0, np.nan),
    )
    val_blood = val_blood.merge(health[["ID", "digest2020"]], on="ID", how="left")
    val_followup = val_blood[
        val_blood[["r4digeste", "digest2020"]].notna().any(axis=1)
    ]
    development = pd.read_csv(DATA / "development_2011.csv.gz", dtype={"person_id": str})
    validation = pd.read_csv(
        DATA / "validation_2015_independent.csv.gz", dtype={"person_id": str}
    )
    full_val_followup = len(val_followup)
    overlap = int(val_followup["ID"].astype(str).isin(development_ids).sum())
    rows = [
        ("development_2011", 1, "Interviewed and age >=45", len(dev_age)),
        ("development_2011", 2, "No digestive disease reported at baseline", len(dev_risk)),
        ("development_2011", 3, "Matched to baseline blood file", len(dev_blood)),
        ("development_2011", 4, "At least one valid follow-up outcome", len(development)),
        ("development_2011", 5, "Incident first reports", int(development["event"].sum())),
        ("validation_2015", 1, "Interviewed and age >=45", len(val_age)),
        ("validation_2015", 2, "No digestive disease reported at baseline", len(val_risk)),
        ("validation_2015", 3, "Matched to baseline blood file", len(val_blood)),
        ("validation_2015", 4, "At least one valid follow-up outcome before independence exclusion", full_val_followup),
        ("validation_2015", 5, "Excluded because ID was in development cohort", overlap),
        ("validation_2015", 6, "Independent temporal evaluation cohort", len(validation)),
        ("validation_2015", 7, "Incident first reports", int(validation["event"].sum())),
        ("validation_2015", 8, "Evaluable at nominal 2020 endpoint", 2302),
    ]
    output = pd.DataFrame(rows, columns=["cohort", "step", "criterion", "n"])
    if set(development["person_id"]) != development_ids:
        raise ValueError("Development ID set changed during flow construction.")
    return output


def characteristics(development: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variable in CONTINUOUS:
        dev = pd.to_numeric(development[variable], errors="coerce")
        val = pd.to_numeric(validation[variable], errors="coerce")
        rows.append(
            {
                "variable": variable,
                "level": "continuous",
                "development_n_nonmissing": int(dev.notna().sum()),
                "development_mean": float(dev.mean()),
                "development_sd": float(dev.std(ddof=1)),
                "development_median": float(dev.median()),
                "development_q1": float(dev.quantile(0.25)),
                "development_q3": float(dev.quantile(0.75)),
                "validation_n_nonmissing": int(val.notna().sum()),
                "validation_mean": float(val.mean()),
                "validation_sd": float(val.std(ddof=1)),
                "validation_median": float(val.median()),
                "validation_q1": float(val.quantile(0.25)),
                "validation_q3": float(val.quantile(0.75)),
                "standardized_difference_validation_minus_development": smd_continuous(dev, val),
            }
        )
    for variable in CATEGORICAL:
        levels = sorted(
            set(pd.to_numeric(development[variable], errors="coerce").dropna())
            | set(pd.to_numeric(validation[variable], errors="coerce").dropna())
        )
        for level in levels:
            dev_valid = pd.to_numeric(development[variable], errors="coerce").dropna()
            val_valid = pd.to_numeric(validation[variable], errors="coerce").dropna()
            dev_count = int(dev_valid.eq(level).sum())
            val_count = int(val_valid.eq(level).sum())
            dev_pct = dev_count / len(dev_valid) if len(dev_valid) else np.nan
            val_pct = val_count / len(val_valid) if len(val_valid) else np.nan
            rows.append(
                {
                    "variable": variable,
                    "level": str(level),
                    "development_n_nonmissing": len(dev_valid),
                    "development_count": dev_count,
                    "development_percent": dev_pct,
                    "validation_n_nonmissing": len(val_valid),
                    "validation_count": val_count,
                    "validation_percent": val_pct,
                    "standardized_difference_validation_minus_development": smd_binary(
                        dev_pct, val_pct
                    ),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    development = pd.read_csv(DATA / "development_2011.csv.gz", dtype={"person_id": str})
    validation = pd.read_csv(
        DATA / "validation_2015_independent.csv.gz", dtype={"person_id": str}
    )
    flow = cohort_flow(set(development["person_id"]))
    flow.to_csv(QA / "manuscript_cohort_flow.csv", index=False)
    table = characteristics(development, validation)
    table.to_csv(QA / "manuscript_baseline_characteristics.csv", index=False)
    print(flow.to_string(index=False))
    print("Characteristics rows:", len(table))


if __name__ == "__main__":
    main()
