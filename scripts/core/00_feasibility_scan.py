import os
"""Read-only feasibility scan for the CHARLS digestive-disease prediction project."""

from pathlib import Path

import numpy as np
import pandas as pd


DATA_ROOT = Path(os.environ["CHARLS_DATA_DIR"])
HARMONIZED = (
    DATA_ROOT
    / "Harmonized CHARLS"
    / "H_CHARLS_D_Data"
    / "H_CHARLS_D_Data.dta"
)
HEALTH_2020 = (
    DATA_ROOT / "2020" / "CHARLS2020r" / "Health_Status_and_Functioning.dta"
)


def main() -> None:
    harmonized_columns = [
        "ID",
        "inw1",
        "r1agey",
        "r1digeste",
        "r2digeste",
        "r3digeste",
        "r4digeste",
    ]
    cohort = pd.read_stata(
        HARMONIZED, columns=harmonized_columns, convert_categoricals=False
    )

    raw_2020 = pd.read_stata(
        HEALTH_2020,
        columns=["ID", "zdisease_10_", "da002_10_", "da003_10_"],
        convert_categoricals=False,
    )
    prior_positive = raw_2020["zdisease_10_"].eq(1)
    prior_disputed = raw_2020["da002_10_"].eq(99)
    current_positive = raw_2020["da003_10_"].eq(1)
    current_negative = raw_2020["da003_10_"].eq(2)

    raw_2020["digest2020"] = np.where(
        (prior_positive & ~prior_disputed) | current_positive,
        1,
        np.where((prior_positive & prior_disputed) | current_negative, 0, np.nan),
    )

    cohort = cohort.merge(
        raw_2020[["ID", "digest2020"]],
        on="ID",
        how="left",
        validate="one_to_one",
    )
    baseline = cohort[cohort["inw1"].eq(1) & cohort["r1agey"].ge(45)]
    risk_set = baseline[baseline["r1digeste"].eq(0)].copy()
    follow_up = ["r2digeste", "r3digeste", "r4digeste", "digest2020"]
    risk_set["has_follow_up"] = risk_set[follow_up].notna().any(axis=1)
    risk_set["event"] = risk_set[follow_up].eq(1).any(axis=1)
    risk_set["event_wave"] = np.select(
        [
            risk_set["r2digeste"].eq(1),
            risk_set["r3digeste"].eq(1),
            risk_set["r4digeste"].eq(1),
            risk_set["digest2020"].eq(1),
        ],
        [2013, 2015, 2018, 2020],
        default=np.nan,
    )
    analyzable = risk_set[risk_set["has_follow_up"]]

    print(f"Baseline age >=45: {len(baseline):,}")
    print(f"Baseline prevalent disease: {baseline['r1digeste'].eq(1).sum():,}")
    print(f"Baseline disease-free risk set: {len(risk_set):,}")
    print(f"With at least one follow-up: {len(analyzable):,}")
    print(
        f"Incident reports: {analyzable['event'].sum():,} "
        f"({analyzable['event'].mean():.2%})"
    )
    print("Earliest event wave:")
    print(analyzable["event_wave"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
