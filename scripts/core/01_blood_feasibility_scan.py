import os
"""Read-only audit of 2011 development and 2015 temporal-validation blood cohorts."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(os.environ["CHARLS_DATA_DIR"])
HARMONIZED = ROOT / "Harmonized CHARLS" / "H_CHARLS_D_Data" / "H_CHARLS_D_Data.dta"
BLOOD_2011 = ROOT / "2011" / "Blood_20140429" / "Blood_20140429.dta"
BLOOD_2015 = ROOT / "2015" / "Blood" / "Blood.dta"
HEALTH_2020 = ROOT / "2020" / "CHARLS2020r" / "Health_Status_and_Functioning.dta"

LABS_2011 = [
    "newbun", "newglu", "newcrea", "newcho", "newtg", "newhdl", "newldl",
    "newcrp", "newhba1c", "newua", "qc1_vb002", "qc1_vb004", "qc1_vb005",
    "qc1_vb006", "qc1_vb009", "cystatinc",
]
LABS_2015 = [
    "bl_bun", "bl_glu", "bl_crea", "bl_cho", "bl_tg", "bl_hdl", "bl_ldl",
    "bl_crp", "bl_hbalc", "bl_ua", "bl_wbc", "bl_hgb", "bl_hct", "bl_mcv",
    "bl_plt", "bl_cysc",
]


def load_core() -> pd.DataFrame:
    columns = [
        "ID", "inw1", "inw3", "r1agey", "r3agey", "r1digeste", "r2digeste",
        "r3digeste", "r4digeste",
    ]
    core = pd.read_stata(HARMONIZED, columns=columns, convert_categoricals=False)
    wave5 = pd.read_stata(
        HEALTH_2020,
        columns=["ID", "zdisease_10_", "da002_10_", "da003_10_"],
        convert_categoricals=False,
    )
    prior_positive = wave5["zdisease_10_"].eq(1)
    prior_disputed = wave5["da002_10_"].eq(99)
    current_positive = wave5["da003_10_"].eq(1)
    current_negative = wave5["da003_10_"].eq(2)
    wave5["digest2020"] = np.where(
        (prior_positive & ~prior_disputed) | current_positive,
        1,
        np.where((prior_positive & prior_disputed) | current_negative, 0, np.nan),
    )
    return core.merge(wave5[["ID", "digest2020"]], on="ID", how="left", validate="one_to_one")


def main() -> None:
    core = load_core()

    blood11 = pd.read_stata(BLOOD_2011, convert_categoricals=False)
    old_id = blood11["ID"].astype(str)
    blood11["ID12"] = old_id.str[:9] + "0" + old_id.str[9:]
    development = core[
        core["inw1"].eq(1) & core["r1agey"].ge(45) & core["r1digeste"].eq(0)
    ].merge(
        blood11[["ID12"] + LABS_2011],
        left_on="ID",
        right_on="ID12",
        how="inner",
        validate="one_to_one",
    )
    development["has_follow_up"] = development[
        ["r2digeste", "r3digeste", "r4digeste", "digest2020"]
    ].notna().any(axis=1)
    development["event"] = development[
        ["r2digeste", "r3digeste", "r4digeste", "digest2020"]
    ].eq(1).any(axis=1)
    development = development[development["has_follow_up"]]

    blood15 = pd.read_stata(BLOOD_2015, convert_categoricals=False)
    validation = core[
        core["inw3"].eq(1) & core["r3agey"].ge(45) & core["r3digeste"].eq(0)
    ].merge(
        blood15[["ID"] + LABS_2015], on="ID", how="inner", validate="one_to_one"
    )
    validation["has_follow_up"] = validation[["r4digeste", "digest2020"]].notna().any(axis=1)
    validation["event"] = validation[["r4digeste", "digest2020"]].eq(1).any(axis=1)
    validation = validation[validation["has_follow_up"]]

    print(
        f"2011 development: n={len(development):,}, events={development['event'].sum():,} "
        f"({development['event'].mean():.2%})"
    )
    common_2011 = [name for name in LABS_2011 if name != "cystatinc"]
    print(f"2011 complete excluding cystatin C: {development[common_2011].notna().all(axis=1).sum():,}")
    print(f"2011 complete including cystatin C: {development[LABS_2011].notna().all(axis=1).sum():,}")
    print(
        f"2015 temporal validation: n={len(validation):,}, events={validation['event'].sum():,} "
        f"({validation['event'].mean():.2%})"
    )
    print(f"2015 complete across all candidate assays: {validation[LABS_2015].notna().all(axis=1).sum():,}")


if __name__ == "__main__":
    main()
