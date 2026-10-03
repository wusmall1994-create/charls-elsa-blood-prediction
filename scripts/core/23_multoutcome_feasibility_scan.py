import os
"""Feasibility scan for a CHARLS cross-outcome incremental-value benchmark.

This is a descriptive gatekeeping analysis. It does not fit prediction models or
select outcomes using model performance. Participant-level outputs stay private;
only aggregate feasibility tables are written to qa_logs/.
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
QA_OUT = PROJECT / "qa_logs"


# The 2020 questionnaire added Parkinson disease at position 13. Arthritis and
# asthma therefore map to 14 and 15, whereas the first 12 positions are unchanged.
OUTCOMES = {
    "hypertension": {"suffix": "hibpe", "index_2020": 1, "proximity": 1},
    "dyslipidemia": {"suffix": "dyslipe", "index_2020": 2, "proximity": 3},
    "diabetes": {"suffix": "diabe", "index_2020": 3, "proximity": 3},
    "cancer": {"suffix": "cancre", "index_2020": 4, "proximity": 0},
    "chronic_lung_disease": {"suffix": "lunge", "index_2020": 5, "proximity": 0},
    "liver_disease": {"suffix": "livere", "index_2020": 6, "proximity": 1},
    "heart_disease": {"suffix": "hearte", "index_2020": 7, "proximity": 1},
    "stroke": {"suffix": "stroke", "index_2020": 8, "proximity": 1},
    "kidney_disease": {"suffix": "kidneye", "index_2020": 9, "proximity": 3},
    "digestive_disease": {"suffix": "digeste", "index_2020": 10, "proximity": 0},
    "psychiatric_problem": {"suffix": "psyche", "index_2020": 11, "proximity": 0},
    "memory_related_disease": {"suffix": "memrye", "index_2020": 12, "proximity": 1},
    "arthritis_or_rheumatism": {"suffix": "arthre", "index_2020": 14, "proximity": 0},
    "asthma": {"suffix": "asthmae", "index_2020": 15, "proximity": 0},
}

PROXIMITY_LABELS = {
    0: "no_direct_diagnostic_overlap",
    1: "biologically_related_not_diagnostic",
    2: "partly_used_in_diagnosis_or_monitoring",
    3: "direct_or_near_direct_diagnostic_overlap",
}


def recode_2020(raw: pd.DataFrame, index: int) -> pd.Series:
    """Reproduce the prespecified 2020 disease-status reconciliation rule."""
    prior = raw[f"zdisease_{index}_"]
    comparison = raw[f"da002_{index}_"]
    current = raw[f"da003_{index}_"]
    prior_positive = prior.eq(1)
    prior_disputed = comparison.eq(99)
    current_positive = current.eq(1)
    current_negative = current.eq(2)
    return pd.Series(
        np.where(
            (prior_positive & ~prior_disputed) | current_positive,
            1,
            np.where((prior_positive & prior_disputed) | current_negative, 0, np.nan),
        ),
        index=raw.index,
        dtype="float64",
    )


def event_summary(data: pd.DataFrame, follow_up: list[str]) -> dict[str, float | int]:
    observed = data[follow_up].notna().any(axis=1)
    analyzable = data.loc[observed]
    event = analyzable[follow_up].eq(1).any(axis=1)
    return {
        "with_follow_up": int(observed.sum()),
        "events": int(event.sum()),
        "event_pct": float(event.mean() * 100) if len(event) else np.nan,
    }


def event_timing(data: pd.DataFrame, follow_up: list[tuple[str, int]]) -> dict[str, int]:
    result = {}
    still_at_risk = pd.Series(True, index=data.index)
    for variable, year in follow_up:
        first = still_at_risk & data[variable].eq(1)
        result[f"events_{year}"] = int(first.sum())
        still_at_risk &= ~first
    return result


def feasibility_tier(development_events: int, evaluation_events: int) -> str:
    if development_events >= 200 and evaluation_events >= 200:
        return "primary_feasible"
    if development_events >= 100 and evaluation_events >= 100:
        return "exploratory_feasible"
    return "insufficient_events_for_primary_benchmark"


def markdown_table(data: pd.DataFrame) -> str:
    """Render a small Markdown table without an optional tabulate dependency."""
    headers = [str(column) for column in data.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in data.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def main() -> None:
    QA_OUT.mkdir(parents=True, exist_ok=True)

    core_columns = ["ID", "inw1", "inw3", "r1agey", "r3agey"]
    for outcome in OUTCOMES.values():
        core_columns.extend(f"r{wave}{outcome['suffix']}" for wave in (1, 2, 3, 4))
    core = pd.read_stata(
        HARMONIZED,
        columns=list(dict.fromkeys(core_columns)),
        convert_categoricals=False,
    )

    raw_columns = ["ID"]
    for outcome in OUTCOMES.values():
        index = outcome["index_2020"]
        raw_columns.extend(
            [f"zdisease_{index}_", f"da002_{index}_", f"da003_{index}_"]
        )
    raw_2020 = pd.read_stata(
        HEALTH_2020,
        columns=list(dict.fromkeys(raw_columns)),
        convert_categoricals=False,
    )
    for name, outcome in OUTCOMES.items():
        raw_2020[f"status2020_{name}"] = recode_2020(
            raw_2020, outcome["index_2020"]
        )
    status_2020 = ["ID", *[f"status2020_{name}" for name in OUTCOMES]]
    core = core.merge(
        raw_2020[status_2020], on="ID", how="left", validate="one_to_one"
    )

    blood_2011 = pd.read_stata(
        BLOOD_2011, columns=["ID"], convert_categoricals=False
    )
    old_id = blood_2011["ID"].astype(str)
    blood_2011_ids = set(old_id.str[:9] + "0" + old_id.str[9:])
    blood_2015 = pd.read_stata(
        BLOOD_2015, columns=["ID"], convert_categoricals=False
    )
    blood_2015_ids = set(blood_2015["ID"].astype(str))
    core_ids = core["ID"].astype(str)

    development_ids_by_outcome: dict[str, set[str]] = {}
    for name, outcome in OUTCOMES.items():
        suffix = outcome["suffix"]
        dev = core.loc[
            core["inw1"].eq(1)
            & core["r1agey"].ge(45)
            & core[f"r1{suffix}"].eq(0)
            & core_ids.isin(blood_2011_ids)
        ]
        dev_follow = [
            f"r2{suffix}",
            f"r3{suffix}",
            f"r4{suffix}",
            f"status2020_{name}",
        ]
        analyzable = dev.loc[dev[dev_follow].notna().any(axis=1)]
        development_ids_by_outcome[name] = set(analyzable["ID"].astype(str))
    common_exclusion_ids = set().union(*development_ids_by_outcome.values())

    rows = []
    for name, outcome in OUTCOMES.items():
        suffix = outcome["suffix"]
        follow_dev = [
            f"r2{suffix}",
            f"r3{suffix}",
            f"r4{suffix}",
            f"status2020_{name}",
        ]
        follow_eval = [f"r4{suffix}", f"status2020_{name}"]

        baseline_2011 = core.loc[
            core["inw1"].eq(1) & core["r1agey"].ge(45) & core_ids.isin(blood_2011_ids)
        ]
        dev = baseline_2011.loc[baseline_2011[f"r1{suffix}"].eq(0)]
        dev_summary = event_summary(dev, follow_dev)
        dev_analyzable = dev.loc[dev[follow_dev].notna().any(axis=1)]

        baseline_2015 = core.loc[
            core["inw3"].eq(1) & core["r3agey"].ge(45) & core_ids.isin(blood_2015_ids)
        ]
        potential = baseline_2015.loc[baseline_2015[f"r3{suffix}"].eq(0)]
        potential_summary = event_summary(potential, follow_eval)

        outcome_nonoverlap = potential.loc[
            ~potential["ID"].astype(str).isin(development_ids_by_outcome[name])
        ]
        outcome_nonoverlap_summary = event_summary(outcome_nonoverlap, follow_eval)
        outcome_nonoverlap_analyzable = outcome_nonoverlap.loc[
            outcome_nonoverlap[follow_eval].notna().any(axis=1)
        ]

        common_nonoverlap = potential.loc[
            ~potential["ID"].astype(str).isin(common_exclusion_ids)
        ]
        common_nonoverlap_summary = event_summary(common_nonoverlap, follow_eval)

        proximity = int(outcome["proximity"])
        row = {
            "outcome": name,
            "harmonized_suffix": suffix,
            "index_2020": int(outcome["index_2020"]),
            "provisional_diagnostic_proximity_level": proximity,
            "provisional_diagnostic_proximity_label": PROXIMITY_LABELS[proximity],
            "2011_blood_age45_n": len(baseline_2011),
            "2011_baseline_prevalent_n": int(baseline_2011[f"r1{suffix}"].eq(1).sum()),
            "development_risk_set_n": len(dev),
            "development_with_follow_up_n": dev_summary["with_follow_up"],
            "development_events_n": dev_summary["events"],
            "development_event_pct": dev_summary["event_pct"],
            "2015_blood_age45_n": len(baseline_2015),
            "2015_baseline_prevalent_n": int(baseline_2015[f"r3{suffix}"].eq(1).sum()),
            "2015_potential_risk_set_n": len(potential),
            "2015_potential_with_follow_up_n": potential_summary["with_follow_up"],
            "2015_potential_events_n": potential_summary["events"],
            "outcome_specific_nonoverlap_n_before_follow_up": len(outcome_nonoverlap),
            "outcome_specific_nonoverlap_with_follow_up_n": outcome_nonoverlap_summary[
                "with_follow_up"
            ],
            "outcome_specific_nonoverlap_events_n": outcome_nonoverlap_summary["events"],
            "outcome_specific_nonoverlap_event_pct": outcome_nonoverlap_summary["event_pct"],
            "common_nonoverlap_n_before_follow_up": len(common_nonoverlap),
            "common_nonoverlap_with_follow_up_n": common_nonoverlap_summary["with_follow_up"],
            "common_nonoverlap_events_n": common_nonoverlap_summary["events"],
            "common_nonoverlap_event_pct": common_nonoverlap_summary["event_pct"],
            "question_series": "harmonized_waves_1_to_4_plus_mapped_2020",
        }
        row.update(
            {
                f"development_{key}": value
                for key, value in event_timing(
                    dev_analyzable,
                    [
                        (f"r2{suffix}", 2013),
                        (f"r3{suffix}", 2015),
                        (f"r4{suffix}", 2018),
                        (f"status2020_{name}", 2020),
                    ],
                ).items()
            }
        )
        row.update(
            {
                f"evaluation_{key}": value
                for key, value in event_timing(
                    outcome_nonoverlap_analyzable,
                    [(f"r4{suffix}", 2018), (f"status2020_{name}", 2020)],
                ).items()
            }
        )
        row["event_feasibility_tier"] = feasibility_tier(
            int(row["development_events_n"]),
            int(row["outcome_specific_nonoverlap_events_n"]),
        )
        rows.append(row)

    results = pd.DataFrame(rows).sort_values(
        ["event_feasibility_tier", "outcome_specific_nonoverlap_events_n"],
        ascending=[True, False],
    )
    csv_path = QA_OUT / "multoutcome_feasibility_matrix.csv"
    results.to_csv(csv_path, index=False)

    display_columns = [
        "outcome",
        "provisional_diagnostic_proximity_level",
        "development_with_follow_up_n",
        "development_events_n",
        "outcome_specific_nonoverlap_with_follow_up_n",
        "outcome_specific_nonoverlap_events_n",
        "outcome_specific_nonoverlap_event_pct",
        "common_nonoverlap_with_follow_up_n",
        "common_nonoverlap_events_n",
        "event_feasibility_tier",
    ]
    display = results[display_columns].copy()
    display["outcome_specific_nonoverlap_event_pct"] = display[
        "outcome_specific_nonoverlap_event_pct"
    ].map(lambda value: f"{value:.1f}")
    md_path = QA_OUT / "14_multoutcome_feasibility_readout.md"
    md_path.write_text(
        "# CHARLS multi-outcome feasibility gate\n\n"
        "This scan is descriptive and precedes all cross-outcome model fitting. "
        "Diagnostic-proximity labels are provisional and require independent clinical adjudication.\n\n"
        "## Event feasibility matrix\n\n"
        + markdown_table(display)
        + "\n\n## Prespecified interpretation rules\n\n"
        "- Primary-feasible: at least 200 development and 200 non-overlapping evaluation events.\n"
        "- Exploratory-feasible: at least 100 events in each cohort but one or both below 200.\n"
        "- Below 100 evaluation events: do not use as a primary cross-outcome benchmark.\n"
        "- Outcome-specific non-overlap is the direct extension of the digestive-disease design.\n"
        "- Common-union exclusion is reported as a stricter comparability sensitivity analysis.\n"
        "- Event counts alone do not determine inclusion; wording consistency, endpoint validity, "
        "diagnostic overlap, missingness, and precision must also be reviewed.\n",
        encoding="utf-8",
    )
    print(display.to_string(index=False))
    print(f"\nCSV: {csv_path}")
    print(f"Readout: {md_path}")


if __name__ == "__main__":
    main()
