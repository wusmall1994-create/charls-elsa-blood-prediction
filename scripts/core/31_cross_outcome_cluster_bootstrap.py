import os
"""Participant-clustered bootstrap for cross-outcome AUROC-increment contrasts."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


PROJECT = Path(os.environ["ANALYSIS_ROOT"])
DATA = PROJECT / "data_private" / "multoutcome_primary_predictions.csv.gz"
QA = PROJECT / "qa_logs"
N_BOOTSTRAP = 2000
SEED = 20260831

PRIMARY = [
    "hypertension",
    "arthritis_or_rheumatism",
    "heart_disease",
    "digestive_disease",
    "chronic_lung_disease",
    "stroke",
]
DIRECT_OVERLAP = ["dyslipidemia", "diabetes"]
DIRECT_OVERLAP_WITHOUT_DIABETES = ["dyslipidemia"]


def delta_auc(y: np.ndarray, base: np.ndarray, blood: np.ndarray, weights=None) -> float:
    return float(
        roc_auc_score(y, blood, sample_weight=weights)
        - roc_auc_score(y, base, sample_weight=weights)
    )


def main() -> None:
    predictions = pd.read_csv(DATA, dtype={"person_id": str})
    predictions = predictions.loc[predictions["evaluable_5y"].astype(bool)]
    outcome_data = {}
    all_ids = set()
    for outcome, group in predictions.groupby("outcome"):
        wide = group.pivot(
            index=["person_id", "event"], columns="model", values="predicted_risk_5y"
        ).reset_index()
        ids = wide["person_id"].astype(str).to_numpy()
        all_ids.update(ids)
        outcome_data[outcome] = {
            "ids": ids,
            "y": wide["event"].astype(int).to_numpy(),
            "base": wide["base_discrete_time"].to_numpy(),
            "blood": wide["blood_enhanced_discrete_time"].to_numpy(),
        }

    global_ids = np.asarray(sorted(all_ids))
    global_index = {person_id: index for index, person_id in enumerate(global_ids)}
    for values in outcome_data.values():
        values["global_positions"] = np.asarray(
            [global_index[person_id] for person_id in values["ids"]], dtype=int
        )

    point = {
        outcome: delta_auc(values["y"], values["base"], values["blood"])
        for outcome, values in outcome_data.items()
    }

    def group_mean(source: dict[str, float], outcomes: list[str]) -> float:
        return float(np.mean([source[outcome] for outcome in outcomes]))

    point_estimands = {
        "mean_delta_auc_primary_nonoverlap": group_mean(point, PRIMARY),
        "mean_delta_auc_direct_overlap": group_mean(point, DIRECT_OVERLAP),
        "direct_overlap_minus_primary": group_mean(point, DIRECT_OVERLAP) - group_mean(point, PRIMARY),
        "mean_delta_auc_direct_overlap_without_diabetes": group_mean(
            point, DIRECT_OVERLAP_WITHOUT_DIABETES
        ),
        "direct_overlap_without_diabetes_minus_primary": group_mean(
            point, DIRECT_OVERLAP_WITHOUT_DIABETES
        )
        - group_mean(point, PRIMARY),
    }

    rng = np.random.default_rng(SEED)
    draws = {estimand: [] for estimand in point_estimands}
    probabilities = np.full(len(global_ids), 1 / len(global_ids))
    for _ in range(N_BOOTSTRAP):
        counts = rng.multinomial(len(global_ids), probabilities)
        sampled_deltas = {}
        for outcome, values in outcome_data.items():
            weights = counts[values["global_positions"]]
            sampled_deltas[outcome] = delta_auc(
                values["y"], values["base"], values["blood"], weights=weights
            )
        sampled_estimands = {
            "mean_delta_auc_primary_nonoverlap": group_mean(sampled_deltas, PRIMARY),
            "mean_delta_auc_direct_overlap": group_mean(
                sampled_deltas, DIRECT_OVERLAP
            ),
            "direct_overlap_minus_primary": group_mean(sampled_deltas, DIRECT_OVERLAP)
            - group_mean(sampled_deltas, PRIMARY),
            "mean_delta_auc_direct_overlap_without_diabetes": group_mean(
                sampled_deltas, DIRECT_OVERLAP_WITHOUT_DIABETES
            ),
            "direct_overlap_without_diabetes_minus_primary": group_mean(
                sampled_deltas, DIRECT_OVERLAP_WITHOUT_DIABETES
            )
            - group_mean(sampled_deltas, PRIMARY),
        }
        for estimand, value in sampled_estimands.items():
            draws[estimand].append(value)

    rows = []
    for estimand, estimate in point_estimands.items():
        values = np.asarray(draws[estimand])
        rows.append(
            {
                "estimand": estimand,
                "estimate": estimate,
                "ci_2_5": float(np.quantile(values, 0.025)),
                "ci_97_5": float(np.quantile(values, 0.975)),
                "n_cluster_ids": len(global_ids),
                "bootstrap_iterations": N_BOOTSTRAP,
            }
        )
    results = pd.DataFrame(rows)
    results.to_csv(QA / "cross_outcome_cluster_bootstrap.csv", index=False)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
