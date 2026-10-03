import os
"""Evidence-based clinical adjudication of laboratory diagnostic proximity."""

from pathlib import Path

import pandas as pd


PROJECT = Path(os.environ["ANALYSIS_ROOT"])
QA = PROJECT / "qa_logs"

RUBRIC = {
    0: "no_diagnostic_overlap",
    1: "supportive_or_exclusionary_only",
    2: "partial_diagnostic_component",
    3: "direct_defining_diagnostic_overlap",
}

ROWS = [
    {
        "outcome": "hypertension",
        "level": 0,
        "overlapping_study_measures": "none in blood block; measured blood pressure is in the non-laboratory reference block",
        "adjudication": "Hypertension diagnosis is based on blood-pressure measurement, not the routine blood block used here.",
        "evidence_source": "WHO hypertension guideline",
        "evidence_url": "https://www.who.int/teams/noncommunicable-diseases/guidelines-for-hypertension",
    },
    {
        "outcome": "dyslipidemia",
        "level": 3,
        "overlapping_study_measures": "total cholesterol, LDL cholesterol, HDL cholesterol, triglycerides",
        "adjudication": "The added block directly contains the lipid measurements used to identify and classify dyslipidemia.",
        "evidence_source": "ESC/EAS dyslipidaemia guideline",
        "evidence_url": "https://www.escardio.org/guidelines/clinical-practice-guidelines/all-esc-practice-guidelines/dyslipidaemias/",
    },
    {
        "outcome": "diabetes",
        "level": 3,
        "overlapping_study_measures": "plasma glucose; HbA1c in full-assay sensitivity",
        "adjudication": "Plasma glucose and HbA1c are direct diagnostic tests for diabetes, subject to confirmation rules.",
        "evidence_source": "ADA Standards of Care in Diabetes 2026",
        "evidence_url": "https://diabetesjournals.org/care/article/49/Supplement_1/S27/163926/2-Diagnosis-and-Classification-of-Diabetes",
    },
    {
        "outcome": "cancer",
        "level": 1,
        "overlapping_study_measures": "complete-blood-count components and routine chemistry",
        "adjudication": "Routine blood results may prompt evaluation and can support selected hematologic-cancer diagnoses, but the broad cancer endpoint generally requires imaging, pathology, or cancer-specific tests.",
        "evidence_source": "US National Cancer Institute diagnostic overview",
        "evidence_url": "https://www.cancer.gov/about-cancer/diagnosis-staging/diagnosis",
    },
    {
        "outcome": "chronic_lung_disease",
        "level": 0,
        "overlapping_study_measures": "none",
        "adjudication": "The included blood measures do not confirm chronic obstructive lung disease; spirometry is required to confirm airflow obstruction.",
        "evidence_source": "GOLD spirometry guidance",
        "evidence_url": "https://goldcopd.org/gold-spirometry-guide/",
    },
    {
        "outcome": "liver_disease",
        "level": 1,
        "overlapping_study_measures": "platelets and nonspecific routine chemistry",
        "adjudication": "The primary block omits ALT, AST, alkaline phosphatase, bilirubin, albumin, INR, and viral serology. Included measures are supportive at most for this broad endpoint.",
        "evidence_source": "AASLD approach to abnormal liver enzymes",
        "evidence_url": "https://www.aasld.org/liver-fellow-network/core-series/back-basics/how-approach-elevated-liver-enzymes",
    },
    {
        "outcome": "heart_disease",
        "level": 0,
        "overlapping_study_measures": "lipids and glucose are risk factors; no troponin or natriuretic peptide",
        "adjudication": "The survey endpoint spans multiple heart problems. The included routine blood block does not establish them; contemporary confirmation relies on disease-specific clinical, ECG, imaging, or invasive testing.",
        "evidence_source": "ESC chronic coronary syndrome guideline",
        "evidence_url": "https://www.escardio.org/guidelines/clinical-practice-guidelines/all-esc-practice-guidelines/chronic-coronary-syndromes/",
    },
    {
        "outcome": "stroke",
        "level": 0,
        "overlapping_study_measures": "none",
        "adjudication": "The included blood measures can describe risk or acute physiology but do not establish stroke; diagnostic imaging is central.",
        "evidence_source": "AHA/ASA acute ischemic stroke guideline 2026",
        "evidence_url": "https://professional.heart.org/en/guidelines-statements/2026-guideline-for-the-early-management-of-patients-with-acute-ischemic-strokestr0000000000000513",
    },
    {
        "outcome": "kidney_disease",
        "level": 2,
        "overlapping_study_measures": "creatinine and cystatin C; blood urea nitrogen is supportive",
        "adjudication": "Creatinine and cystatin C directly contribute to GFR estimation, but a single measurement does not establish chronic kidney disease without chronicity or other markers of kidney damage. The survey endpoint is broader than CKD alone.",
        "evidence_source": "KDIGO 2024 CKD guideline",
        "evidence_url": "https://kdigo.org/wp-content/uploads/2024/03/KDIGO-2024-CKD-Guideline.pdf",
    },
    {
        "outcome": "digestive_disease",
        "level": 1,
        "overlapping_study_measures": "hemoglobin, white-cell count, CRP and routine chemistry",
        "adjudication": "These measures may indicate inflammation, anemia, or complications, but they do not define the heterogeneous stomach/digestive endpoint, for which endoscopy, imaging, pathology, or disease-specific tests are often used.",
        "evidence_source": "NIDDK digestive diagnostic tests",
        "evidence_url": "https://www.niddk.nih.gov/health-information/diagnostic-tests",
    },
    {
        "outcome": "psychiatric_problem",
        "level": 1,
        "overlapping_study_measures": "routine laboratory tests may exclude medical mimics",
        "adjudication": "Routine blood tests can help identify alternative medical causes but do not define the broad emotional or psychiatric endpoint.",
        "evidence_source": "American Psychiatric Association overview",
        "evidence_url": "https://www.psychiatry.org/patients-families/psychiatric-medications-an-overview-1",
    },
    {
        "outcome": "memory_related_disease",
        "level": 1,
        "overlapping_study_measures": "routine laboratory tests may exclude reversible causes; no Alzheimer-specific biomarker is included",
        "adjudication": "Routine tests can support differential diagnosis, but the included panel does not diagnose dementia or Alzheimer disease.",
        "evidence_source": "US National Institute on Aging diagnostic overview",
        "evidence_url": "https://www.nia.nih.gov/health/alzheimers-symptoms-and-diagnosis/how-alzheimers-disease-diagnosed",
    },
    {
        "outcome": "arthritis_or_rheumatism",
        "level": 1,
        "overlapping_study_measures": "CRP is supportive for inflammatory activity; RF and anti-CCP are absent",
        "adjudication": "CRP is nonspecific and the endpoint combines multiple arthritic conditions; the included panel does not directly define arthritis or rheumatism.",
        "evidence_source": "American College of Rheumatology rheumatoid arthritis resources",
        "evidence_url": "https://rheumatology.org/rheumatoid-arthritis-guideline",
    },
    {
        "outcome": "asthma",
        "level": 0,
        "overlapping_study_measures": "none; total white-cell count is not a blood eosinophil count",
        "adjudication": "Asthma confirmation is based on characteristic symptoms and variable expiratory airflow. The included blood block does not confirm the diagnosis.",
        "evidence_source": "GINA 2026 strategy report",
        "evidence_url": "https://ginasthma.org/wp-content/uploads/2026/05/GINA-2026-Strategy-Report-WMS.pdf",
    },
]


def markdown_table(data: pd.DataFrame) -> str:
    headers = list(data.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in data.itertuples(index=False, name=None):
        values = [str(value).replace("|", "\\|") for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    data = pd.DataFrame(ROWS)
    data["proximity_label"] = data["level"].map(RUBRIC)
    data["analysis_group"] = data["level"].map(
        {
            0: "no_direct_overlap",
            1: "supportive_only",
            2: "partial_diagnostic_component",
            3: "direct_diagnostic_overlap",
        }
    )
    columns = [
        "outcome",
        "level",
        "proximity_label",
        "analysis_group",
        "overlapping_study_measures",
        "adjudication",
        "evidence_source",
        "evidence_url",
    ]
    data = data[columns].sort_values(["level", "outcome"], ascending=[False, True])
    data.to_csv(QA / "clinical_diagnostic_proximity_adjudication.csv", index=False)

    display = data[
        [
            "outcome",
            "level",
            "analysis_group",
            "overlapping_study_measures",
            "adjudication",
            "evidence_source",
        ]
    ]
    text = """# Clinical adjudication of diagnostic proximity

## Locked rubric

- Level 3: an included blood measure is a direct defining diagnostic test for the survey outcome.
- Level 2: an included measure is a direct diagnostic component for an important disease construct, but a single measurement is insufficient because persistence, another specimen, or another modality is required.
- Level 1: included measures are supportive, assess complications, or help exclude alternatives, but do not define the outcome.
- Level 0: the included blood block has no material diagnostic role; associations with risk or prognosis do not count as diagnostic proximity.

The classification evaluates the measures actually present in the CHARLS blood block, not laboratory medicine in general. It is an evidence-informed secondary classification of the completed benchmark. The available audit trail does not establish blinded independent clinical adjudication or prospective registration; neither is claimed.

## Adjudicated outcomes

""" + markdown_table(display) + """

## Consequence for the analysis hierarchy

Diabetes and dyslipidemia are the only direct-overlap comparators. Kidney disease is retained as a separate partial-component comparator rather than pooled with them. All other outcomes have either supportive-only or no direct overlap. This refinement prevents the kidney result from being used to manufacture or refute a simple direct-overlap gradient.
"""
    (QA / "16_clinical_diagnostic_proximity_adjudication.md").write_text(
        text, encoding="utf-8"
    )
    print(data[["outcome", "level", "analysis_group"]].to_string(index=False))


if __name__ == "__main__":
    main()
