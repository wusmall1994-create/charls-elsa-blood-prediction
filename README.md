# Routine blood measures in CHARLS and ELSA

Statistical code for evaluating the incremental predictive value of a seven-measure blood panel across seven subsequent diagnosis-report outcomes. The analysis compares non-laboratory and simplified laboratory references, assesses calibration and illustrative net benefit, and evaluates frozen-model transport between cohorts.

## Data and environment

Obtain authorized CHARLS and ELSA data from https://charls.pku.edu.cn/ and https://www.elsa-project.ac.uk/accessing-elsa-data. This repository distributes no participant records, fitted individual predictions, manuscript files or aggregate study results. Python package requirements are in requirements.txt; the analysis environment uses Python 3.12.

Set ANALYSIS_ROOT to a writable private analysis directory, CHARLS_DATA_DIR to the authorized CHARLS releases and ELSA_DATA_DIR to the authorized ELSA releases. Create data_private, qa_logs and results under ANALYSIS_ROOT. Analysis modules create analysis_work for aggregate outputs. Shared code imports resolve within this repository. Keep all source and derived participant-level records private.

## Analysis modules

| Location | Function |
| --- | --- |
| scripts/core/ | Shared cohort, model and response-weighting functions; CHARLS 14-measure benchmark; ELSA and common-panel evaluation |
| scripts/analyses/transport/ | Frozen-model transfer and paired performance comparisons |
| scripts/analyses/competing_recalibration/ | Mortality ascertainment, competing events, incomplete-outcome scenarios and cross-fitted recalibration |
| scripts/analyses/decision_curves/ | Net benefit for common-panel and transported predictions |
| scripts/analyses/panel_comparisons/ | Simplified indicators, biochemical restriction, landmark prediction and refitting stability |
| scripts/analyses/panel_performance/ | Three-model calibration, Brier scores, decision curves, glucose-stratum contrasts and figures |

## Execution dependencies

1. CHARLS: 23 examines outcome feasibility, 24 constructs cohorts, 25 fits the benchmark and 26 evaluates paired uncertainty. Modules 27–35 provide the assay, nonlinear, cross-outcome, mortality, response-weighting and participant-overlap checks. Modules 02, 04 and 07 supply shared functions; module 01 audits blood availability.
2. ELSA: 37 checks mappings, 38 constructs cohorts, 39 fits nested models and 40 verifies saved outputs.
3. Common panel: 41_build_common_panel_cohorts.py constructs the common-panel cohorts; 42_replication_sensitivity_suite.py fits and evaluates paired models and sensitivities.
4. Transport: run 45_transport_and_interpretation.py and verify_transport.py.
5. Competing events and updating: run 46–49, then 51_verify_competing.py. Decision curves in 55 require the common-panel predictions and transported/recalibrated outputs.
6. Panel comparisons: run 61_feasibility.py, then 62_panel_analyses.py for each cohort/outcome and its stability run; finish with 63_verify_panel.py and 64_export_panel.py.
7. Performance: run 68_panel_performance.py, then 69_panel_figures.py. Module 70_glucose_stratum_net_benefit.py evaluates exploratory between-stratum differences in complete-minus-simplified net benefit using 20,000 participant bootstrap samples within glucose-availability strata. Models remain paired and predictions fixed; sampling is not stratified by outcome. The 10% threshold is the focus of interpretation, with 5% and 20% supporting. Intervals are pointwise and not multiplicity-adjusted.

```text
python scripts/analyses/panel_performance/70_glucose_stratum_net_benefit.py --predictions <private-ELSA_diabe_all_predictions.pkl> --people <private-elsa_w6_people.pkl> --output-dir <aggregate-output-directory>
```

Only load trusted private pickle files. Module 70 exports aggregate JSON/CSV estimates and input checksums; it does not export individual records.

Inspect each script's arguments and input paths before execution. These are modular analysis stages, not a single automated command. Random seeds, wave-specific mappings, outcome definitions and scoring conventions are specified in the scripts. Date-shaped random seeds are numerical parameters, not version labels.

## Interpretation and reproducibility

The endpoints are subsequent diagnosis reports. The seven-outcome common-panel comparison is principal; the 13 modeled CHARLS outcomes provide supporting evidence, with cancer descriptive. Simplified-panel, restriction, landmark, transport, recalibration and decision-curve analyses are exploratory. Diagnostic-proximity classification is post hoc. The study was not prospectively registered. Conditional bootstrap intervals do not represent full retraining uncertainty; refitting checks are reported separately.

Syntax and internal references are checked. Saved-result and saved-model verification do not constitute an independent end-to-end rerun from raw releases. Such a rerun has not been completed.

## Privacy and licence

Do not commit source data, derived individual records, private model objects, credentials or local environment files. Code is licensed under MIT; the CHARLS and ELSA data remain subject to their custodians' access conditions.
