# Routine blood measures in CHARLS and ELSA

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23562784.svg)](https://doi.org/10.5281/zenodo.23562784)

Analysis code for **Incremental predictive value of routine blood panels for physician-diagnosis reports in two ageing cohorts**.

The principal analysis compares a seven-measure blood panel with core and richer non-laboratory models across seven subsequent physician-diagnosis-report outcomes. Exploratory analyses assess informative laboratory comparators, measurement availability, biochemical restriction, landmark prediction, frozen model transport, target recalibration and illustrative net benefit. A 14-measure CHARLS benchmark supplies supporting results across 13 modeled outcomes; cancer is descriptive.

## Data access and private workspace

Obtain authorized CHARLS data from https://charls.pku.edu.cn/ and ELSA data from https://www.elsa-project.ac.uk/accessing-elsa-data. Registration and data-use conditions are controlled by the data custodians. This repository contains source code and citation metadata; it excludes participant records, individual predictions, private fitted objects, manuscripts and editing logs.

Set `ANALYSIS_ROOT` to a writable private workspace, `CHARLS_DATA_DIR` to the authorized CHARLS releases and `ELSA_DATA_DIR` to the authorized ELSA releases. Create `data_private`, `qa_logs` and `results` under `ANALYSIS_ROOT`; analysis stages also use `analysis_work`. Keep the code checkout separate from this private workspace. Detailed wave and field mappings are defined in `scripts/core/`.

## Software

Python 3.12 is used for the analyses. `requirements.txt` specifies compatible package ranges; `requirements-tested.txt` records the package versions in the environment used to check the released code and run the paired pipeline-refitting analysis. `pyreadstat` is an optional raw-file import dependency listed in `requirements.txt` and is not needed for the trusted intermediate-file checks. R is used for the transport decision-curve figure; package versions and commands are in `R-dependencies.md`. No software installation is performed by the scripts.

## Analysis modules and execution order

| Stage | Location and entry points | Inputs and outputs |
| --- | --- | --- |
| CHARLS construction and supporting benchmark | `scripts/core/23`–`35`; shared functions in `02`, `04`, `07`; blood audit in `01` | Authorized releases to private cohorts, predictions and aggregate summaries |
| ELSA construction and nested evaluation | `scripts/core/37`–`40` | Authorized interview/nurse releases to private people/interval files and predictions |
| Seven-outcome common panel | `scripts/core/41_build_common_panel_cohorts.py`, `42_replication_sensitivity_suite.py` | Cohorts to paired common-panel predictions and aggregate metrics |
| Frozen transport | `scripts/analyses/transport/45_transport_and_interpretation.py`, `verify_transport.py` | Source pipelines and target cohorts to private transferred predictions and aggregate checks |
| Mortality, missing outcomes and updating | `scripts/analyses/competing_recalibration/46`–`51` | Cohorts/predictions to competing-event and outcome-assignment estimates |
| Reference/panel decision curves | `scripts/analyses/decision_curves/55_decision_curves.py` | Paired predictions to threshold-specific net benefit |
| Simplified comparators and restrictions | `scripts/analyses/panel_comparisons/61`–`64` | Matched cohorts to simplified/full predictions, restriction and landmark results, fixed-penalty refitting checks |
| Calibration, Brier scores and availability contrast | `scripts/analyses/panel_performance/68`–`70` | Paired predictions to prediction-error, calibration and between-stratum net-benefit summaries |
| HbA1c comparators and common-core transport | `scripts/analyses/panel_performance/71_comparator_transport.py` | Matched cohorts to nested/alternative-assay comparisons, frozen and cross-fitted recalibrated predictions |
| Paired pipeline refitting | `scripts/analyses/panel_performance/72_refitting_validation.py` | Private cohorts to 100 paired bootstrap repetitions, tuning metadata and descriptive metric summaries |
| Transport decision-curve figure | `scripts/figures/` | Private evaluated predictions to aggregate curve CSV, then PNG/SVG |

Inspect each script's arguments and input paths before execution. These are dependent analysis stages, not a single-command pipeline. Module 62 runs separately for CHARLS/ELSA and diabetes/lipid-related reports, with its stability option where applicable. Run its verification and export stages after the model jobs. Module 68 precedes 69; module 71 uses common-panel cohorts and saved module-62 diabetes predictions. Only load trusted pickle and model files.

### HbA1c comparisons and recalibration

```text
python scripts/analyses/panel_performance/71_comparator_transport.py --data-private <private-data-directory> --common-panel-dir <common-panel-directory> --panel-predictions-dir <panel-predictions-directory> --output-private <private-model-output-directory> --output-dir <aggregate-output-directory>
```

The common-panel directory supplies ELSA wave-4 and wave-6 people/interval files. The private-data directory supplies CHARLS diabetes/lipid development and evaluation files and ELSA outer-fold assignments. The panel-predictions directory supplies `CHARLS_diabe_all_predictions.pkl` and `ELSA_diabe_all_predictions.pkl`. Tasks `hba1c`, `frozen` and `evaluate` can be run separately with `--task`.

Glucose plus HbA1c versus panel plus HbA1c is a nested addition of six measures. Seven-measure panel versus glucose plus HbA1c compares alternative assay combinations. Transport uses common-core non-laboratory information; recalibration estimates intercept-only or intercept-and-slope functions outside each scored target fold, using identical target folds across models. Net-benefit thresholds are illustrative, with 10% the focus and 5%/20% supporting.

### Paired pipeline-refitting sensitivity analysis

```text
python scripts/analyses/panel_performance/72_refitting_validation.py --data-private <private-data-directory> --common-panel-dir <common-panel-directory> --code-dir <repository-root> --output-dir <aggregate-output-directory> --repetitions 100 --workers 6
```

Each repetition repeats preprocessing and five-fold penalty selection. ELSA uses grouped nested evaluation; duplicated original identities remain together in all folds. Transport independently resamples source and target participants and re-estimates intercept-and-slope calibration outside scored target folds. Output files contain aggregate replicate metrics, tuning metadata and descriptive summaries, with no individual predictions or fitted models. An existing repetition file is reused to allow interrupted runs to resume; use an empty output directory for a fresh run. Repetition numbers and date-shaped random seeds are statistical parameters.

### Aggregate figure source and rendering

```text
python scripts/figures/export_transport_decision_source.py --evaluated-dir <private-model-output-directory> --output-dir <aggregate-curve-directory>
Rscript scripts/figures/transport_decision_curves.R <aggregate-curve-directory>/decision_curve_source.csv <figure-output-stem>
```

The exporter reads `CHARLS_to_ELSA_diabe_evaluated.pkl` and `ELSA_to_CHARLS_diabe_evaluated.pkl` produced by module 71, exporting only aggregate threshold curves and input checksums. The R script draws frozen/recalibrated curves in both directions. Best default is the maximum of assess-all and assess-none. Curves are point estimates; conditional intervals and refitting ranges are reported separately.

## Interpretation and reproducibility boundaries

Endpoints reflect subsequent recognition and reporting of physician diagnoses, not biological disease onset. Diagnostic-proximity classification is post hoc; the study was not prospectively registered. Conditional paired bootstrap intervals are pointwise and unadjusted for multiplicity. Full pipeline-refitting medians, percentiles and sign counts are descriptive sensitivity results, not equivalence tests, clinical benefit probabilities or independent population validation.

The released scripts have syntax, entry-point and aggregate-result checks. These checks and the completed paired refitting analysis do not constitute an independent end-to-end rerun from the authorized raw releases.

## Citation and licence

Liang, H., Wu, X., Qiu, R. & Huang, S. (2026). *Routine blood measures in CHARLS and ELSA: analysis code*, version 1.0.0. Zenodo. https://doi.org/10.5281/zenodo.23562784

Use `CITATION.cff` for software citation metadata. The MIT licence applies to code only. CHARLS and ELSA source and derived participant-level data remain subject to their custodians' access conditions.
