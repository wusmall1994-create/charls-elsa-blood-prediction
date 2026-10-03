# Routine blood measures in CHARLS and ELSA: statistical analysis code

This repository contains the statistical analysis code used to evaluate the incremental predictive value of routine blood measures for subsequent chronic disease reports in CHARLS and ELSA.

## Scope

The repository contains code for cohort construction, discrete-time models, nested cross-validation, paired bootstrap comparisons, transport and recalibration, competing-event analyses, incomplete-outcome scenarios, simplified-panel and biochemical-restriction analyses, landmark analyses, calibration, Brier scores, and decision curves.

No manuscript files, participant-level data, fitted participant-level predictions, personal identifiers, or aggregate study results are included.

## Data access

CHARLS and ELSA are third-party datasets. Researchers must obtain the data directly from the respective custodians and comply with their registration and data-use conditions. The authors cannot redistribute these data.

- CHARLS: https://charls.pku.edu.cn/
- ELSA: https://www.elsa-project.ac.uk/accessing-elsa-data

## Configuration

Set these environment variables before running the scripts:

- `ANALYSIS_ROOT`: a writable local project directory. Derived participant-level files and outputs are written below this directory and must remain private.
- `CHARLS_DATA_DIR`: directory containing the authorized CHARLS releases.
- `ELSA_DATA_DIR`: directory containing the authorized ELSA releases.

The scripts retain the file names and wave-specific variable mappings used in the analysis. Because the custodians distribute data under access conditions, the repository does not include test extracts or synthetic copies of the source files.

## Code organization

- `scripts/core/`: cohort construction, primary models, validation, sensitivity analyses, ELSA replication, and statistical audits.
- `scripts/extensions/`: model transport, recalibration, competing-event analyses, decision curves, simplified-panel comparisons, biochemical restrictions, landmark analyses, and performance figures.

Numbered filenames indicate the analytical sequence. Some later scripts require private intermediate files created by earlier scripts. Manuscript-generation and document-editing programs have been excluded.

## Environment

Python dependencies are listed in `requirements.txt`. The analysis was developed for Python 3.12. Create an isolated environment and install the listed packages before running the code.

## Privacy and data governance

Do not commit source data, derived participant-level files, model objects containing participant-level information, or local `.env` files. The `.gitignore` excludes the principal restricted file types and directories, but users remain responsible for complying with the CHARLS and ELSA terms of use.

## Licence

Code is released under the MIT License. The CHARLS and ELSA data remain subject to their own access and use conditions.
