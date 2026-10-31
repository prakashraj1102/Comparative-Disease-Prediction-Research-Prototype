# Data and Access Notes

Raw training and test datasets are not included in this public repository. The application runs from its saved model artifacts and configuration files and does not require the raw experiment CSV files at inference time.

## Stable-FCI experiment input schema

The notebook expects an encoded table with 4,920 rows and 173 columns in the archived run: 41 one-hot disease columns and 132 binary symptom columns. It then creates a stratified 80/20 train/test split (random state 42).

## STAR experiment input schema

The notebook expects four separate CSV inputs:

- training disease indicators;
- training symptom indicators;
- test disease indicators;
- test symptom indicators.

The archived run contains 4,920 training rows and 42 test rows, with 41 disease columns and 132 symptom columns. Disease/symptom rows must correspond in the same order. The notebook checks row counts but these input files do not contain patient IDs to prove row-level alignment.

## Obtain the data

Before adding download links or dataset citations, verify the dataset's original source, version, citation requirements, and redistribution license. Do not upload any dataset unless its terms allow public redistribution. Do not add patient-identifiable or otherwise restricted data.

## Re-running experiments

The published notebooks are cleaned copies of the experiment notebooks. To rerun them, obtain the source data separately, review and edit the notebook's input-path configuration for the local environment, and install compatible research dependencies. The application `requirements.txt` is specifically pinned for the serialized deployment models; it should not be assumed to be a complete dependency lockfile for re-running the full causal discovery notebooks.
