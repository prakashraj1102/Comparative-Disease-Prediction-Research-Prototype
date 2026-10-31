# Dataset Description and Provenance

## Dataset structure

The experiments use a binary disease-symptom classification dataset with:

- 4,920 rows in the Stable-FCI source dataset.
- 41 one-hot disease-label columns.
- 132 binary symptom columns.
- 173 columns in total (41 disease columns + 132 symptom columns).

Each row represents one example and has exactly one active disease indicator in the validated source candidate. Symptom indicators use 0/1 values.

The STAR experiment uses separately supplied disease and symptom matrices:

- Training: 4,920 rows.
- External test: 42 rows.
- 41 disease columns and 132 symptom columns in the corresponding matrices.

The STAR notebook validates binary values, missingness, one-hot disease rows, row counts between the disease and symptom matrices, and disease-column correspondence. It does not have patient identifiers in these split files, so its row-order alignment assumption cannot be independently proven from those files alone.

## Feature identifiers

The project preserves original symptom identifiers and column order. This matters because each fitted classifier expects a fixed feature schema. For example, `fluid_overload` and `fluid_overload.1` remain separate identifiers in the STAR configuration. They have not been merged or renamed in the published application configuration.

The application may display a more readable label, but the internal feature identifier must remain the exact identifier expected by the model.

## Dataset files are not bundled

Raw training/test CSV datasets are not included in this repository package. The application does not require those raw datasets at inference time because it uses the saved models and their feature configurations.

To rerun the original experiments or repeat the full Stage 4 validation, a researcher must obtain the corresponding source data separately and prepare files in the schemas expected by the notebooks/validation script.

## Provenance caveat for Stable-FCI

Two different local copies of the Stable-FCI encoded dataset were found during the validation audit. Both had the expected 4,920 × 173 structure and both reproduced the archived test metrics for all three Stable-FCI classifiers under the validation script. Their SHA-256 hashes differ, so the available evidence does not establish which physical file was the unique original file loaded for the historical notebook run.

This uncertainty is recorded rather than silently resolved. It does not affect application inference, which uses the six saved model files and their frozen feature configurations, not the raw training CSV.

## Source citation and redistribution

Before publishing a raw dataset or claiming a particular dataset source/license, verify the original source and its redistribution terms. The current archived experiment outputs do not by themselves establish permission to redistribute the source data or derived model artifacts. Keep restricted or private data out of the public repository.
