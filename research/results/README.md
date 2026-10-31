# Archived experimental results

This directory contains selected, source-faithful outputs from the original Stable-FCI and STAR CI-selected experiment archives. The CSV/JSON/text/PDF/image contents have not been recalculated or edited. Two large CSV logs are gzip-compressed for repository storage; decompress them to recover the original CSV byte stream.

## Stable-FCI

- `causal_discovery/`: PAG/graph figures, edge endpoint tables, sepsets, and the complete CI-test log.
- `feature_selection/`: the selected symptom list.
- `experiment_metadata/`: source dataset shape/column metadata and train/test split metadata.
- `model_comparison/`: selected-feature vs all-symptom comparison and shared figure.
- `model_evaluation/`: per-classifier test/train confusion matrices, CV outputs, summaries, tuning plots, feature coefficients/importances, and baseline comparisons.

The archived `ci_tests.csv` was approximately 503 MB as plain CSV and is stored as `causal_discovery/ci_tests.csv.gz`. It contains CI-test metadata (`depth`, tested variables, conditioning set, p-value, independence decision), not patient-level rows.

## STAR CI-selected

- `ci_screening/`: summary and detailed disease-symptom CI screening records. The detailed 21.7 MB CSV is gzip-compressed.
- `feature_selection/`: exact selected symptom list and all available per-disease STAR graph PDFs, plus archived PNG previews where available.
- `experiment_metadata/`: archived experiment configuration and metric summary.
- `model_evaluation/`: model comparison tables, classification report, saved prediction probability table, confusion matrix, comparison plots, overfitting-gap plot, and learning curves.

## Deliberate exclusions

- Serialized model files (`.joblib`) are not duplicated here. The six deployed CI-selected model artifacts belong under the repository's `artifacts/` directory.
- All-symptom baseline `.joblib` files are omitted; their saved evaluation tables and plots are retained.
- The raw source training/test datasets are not included. See `data/README.md` when available for source and access instructions.
- The 41 per-disease STAR graph PDFs are included; the archive provided PNG previews for 39 of the 41 graphs.
- Several similarly named Stable-FCI comparison CSV exports contained the same reported rows with formatting/blank-row differences. The canonical `model_comparison_summary.csv` is retained; the duplicate variants are omitted.

## Important interpretation notes

- Stable-FCI and STAR were run under different training/testing protocols. Their metrics must not be interpreted as a head-to-head comparison on a shared test set.
- The Stable-FCI archive records 4,920 rows, a 3,936/984 stratified split, and `random_state=42`.
- The STAR archive records 4,920 training rows and 42 external test rows.
- The source archives contain a STAR selected-feature identifier `fluid_overload.1`; it is preserved exactly. No correction or merging is performed here.
- Results are archived as generated. This package does not endorse a clinical interpretation of model scores.
