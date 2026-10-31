# Stage 4 — Prediction Validation

This validation package checks that the deployed Streamlit prediction engine correctly uses the existing six trained models and reproduces the notebook's saved inference outputs where row-level references exist.

## What this test does

- Loads the six existing model files using the project's pinned scikit-learn and XGBoost versions.
- Recreates the Stable-FCI notebook's exact stratified 80/20 split (`test_size=0.20`, `random_state=42`) from the original 4,920-row encoded dataset.
- Recomputes Stable-FCI test accuracy, macro precision, macro recall, macro F1 and top-3 accuracy for all three models, then checks them against the saved experiment summaries.
- Checks the Streamlit application's symptom-to-feature mapping and top-three output against direct predictions from the original model objects, using held-out records across all 41 Stable-FCI disease classes.
- Replays all 42 STAR test symptom rows. For STAR Logistic Regression, it compares all 41 class probabilities and predicted labels against the original notebook's archived `prediction_probabilities.csv`.
- Checks the STAR application input mapping and top-three output for all 42 rows with all three STAR models.

No model is fitted, no causal discovery algorithm is run, and no model artifact is changed.

## Exact source data files required

The supplied model ZIPs do not contain the two input matrices needed to reconstruct the complete held-out test inputs. Keep these original files on your computer and pass their absolute paths to the script:

1. **Stable-FCI:** the exact `train_encoded.csv` used in the Stable-FCI notebook. It must contain 4,920 rows, 41 `disease_` one-hot columns and the same 132 symptom columns.
2. **STAR:** the exact `Test_Only_Symptoms.csv` used in the STAR notebook. It must contain 42 rows and the same 132 symptom columns.

The test does not need the STAR training files or the STAR `Test_Only_Diseases.csv`, because it validates inference parity against the notebook's saved predictions rather than recalculating STAR classification metrics.

Do not substitute a similarly named or modified dataset. The script intentionally stops if row counts, columns, binary values or split dimensions do not match.

## Run on Ubuntu

Activate the same environment that passed Stage 2:

```bash
source .venv-app/bin/activate
```

Confirm versions:

```bash
python -c "import sklearn, xgboost; print(sklearn.__version__, xgboost.__version__)"
```

Expected: `1.9.0 3.1.3`.

Run the validation (replace the two file paths with the actual locations):

```bash
python scripts/validate_stage4.py \
  --stable-dataset "/absolute/path/to/train_encoded.csv" \
  --star-test-symptoms "/absolute/path/to/Test_Only_Symptoms.csv"
```

On success, the script creates:

- `reports/stage4_validation/stage4_validation_summary.json`
- `reports/stage4_validation/stage4_validation_details.csv`

A non-zero exit code means validation did not pass. Read the first reported failure; do not edit the model artifacts to bypass it.

## Archived STAR reference case

The package also contains:

- `../../reference_reports/star_ci_reference_prediction_probabilities.csv`: all 42 original STAR Logistic Regression test-row outputs and 41 probability columns.
- `../../reference_reports/star_ci_sample0_reference.json`: the notebook's printed first example, including its three observed symptoms and saved probability vector.

The script uses the complete CSV for row-by-row validation.

## Tolerances and interpretation

- Probability comparisons use an absolute tolerance of `1e-10`.
- The Stable-FCI aggregate metrics are compared against the saved notebook JSON outputs.
- Stable-FCI and STAR retain their different train/test protocols; this validation does not try to unify them.
- A PASS means the app reproduces the saved model inference behavior on the supplied experiment inputs. It does not prove clinical validity or that `predict_proba` scores are calibrated probabilities.

## Files preserved

The original notebooks, trained models, feature configurations, and model hashes are not modified by the validation script. Input CSV files remain at their original locations and are not copied into the application directory.
