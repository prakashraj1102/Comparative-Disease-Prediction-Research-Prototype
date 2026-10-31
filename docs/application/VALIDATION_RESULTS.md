# Stage 4 inference validation results

**Overall result: PASS.** The user ran the Stage 4 validation in the pinned Ubuntu environment. All six saved classifiers passed the configured inference checks. No model training or causal discovery was performed.

## Stable-FCI

Two structurally valid 4,920-row candidate source datasets were evaluated independently. Both reproduced the archived held-out metrics for Logistic Regression, Random Forest, and XGBoost, and each passed application parity checks on 41 representative records per model.

The two candidates were distinct files with different content hashes. The available evidence establishes that both reproduce the saved metrics within the validation script's configured tolerance; it does not establish which copy was originally loaded when the notebook was executed. This provenance uncertainty does not affect runtime inference because the application uses the saved model files and feature configurations, not the source training CSV.

## STAR CI-selected

The 42-row external symptom input passed application inference checks for all three classifiers. STAR Logistic Regression additionally matched the archived probability output for all 42 rows and all 41 disease classes.

## Additional checks

- Single-symptom input: PASS.
- Multiple-symptom input: PASS.
- All 132 symptoms selected: PASS.
- Empty-selection handling: PASS.
- All six models loaded and returned 41-class probability outputs: PASS.

## Warning

XGBoost emitted an upstream serialized-snapshot/pickle compatibility warning while loading its joblib artifact. The warning was non-fatal; all six models loaded and passed inference checks with the pinned scikit-learn 1.9.0 and XGBoost 3.1.3 versions. Keep XGBoost pinned and do not modify or convert the saved artifacts as part of this application stage.

## Scope

This validation demonstrates inference consistency with the saved experiment artifacts on the supplied test inputs. It does not establish clinical validity, probability calibration, generalisation to external populations, or superiority of one causal feature-selection approach over the other. The original Stable-FCI and STAR evaluation protocols remain different.
