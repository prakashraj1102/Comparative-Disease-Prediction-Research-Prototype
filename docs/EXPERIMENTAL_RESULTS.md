# Experimental Results

## How to read these results

The values below are copied from the archived experiment summaries. They are reported in their original contexts. Stable-FCI and STAR CI-selected used different test protocols and sample sizes, so their metrics are **not a controlled head-to-head comparison**.

## Stable-FCI-selected classifiers

The Stable-FCI experiment used a stratified 80/20 split of 4,920 rows: 3,936 training rows and 984 held-out test rows. Each of the 41 disease classes contributed 96 training rows and 24 test rows.

| Classifier | Test accuracy | Macro precision | Macro recall | Macro F1 | Top-3 accuracy | Macro OVR ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 80.79% | 76.70% | 80.79% | 77.49% | 92.89% | 98.92% |
| Random Forest | 80.89% | 76.75% | 80.89% | 77.56% | 92.89% | 98.93% |
| XGBoost | 80.79% | 76.70% | 80.79% | 77.49% | 92.89% | 98.90% |

Selected-feature model settings recorded in the archived summaries:

| Classifier | Best recorded settings | Best 5-fold CV macro F1 |
|---|---|---:|
| Logistic Regression | `C = 10.0` | 0.7805 |
| Random Forest | `n_estimators = 300`, `max_depth = None` | 0.7807 |
| XGBoost | `n_estimators = 500`, `max_depth = 9`, `learning_rate = 0.01` | 0.7804 |

The all-symptom Stable-FCI notebook baseline reports 100% test metrics for all three classifiers. This striking result should be treated cautiously and investigated in future work for dataset construction effects, feature/label relationships, and possible leakage or unusually strong dataset regularities. It should not be interpreted as evidence of clinical performance.

## STAR CI-selected classifiers

The STAR experiment used 4,920 training rows and a separate 42-row external test set. The reported selected-feature metrics are:

| Classifier | Test accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 | Top-3 accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 97.62% | 98.78% | 98.78% | 98.37% | 97.62% | 97.62% |
| Random Forest | 97.62% | 98.78% | 98.78% | 98.37% | 97.62% | 97.62% |
| XGBoost | 97.62% | 98.78% | 98.78% | 98.37% | 97.62% | 97.62% |

The archived Logistic Regression summary also reports Top-5 accuracy of 100.00% and log loss of 0.0772. The selected `C` was 1.0, with best 5-fold CV macro F1 of approximately 0.9963.

The 42-row test set is very small for 41 classes: the saved classification report shows support of one example for most classes and two for one class. These results are therefore highly sensitive to individual cases and should be treated as preliminary external-test evidence, not as a reliable estimate of population-level performance.

The STAR all-symptom baseline is also present in the archived comparison table. It reports 100% across the listed metrics for Logistic Regression and 97.62% accuracy for the Random Forest and XGBoost baselines. Interpret those comparisons in light of the same small test size.

## Stage 4 application validation

The application was validated in the pinned environment using the archived models and supplied source test files:

- Stable-FCI: both structurally valid source dataset candidates reproduced the archived held-out metrics for Logistic Regression, Random Forest, and XGBoost. Application parity was checked on 41 representative rows per model.
- STAR: all three models passed inference checks on the 42-row symptom test input. STAR Logistic Regression additionally matched the archived 41-class probability vectors for all 42 rows.
- Input edge cases: single-symptom, multiple-symptom, all-symptom, and empty-selection handling passed.

The final integrated project was subsequently tested from its permanent repository directory: configuration verification passed, all six models loaded and returned 41-class probability vectors, and the automated suite reported 20 passed tests. The XGBoost serialized-snapshot warning was non-fatal and is documented in the application provenance notes.

## Evidence locations

See:

- `research/results/stable_fci/model_evaluation/`
- `research/results/stable_fci/model_comparison/`
- `research/results/star_ci/model_evaluation/`
- `research/results/star_ci/ci_screening/`
- `docs/application/VALIDATION_RESULTS.md`
- `research/results/RESULTS_FILE_MANIFEST.json`
