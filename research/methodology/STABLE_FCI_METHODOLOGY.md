# Stable-FCI Methodology

## 1. Objective

The Stable-FCI experiment uses conditional-independence tests to estimate a Partial Ancestral Graph (PAG) over disease and symptom variables. Symptoms that remain adjacent to at least one disease node in the final PAG are then used as features for multiclass classification.

The implementation is contained in `research/notebooks/01_stable_fci_causal_feature_selection.ipynb`. Archived evidence is in `research/results/stable_fci/`.

## 2. Data and split

The encoded dataset contains 4,920 rows, 41 disease indicator columns, and 132 binary symptom columns (173 columns total). The notebook creates a stratified train/test split before causal discovery:

- Test fraction: 20%.
- Random state: 42.
- Training rows: 3,936.
- Test rows: 984.
- Per disease: 96 training and 24 test examples in the archived split.

Stable-FCI is run on the training rows only. The held-out test data is not passed to the causal discovery stage or symptom-selection stage.

## 3. CI test and Stable-FAS skeleton search

The CI test is the G² (likelihood-ratio) test for binary variables, with significance threshold `alpha = 0.05`. In the stable Fast Adjacency Search implementation, the adjacency structure is frozen at each conditioning depth. Independent tests at a depth are dispatched in parallel, and edges are removed only after the tests for that depth have completed. This preserves the stable, order-independent adjacency update semantics described in the notebook.

Configured bounds:

- Maximum conditioning depth: 10.
- Parallel workers: all available cores (`n_jobs = -1` in the experiment configuration).

The algorithm stores separating-set information and a detailed CI-test log.

## 4. Background knowledge

Before orientation, disease nodes are assigned tier 0 and symptom nodes tier 1. The background-knowledge constraint forbids a definite `Symptom -> Disease` orientation. This is an explicit domain assumption encoded in the experiment; it is not learned from the CI tests alone.

## 5. Collider detection, Possible-D-SEP, and orientation

The notebook applies collider detection using the separating sets from Stable-FAS, then runs a bounded Possible-D-SEP edge-removal stage. Possible-D-SEP uses:

- Maximum path length: 10.
- Maximum conditioning-set size: 10.

The notebook then applies the FCI orientation rules R1–R10 (including the knowledge-aware routines used in the implementation) iteratively until no further change is made or the implementation's 50-iteration safety cap is reached. The result is represented as a PAG, which may contain partially oriented/uncertain edge marks; a PAG is not a fully directed DAG.

## 6. Symptom-selection rule

A symptom is selected if it has at least one remaining edge, of any endpoint-mark type, to at least one disease node in the final PAG. The final saved list contains 37 of the original 132 symptoms.

- Retained: 37 / 132 = 28.03%.
- Reduced: 95 / 132 = 71.97%.

The selected list is stored at `research/results/stable_fci/feature_selection/stable_fci_selected_symptoms.csv`.

Adjacency in the estimated PAG under the specified assumptions and search bounds is the selection criterion. It should not be rephrased as proof that a symptom causes a disease or that every retained edge is a direct causal effect.

## 7. Downstream classifiers

Three multiclass classifiers are trained on the Stable-FCI-selected 37-feature matrix:

- Logistic Regression (L2 regularisation; `lbfgs` solver).
- Random Forest.
- XGBoost (`multi:softprob` objective in the saved experiment implementation).

A parallel all-symptoms baseline uses all 132 symptom columns. Each classifier/feature branch is tuned on the training partition using five-fold stratified cross-validation and macro F1. The held-out test set is reserved for final evaluation.

Recorded best settings for the Stable-FCI-selected branch:

| Model | Best settings | Best CV macro F1 |
|---|---|---:|
| Logistic Regression | `C = 10.0` | 0.7805 |
| Random Forest | `n_estimators = 300`, `max_depth = None` | 0.7807 |
| XGBoost | `n_estimators = 500`, `max_depth = 9`, `learning_rate = 0.01`; `subsample = 0.8`, `colsample_bytree = 0.8` | 0.7804 |

The models are fitted on the full 3,936-row training partition with their selected settings and evaluated on the 984-row held-out test partition.

## 8. Recorded selected-feature test results

| Model | Accuracy | Macro precision | Macro recall | Macro F1 | Top-3 accuracy |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 80.79% | 76.70% | 80.79% | 77.49% | 92.89% |
| Random Forest | 80.89% | 76.75% | 80.89% | 77.56% | 92.89% |
| XGBoost | 80.79% | 76.70% | 80.79% | 77.49% | 92.89% |

Full precision values and additional metrics are in the per-model JSON summaries under `research/results/stable_fci/model_evaluation/`.

## 9. Evaluation nuance

Stable-FCI feature selection is fixed once using the complete training partition before the downstream classifier cross-validation. The held-out test set is not used in feature selection or hyperparameter tuning. However, the feature-selection step is not repeated separately within each CV fold. Fully nested cross-validation would be required to evaluate the entire feature-selection-plus-classifier-tuning pipeline within the inner/outer resampling process.

The archived all-symptom baseline reports perfect test metrics for all three model families. This result deserves separate dataset-structure and leakage investigation and should not be treated as evidence of clinical generalisation.
