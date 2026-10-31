# STAR CI-Selected Methodology

## 1. Objective

The STAR CI-selected experiment applies a custom two-stage, disease-specific G² conditional-independence screening procedure. It retains symptoms separately for each disease, then unions those disease-specific retained lists into one common feature matrix for multiclass classification.

The implementation is in `research/notebooks/02_star_ci_selected_feature_screening.ipynb`. Archived evidence is in `research/results/star_ci/`.

## 2. Data inputs and protocol

The notebook loads separate disease and symptom matrices for training and testing:

- Training: 4,920 rows.
- External test: 42 rows.
- Disease classes: 41.
- Original symptom features: 132.

The notebook checks that all fields are binary and non-missing, that every disease row is one-hot, that corresponding disease/symptom files have equal row counts, and that train/test disease columns can be aligned. Because these files do not carry patient IDs, equal row counts are necessary but not sufficient to prove that row i in the disease matrix corresponds to row i in the symptom matrix. The workflow assumes both matrices were derived from the same original row ordering.

CI screening is run on the training disease and symptom matrices only. The test set is not used to select symptoms.

## 3. G² statistic

For a 2×2 contingency table with observed counts `O` and expected counts `E`, the notebook computes:

`G² = 2 × Σ O_ij × ln(O_ij / E_ij)`

with zero-count cells contributing zero. The p-value is obtained from the chi-square survival function using the table's degrees of freedom. The significance threshold is `alpha = 0.05`.

The notebook records each test's statistic, degrees of freedom, p-value, status, reason, and decision. Degenerate strata are handled explicitly. If a conditional test has no usable strata, it is marked invalid and treated as dependent so an unevaluable test does not remove a disease-symptom candidate.

## 4. Two-stage screening for each disease-symptom pair

For each disease indicator `D` and symptom indicator `S`:

### Stage 1 — Unconditional test

Test `D ⟂ S` with no conditioning variable.

- If p-value `>= 0.05`, the pair is labelled independent and the candidate relationship is removed; Stage 2 is skipped.
- If p-value `< 0.05`, continue to Stage 2.
- Invalid tests are treated as dependent, so they do not remove the candidate.

### Stage 2 — One paired conditioning variable at a time

For each symptom `S_j` in the fixed original symptom-column order, the notebook constructs a paired binary variable:

`P_j = D AND S_j`

It tests `D ⟂ S | P_j` with one conditioning variable at a time. The conditioning variable is stratified into 0 and 1. Valid 2×2 G² statistics and degrees of freedom from usable strata are summed; the combined p-value is computed from the chi-square distribution.

- At the first valid independent result (`p >= 0.05`), the D–S candidate is removed and the remaining Stage 2 sequence stops.
- If all Stage 2 tests remain dependent, the symptom is retained for that disease.
- If a test is invalid because all strata are unusable, it is treated as dependent and the sequence continues.

This is the specific custom screening procedure implemented in the notebook. It is not the same as a general PC/FCI search over all conditioning subsets of increasing size.

## 5. Disease-specific star representations and feature union

The procedure produces one retained symptom set per disease. The notebook renders a disease-centred star-shaped graph for each disease, with retained symptoms as connected nodes. The graph is a visual representation of the screening result. A display arrow from disease to symptom is not an additional causal direction learned by the CI test.

A multiclass classifier needs one common feature matrix, so the notebook takes the union of all per-disease retained sets. It preserves the original symptom-column order when building this union.

The archived union contains 106 of the 132 original symptoms:

- Retained: 106 / 132 = 80.30%.
- Reduced: 26 / 132 = 19.70%.

The saved list is `research/results/star_ci/feature_selection/selected_symptoms.txt`.

## 6. Classifiers and tuning

Three classifier families are evaluated on the 106-feature CI-selected matrix and, separately, on a 132-feature all-symptom baseline:

- Logistic Regression, with `C` selected by five-fold stratified `GridSearchCV` using macro F1.
- Random Forest, with tree depth and minimum leaf size tuned by five-fold stratified `GridSearchCV` using macro F1.
- XGBoost, with integer-encoded training labels and a saved class-order check to preserve `predict_proba` column alignment.

The same train/test inputs and five-fold stratified CV object are used for the selected-feature and all-feature branches within this STAR experiment. The 42-row external test set is used for test metrics after tuning on training data.

## 7. Recorded CI-selected test results

| Model | Accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 | Top-3 accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 97.62% | 98.78% | 98.78% | 98.37% | 97.62% | 97.62% |
| Random Forest | 97.62% | 98.78% | 98.78% | 98.37% | 97.62% | 97.62% |
| XGBoost | 97.62% | 98.78% | 98.78% | 98.37% | 97.62% | 97.62% |

The Logistic Regression summary also records Top-5 accuracy of 100.00%, log loss of 0.0772, selected `C = 1.0`, and best CV macro F1 of approximately 0.9963.

## 8. Interpretation caveats

The external test set has only 42 observations across 41 classes. The archived classification report shows support of one observation for most disease classes and two for one class. Per-class metrics and overall performance estimates are therefore highly sensitive to individual test cases.

The STAR procedure's retained relationships are CI-screening decisions under the specified binary encoding, alpha, paired-variable construction, fixed search order, and invalid-test policy. The method should not be described as producing a complete PAG or proving causal effects.
