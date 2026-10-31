# Saved model artifact provenance

The six joblib files below were copied from the two experiment output archives without retraining or modifying the fitted models.

| Approach | Classifier | Repository path | Size (bytes) | SHA-256 |
|---|---|---|---:|---|
| stable_fci | logistic_regression | `artifacts/stable_fci/logistic_regression/stable_fci_logistic_regression.joblib` | 14943 | `b5c6fc20f5ef32c2571e7ade686ba318ded94ef22dd2f0163f54abb662bb4653` |
| stable_fci | random_forest | `artifacts/stable_fci/random_forest/stable_fci_random_forest.joblib` | 16234641 | `7f8be33ccd5695801f3a9299fbc0c8bf486c78a863bea9d0d72f0dff930272c9` |
| stable_fci | xgboost | `artifacts/stable_fci/xgboost/stable_fci_xgboost.joblib` | 19636772 | `751061026fcb1c197746e0ae16d3e91b6775310841ba21f728a6840fd26423fc` |
| star_ci | logistic_regression | `artifacts/star_ci/logistic_regression/logistic_regression_model.joblib` | 39199 | `15ae446934e422649ca4667ec094a8f1a6df6bf9c361975b2c88f8bd1c6b34cd` |
| star_ci | random_forest | `artifacts/star_ci/random_forest/random_forest_model_ci_selected.joblib` | 21821473 | `1be408c13bf4e6833f85fb7204cf5c004765a590718f4d5a69f54ce01b95943d` |
| star_ci | xgboost | `artifacts/star_ci/xgboost/xgboost_model_ci_selected.joblib` | 6277271 | `1a7d4cd01a0b12838cbccfa05ac8de0dc7c5d03ef3c8e2a51d55c3d087809d00` |

Required serialized-model runtime: scikit-learn 1.9.0 and XGBoost 3.1.3. See the root `artifact_manifest.json` for the machine-readable manifest.
