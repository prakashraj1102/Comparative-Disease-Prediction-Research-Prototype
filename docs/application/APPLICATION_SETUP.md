# Application setup and usage

This document describes how to install and run the comparative disease-prediction application. The repository-level `README.md` will provide the broader research overview.

## What the application deploys

- Stable-FCI classifiers: Logistic Regression, Random Forest, and XGBoost using the saved 37-feature schema.
- STAR CI-selected classifiers: Logistic Regression, Random Forest, and XGBoost using the saved 106-feature schema.
- Shared symptom input: 132 binary symptom identifiers and 41 disease classes.
- The same user input is independently projected into each approach's saved feature order.

The application does not rerun causal discovery, retrain classifiers, merge the feature sets, average model scores, or create an ensemble.

## Environment

Use Python 3.14 and install the pinned model dependencies. From the repository root:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Check the key versions:

```bash
python -c "import sklearn, xgboost, streamlit; print(sklearn.__version__, xgboost.__version__, streamlit.__version__)"
```

Expected versions: scikit-learn 1.9.0, XGBoost 3.1.3, and Streamlit 1.66.1.

## Launch

```bash
streamlit run app.py
```

Open the local address printed by Streamlit (normally `http://localhost:8501`).

## Prediction workflow

1. Search or browse the 132 symptoms.
2. Select the symptoms to encode as present (1); unchecked symptoms are encoded as absent (0).
3. Select an ML classifier independently for Stable-FCI and STAR CI-selected.
4. Request predictions from both approaches.
5. Review the top-three model outputs, overlap, rank alignment, and downloadable comparison table.

A symptom not retained by an approach's feature-selection process is not used by that approach. The application reports the feature coverage separately. The identifiers `fluid_overload` and `fluid_overload.1` remain distinct in the saved schema.

## Verification

Run from the repository root in the pinned environment:

```bash
python scripts/verify_configuration.py
python scripts/verify_models.py
python -m pytest -q
```

For the full experiment-input validation, see `STAGE4_VALIDATION_GUIDE.md` in this folder. The original datasets are not bundled; the validation script accepts local paths and does not copy or modify the datasets.

## Interpretation and limitations

The displayed percentages are each model's `predict_proba` outputs expressed as percentages. They are model-estimated scores, not calibrated clinical probabilities or diagnostic certainty. The two experiments retain their original and different evaluation protocols; the interface does not present their historical metrics as a controlled head-to-head benchmark.

This is a research prototype, not a medical diagnostic system. It must not be used to make healthcare decisions. The input format has no separate unknown/missing symptom state; unchecked means absent, consistent with the binary training representation.

The XGBoost joblib objects emit a non-fatal serialized-snapshot compatibility warning in the verified environment. Keep XGBoost pinned to 3.1.3 and do not convert or overwrite the saved model files as part of routine application setup.
