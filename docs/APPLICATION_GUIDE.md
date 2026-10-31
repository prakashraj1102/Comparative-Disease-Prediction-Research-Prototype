# Application Setup and Usage

## Purpose

The Streamlit application accepts a binary symptom checklist and obtains separate predictions from the Stable-FCI and STAR CI-selected model pipelines.

## Environment

The checked-in `requirements.txt` pins the serialized-model compatibility versions:

- scikit-learn 1.9.0
- XGBoost 3.1.3
- Streamlit 1.66.1
- nbformat 5.11.1 for notebook integrity verification

A Python 3.14 environment was used for the recorded verification. Create a clean environment from the repository root:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Verify before launching

```bash
python scripts/verify_configuration.py
python scripts/verify_models.py
python -m pytest -q
```

If the research-notebook integrity script and research-results manifest script are present:

```bash
python scripts/verify_notebook_integrity.py
python scripts/verify_research_results.py
```

## Launch

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit (normally `http://localhost:8501`).

## User workflow

1. Search or browse the master symptom list.
2. Select symptoms that are present. Selected symptoms are encoded as 1; unchecked symptoms are encoded as 0.
3. Choose a classifier independently for Stable-FCI and STAR CI-selected.
4. Run prediction.
5. Review the top-three outputs for each approach.
6. Review overlap, rank alignment, and approach-specific differences.
7. Export the comparison table if required.

## Important input detail

The two feature sets do not include the same symptoms. The application shows feature coverage and projects the user's master symptom selection into each exact trained feature order. A symptom not selected by an approach does not enter that model's input.

The identifiers `fluid_overload` and `fluid_overload.1` are intentionally preserved as distinct model features in the current STAR schema. This is a known data-schema issue deferred from the research experiment; the application does not merge them.

## Interpretation

Scores are raw `predict_proba` outputs represented as percentages. They have not been established as calibrated probabilities. The application is a research demonstration, not a clinical decision-support or diagnostic tool.
