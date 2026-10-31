# Comparative Disease Prediction Using Stable-FCI and STAR CI-Selected Features

## Overview

This project investigates two different conditional-independence (CI)-based symptom-selection approaches and uses their resulting feature sets in multiclass disease-classification models. The project then deploys the saved classifiers in a Streamlit application so the same user-selected symptoms can be evaluated independently by both approaches.

The two research pipelines are intentionally kept separate:

- **Stable-FCI pipeline:** Stable-FCI is applied to the training partition of a binary disease-symptom dataset. Symptoms adjacent to disease nodes in the resulting Partial Ancestral Graph (PAG) are selected for downstream classification.
- **STAR CI-selected pipeline:** a custom two-stage, disease-specific G² conditional-independence screening procedure retains symptoms for each disease. The disease-specific retained sets are unioned into one feature set for multiclass classification.

Each approach has three saved classifiers: Logistic Regression, Random Forest, and XGBoost. The application loads these six saved estimators; it does not rerun causal discovery or retrain the models.

## Research question

Given the same symptom input, how do predictions from classifiers trained on the Stable-FCI-selected feature set differ from predictions made by classifiers trained on the STAR CI-selected feature set?

The application makes those differences inspectable. It does not average the model scores or create an ensemble.

## Main components

| Component | Contents |
|---|---|
| `research/notebooks/` | Cleaned publication copies of the two experiment notebooks. Code-cell source is preserved; saved outputs and execution counters are cleared. |
| `research/results/` | Curated, source-faithful graphs, edge tables, CI-screening logs, metrics, confusion matrices, plots, and feature-selection records. |
| `research/methodology/` | Detailed descriptions of both feature-selection procedures and their ML evaluation protocols. |
| `artifacts/` | The six fitted CI-selected model artifacts used by the application. |
| `configurations/` | Master symptom identifiers, selected feature orders, disease-label mapping, model registry, and audit metadata. |
| `core/` and `app.py` | Input preparation, model loading, inference, result comparison, analytics, and Streamlit interface. |
| `scripts/` and `tests/` | Configuration/model verification, Stage 4 validation, notebook/results integrity checks, and automated tests. |

## Scope

The application accepts a binary symptom checklist, projects that input into each approach's saved feature schema, and displays the top-three model outputs. Users can select the classifier independently for Stable-FCI and STAR CI-selected.

This project is a research prototype. It is not a clinical diagnostic system and has not been clinically validated.

## Key experiment facts

| Item | Stable-FCI | STAR CI-selected |
|---|---:|---:|
| Disease classes | 41 | 41 |
| Original symptom variables | 132 | 132 |
| Selected symptom variables | 37 | 106 |
| Training rows | 3,936 | 4,920 |
| Test rows | 984 | 42 |
| Downstream classifiers | 3 | 3 |

The training/testing protocols differ. Consequently, the historical metrics must not be interpreted as a controlled, head-to-head performance comparison. See [Experimental results](EXPERIMENTAL_RESULTS.md) and [Limitations](LIMITATIONS_AND_FUTURE_WORK.md).

## How to explore the repository

Start with this overview, then read the two methodology documents and the experimental-results summary. For implementation details, inspect the notebooks and the application code. The archived result files provide supporting evidence for the reported figures and metrics.
