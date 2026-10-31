# Research notebooks

This folder contains publication copies of the two research notebooks used in the comparative disease-prediction project.

## Notebooks

1. `01_stable_fci_causal_feature_selection.ipynb` — Stable-FCI-based causal feature discovery followed by Logistic Regression, Random Forest, and XGBoost classification.
2. `02_star_ci_selected_feature_screening.ipynb` — disease-specific CI screening, union of retained symptoms, and downstream Logistic Regression, Random Forest, and XGBoost classification, with an all-symptoms baseline.

## Why outputs are cleared

The original notebooks contained machine-specific paths, stale execution traces, dataset previews, and large verbose CI logs. The publication copies therefore retain the complete code-cell source but clear all saved cell outputs and execution counters. This is an output-cleaning operation, not a change to algorithm code or model training logic. Archived metrics, configurations, and reproducibility evidence should be published separately under the project's research results and documentation folders.

The original uploaded notebook files should be retained by the project owner as untouched archival originals.

## Data requirements

The notebooks do not include raw data. Obtain the permitted input files separately and follow the expected filenames documented in the publication note at the top of each notebook. Review the dataset's licence and redistribution conditions before sharing any data or derived files.

## Running

Open a notebook in JupyterLab or VS Code after creating a dedicated Python environment and installing compatible dependencies. The Stable-FCI notebook requires a `causal-learn` version that provides the FCI orientation rules used by the notebook. The fitted-model application has its own pinned runtime requirements; do not assume the notebook and application environments are interchangeable.

Running the notebooks repeats CI screening/causal discovery and model fitting. It is not necessary merely to launch the separately packaged Streamlit application.

## Important Stable-FCI input note

The current Stable-FCI notebook loader detects disease fields by the `disease_` prefix. Although the configuration cell still contains legacy `DISEASE_COLUMN` / `DISEASE_COLUMNS` options and explanatory comments for a categorical-label alternative, those settings are not used by the current loader. Use the audited one-hot disease-column format for this experiment.

## Integrity check

From the publication-pack root, run:

```bash
python scripts/verify_notebook_integrity.py
```

The script validates notebook format, file hashes, code-cell source hashes, and confirms that outputs remain cleared. It requires `nbformat`.
