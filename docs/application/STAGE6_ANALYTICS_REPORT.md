# Stage 6 — Prediction Analytics and Visual Comparison

## Purpose

Stage 6 improves the interpretation of the two top-three prediction lists for one user input. It does not retrain or alter Stable-FCI, STAR CI-selected, any classifier, preprocessing, or feature selection.

## Added analytics

1. **Top-ranked agreement:** whether both selected classifiers return the same disease at rank 1.
2. **Top-three overlap:** shared diseases and Jaccard overlap (`intersection / union`) between the two top-three disease sets.
3. **Exact rank matches:** number of rank positions at which both approaches list the same disease.
4. **Mean absolute rank shift:** average absolute difference between Stable-FCI and STAR ranks for diseases present in both lists. It is unavailable when there are no shared diseases.
5. **Shared and approach-specific disease lists.**
6. **Ordinal rank visual:** separate black and gray bars indicate list positions. A longer bar means a better rank only; the bars do not represent probability values.
7. **Detailed output table and CSV export:** includes each disease in the union of the lists, both ranks, each separate model score, top-three membership, and rank shift where shared.

## Interpretation restrictions

- The two model score columns are kept separate.
- No score averaging, ensembling, calibration claim, or cross-model score difference metric has been added.
- Jaccard overlap and rank-shift statistics describe only the two selected classifiers for the current symptom input.
- These per-input analytics are not model accuracy, population-level reliability, clinical confidence, or evidence that either causal discovery approach is superior.
- Existing Stable-FCI and STAR experiment protocols remain unchanged.

## Files changed

- `app.py`: analytics tabs, metrics, rank visualisation, detailed table, and CSV download.
- `core/prediction_analytics.py`: pure calculation functions for overlap and rank alignment.
- `tests/test_prediction_analytics.py`: unit tests for complete agreement, reordered rankings, partial overlap, no overlap, empty lists, and malformed duplicate entries.
- `README.md`: Stage 6 documentation and validation instructions.

No model artifact, feature configuration, disease-label configuration, input processor, model loader, prediction engine, or result-comparison implementation was modified.

## Validation performed in the build environment

- Unit tests: 19 passed, 1 skipped.
- Python compilation: passed.
- Configuration and model-artifact hash audit: passed.
- No Streamlit browser-rendering test was performed in the build environment because Streamlit is not installed there. The user's pinned environment has Streamlit 1.66.1; a local visual check remains part of manual acceptance.
- No model loading, model fitting, or causal discovery was run as part of Stage 6.

## Manual acceptance checklist

1. Launch `streamlit run app.py` in the project's pinned virtual environment.
2. Generate a prediction using any selected pair of classifiers.
3. Confirm the analytics section shows top-rank agreement, top-three overlap, exact rank matches, and mean rank shift.
4. Open the Overlap overview, Rank alignment, and Detailed outputs & export tabs.
5. Confirm a disease missing from one top-three list displays as "Not in top three".
6. Download and inspect `stable_fci_star_ci_prediction_comparison.csv`.
7. Repeat with a symptom input producing no shared top-three diseases, if available.
8. Confirm the black-and-white theme remains legible and the prediction results are unchanged.
