# Reproducibility and Validation

## Environment pins

The serialized models were verified using scikit-learn 1.9.0 and XGBoost 3.1.3. Streamlit is pinned to 1.66.1 in the application requirements. Use the checked-in `requirements.txt` and do not casually upgrade the serialized-model dependencies.

## Reproduction assets

- Cleaned experiment notebooks: `research/notebooks/`.
- Archived Stable-FCI outputs: `research/results/stable_fci/`.
- Archived STAR outputs: `research/results/star_ci/`.
- Six saved fitted models: `artifacts/`.
- Ordered input/class configurations: `configurations/`.
- Model SHA-256 manifest: `artifact_manifest.json`.
- Results SHA-256 manifest: `research/results/RESULTS_FILE_MANIFEST.json`.

The cleaned notebooks preserve all original code-cell source according to `notebook_source_integrity.json`; their saved cell outputs and execution counters were cleared to reduce notebook size and remove machine-specific output text. The archived numeric reports and plots are preserved separately under `research/results/`.

## Checks run in the project directory

The recorded successful run reported:

- `verify_configuration.py`: PASS; 132 master symptoms, 37 Stable-FCI features, 106 STAR features, 41 disease classes, and six artifact hashes verified.
- `verify_models.py`: PASS; all six models loaded, accepted their required feature counts (37 or 106), and returned 41-class probability vectors with row sums of 1.0.
- `pytest`: 20 passed.
- `verify_notebook_integrity.py`: PASS; 55 Stable-FCI code cells and 37 STAR code cells retained with matching source hashes.
- Streamlit: server started successfully from the permanent project directory.

The Stage 4 experiment-input validator also passed:

- Stable-FCI: both distinct structurally valid source CSV candidates reproduced the archived test metrics for all three models; app parity was checked on 41 representative rows per model.
- STAR: all three models passed the 42-row external input inference checks; Logistic Regression matched the archived 42 × 41 probability matrix.

The Stage 4 outputs are documented in `docs/application/VALIDATION_RESULTS.md`.

## Reproduction limitations

The raw train/test datasets are not bundled. A reader can run the application using the saved models but must obtain the data separately to rerun the complete original experiments. The exact original Stable-FCI local source file could not be uniquely identified because two different candidate files reproduced the same archived metrics under the recorded validator.

For STAR, disease and symptom matrices are separate files without patient IDs. The notebook verifies equal row counts and assumes the two files retain matching row order; the files themselves cannot prove patient-level row alignment.

## Model serialization warning

Loading the saved XGBoost joblib artifacts emits a non-fatal upstream warning about Python pickle snapshots and cross-version serialization. In the verified environment all models loaded and passed inference. Keep XGBoost pinned to 3.1.3 and preserve the original artifact files. A future portability project may convert/re-export models only through a separate, explicitly validated migration process.
