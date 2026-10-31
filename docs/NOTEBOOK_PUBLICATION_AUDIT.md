# Notebook Publication Audit

## Scope

The two user-provided notebooks were inspected without executing any code. Publication copies were created. Original notebook files were not modified.

## Findings

### 01_stable_fci_causal_feature_selection.ipynb

- Original file: `Stable_FCI_Logistic_Regression_RandomForest_XGBoost(1)(1).ipynb`
- Original size: 654,380 bytes
- Publication copy size: 128,836 bytes
- Original cells: 97 (including 55 code cells)
- Saved outputs cleared: 85
- Output payload removed: 517,506 bytes
- Code cell source preserved byte-for-byte at the cell-source level: **Yes**
- Secret-pattern scan of source cells: `{'email_like': 0, 'api_key_assignment': 0, 'private_key_header': 0}`
- Absolute local paths in source cells: 0

### 02_star_ci_selected_feature_screening.ipynb

- Original file: `Star_ThreeModel_Comparison(2).ipynb`
- Original size: 16,151,490 bytes
- Publication copy size: 113,478 bytes
- Original cells: 62 (including 37 code cells)
- Saved outputs cleared: 54
- Output payload removed: 14,896,617 bytes
- Code cell source preserved byte-for-byte at the cell-source level: **Yes**
- Secret-pattern scan of source cells: `{'email_like': 0, 'api_key_assignment': 0, 'private_key_header': 0}`
- Absolute local paths in source cells: 0

## Privacy and output review

- The original code/Markdown source contained no matches for common email-address, API-key/secret-assignment, or private-key-header patterns. This is a targeted automated scan, not a guarantee that no sensitive information exists anywhere in the wider project.
- The original saved outputs contained machine-specific `/home/...` paths. All saved outputs were cleared, including text, dataframe previews, plots, warnings, and verbose CI-test logs. This also removed the very large STAR CI screening display (approximately 14.9 MB of serialized output payload).
- No raw CSV dataset was embedded as a file in either notebook; however, dataframe previews were present in saved outputs and were removed.
- No code source contained an absolute `/home`, `/mnt`, `/media`, or `/Users` path.

## Publication changes made

1. Added a reproducibility/publication note describing required data files and the fact that data are not bundled.
2. Cleared saved outputs and execution counts from every code cell. This removes local path strings, large logs, dataset row previews, and stale notebook output state.
3. Normalized kernel display names to `Python 3`.
4. Updated the STAR notebook title/summary to reflect all three classifiers; no code cells were changed.
5. Added a notebook-folder README.

## Important methodological notes to document in the full repository

- Stable-FCI is configured with `MAX_DEPTH = 10` and `MAX_PATH_LENGTH = 10`. Describe it as the bounded configuration used in this experiment; do not imply unlimited search.
- The Stable-FCI experiment uses a stratified 80/20 train/test split from the 4,920-row one-hot dataset. Its final feature list contains 37 symptoms.
- The STAR experiment uses four separate train/test disease/symptom files and, in the archived run, 4,920 training records and 42 external test records. Its union feature list contains 106 symptoms.
- The two experiments use different test protocols. Do not present their evaluation scores as a controlled head-to-head comparison on the same held-out records.
- Probability outputs should be called model scores or estimated probabilities unless calibration has been assessed.
- The STAR test set is small (42 records); clearly disclose this when discussing performance.
- Inputs and model artefacts may have separate licence/redistribution restrictions. Verify these before making the repository public.


## Additional code/documentation discrepancy identified

The Stable-FCI configuration cell still includes `DISEASE_COLUMN` and `DISEASE_COLUMNS` options and comments describing a categorical-label input path. The current loading code does not consume those settings; it selects disease columns by the `disease_` prefix. The publication note now explicitly documents the behavior and instructs readers to use the audited one-hot schema. The code itself was left unchanged.

## Not changed

- No code-cell source was changed.
- No algorithm, CI-test, feature-selection, training, prediction, or evaluation logic was changed.
- No notebook was executed or retrained during this preparation.
- The original uploaded notebooks remain untouched in the working source area.

## Remaining work before public release

- Add the exact notebook dependency/environment specifications after a separate compatibility audit. The Stable-FCI notebook pins `causal-learn==0.1.4.8` in its setup cell; the STAR notebook has an unpinned `xgboost` install line, so a reproducible environment specification is still needed.
- The Stable-FCI configuration has `USE_GOOGLE_DRIVE = True` by default, but its mount cell gracefully skips when the Colab module is unavailable.
- Document dataset provenance and terms without uploading restricted data.
- Publish curated experiment result files and plots from the selected archived runs, with provenance and run IDs.
- Review saved model artefact redistribution rights and security risks of loading joblib/pickle files.
- Audit the full repository, including Git history and ignored files, before the first public push.