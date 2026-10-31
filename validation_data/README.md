# Validation inputs (not bundled)

Keep original experimental datasets outside this application folder. Stage 4 requires only these two source files:

- Exact Stable-FCI source: `train_encoded.csv` (4,920 rows; 41 disease columns + 132 symptom columns)
- Exact STAR external test input: `Test_Only_Symptoms.csv` (42 rows × 132 symptoms)

These data files are deliberately not included in this ZIP. Use `--stable-dataset` and `--star-test-symptoms` absolute paths when running `scripts/validate_stage4.py`.
