# Limitations and Future Work

## Dataset and validation limitations

1. **Dataset provenance:** The exact historical Stable-FCI source-file copy is not uniquely established. Two candidate datasets with different hashes both reproduced the saved test metrics. Raw datasets are therefore not bundled, and source/license details must be verified before public redistribution.
2. **Different evaluation protocols:** Stable-FCI uses a stratified 80/20 split (3,936 train / 984 test), while STAR uses 4,920 training rows and 42 separate external test rows. Historical metrics cannot be directly compared to conclude that one approach is better.
3. **Small STAR external test set:** 42 test examples across 41 classes provide very limited per-class support. The high metrics are sensitive to a small number of observations and require validation on larger independent cohorts.
4. **STAR file alignment:** disease and symptom splits do not contain patient identifiers. The notebook checks row counts but assumes the files preserve corresponding row order.
5. **Unresolved duplicate-like feature identifier:** `fluid_overload` and `fluid_overload.1` are preserved as separate STAR feature identifiers. Their relationship should be investigated in a future data-quality stage without silently changing the current model schema.
6. **Stable-FCI search bounds:** the FCI search is bounded by maximum conditioning depth 10 and maximum path length 10; these settings trade exhaustive search for computational tractability.
7. **Nested feature-selection validation:** Stable-FCI feature selection is performed once on the full training partition before the downstream classifier's internal cross-validation. The held-out test partition is kept separate, but fully nested cross-validation would be needed to assess the complete feature-selection-plus-model-selection procedure within each outer training fold.
8. **Very high all-feature baseline scores:** the archived Stable-FCI all-symptom baselines report perfect test metrics. This warrants additional investigation into dataset structure, feature-label relationships, and potential leakage before drawing conclusions about generalisation.

## Model and application limitations

1. `predict_proba` outputs have not been subjected to a calibration study. Percentages in the UI are model scores, not calibrated clinical risk estimates.
2. The input is binary and has no explicit unknown/missing state. Unchecked symptoms are treated as absent.
3. The application does not provide clinical explanations, differential diagnosis reasoning, treatment recommendations, or clinician oversight.
4. Neither pipeline has been clinically validated, prospectively evaluated, or assessed for performance across independent populations or demographic subgroups.
5. The two approaches are compared at inference time for the same symptom input, but the UI's agreement/overlap measures do not establish which prediction is correct.
6. The XGBoost pickled snapshots emit a non-fatal compatibility warning. Their dependencies are pinned and their current inference has been validated, but long-term serialization portability remains a maintenance concern.

## Future work

- Verify the original dataset source, citation, and redistribution terms.
- Obtain larger independent test cohorts with patient-level identifiers and a documented split protocol.
- Revisit the STAR disease/symptom row-alignment assumption using stable IDs or a verifiable shared source ordering.
- Investigate `fluid_overload` versus `fluid_overload.1` at source-data level and decide through an auditable schema revision; do not alter the current deployed schema silently.
- Run nested evaluation where feature selection and model tuning are both performed inside the outer training folds.
- Evaluate probability calibration on an appropriate independent dataset before presenting model scores as calibrated probabilities.
- Conduct external validation, subgroup evaluation, and error analysis before any clinical-facing claims.
- Improve model serialization portability in a separate migration with equivalence tests.
- Document software/data citations and select an appropriate code license after checking dataset/model redistribution rights.

## Intended use

This repository is intended for research inspection, education, and software demonstration. It is not intended for medical diagnosis or treatment decisions.
