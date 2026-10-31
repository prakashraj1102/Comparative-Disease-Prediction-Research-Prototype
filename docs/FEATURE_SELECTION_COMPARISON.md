# Feature-Selection Comparison

## Summary

Both experiments start with the same 132 symptom identifiers and 41 disease classes, but retain substantially different feature sets.

| Measure | Stable-FCI | STAR CI-selected |
|---|---:|---:|
| Original symptom features | 132 | 132 |
| Selected features | 37 | 106 |
| Retained | 28.03% | 80.30% |
| Removed | 95 | 26 |
| Reduction | 71.97% | 19.70% |

The saved feature lists have:

- 29 symptoms selected by both approaches.
- 8 symptoms selected only by Stable-FCI.
- 77 symptoms selected only by STAR CI-selected.
- 18 symptoms selected by neither approach.

These counts describe overlap between the saved lists; they do not measure which approach is more correct.

## Stable-FCI selection rule

Stable-FCI produces a PAG over disease and symptom variables. A symptom is retained if it has at least one remaining edge of any endpoint-mark type to at least one disease node in the final PAG. An edge may be directed, bidirected, or partially/uncertainly oriented. Retention therefore reflects adjacency in the final PAG under the configured CI tests, search bounds, and background knowledge; it should not automatically be described as proof of a causal effect.

## STAR CI-selected union rule

The STAR procedure first creates a retained symptom list for each disease. The final multiclass feature set is the union of those disease-specific lists. The union follows the original symptom-column order so the model feature schema is deterministic for the same input data and screening settings.

## Important interpretation

The two feature sets answer different methodological questions:

- Stable-FCI uses a constraint-based FCI/PAG workflow with a disease-before-symptom background-knowledge constraint.
- STAR CI-selected uses a custom two-stage disease-specific CI screening procedure and a union of retained symptoms.

The STAR star-shaped graph visualisations are representations of the screening output. The CI screening decision itself does not establish a complete causal graph or independently infer the direction of every disease-symptom relationship. In the notebook, arrows drawn disease-to-symptom are a display convention, not a separate causal-orientation result from the CI test.

The frontend receives the same 132-feature symptom checklist but projects it independently into the saved 37-feature Stable-FCI schema and 106-feature STAR schema. Features not selected by a given approach are not supplied to that approach's classifier.
