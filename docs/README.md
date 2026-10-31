# Project Documentation Index

This folder contains the project overview, dataset notes, application architecture, experiment results, reproducibility instructions, and limitations.

## Start here

1. [Project overview](PROJECT_OVERVIEW.md) — purpose, scope, and project components.
2. [Dataset description](DATASET_DESCRIPTION.md) — data shape, feature types, and provenance caveats.
3. [Feature-selection comparison](FEATURE_SELECTION_COMPARISON.md) — how the two selected feature sets differ.
4. [Experimental results](EXPERIMENTAL_RESULTS.md) — reported results with their original evaluation protocols.
5. [System architecture](SYSTEM_ARCHITECTURE.md) — how the trained models are used by the application.
6. [Application guide](APPLICATION_GUIDE.md) — setup and user workflow.
7. [Reproducibility and validation](REPRODUCIBILITY.md) — pinned dependencies and validation evidence.
8. [Limitations and future work](LIMITATIONS_AND_FUTURE_WORK.md) — known caveats and responsible interpretation.

## Methodology documents

The detailed experiment descriptions are in [`research/methodology/`](../research/methodology/):

- [Stable-FCI methodology](../research/methodology/STABLE_FCI_METHODOLOGY.md)
- [STAR CI-selected methodology](../research/methodology/STAR_CI_SELECTED_METHODOLOGY.md)
- [ML evaluation protocols](../research/methodology/ML_EVALUATION_PROTOCOLS.md)

## Evidence

Archived source-faithful outputs are under [`research/results/`](../research/results/). The results pack includes a SHA-256 manifest. Its own README explains the contents and deliberate exclusions.

The application-specific setup and artifact provenance notes are in [`docs/application/`](application/).
