# Research Results Archive Audit

## Source archives

- Stable-FCI: `Stable FCI All Model 02 compressed.zip` (66 ZIP entries; 528.69 MiB uncompressed).
- STAR: `Star_All_Three_Model compressed.zip` (106 ZIP entries; 82.98 MiB uncompressed).

## Curated result package

- Stable-FCI CI-test log: 54,686,755 bytes as gzip; SHA-256 `d4b2a863a6051dae1232339646b24e09f80c162f115e5395337945e14790f776`.
- STAR detailed CI-screening log: 899,376 bytes as gzip; SHA-256 `b38e669b098f9e0087b39b92f299184cf530edf531fe393ec9f23eb433db96cf`.
- STAR graph files: 41 PDF graphs and 39 PNG previews.
- Selected model evaluation data, figures, and feature-selection outputs are organised under `research/results/`.

## Preservation policy

No metric values were recomputed or edited. Source reports and visualisations were copied as archived. Two large CSV logs were gzip-compressed without changing their decompressed content. A small number of redundant comparison exports and serialized model duplicates were excluded; the deployed selected-feature model artifacts belong in `artifacts/`, not here.

## Important caveat

The two Stable-FCI dataset candidates previously identified both reproduced the archived metrics during Stage 4 validation, but their hashes differ. The exact historical absolute path used in the original run is not conclusively established. Therefore this results package does not claim that one particular local dataset path is the unique historical source.

## File inventory

A machine-readable SHA-256 manifest is provided in `research/results/RESULTS_FILE_MANIFEST.json`.
