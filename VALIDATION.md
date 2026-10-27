# Release validation

Version 1.0.0 was validated with the software versions listed in `requirements-tested.txt`.

## Repository tests

- Five synthetic-data and repository-structure tests passed.
- All Python entry points parsed successfully.
- The tracked source tree contained no research rasters, model arrays, compressed data archives, machine-specific paths, or processing-version identifiers.

## Companion-release checks

- 103 manifest members matched their recorded byte counts and SHA-256 digests.
- All CSV files opened successfully and had unique column names.
- Fifteen GeoTIFF files had valid georeferencing and readable data.
- The model feature archive contained four specifications and 249 consistently ordered objects.

## Numerical reproduction

- Weighted point estimates matched the released metric ledger.
- Delete-one jackknife standard errors and confidence intervals matched at the released precision.
- All 45 validation splits were rerun for four specifications and five ensemble members.
- The maximum absolute difference between recomputed and released object probabilities was `4.994e-09`.
- The maximum absolute difference between recomputed and released ensemble standard deviations was `5.000e-09`.
- All five members of the selected model loaded successfully and produced probabilities for 249 model-development objects.

The validation tolerances enforced by the code are `1e-7` for proportion point estimates, `1e-4 ha` for area point estimates, `1e-6` for jackknife statistics, and `1e-6` for reproduced model probabilities.
