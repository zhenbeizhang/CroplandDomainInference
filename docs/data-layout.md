# Companion data layout

The code repository and data release are separate records. This keeps source code easy to review while preserving the data release as a versioned, checksummed research object.

## Required files

### Design-based inference

```text
data/inference/calibrated_probability_sample.csv
data/inference/core_metrics.csv
data/inference/cell_delete_replicates.csv
data/inference/map_area_totals.csv
```

These files provide sample-unit labels and weights, the reported numerical ledger, delete-one replicate estimates, and map-derived census areas.

### Model reproduction

```text
data/model/model_objects.csv
data/model/model_features.npz
data/model/model_split_membership.csv
data/model/model_validation_predictions.csv
data/model/trained_selected_model/member_01.txt
data/model/trained_selected_model/member_02.txt
data/model/trained_selected_model/member_03.txt
data/model/trained_selected_model/member_04.txt
data/model/trained_selected_model/member_05.txt
```

`model_features.npz` contains one base feature matrix and one seven-depth training array for each of the four specifications. Missing annual observations are stored as `NaN` and handled natively by LightGBM. Object order is fixed by the `object_id` vector in the archive and checked against `model_objects.csv`.

### Release integrity

```text
MANIFEST_SHA256.csv
```

The manifest lists every data-release member other than itself, together with byte count and SHA-256 digest. `verify_release.py` also opens every CSV and GeoTIFF and checks the structure of the model feature archive.

## Stable identifiers

- Probability-sample units: `S0001`–`S0450`
- Probability-sample objects: `P0001`–`P0442`
- Model-development objects: `O0001`–`O0249`

These identifiers are stable within the public release and are used consistently across all released tables and arrays.

## Coordinate and raster conventions

- Sample coordinates are longitude and latitude in WGS84.
- Model-object projected coordinates use EPSG:32645.
- Released maps use EPSG:32645 at 30 m resolution.
- Probability rasters store values from 0 to 1 within D2 and `-9999` outside D2.
- Binary maps use 0 outside D2, 1 for cultivated, and 2 for not cultivated.
