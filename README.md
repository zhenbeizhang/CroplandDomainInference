# Historical cropland domain inference

This repository provides the Python workflows used to reproduce the principal inference and model-validation results reported in **“Spatial support governs accuracy and area inference in satellite maps of historical cropland status.”**

The analysis treats the spatial domain of a map as part of the estimand. It links a broad historical-cropland population (D1), the nested support of the delivered probability map (D2), probability-sample reference observations, and design-based area accounting. The released workflows reproduce the result that a conditional producer's accuracy of 81.2% within D2 corresponds to an end-to-end producer's accuracy of 70.3% across D1, and that the mapped area of 126.6 thousand ha becomes a design-adjusted estimate of 88.6 thousand ha after domain and classification errors are accounted for.

## Workflows

| Script | Purpose |
|---|---|
| `scripts/reproduce_inference.py` | Recomputes weighted accuracy, coverage, four-stage area accounting, and delete-one jackknife intervals |
| `scripts/fit_and_validate_model.py` | Refits four LightGBM specifications over 45 validation splits and checks the released object probabilities |
| `scripts/predict_selected_model.py` | Applies the five released members of the selected model to the model-development objects |
| `scripts/verify_release.py` | Checks the companion data release against its SHA-256 manifest and validates its tabular, raster, and feature-array structure |
| `scripts/run_reproduction.py` | Runs the release check, inference reproduction, and selected-model check; optionally runs the full validation experiment |

The full validation experiment fits 900 small models: 45 splits × 4 specifications × 5 ensemble members.

## Companion data

The scripts use the companion analysis-data release archived in Zenodo at <https://doi.org/10.5281/zenodo.23681024>. Keep that release outside the Git repository and pass its root directory with `--data-root` or `--release-root`. The expected structure begins with:

```text
companion-data-release/
├── MANIFEST_SHA256.csv
└── data/
    ├── inference/
    ├── model/
    ├── maps/
    ├── sensitivity/
    └── figure_source/
```

The detailed file contract is given in [`docs/data-layout.md`](docs/data-layout.md).

## Installation

Python 3.12 is recommended.

```text
python -m venv .venv
```

Activate the environment and install the dependencies:

```text
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Exact versions used for release verification are recorded in `requirements-tested.txt`.

## Reproduce the results

Run the standard checks and write newly generated tables to a separate output directory:

```text
python scripts/run_reproduction.py --data-root /path/to/companion-data-release --output outputs
```

Add `--full-validation` to rerun all 45 validation splits:

```text
python scripts/run_reproduction.py --data-root /path/to/companion-data-release --output outputs --full-validation
```

Each workflow can also be run separately:

```text
python scripts/verify_release.py --release-root /path/to/companion-data-release
python scripts/reproduce_inference.py --data-root /path/to/companion-data-release --output outputs/inference
python scripts/predict_selected_model.py --data-root /path/to/companion-data-release --output outputs/selected_model_predictions.csv
python scripts/fit_and_validate_model.py --data-root /path/to/companion-data-release --output outputs/model_validation
```

The scripts read released inputs but do not alter them.

## Reproducibility scope

The public workflows begin from analysis-ready probability-sample records and model-ready predictor arrays. They reproduce the reported design-based estimates, uncertainty intervals, selected-model predictions, and cross-validation probabilities. The companion analysis data are archived at <https://doi.org/10.5281/zenodo.23681024>. Raw GF-2 and dated very-high-resolution imagery are governed by provider terms and are not part of the code or data release. The OasisCrop 1 m cultivated-land input used to derive morphology predictors is available separately at <https://doi.org/10.5281/zenodo.21252320>.

Method-to-output traceability is summarized in [`docs/reproducibility.md`](docs/reproducibility.md).

## Tests

Run the repository tests with:

```text
python -m pip install -r requirements-dev.txt
python -m pytest
```

Tests use synthetic data and therefore do not require the companion release. Integration checks are performed by the reproduction scripts themselves.

## Citation and licence

Citation metadata are provided in `CITATION.cff`. Please cite both the associated article and the archived software release when using this code.

The code is released under the MIT License. Licences for the companion data and third-party inputs are stated in the data repository.

## Contact

Zhenfeng Shao, Wuhan University — shaozhenfeng@whu.edu.cn
