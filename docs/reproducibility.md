# Reproducibility map

## Design-based inference

Let `w` be the calibrated area weight, `H` confirmed historical cropland, `C` cultivation detected in 2024, `R = H(1-C)` the reference not-cultivated class, and `M` the mapped not-cultivated class.

The inference workflow reconstructs:

- candidate reliability within D2;
- producer's accuracy for the not-cultivated class within D2;
- coverage of D1 reference-positive area by D2;
- end-to-end producer's accuracy across D1;
- mapped and design-adjusted area estimates; and
- delete-one jackknife standard errors and 95% confidence intervals.

The released estimates satisfy, before display rounding:

```text
D1 end-to-end producer's accuracy
= D2-conditional producer's accuracy × D2 coverage of D1 reference-positive area
= 0.8123628945 × 0.8657645945
= 0.7033150319
```

The area sequence is:

```text
Mapped D2 area classified as not cultivated                         126,559.53 ha
After correcting cultivation-status classification error           113,124.63 ha
After removing mapped positives without historical support          76,226.16 ha
After adding reference-positive omission in D1 outside D2           88,636.60 ha
95% confidence interval                                      65,436.05–111,837.16 ha
```

## Model validation

Four prespecified specifications are compared:

- M0: raw annual trajectory features with historical-object geometry;
- M1: raw annual trajectory features without geometry;
- M2: county-year robust standardized trajectory features without geometry;
- M3: M2 plus 26 GF-2-derived soft morphology predictors.

Each training split is augmented over seven forced historical-depth settings. Five LightGBM members are fitted per specification and split. Validation comprises repeated object-random folds, buffered 30 km spatial blocks, and leave-one-county-out tests. The selected M3 specification contains 154 predictors.

`fit_and_validate_model.py` reconstructs all object probabilities from the released feature arrays and split membership, aggregates repeated held-out predictions, and compares them with `model_validation_predictions.csv` using an absolute tolerance of `1e-6`.

## Source-data boundary

The reproducible analytical unit begins with probability-sample records and model-ready feature arrays. The code does not require restricted source imagery to verify the statistical claims. Provenance and licensing for CLCD, GLC-FCS30D, GFSAD–LGRIP, Sentinel-2, OasisCrop/GF-2, and dated very-high-resolution imagery are recorded in the companion data release.
