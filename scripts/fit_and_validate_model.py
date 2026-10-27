"""Reproduce spatial validation predictions for the four model specifications."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.utils.class_weight import compute_sample_weight


ROOT = Path(__file__).resolve().parents[1]
BASE_SEED = 63063
N_MEMBERS = 5
VARIANTS = (
    "M0_RAW_T1_GEOMETRY",
    "M1_RAW_T1_NO_GEOMETRY",
    "M2_CYRZ_T1_NO_GEOMETRY",
    "M3_CYRZ_T1_NO_GEOMETRY_GF2_SOFT",
)

# LightGBM assigns generic feature labels when fitted from NumPy arrays. Some
# scikit-learn versions warn when prediction is also supplied as a NumPy array,
# although the column order is fixed and checked by the released archive.
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
    category=UserWarning,
)


def parameters(seed: int, workers: int) -> dict[str, object]:
    return {
        "objective": "binary",
        "learning_rate": 0.025,
        "n_estimators": 151,
        "num_leaves": 7,
        "max_depth": 3,
        "min_child_samples": 12,
        "min_split_gain": 0.01,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.70,
        "reg_alpha": 0.25,
        "reg_lambda": 2.0,
        "random_state": seed,
        "n_jobs": workers,
        "verbosity": -1,
    }


def fit_ensemble(
    values: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    seed: int,
    workers: int,
):
    models = []
    for member in range(N_MEMBERS):
        model = LGBMClassifier(**parameters(seed + member * 1009, workers))
        model.fit(values, target, sample_weight=weights, callbacks=[lgb.log_evaluation(0)])
        models.append(model)
    return models


def predict_ensemble(models, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    predictions = np.vstack(
        [model.predict_proba(values, validate_features=False)[:, 1] for model in models]
    )
    return predictions.mean(axis=0), predictions.std(axis=0, ddof=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT,
        help="Root of the companion data release (default: repository root)",
    )
    parser.add_argument("--workers", type=int, default=8, help="LightGBM worker threads")
    parser.add_argument("--output", type=Path, help="Optional directory for reproduced predictions")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = args.data_root.resolve() / "data" / "model"
    required = [
        model_dir / "model_objects.csv",
        model_dir / "model_split_membership.csv",
        model_dir / "model_validation_predictions.csv",
        model_dir / "model_features.npz",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing companion-release files:\n" + "\n".join(missing))
    objects = pd.read_csv(required[0])
    memberships = pd.read_csv(required[1])
    released = pd.read_csv(required[2])
    arrays = np.load(required[3])
    object_ids = arrays["object_id"].astype(str)
    if not np.array_equal(object_ids, objects["object_id"].astype(str).to_numpy()):
        raise ValueError("Object order differs between model_objects.csv and model_features.npz")
    target = objects["not_cultivated_label"].to_numpy(dtype=int)
    index_by_id = {object_id: index for index, object_id in enumerate(object_ids)}

    raw_rows = []
    for split_number, split in memberships.groupby("split_number", sort=True):
        if split["scheme"].nunique() != 1:
            raise ValueError(f"Split {split_number} contains more than one validation scheme")
        scheme = str(split["scheme"].iloc[0])
        train_ids = split.loc[split["role"].eq("train"), "object_id"].astype(str)
        test_ids = split.loc[split["role"].eq("test"), "object_id"].astype(str)
        train = np.asarray([index_by_id[value] for value in train_ids], dtype=int)
        test = np.asarray([index_by_id[value] for value in test_ids], dtype=int)
        seed = BASE_SEED + int(split_number) * 10_000 + 5_000
        for variant in VARIANTS:
            augmented = arrays[f"{variant}__augmented"]
            base = arrays[f"{variant}__base"]
            train_values = np.concatenate([augmented[depth, train, :] for depth in range(7)], axis=0)
            train_target = np.tile(target[train], 7)
            base_weights = np.clip(compute_sample_weight("balanced", target[train]), 0.25, 4.0) / 7.0
            weights = np.tile(base_weights, 7)
            models = fit_ensemble(train_values, train_target, weights, seed, args.workers)
            score, score_sd = predict_ensemble(models, base[test, :])
            for position, object_index in enumerate(test):
                raw_rows.append(
                    {
                        "object_id": object_ids[object_index],
                        "variant": variant,
                        "scheme": scheme,
                        "not_cultivated_score": float(score[position]),
                        "ensemble_sd": float(score_sd[position]),
                    }
                )
        print(f"Completed split {int(split_number):02d}/45", flush=True)

    raw = pd.DataFrame(raw_rows)
    aggregated = (
        raw.groupby(["object_id", "variant", "scheme"], as_index=False)
        .agg(
            not_cultivated_score=("not_cultivated_score", "mean"),
            ensemble_sd=("ensemble_sd", "mean"),
            prediction_replicates=("not_cultivated_score", "size"),
        )
    )
    comparison = aggregated.merge(
        released[
            [
                "object_id",
                "variant",
                "scheme",
                "not_cultivated_score",
                "ensemble_sd",
                "prediction_replicates",
            ]
        ],
        on=["object_id", "variant", "scheme"],
        suffixes=("_recomputed", "_released"),
        validate="one_to_one",
    )
    if len(comparison) != len(released):
        raise RuntimeError(f"Expected {len(released)} aggregated rows; reproduced {len(comparison)}")
    comparison["score_absolute_difference"] = (
        comparison["not_cultivated_score_recomputed"]
        - comparison["not_cultivated_score_released"]
    ).abs()
    comparison["sd_absolute_difference"] = (
        comparison["ensemble_sd_recomputed"] - comparison["ensemble_sd_released"]
    ).abs()
    if not comparison["prediction_replicates_recomputed"].equals(
        comparison["prediction_replicates_released"]
    ):
        raise RuntimeError("Prediction replicate counts differ from the released results")
    max_score_difference = float(comparison["score_absolute_difference"].max())
    max_sd_difference = float(comparison["sd_absolute_difference"].max())
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        raw.to_csv(args.output / "model_validation_predictions_by_split.csv", index=False)
        comparison.to_csv(args.output / "model_validation_comparison.csv", index=False)
    print(f"Maximum probability difference: {max_score_difference:.3e}")
    print(f"Maximum ensemble-SD difference: {max_sd_difference:.3e}")
    if max_score_difference > 1e-6 or max_sd_difference > 1e-6:
        raise RuntimeError("Model reproduction differs from the released predictions")
    print("Model-validation reproduction passed.")


if __name__ == "__main__":
    main()
