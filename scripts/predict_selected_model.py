"""Apply the five released members of the selected model to model objects."""

from __future__ import annotations

import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
VARIANT = "M3_CYRZ_T1_NO_GEOMETRY_GF2_SOFT"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT,
        help="Root of the companion data release (default: repository root)",
    )
    parser.add_argument("--output", type=Path, help="Optional output CSV path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = args.data_root.resolve() / "data" / "model"
    archive = np.load(model_dir / "model_features.npz")
    values = archive[f"{VARIANT}__base"]
    feature_names = archive[f"{VARIANT}__feature_names"]
    objects = pd.read_csv(model_dir / "model_objects.csv")
    if values.shape != (len(objects), len(feature_names)):
        raise ValueError("Selected-model feature dimensions do not agree")

    predictions = []
    for path in sorted((model_dir / "trained_selected_model").glob("member_*.txt")):
        # model_str avoids native-library path limitations on systems where the
        # repository is stored below a directory containing non-ASCII text.
        model = lgb.Booster(model_str=path.read_text(encoding="utf-8"))
        if model.num_feature() != values.shape[1]:
            raise ValueError(f"Feature count differs for {path.name}")
        predictions.append(model.predict(values))
    if len(predictions) != 5:
        raise ValueError("Expected five selected-model members")

    matrix = np.vstack(predictions)
    output = objects[["object_id", "county", "not_cultivated_label", "in_D2"]].copy()
    output["selected_model_probability"] = matrix.mean(axis=0)
    output["ensemble_sd"] = matrix.std(axis=0, ddof=0)
    output["predicted_not_cultivated"] = output["selected_model_probability"].ge(0.5)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(args.output, index=False)
        print(f"Wrote {len(output)} object predictions to {args.output}")
    else:
        print(output.groupby("county")["selected_model_probability"].agg(["count", "mean"]).to_string())
        print("\nSelected-model members loaded and applied successfully.")


if __name__ == "__main__":
    main()
