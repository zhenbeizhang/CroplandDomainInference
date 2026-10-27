"""Verify package membership, file digests, and basic data readability."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--release-root",
        type=Path,
        default=ROOT,
        help="Root of the companion data release (default: repository root)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.release_root.resolve()
    manifest_path = root / "MANIFEST_SHA256.csv"
    if not manifest_path.exists():
        raise FileNotFoundError("MANIFEST_SHA256.csv is missing")
    manifest = pd.read_csv(manifest_path)
    required = {"relative_path", "bytes", "sha256"}
    if not required.issubset(manifest.columns):
        raise ValueError(f"Manifest columns must include {sorted(required)}")
    if manifest["relative_path"].duplicated().any():
        raise ValueError("Manifest contains duplicate paths")

    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    expected = set(manifest["relative_path"].astype(str))
    if actual != expected:
        raise RuntimeError(
            f"Package membership differs from manifest; missing={sorted(expected-actual)}, "
            f"unlisted={sorted(actual-expected)}"
        )

    failures = []
    for row in manifest.itertuples(index=False):
        path = root / str(row.relative_path)
        if path.stat().st_size != int(row.bytes):
            failures.append(f"size: {row.relative_path}")
        if sha256(path) != str(row.sha256):
            failures.append(f"sha256: {row.relative_path}")
    if failures:
        raise RuntimeError("Manifest verification failed:\n" + "\n".join(failures))

    for path in root.rglob("*.csv"):
        if path == manifest_path:
            continue
        frame = pd.read_csv(path, low_memory=False)
        if frame.columns.duplicated().any():
            raise ValueError(f"Duplicate columns in {path.relative_to(root)}")
        object_columns = frame.select_dtypes(include=["object", "string"])
        for column in object_columns:
            values = object_columns[column].dropna().astype(str).str.lstrip()
            dangerous = values.str.match(r"^[=+@]|^-[A-Za-z]")
            if dangerous.any():
                raise ValueError(f"Spreadsheet formula-like text in {path.relative_to(root)}:{column}")

    for path in root.rglob("*.tif"):
        with rasterio.open(path) as dataset:
            if dataset.crs is None or dataset.transform.a == 0 or dataset.transform.e == 0:
                raise ValueError(f"Invalid georeferencing in {path.relative_to(root)}")
            if dataset.count != 1:
                raise ValueError(f"Expected one band in {path.relative_to(root)}")
            values = dataset.read(1, masked=True)
            if values.count() == 0:
                raise ValueError(f"Raster has no valid pixels: {path.relative_to(root)}")

    archive = np.load(root / "data" / "model" / "model_features.npz")
    variants = [key.removesuffix("__base") for key in archive.files if key.endswith("__base")]
    if len(variants) != 4 or archive["object_id"].shape != (249,):
        raise ValueError("Unexpected model feature archive structure")
    for variant in variants:
        base = archive[f"{variant}__base"]
        augmented = archive[f"{variant}__augmented"]
        names = archive[f"{variant}__feature_names"]
        if augmented.shape != (7, *base.shape) or base.shape[1] != len(names):
            raise ValueError(f"Feature archive dimensions do not agree for {variant}")

    print(f"Verified {len(expected)} files, all CSV tables, 15 GeoTIFFs, and the model feature archive.")


if __name__ == "__main__":
    main()
