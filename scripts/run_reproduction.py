"""Run the standard reproduction sequence against the companion data release."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True, help="Root of the companion data release")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs", help="Output directory")
    parser.add_argument("--workers", type=int, default=8, help="LightGBM worker threads")
    parser.add_argument(
        "--full-validation",
        action="store_true",
        help="Also rerun all four model specifications over 45 validation splits",
    )
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    python = sys.executable

    run([python, str(SCRIPTS / "verify_release.py"), "--release-root", str(data_root)])
    run(
        [
            python,
            str(SCRIPTS / "reproduce_inference.py"),
            "--data-root",
            str(data_root),
            "--output",
            str(output / "inference"),
        ]
    )
    run(
        [
            python,
            str(SCRIPTS / "predict_selected_model.py"),
            "--data-root",
            str(data_root),
            "--output",
            str(output / "selected_model_predictions.csv"),
        ]
    )
    if args.full_validation:
        run(
            [
                python,
                str(SCRIPTS / "fit_and_validate_model.py"),
                "--data-root",
                str(data_root),
                "--output",
                str(output / "model_validation"),
                "--workers",
                str(args.workers),
            ]
        )
    print("\nReproduction sequence completed successfully.")


if __name__ == "__main__":
    main()
