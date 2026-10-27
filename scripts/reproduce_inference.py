"""Recompute the paper's principal design-based estimates.

All inputs are resolved relative to this release package. The script checks
the weighted point estimates and delete-one jackknife uncertainty against the
reported values and optionally writes the recomputed tables.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
T_CRITICAL_DF_440 = 1.9653701191315174


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else np.nan


def weighted_total(weights: np.ndarray, mask: np.ndarray) -> float:
    return float(weights[mask].sum())


def compute_point_estimates(sample: pd.DataFrame, mapped_area_ha: float) -> dict[str, float]:
    weights = sample["analysis_weight_ha"].to_numpy(dtype=float)
    historical = sample["reference_h"].to_numpy(dtype=float) == 1
    cultivated = sample["reference_active"].to_numpy(dtype=float) == 1
    not_cultivated = historical & ~cultivated
    d1 = sample["D1_2017"].astype(bool).to_numpy()
    d2 = sample["D2_2017"].astype(bool).to_numpy()
    mapped = sample["map_inactive_2017"].astype(bool).to_numpy()

    d2_historical = weighted_total(weights, d2 & historical)
    d2_total = weighted_total(weights, d2)
    d2_true_positive = weighted_total(weights, d2 & not_cultivated & mapped)
    d2_false_negative = weighted_total(weights, d2 & not_cultivated & ~mapped)
    d1_true_positive = weighted_total(weights, d1 & not_cultivated & mapped)
    d1_false_negative = weighted_total(weights, d1 & not_cultivated & ~mapped)
    d2_not_cultivated = weighted_total(weights, d2 & not_cultivated)
    d1_not_cultivated = weighted_total(weights, d1 & not_cultivated)

    conditional_pa = safe_ratio(d2_true_positive, d2_true_positive + d2_false_negative)
    coverage = safe_ratio(d2_not_cultivated, d1_not_cultivated)
    end_to_end_pa = safe_ratio(d1_true_positive, d1_true_positive + d1_false_negative)

    active_mapped = weighted_total(weights, d2 & historical & cultivated & mapped)
    omitted_within_d2 = weighted_total(weights, d2 & not_cultivated & ~mapped)
    mapped_without_historical_support = weighted_total(weights, d2 & ~historical & mapped)
    omitted_outside_d2 = weighted_total(weights, d1 & ~d2 & not_cultivated)
    stage2 = mapped_area_ha - active_mapped + omitted_within_d2
    stage3 = stage2 - mapped_without_historical_support
    stage4 = stage3 + omitted_outside_d2

    return {
        "candidate_reliability_D2": safe_ratio(d2_historical, d2_total),
        "conditional_inactive_PA": conditional_pa,
        "reference_inactive_coverage_D2_given_D1": coverage,
        "end_to_end_inactive_PA_D1": end_to_end_pa,
        "PA_factorization_conditional_times_coverage": conditional_pa * coverage,
        "raw_mapped_inactive_probability_frame_ha": mapped_area_ha,
        "stage2_conditional_error_corrected_ha": stage2,
        "stage3_candidate_intrusion_propagated_ha": stage3,
        "stage4_D1_omission_added_ha": stage4,
        "estimated_D2_probability_frame_ha": weighted_total(weights, d2),
        "estimated_D1_probability_frame_ha": weighted_total(weights, d1),
    }


def jackknife_summary(replicates: pd.DataFrame, reported: pd.DataFrame) -> pd.DataFrame:
    lookup = reported.set_index("metric")["estimate"].to_dict()
    degrees_of_freedom = int(reported["df"].iloc[0])
    if degrees_of_freedom != 440:
        raise ValueError(f"Expected 440 reference degrees of freedom; found {degrees_of_freedom}.")
    rows = []
    for metric, metric_rows in replicates.groupby("metric"):
        variance = 0.0
        for _, group in metric_rows.groupby("variance_stratum"):
            m_selected = int(group["m_selected"].iloc[0])
            finite_population_multiplier = 1.0 - float(group["sampling_fraction"].iloc[0])
            group_mean = float(group["estimate"].mean())
            variance += (
                finite_population_multiplier
                * (m_selected - 1)
                / m_selected
                * float(np.square(group["estimate"] - group_mean).sum())
            )
        standard_error = math.sqrt(max(variance, 0.0))
        estimate = float(lookup[metric])
        low = estimate - T_CRITICAL_DF_440 * standard_error
        high = estimate + T_CRITICAL_DF_440 * standard_error
        if not metric.endswith("_ha"):
            low, high = max(0.0, low), min(1.0, high)
        rows.append(
            {
                "metric": metric,
                "estimate": estimate,
                "se_jackknife_recomputed": standard_error,
                "ci95_low_recomputed": low,
                "ci95_high_recomputed": high,
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT,
        help="Root of the companion data release (default: repository root)",
    )
    parser.add_argument("--output", type=Path, help="Optional directory for recomputed CSV files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = args.data_root.resolve() / "data" / "inference"
    required = [
        data / "calibrated_probability_sample.csv",
        data / "core_metrics.csv",
        data / "cell_delete_replicates.csv",
        data / "map_area_totals.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing companion-release files:\n" + "\n".join(missing))
    sample = pd.read_csv(required[0])
    reported = pd.read_csv(required[1])
    replicates = pd.read_csv(required[2])
    map_totals = pd.read_csv(required[3])
    total_row = map_totals.loc[map_totals["county"].eq("five_counties_total")]
    if len(total_row) != 1:
        raise ValueError("map_area_totals.csv must contain one five_counties_total row")
    mapped_area = float(total_row["mapped_not_cultivated_area_ha"].iloc[0])

    recomputed = compute_point_estimates(sample, mapped_area)
    reported_lookup = reported.set_index("metric")["estimate"].to_dict()
    failures = []
    for metric, estimate in recomputed.items():
        expected = float(reported_lookup[metric])
        tolerance = 1e-7 if not metric.endswith("_ha") else 1e-4
        if not math.isclose(estimate, expected, rel_tol=0.0, abs_tol=tolerance):
            failures.append(f"{metric}: recomputed={estimate}, reported={expected}")

    jackknife = jackknife_summary(replicates, reported)
    merged = jackknife.merge(
        reported[["metric", "se_jackknife", "ci95_low", "ci95_high"]],
        on="metric",
        validate="one_to_one",
    )
    for row in merged.itertuples(index=False):
        for recalculated, reference in (
            ("se_jackknife_recomputed", "se_jackknife"),
            ("ci95_low_recomputed", "ci95_low"),
            ("ci95_high_recomputed", "ci95_high"),
        ):
            if not math.isclose(
                float(getattr(row, recalculated)),
                float(getattr(row, reference)),
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                failures.append(f"{row.metric}: {recalculated} differs from {reference}")

    point_table = pd.DataFrame(
        [{"metric": metric, "estimate_recomputed": value} for metric, value in recomputed.items()]
    )
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        point_table.to_csv(args.output / "recomputed_point_estimates.csv", index=False)
        merged.to_csv(args.output / "recomputed_jackknife_intervals.csv", index=False)
    print(point_table.to_string(index=False))
    if failures:
        raise RuntimeError("Inference reproduction failed:\n" + "\n".join(failures))
    print("\nAll point-estimate and jackknife checks passed.")


if __name__ == "__main__":
    main()
