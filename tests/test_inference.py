from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_safe_ratio():
    module = load_script("reproduce_inference")
    assert module.safe_ratio(3.0, 4.0) == 0.75
    assert np.isnan(module.safe_ratio(1.0, 0.0))


def test_point_estimate_factorization_and_area_accounting():
    module = load_script("reproduce_inference")
    sample = pd.DataFrame(
        {
            "analysis_weight_ha": [10.0, 20.0, 30.0, 40.0],
            "reference_h": [1, 1, 1, 0],
            "reference_active": [0, 1, 0, 0],
            "D1_2017": [True, True, True, True],
            "D2_2017": [True, True, False, False],
            "map_inactive_2017": [True, True, False, True],
        }
    )
    result = module.compute_point_estimates(sample, mapped_area_ha=100.0)
    assert result["candidate_reliability_D2"] == 1.0
    assert result["conditional_inactive_PA"] == 1.0
    assert result["reference_inactive_coverage_D2_given_D1"] == 0.25
    assert result["end_to_end_inactive_PA_D1"] == 0.25
    assert result["PA_factorization_conditional_times_coverage"] == 0.25
    assert result["stage2_conditional_error_corrected_ha"] == 80.0
    assert result["stage3_candidate_intrusion_propagated_ha"] == 80.0
    assert result["stage4_D1_omission_added_ha"] == 110.0
