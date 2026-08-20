from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/physical/isone_2024_2026_9bus_fixed_renewable_v1"
REPORT = ROOT / "reports/fixed_renewable_physics_v1"


def _read(name: str) -> pd.DataFrame:
    return pd.read_parquet(OUT / name)


def test_train_selection_is_exactly_168_and_one_to_one():
    frame = _read("train_168h_inputs.parquet")
    assert len(frame) == 168
    assert set(frame["split"]) == {"TRAIN"}
    assert frame["timestamp_utc"].is_unique
    assert frame["timestamp_utc"].is_monotonic_increasing


def test_load_weights_and_gross_allocation():
    manifest = json.loads((OUT / "MANIFEST.json").read_text(encoding="utf-8"))
    weights = np.array(list(manifest["weights"].values()), dtype=float)
    assert np.isclose(weights.sum(), 1.0)
    assert manifest["source_audit"]["one_to_one_alignment"]


def test_btm_is_negative_load_and_not_generator():
    c1 = _read("train_168h_c1_btm_solar_only.parquet")
    c2 = _read("train_168h_c2_btm_solar_fixed_wind.parquet")
    assert (c1["solar_as_generator_mw"] == 0).all()
    assert (c2["solar_as_generator_mw"] == 0).all()
    assert (c1["estimated_btm_solar_system_mw"] >= 0).all()


def test_fixed_wind_is_not_available_power_or_opf_variable():
    c2 = _read("train_168h_c2_btm_solar_fixed_wind.parquet")
    assert (c2["wind_as_opf_decision_variable"] == False).all()
    assert (c2["wind_available_power_known"] == False).all()
    assert (c2["wind_semantic_class"] == "DISPATCH_EXPECTED_WIND_GENERATION_PROXY").all()


def test_forecast_rejection_and_no_utility_solar():
    for name in ("train_168h_c0_no_renewable.parquet", "train_168h_c1_btm_solar_only.parquet", "train_168h_c2_btm_solar_fixed_wind.parquet"):
        frame = _read(name)
        assert not frame["forecast_fields_used_by_physical_solver"].any()
        assert np.isclose(frame["utility_solar_injection_mw"], 0.0).all()


def test_system_and_nodal_balance():
    c2 = _read("train_168h_c2_btm_solar_fixed_wind.parquet")
    assert c2["system_balance_residual_mw"].abs().max() <= 1e-7
    assert c2["max_abs_nodal_residual_mw"].abs().max() <= 1e-7


def test_generator_and_branch_limits():
    manifest = json.loads((OUT / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["hard_checks"]["generator_bounds"]
    assert manifest["hard_checks"]["branch_limits"]


def test_zero_equivalence_decomposition_and_effect():
    zero = json.loads((REPORT / "ZERO_RENEWABLE_EQUIVALENCE.json").read_text(encoding="utf-8"))
    assert zero["status"] == "PASS_ZERO_RENEWABLE_EQUIVALENCE"
    deltas = _read("train_168h_counterfactual_deltas.parquet")
    assert len(deltas) == 168
    assert deltas.filter(regex="delta_C1_minus_C0").abs().to_numpy().max() >= 0
    assert deltas.filter(regex="delta_C2_minus_C0").abs().to_numpy().max() > 0


def test_load_imputation_provenance_is_preserved():
    frame = _read("train_168h_inputs.parquet")
    assert set(frame["load_imputation_status"]) <= {"OBSERVED", "IMPUTED_LINEAR_TIME_INTERNAL"}


def test_deterministic_and_create_only():
    manifest = json.loads((OUT / "MANIFEST.json").read_text(encoding="utf-8"))
    repro = json.loads((REPORT / "TRAIN_168H_REPRODUCIBILITY.json").read_text(encoding="utf-8"))
    assert manifest["hard_checks"]["deterministic_rerun"]
    assert repro["deterministic_rerun_pass"]
    assert manifest["outputs_created_only"]
    assert not manifest["old_renewable_physics_overwritten"]
