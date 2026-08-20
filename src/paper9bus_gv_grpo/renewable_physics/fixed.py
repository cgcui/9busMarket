from __future__ import annotations

import json
from typing import Iterable

import numpy as np

from ..action_space_v2_simulator.case_data import FrozenCase
from ..action_space_v2_simulator.dcopf import DCOPFResult


CASE_C0 = "C0_NO_RENEWABLE"
CASE_C1 = "C1_BTM_SOLAR_ONLY"
CASE_C2 = "C2_BTM_SOLAR_PLUS_FIXED_WIND"


def base_load_weights(case: FrozenCase) -> np.ndarray:
    base = np.asarray(case.load_base_mw, dtype=float)
    if base.sum() <= 0 or np.any(base < 0):
        raise ValueError("frozen base load must be non-negative and positive")
    weights = base / base.sum()
    if not np.isclose(weights.sum(), 1.0, atol=1e-12):
        raise ValueError("base-load weights do not sum to one")
    return weights


def allocate_system_value(system_mw: float, weights: np.ndarray) -> np.ndarray:
    value = float(system_mw) * np.asarray(weights, dtype=float)
    if not np.all(np.isfinite(value)):
        raise ValueError("non-finite allocated physical input")
    return value


def residual_load(case_id: str, gross_bus_mw: np.ndarray, btm_bus_mw: np.ndarray, wind_bus_mw: np.ndarray) -> np.ndarray:
    gross = np.asarray(gross_bus_mw, dtype=float)
    btm = np.asarray(btm_bus_mw, dtype=float)
    wind = np.asarray(wind_bus_mw, dtype=float)
    if case_id == CASE_C0:
        out = gross.copy()
    elif case_id == CASE_C1:
        out = gross - btm
    elif case_id == CASE_C2:
        out = gross - btm - wind
    else:
        raise ValueError(f"unknown fixed-renewable case: {case_id}")
    if not np.all(np.isfinite(out)):
        raise ValueError("non-finite residual load")
    return out


def nodal_residuals(case: FrozenCase, result: DCOPFResult, load_mw: np.ndarray) -> np.ndarray:
    generation = np.zeros(len(case.bus_ids), dtype=float)
    for g in case.generators:
        generation[int(g["bus"]) - 1] += result.pg_mw[str(g["generator_id"])]
    net_export = np.zeros(len(case.bus_ids), dtype=float)
    for branch in case.branches:
        if int(branch["status"]) == 0:
            continue
        flow = result.branch_flows_mw[int(branch["branch_id"])]
        net_export[int(branch["from_bus"]) - 1] += flow
        net_export[int(branch["to_bus"]) - 1] -= flow
    return generation - np.asarray(load_mw, dtype=float) - net_export


def json_vector(values: Iterable[float]) -> str:
    return json.dumps([float(x) for x in values], separators=(",", ":"))
