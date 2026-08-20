"""Renewable-aware 9-bus interface with gross-load mode as the current freeze.

The IEEE-9 physical case remains fixed.  This module maps the public ISO-NE
gross load forecast to the frozen IEEE-9 load pattern.  Wind and solar are
reserved as optional future inputs, but are deliberately not subtracted in
the current protocol.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


FROZEN_CASE9_LOAD_MW = np.asarray([0.0, 0.0, 0.0, 0.0, 90.0, 0.0, 100.0, 0.0, 125.0], dtype=float)
FROZEN_CASE9_TOTAL_LOAD_MW = float(FROZEN_CASE9_LOAD_MW.sum())
ALPHA_MIN = 0.5
ALPHA_MAX = 1.5


def _json_value(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value


def _hourly_gross_load(row: Mapping[str, Any]) -> float:
    values = _json_value(row.get("hourly_load_mw"))
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError("hourly_load_mw must be a sequence")
    hour = int(row["target_he"])
    if hour < 1 or hour > len(values):
        raise ValueError(f"target_he {hour} is outside hourly_load_mw")
    value = float(values[hour - 1])
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"invalid gross load forecast: {value}")
    return value


@dataclass(frozen=True)
class GrossLoadMapping:
    """TRAIN-only monotone mapping from ISO-NE MW to IEEE-9 alpha."""

    reference_train_median_mw: float
    alpha_min: float = ALPHA_MIN
    alpha_max: float = ALPHA_MAX
    fit_split: str = "TRAIN"
    mapping_id: str = "ISO2Y-GrossLoad-to-IEEE9-MedianRatio-v1"

    def alpha(self, gross_load_mw: float) -> float:
        value = float(gross_load_mw) / float(self.reference_train_median_mw)
        return float(np.clip(value, self.alpha_min, self.alpha_max))

    def bus_loads(self, gross_load_mw: float) -> np.ndarray:
        return FROZEN_CASE9_LOAD_MW * self.alpha(gross_load_mw)

    def as_dict(self, train_stats: Mapping[str, Any] | None = None) -> dict[str, Any]:
        out = {
            "mapping_id": self.mapping_id,
            "formula": "alpha_t = clip(gross_load_forecast_t / TRAIN_median_gross_load_forecast, 0.5, 1.5)",
            "reference_train_median_mw": float(self.reference_train_median_mw),
            "alpha_min": float(self.alpha_min),
            "alpha_max": float(self.alpha_max),
            "fit_split": self.fit_split,
            "case9_base_load_mw_by_bus": {str(i + 1): float(v) for i, v in enumerate(FROZEN_CASE9_LOAD_MW)},
            "case9_base_total_load_mw": FROZEN_CASE9_TOTAL_LOAD_MW,
            "renewable_adjustment": "DISABLED_FORECASTS_NOT_ADDED",
        }
        if train_stats:
            out["train_gross_load_stats"] = dict(train_stats)
        return out


def fit_gross_load_mapping(public_state: pd.DataFrame) -> tuple[GrossLoadMapping, dict[str, Any]]:
    train = public_state[public_state["split"].astype(str) == "TRAIN"]
    values = [_hourly_gross_load(row) for row in train.to_dict("records")]
    if not values:
        raise ValueError("TRAIN has no gross load forecast values")
    reference = float(np.median(np.asarray(values, dtype=float)))
    mapping = GrossLoadMapping(reference_train_median_mw=reference)
    stats = {
        "rows": len(values),
        "min_mw": float(np.min(values)),
        "median_mw": reference,
        "max_mw": float(np.max(values)),
        "p01_mw": float(np.quantile(values, 0.01)),
        "p99_mw": float(np.quantile(values, 0.99)),
    }
    return mapping, stats


def build_gross_load_bridge(public_state: pd.DataFrame, mapping: GrossLoadMapping) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in public_state.to_dict("records"):
        gross = _hourly_gross_load(row)
        alpha = mapping.alpha(gross)
        loads = mapping.bus_loads(gross)
        target_date = str(row["target_local_date"])
        target_he = int(row["target_he"])
        state_id = f"isone2y_grossload_{target_date.replace('-', '')}_he{target_he:02d}"
        source_hash = str(row.get("public_state_hash", ""))
        if not source_hash:
            source_hash = hashlib.sha256(f"{target_date}|{target_he}|{gross:.6f}".encode()).hexdigest()
        item: dict[str, Any] = {
            "protocol": "Paper9Bus-ISONE2Y-RenewableAware-GrossLoad-v1",
            "physical_state_id": state_id,
            "target_hour_utc": str(row["target_hour_utc"]),
            "target_local_date": target_date,
            "target_he": target_he,
            "split": str(row["split"]),
            "public_state_hash": source_hash,
            "gross_load_forecast_mw": gross,
            "case9_alpha": alpha,
            "total_case9_load_mw": float(loads.sum()),
            "renewable_input_mode": "DISABLED_FORECASTS_NOT_ADDED",
            "wind_forecast_used": False,
            "solar_forecast_used": False,
            "net_load_forecast_used": False,
            "load_mapping_id": mapping.mapping_id,
            "load_source": "ISO-NE hourly_load_mw day-ahead gross-load forecast",
        }
        item.update({f"load_bus_{i + 1}_mw": float(value) for i, value in enumerate(loads)})
        rows.append(item)
    out = pd.DataFrame(rows).sort_values(["target_local_date", "target_he"]).reset_index(drop=True)
    if len(out) != len(public_state):
        raise ValueError("load bridge row count changed")
    return out


def bridge_summary(bridge: pd.DataFrame, mapping: GrossLoadMapping) -> dict[str, Any]:
    return {
        "rows": int(len(bridge)),
        "train_rows": int((bridge["split"] == "TRAIN").sum()),
        "dev_rows": int((bridge["split"] == "DEV").sum()),
        "date_min": str(bridge["target_local_date"].min()),
        "date_max": str(bridge["target_local_date"].max()),
        "alpha_min": float(bridge["case9_alpha"].min()),
        "alpha_max": float(bridge["case9_alpha"].max()),
        "alpha_clipped_low_rows": int((bridge["gross_load_forecast_mw"] / mapping.reference_train_median_mw < mapping.alpha_min).sum()),
        "alpha_clipped_high_rows": int((bridge["gross_load_forecast_mw"] / mapping.reference_train_median_mw > mapping.alpha_max).sum()),
        "renewable_forecast_values_used": False,
        "net_load_forecast_used": False,
        "status": "PASS_GROSS_LOAD_BRIDGE_RENEWABLE_INPUTS_DISABLED",
    }
