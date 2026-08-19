"""Public-Energy-State-v1: deterministic, auditable public feature handling."""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any

import numpy as np


FORBIDDEN_FIELD_TOKENS = (
    "hidden",
    "private",
    "oracle",
    "payoff",
    "regret",
    "future_realized",
    "realized_future",
    "opponent_profit",
    "opponent_dispatch",
    "g3_state",
    "hidden_state",
)


def canonical_json(value: Any) -> str:
    """Return stable JSON bytes for hashing and reproducibility checks."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def state_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def contains_forbidden_key(value: Any, path: str = "") -> list[str]:
    """Find forbidden semantic keys recursively; values are never inspected as labels."""
    hits: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            name = str(key).lower()
            current = f"{path}.{key}" if path else str(key)
            if any(token in name for token in FORBIDDEN_FIELD_TOKENS):
                hits.append(current)
            hits.extend(contains_forbidden_key(child, current))
    elif isinstance(value, (list, tuple)):
        for i, child in enumerate(value):
            hits.extend(contains_forbidden_key(child, f"{path}[{i}]"))
    return hits


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (Mapping, list, tuple)):
        return True
    if isinstance(value, str):
        return bool(value)
    return _finite(value)


def _quantiles(values: list[float], low: float = 1 / 3, high: float = 2 / 3) -> dict[str, float]:
    clean = np.asarray([float(x) for x in values if _finite(x)], dtype=float)
    if clean.size == 0:
        return {"low": None, "high": None}
    return {"low": float(np.quantile(clean, low)), "high": float(np.quantile(clean, high))}


def fit_interpretation_rules(train_rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Fit all data-dependent thresholds on TRAIN only."""
    loads = [r.get("current_load_mw") for r in train_rows]
    spreads = [r.get("historical_lmp_spread", r.get("day_ahead_lmp_spread")) for r in train_rows]
    pressure = [r.get("forecast_change_vs_current_pct") for r in train_rows]
    return {
        "schema_version": "Public-Energy-State-v1",
        "fit_split": "TRAIN",
        "quantile_definition": {"low": 1 / 3, "high": 2 / 3, "method": "numpy.quantile"},
        "load_level": _quantiles(loads),
        "price_dispersion": _quantiles(spreads),
        "tomorrow_demand_pressure": _quantiles(pressure),
        "fixed_rules": {
            "network_stress": "binding_branch_count == 0 -> LOW; 1..2 -> ELEVATED; >2 -> HIGH",
            "congestion_status": "binding_branch_count > 0 -> CONGESTED; otherwise UNCONGESTED",
            "generator_utilization": "<0.40 LOW; 0.40..0.80 MEDIUM; >0.80 HIGH",
            "net_load": "load_forecast - wind_forecast - solar_forecast, only when all legal inputs exist",
        },
        "final_accessed": False,
    }


def _three_level(value: Any, thresholds: Mapping[str, Any], labels: tuple[str, str, str] = ("LOW", "MEDIUM", "HIGH")) -> str | None:
    if not _finite(value) or thresholds.get("low") is None or thresholds.get("high") is None:
        return None
    x = float(value)
    return labels[0] if x <= float(thresholds["low"]) else labels[2] if x >= float(thresholds["high"]) else labels[1]


def _fixed_generator_level(value: Any) -> str | None:
    if not _finite(value):
        return None
    x = float(value)
    return "LOW" if x < 0.40 else "MEDIUM" if x <= 0.80 else "HIGH"


def interpret_public_state(state: Mapping[str, Any], rules: Mapping[str, Any]) -> dict[str, Any]:
    """Derive semantic labels from public raw values only."""
    current = state.get("current_energy_state", {})
    forecast = state.get("day_ahead_load_forecast", {})
    market = state.get("market", {})
    network = state.get("network", {})
    own = state.get("own_generator", {})
    out: dict[str, Any] = {}

    load_level = _three_level(current.get("total_load_mw"), rules.get("load_level", {}))
    if load_level:
        out["load_level"] = load_level
    trend = current.get("load_trend_value")
    if _finite(trend):
        out["load_trend"] = "FALLING" if float(trend) < -1e-9 else "RISING" if float(trend) > 1e-9 else "STABLE"
    pressure = forecast.get("change_vs_current_pct")
    pressure_label = _three_level(pressure, rules.get("tomorrow_demand_pressure", {}))
    if pressure_label:
        out["tomorrow_demand_pressure"] = pressure_label
    spread_label = _three_level(market.get("historical_lmp_spread", market.get("day_ahead_lmp_spread")), rules.get("price_dispersion", {}))
    if spread_label:
        out["price_dispersion"] = spread_label
    branches = network.get("binding_branch_count")
    if _finite(branches):
        n = int(float(branches))
        out["network_stress"] = "LOW" if n == 0 else "ELEVATED" if n <= 2 else "HIGH"
        out["congestion_status"] = "CONGESTED" if n > 0 else "UNCONGESTED"
    utilization = own.get("capacity_utilization")
    utilization_label = _fixed_generator_level(utilization)
    if utilization_label:
        out["generator_utilization"] = utilization_label
    if "upward_headroom_mw" in own and _finite(own.get("upward_headroom_mw")):
        out["generator_headroom"] = "LOW" if float(own["upward_headroom_mw"]) < 0.10 else "MEDIUM"
    wind = state.get("renewable_forecast", {}).get("wind_mw")
    solar = state.get("renewable_forecast", {}).get("solar_mw")
    load = forecast.get("hourly_load_mw")
    if isinstance(load, list) and isinstance(wind, list) and isinstance(solar, list) and len(load) == len(wind) == len(solar):
        net = [float(a) - float(b) - float(c) for a, b, c in zip(load, wind, solar)]
        out["net_load_ramp"] = "RISING" if np.diff(net).max(initial=0.0) > abs(np.diff(net).min(initial=0.0)) else "FALLING"
    return out


def build_public_energy_state(row: Mapping[str, Any], rules: Mapping[str, Any]) -> dict[str, Any]:
    """Build the nested state card, omitting unavailable fields."""
    input_hits = contains_forbidden_key(row)
    if input_hits:
        raise ValueError(f"forbidden input fields: {input_hits}")
    card: dict[str, Any] = {"schema_version": "Public-Energy-State-v1"}
    current: dict[str, Any] = {}
    for source_key, output_key in (("current_load_mw", "total_load_mw"), ("total_load_mw", "total_load_mw"), ("load_factor", "load_factor"), ("load_trend_value", "load_trend_value"), ("load_history_last_4h_mw", "load_history_last_4h_mw")):
        if _present(row.get(source_key)):
            current[output_key] = row[source_key]
    if current:
        card["current_energy_state"] = current

    forecast: dict[str, Any] = {}
    forecast_keys = ("hourly_load_mw", "forecast_zone_load_mw", "peak_load_mw", "peak_hour", "minimum_load_mw", "daily_energy_mwh", "max_up_ramp_mw_per_h", "max_down_ramp_mw_per_h", "change_vs_current_pct")
    for key in forecast_keys:
        if _present(row.get(key)):
            forecast[key] = row[key]
    if forecast:
        card["day_ahead_load_forecast"] = forecast

    renewable = {}
    for key in ("wind_mw", "solar_mw", "net_load_mw", "net_load_peak_mw", "net_load_ramp"):
        if _present(row.get(key)):
            renewable[key] = row[key]
    load_forecast = row.get("hourly_load_mw")
    wind_forecast = row.get("wind_mw")
    solar_forecast = row.get("solar_mw")
    if all(isinstance(x, (list, tuple)) for x in (load_forecast, wind_forecast, solar_forecast)) and len(load_forecast) == len(wind_forecast) == len(solar_forecast):
        net_load = [float(a) - float(b) - float(c) for a, b, c in zip(load_forecast, wind_forecast, solar_forecast)]
        renewable.setdefault("net_load_mw", net_load)
        renewable.setdefault("net_load_peak_mw", float(max(net_load)))
        renewable.setdefault("net_load_ramp", float(max(abs(x) for x in np.diff(net_load))) if len(net_load) > 1 else 0.0)
    if renewable:
        card["renewable_forecast"] = renewable

    own = {}
    for key in ("dispatch_mw", "capacity_utilization", "upward_headroom_mw", "downward_headroom_mw", "own_lmp"):
        if _present(row.get(key)):
            own[key] = row[key]
    if own:
        card["own_generator"] = own

    market = {}
    for key in ("historical_lmp_mean", "historical_lmp_min", "historical_lmp_max", "historical_lmp_spread"):
        if _present(row.get(key)):
            market[key] = row[key]
    if market:
        card["market"] = market

    network = {}
    for key in ("binding_branch_count", "binding_event_count", "network_signal"):
        if _present(row.get(key)):
            network[key] = row[key]
    if network:
        card["network"] = network

    history = {}
    for key in ("load_last_4h_mw", "lmp_last_4h"):
        if _present(row.get(key)):
            history[key] = row[key]
    if history:
        card["recent_history"] = history

    card["public_interpretation"] = interpret_public_state(card, rules)
    hits = contains_forbidden_key(card)
    if hits:
        raise ValueError(f"forbidden public fields: {hits}")
    return card


def format_public_energy_state_prompt(card: Mapping[str, Any]) -> str:
    """Deterministic human-readable prompt; no LLM-generated interpretation."""
    def lines(value: Any, indent: int = 0) -> list[str]:
        pad = " " * indent
        if isinstance(value, Mapping):
            out = []
            for key, child in value.items():
                label = str(key).replace("_", " ")
                if isinstance(child, (Mapping, list)):
                    out.append(f"{pad}{label}:")
                    out.extend(lines(child, indent + 2))
                else:
                    out.append(f"{pad}{label}: {child}")
            return out
        if isinstance(value, list):
            return [f"{pad}{json.dumps(value, ensure_ascii=False, separators=(',', ':'))}"]
        return [f"{pad}{value}"]

    body = {k: v for k, v in card.items() if k not in {"schema_version", "public_interpretation"}}
    out = ["PUBLIC ENERGY STATE", *lines(body), "", "DETERMINISTIC INTERPRETATION", *lines(card.get("public_interpretation", {}), 2)]
    return "\n".join(out)
