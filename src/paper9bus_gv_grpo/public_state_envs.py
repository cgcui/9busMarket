"""Environment-specific Public-State-v1 builders.

Paper9Bus and ISO2Y deliberately have separate input contracts.  Neither
builder accepts target, payoff, regret, or hidden-opponent fields.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import numpy as np

from .audit_hardening import (
    PAPER9BUS_PUBLIC_STATE_X1_V1,
    PAPER9BUS_PUBLIC_STATE_X2_AUDIT_V1,
    validate_paper9bus_x1_card,
)
from .public_state import (
    _finite,
    _three_level,
    build_public_energy_state,
    canonical_json,
    contains_forbidden_key,
)


def _json_value(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


def _obs(row: Mapping[str, Any]) -> dict[str, Any]:
    value = row.get("observation_json", row.get("public_observation", {}))
    value = _json_value(value)
    return dict(value) if isinstance(value, Mapping) else {}


def fit_paper9bus_interpretation_rules(train_rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    loads = [r.get("total_load_mw", r.get("total_load")) for r in train_rows]
    spreads = []
    for row in train_rows:
        obs = _obs(row)
        value = obs.get("system_lmp_spread")
        if _finite(value):
            spreads.append(float(value))
    def quantile(xs):
        arr = np.asarray([float(x) for x in xs if _finite(x)], dtype=float)
        return {"low": float(np.quantile(arr, 1 / 3)), "high": float(np.quantile(arr, 2 / 3))} if arr.size else {"low": None, "high": None}
    return {
        "schema_version": "Paper9Bus-Public-State-v1",
        "environment": "Paper9Bus-3Gen-C3",
        "fit_split": "TRAIN",
        "quantile_definition": {"low": 1 / 3, "high": 2 / 3, "method": "numpy.quantile"},
        "load_level": quantile(loads),
        "price_dispersion": quantile(spreads),
        "fixed_rules": {
            "generator_utilization": "<0.40 LOW; 0.40..0.80 MEDIUM; >0.80 HIGH",
            "network_stress": "max utilization <0.80 LOW; <0.95 ELEVATED; otherwise HIGH",
            "congestion_status": "binding_branch_count > 0 -> CONGESTED; otherwise UNCONGESTED",
        },
        "final_accessed": False,
    }


def build_paper9bus_public_state(
    row: Mapping[str, Any],
    rules: Mapping[str, Any],
    *,
    include_bus_loads: bool = False,
    feature_set_id: str = PAPER9BUS_PUBLIC_STATE_X1_V1,
) -> dict[str, Any]:
    """Build a Paper9Bus public state without Paper9Bus forecasts.

    X1 is the safe default.  The earlier bus-level engineering audit must
    explicitly request ``feature_set_id=X2`` and ``include_bus_loads=True``.
    """
    obs = _obs(row)
    public: dict[str, Any] = {}
    total = row.get("total_load_mw", row.get("total_load"))
    if _finite(total):
        public["current_energy_state"] = {"total_load_mw": float(total)}
    bus = [row.get(f"load_bus_{i}") for i in range(1, 10)]
    if include_bus_loads and all(_finite(x) for x in bus):
        public.setdefault("current_energy_state", {})["bus_loads_mw"] = [float(x) for x in bus]

    own = {}
    for source, target in (("g1_dispatch", "dispatch_mw"), ("g1_lmp", "own_lmp")):
        if _finite(obs.get(source)):
            own[target] = float(obs[source])
    if own:
        public["own_generator"] = own
    market = {}
    for source, target in (("system_lmp_mean", "system_lmp_mean"), ("system_lmp_min", "system_lmp_min"), ("system_lmp_max", "system_lmp_max"), ("system_lmp_spread", "lmp_spread")):
        if _finite(obs.get(source)):
            market[target] = float(obs[source])
    if market:
        public["market"] = market
    network = {}
    for source, target in (("binding_branch_count", "binding_branch_count"), ("max_branch_utilization", "max_branch_utilization")):
        if _finite(obs.get(source)):
            network[target] = float(obs[source])
    if network:
        public["network"] = network

    interpretation: dict[str, Any] = {}
    if public.get("current_energy_state", {}).get("total_load_mw") is not None:
        interpretation["load_level"] = _three_level(public["current_energy_state"]["total_load_mw"], rules.get("load_level", {}))
    if public.get("market", {}).get("lmp_spread") is not None:
        interpretation["price_dispersion"] = _three_level(public["market"]["lmp_spread"], rules.get("price_dispersion", {}))
    branches = public.get("network", {}).get("binding_branch_count")
    utilization = public.get("network", {}).get("max_branch_utilization")
    if _finite(utilization):
        interpretation["network_stress"] = "LOW" if utilization < 0.80 else "ELEVATED" if utilization < 0.95 else "HIGH"
    elif _finite(branches):
        interpretation["network_stress"] = "LOW" if int(branches) == 0 else "ELEVATED" if int(branches) <= 2 else "HIGH"
    if _finite(branches):
        interpretation["congestion_status"] = "CONGESTED" if int(branches) > 0 else "UNCONGESTED"
    public["schema_version"] = "Paper9Bus-Public-State-v1"
    public["environment"] = "Paper9Bus-3Gen-C3"
    public["public_interpretation"] = {k: v for k, v in interpretation.items() if v is not None}
    if contains_forbidden_key(public):
        raise ValueError("forbidden field in Paper9Bus public state")
    if feature_set_id == PAPER9BUS_PUBLIC_STATE_X1_V1:
        if include_bus_loads:
            raise ValueError("X1 cannot include bus loads; request the explicit X2 audit feature set")
        validate_paper9bus_x1_card(public)
    elif feature_set_id != PAPER9BUS_PUBLIC_STATE_X2_AUDIT_V1:
        raise ValueError(f"unknown Paper9Bus feature set: {feature_set_id}")
    return public


def build_iso2y_public_state(row: Mapping[str, Any], rules: Mapping[str, Any]) -> dict[str, Any]:
    """Build ISO2Y from the verified public parquet; unavailable fields omit."""
    source = {key: _json_value(value) for key, value in dict(row).items()}
    card = build_public_energy_state(source, rules)
    card["environment"] = "ISO-NE-2020-2021-public-pack"
    card["schema_version"] = "ISO2Y-Public-Energy-State-v1"
    if contains_forbidden_key(card):
        raise ValueError("forbidden field in ISO2Y public state")
    return card


def public_state_bytes(card: Mapping[str, Any]) -> str:
    return canonical_json(card)
