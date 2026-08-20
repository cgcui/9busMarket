from __future__ import annotations

from .case_data import FrozenCase


def bid_costs(case: FrozenCase, k_g1: float, k_g3: float) -> dict[str, float]:
    if float(k_g1) <= 0 or float(k_g3) <= 0:
        raise ValueError("bid multipliers must be positive")
    true = {str(g["generator_id"]): float(g["true_marginal_cost_usd_per_mwh"]) for g in case.generators}
    return {"G1": true["G1"] * float(k_g1), "G2": true["G2"], "G3": true["G3"] * float(k_g3)}


def true_generation_cost(case: FrozenCase, generator_id: str, dispatch_mw: float) -> float:
    g = case.generator(generator_id)
    return float(g["true_marginal_cost_usd_per_mwh"]) * float(dispatch_mw)


def crossing_multiplier(case: FrozenCase, k_g3: float) -> float:
    g1 = case.generator("G1"); g3 = case.generator("G3")
    return float(g3["true_marginal_cost_usd_per_mwh"]) / float(g1["true_marginal_cost_usd_per_mwh"]) * float(k_g3)
