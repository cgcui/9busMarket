from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

from .bidding import bid_costs, true_generation_cost
from .case_data import FrozenCase


@dataclass(frozen=True)
class DCOPFResult:
    solver_status: int
    solver_status_name: str
    objective_bid_cost: float
    pg_mw: dict[str, float]
    theta_rad: dict[int, float]
    branch_flows_mw: dict[int, float]
    nodal_lmp: dict[int, float]
    active_generator_bounds: tuple[str, ...]
    active_branch_constraints: tuple[str, ...]
    binding_signature: str
    max_branch_utilization: float
    balance_residual_mw: float
    bid_cost_coefficient: dict[str, float]
    focal: dict[str, float]
    solver_meta: dict

    def dispatch_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.pg_mw, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _index(case: FrozenCase):
    buses = {b: i for i, b in enumerate(case.bus_ids)}
    gens = {g["generator_id"]: len(case.bus_ids) + i for i, g in enumerate(case.generators)}
    return buses, gens


def _flow_coeff(case: FrozenCase, branch: dict, buses: dict[int, int]) -> np.ndarray:
    n = len(case.bus_ids) + len(case.generators)
    a = np.zeros(n, dtype=np.float64)
    if int(branch["status"]) == 0:
        return a
    f = buses[int(branch["from_bus"])]
    t = buses[int(branch["to_bus"])]
    tap = float(branch.get("tap", 1.0) or 1.0)
    if tap == 0:
        raise ValueError("zero transformer tap")
    susceptance_mw_per_rad = float(case.base_mva) / float(branch["x_pu"])
    phase = np.deg2rad(float(branch.get("phase_shift_deg", 0.0)))
    a[f] = susceptance_mw_per_rad / tap
    a[t] = -susceptance_mw_per_rad / tap
    return a, -susceptance_mw_per_rad * phase / tap


def solve_dcopf(case: FrozenCase, load_mw: np.ndarray, k_g1: float, k_g3: float,
                *, focal_generator: str = "G1", finite_difference: bool = False) -> DCOPFResult:
    load = np.asarray(load_mw, dtype=np.float64)
    if load.shape != (len(case.bus_ids),):
        raise ValueError("load must have one MW value per bus")
    buses, gens = _index(case); n = len(case.bus_ids) + len(case.generators)
    bids = bid_costs(case, k_g1, k_g3)
    c = np.zeros(n, dtype=np.float64)
    for g in case.generators:
        c[gens[g["generator_id"]]] = bids[g["generator_id"]]

    A_eq = np.zeros((len(case.bus_ids), n), dtype=np.float64)
    b_eq = load.copy()
    for bus, i in buses.items():
        for g in case.generators:
            if int(g["bus"]) == bus:
                A_eq[i, gens[g["generator_id"]]] = 1.0
    branch_coefficients = []
    for branch in case.branches:
        coeff, constant = _flow_coeff(case, branch, buses)
        branch_coefficients.append((coeff, constant))
        # gen - load - outgoing_flow = 0, equivalently gen - outgoing_flow = load
        if int(branch["status"]) == 1:
            f = buses[int(branch["from_bus"])]
            t = buses[int(branch["to_bus"])]
            A_eq[f, :] -= coeff
            A_eq[t, :] += coeff
            b_eq[f] += constant
            b_eq[t] -= constant

    A_ub = []; b_ub = []; ub_names = []
    for branch, (coeff, constant) in zip(case.branches, branch_coefficients):
        if int(branch["status"]) == 0:
            continue
        limit = float(branch["rate_a_mw"])
        A_ub.append(coeff); b_ub.append(limit - constant); ub_names.append(f"branch_{int(branch['branch_id'])}_upper")
        A_ub.append(-coeff); b_ub.append(limit + constant); ub_names.append(f"branch_{int(branch['branch_id'])}_lower")

    bounds = [(None, None)] * len(case.bus_ids)
    for g in case.generators:
        bounds.append((float(g["pmin_mw"]), float(g["pmax_mw"])))
    ref_i = buses[int(case.reference_bus)]
    bounds[ref_i] = (0.0, 0.0)
    options = {"presolve": True, "dual_feasibility_tolerance": 1e-7, "primal_feasibility_tolerance": 1e-7}
    solution = linprog(c, A_ub=np.asarray(A_ub), b_ub=np.asarray(b_ub), A_eq=A_eq, b_eq=b_eq,
                       bounds=bounds, method="highs", options=options)
    status_name = {0: "OPTIMAL", 1: "ITERATION_LIMIT", 2: "INFEASIBLE", 3: "UNBOUNDED", 4: "NUMERICAL_ERROR"}.get(int(solution.status), "UNKNOWN")
    if not solution.success:
        raise RuntimeError(f"DCOPF {status_name}: {solution.message}")
    x = np.asarray(solution.x, dtype=np.float64)
    theta = {bus: float(x[i]) for bus, i in buses.items()}
    pg = {str(g["generator_id"]): float(x[gens[g["generator_id"]]]) for g in case.generators}
    flows = {}
    active_branches = []
    for branch, (coeff, constant) in zip(case.branches, branch_coefficients):
        val = float(coeff @ x + constant) if int(branch["status"]) else 0.0
        bid = int(branch["branch_id"]); flows[bid] = val
        lim = float(branch["rate_a_mw"])
        if int(branch["status"]) and abs(abs(val) - lim) <= 2e-6:
            active_branches.append(str(bid))
    active_gens = []
    for g in case.generators:
        v = pg[g["generator_id"]]
        if abs(v - float(g["pmin_mw"])) <= 2e-6: active_gens.append(f"{g['generator_id']}_Pmin")
        if abs(v - float(g["pmax_mw"])) <= 2e-6: active_gens.append(f"{g['generator_id']}_Pmax")
    # scipy's equality marginal is d(min objective)/d(load) for A_eq x = b_eq;
    # the sign/scaling is tested independently by finite-difference validation.
    lmp = {bus: float(solution.eqlin.marginals[i]) for bus, i in buses.items()}
    residual = A_eq @ x - b_eq
    util = [abs(flows[int(b["branch_id"])]) / float(b["rate_a_mw"]) for b in case.branches if int(b["status"])]
    focal_dispatch = pg[focal_generator]; focal_lmp = lmp[int(case.generator(focal_generator)["bus"])]
    revenue = focal_lmp * focal_dispatch
    true_cost = true_generation_cost(case, focal_generator, focal_dispatch)
    return DCOPFResult(int(solution.status), status_name, float(solution.fun), pg, theta, flows, lmp,
                       tuple(active_gens), tuple(active_branches), ",".join(active_branches), float(max(util, default=0.0)),
                       float(np.max(np.abs(residual))), bids,
                       {"dispatch_mw": focal_dispatch, "lmp": focal_lmp, "revenue": revenue,
                        "true_generation_cost": true_cost, "profit": revenue - true_cost},
                       {"backend": "scipy.optimize.linprog", "method": "highs", "message": str(solution.message),
                        "primal_feasibility_tolerance": 1e-7, "dual_feasibility_tolerance": 1e-7})
