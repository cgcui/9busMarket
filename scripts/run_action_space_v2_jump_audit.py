"""Audit dispatch jump structure from the existing Action-Space-v2 cell bank.

This script is deliberately cell-bank-only: it never calls the OPF solver.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paper9bus_gv_grpo.action_space_v2_simulator.bidding import crossing_multiplier  # noqa: E402
from paper9bus_gv_grpo.action_space_v2_simulator.case_data import load_frozen_case  # noqa: E402
from paper9bus_gv_grpo.schema import ACTION_VALUES  # noqa: E402


CELL_ROOT = ROOT / "data/physical/action_space_v2_renewable_train_probe_v1/full_train_c2_cells"
REPORT_ROOT = ROOT / "reports/action_space_v2_jump_audit"
SOLVER_PRIMAL_TOL = 1e-7  # frozen dcopf.py scipy HiGHS option
DISPATCH_TOL = max(100.0 * SOLVER_PRIMAL_TOL, 1e-6)
LMP_TOL = 1e-7
PROFIT_TOL = 1e-7
EPS = 1e-8


def read_cells() -> pd.DataFrame:
    paths = sorted(CELL_ROOT.glob("part-*.parquet"))
    if len(paths) != 12:
        raise RuntimeError(f"expected 12 monthly partitions, found {len(paths)}")
    columns = [
        "timestamp_utc", "physical_example_id", "k_g1", "k_g3",
        "G1_dispatch_mw", "G2_dispatch_mw", "G3_dispatch_mw",
        "G1_lmp_usd_per_mwh", "G1_profit", "binding_branch_status",
        "raw_residual_mw", "solver_residual_mw", "gross_load_mw",
        "btm_solar_mw", "wind_proxy_mw", "surplus_export_mw", "max_branch_utilization",
    ]
    frame = pd.concat([pd.read_parquet(path, columns=columns) for path in paths], ignore_index=True)
    frame["timestamp_utc"] = frame["timestamp_utc"].astype(str)
    return frame.sort_values(["timestamp_utc", "k_g3", "k_g1"], kind="mergesort").reset_index(drop=True)


def cluster_count(values: np.ndarray, tolerance: float) -> int:
    if len(values) == 0:
        return 0
    ordered = np.sort(np.asarray(values, dtype=float))
    return int(1 + np.sum(np.abs(np.diff(ordered)) > tolerance))


def active_generator_signature(row: pd.Series, case) -> str:
    values = []
    for generator in case.generators:
        gid = str(generator["generator_id"])
        dispatch = float(row[f"{gid}_dispatch_mw"])
        pmin = float(generator["pmin_mw"]); pmax = float(generator["pmax_mw"])
        if abs(dispatch - pmin) <= DISPATCH_TOL:
            values.append(f"{gid}_Pmin")
        if abs(dispatch - pmax) <= DISPATCH_TOL:
            values.append(f"{gid}_Pmax")
    return ",".join(values)


def describe(values: pd.Series) -> dict[str, float]:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return {key: float("nan") for key in ("mean", "median", "p75", "p90", "p95", "p99", "max")}
    return {
        "mean": float(values.mean()), "median": float(values.median()),
        "p75": float(values.quantile(.75)), "p90": float(values.quantile(.90)),
        "p95": float(values.quantile(.95)), "p99": float(values.quantile(.99)),
        "max": float(values.max()),
    }


def interval_crossing(case, k3: float, left: float, right: float) -> bool:
    crossing = float(crossing_multiplier(case, k3))
    return left - EPS <= crossing <= right + EPS


def build_audit(cells: pd.DataFrame, case) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    expected_curves = 8760 * 6
    expected_rows = expected_curves * 43
    if len(cells) != expected_rows:
        raise RuntimeError(f"cell bank has {len(cells)} rows, expected {expected_rows}")
    curve_keys = cells.groupby(["timestamp_utc", "k_g3"], sort=False).size()
    if len(curve_keys) != expected_curves or not bool((curve_keys == 43).all()):
        raise RuntimeError("cell bank is not exactly 52,560 complete 43-point curves")

    events: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    for (timestamp, k3), group in cells.groupby(["timestamp_utc", "k_g3"], sort=True):
        group = group.sort_values("k_g1", kind="mergesort").reset_index(drop=True).copy()
        group["active_generator_status"] = group.apply(lambda row: active_generator_signature(row, case), axis=1)
        d1 = group["G1_dispatch_mw"].to_numpy(float)
        d2 = group["G2_dispatch_mw"].to_numpy(float)
        d3 = group["G3_dispatch_mw"].to_numpy(float)
        profit = group["G1_profit"].to_numpy(float)
        deltas = np.column_stack((np.diff(d1), np.diff(d2), np.diff(d3)))
        g1_jump = np.abs(deltas[:, 0]) > DISPATCH_TOL
        system_jump = np.max(np.abs(deltas), axis=1) > DISPATCH_TOL
        binding_changed = group["binding_branch_status"].astype(str).to_numpy()[1:] != group["binding_branch_status"].astype(str).to_numpy()[:-1]
        active_changed = group["active_generator_status"].to_numpy()[1:] != group["active_generator_status"].to_numpy()[:-1]
        structural = system_jump & (binding_changed | active_changed)
        cliff = deltas[:, 0] < -DISPATCH_TOL
        localmax = ((profit[1:-1] - profit[:-2]) > PROFIT_TOL) & ((profit[1:-1] - profit[2:]) >= -PROFIT_TOL)
        best_idx = int(np.argmax(profit))
        for i in np.flatnonzero(g1_jump):
            left = float(group.iloc[i].k_g1); right = float(group.iloc[i + 1].k_g1)
            crossing = interval_crossing(case, float(k3), left, right)
            network = bool(binding_changed[i])
            generator = bool(active_changed[i])
            if crossing and (network or generator):
                classification = "MULTIPLE_MECHANISMS"
            elif crossing:
                classification = "BID_CROSSING_ALIGNED"
            elif network:
                classification = "NETWORK_ACTIVE_SET_DRIVEN"
            elif generator:
                classification = "GENERATOR_BOUND_DRIVEN"
            else:
                classification = "UNCLASSIFIED"
            events.append({
                "timestamp_utc": str(timestamp), "k_g3": float(k3), "k_g1_left": left, "k_g1_right": right,
                "delta_G1_dispatch_mw": float(deltas[i, 0]), "delta_G2_dispatch_mw": float(deltas[i, 1]),
                "delta_G3_dispatch_mw": float(deltas[i, 2]), "delta_G1_lmp_usd_per_mwh": float(group.iloc[i + 1].G1_lmp_usd_per_mwh - group.iloc[i].G1_lmp_usd_per_mwh),
                "delta_profit": float(profit[i + 1] - profit[i]), "g1_dispatch_cliff": bool(cliff[i]),
                "system_dispatch_jump": bool(system_jump[i]), "structural_transition": bool(structural[i]),
                "binding_set_changed": bool(binding_changed[i]), "active_set_changed": bool(active_changed[i]),
                "theoretical_crossing": float(crossing_multiplier(case, float(k3))), "crossing_in_interval": crossing,
                "classification": classification,
            })
        curves.append({
            "timestamp_utc": str(timestamp), "physical_example_id": str(group.iloc[0].physical_example_id), "k_g3": float(k3),
            "n_G1_dispatch_regimes": cluster_count(d1, DISPATCH_TOL),
            "n_system_dispatch_regimes": _vector_regime_count(np.column_stack((d1, d2, d3)), DISPATCH_TOL),
            "n_full_dispatch_vector_regimes": _vector_regime_count(np.column_stack((d1, d2, d3)), DISPATCH_TOL),
            "n_binding_regimes": int(group.binding_branch_status.astype(str).nunique()),
            "n_active_set_regimes": int(group.active_generator_status.nunique()),
            "n_G1_dispatch_jumps": int(g1_jump.sum()), "n_G1_dispatch_cliffs": int(cliff.sum()),
            "n_system_dispatch_jumps": int(system_jump.sum()), "n_structural_transitions": int(structural.sum()),
            "n_local_profit_maxima": int(localmax.sum()), "best_probe_k_g1": float(group.iloc[best_idx].k_g1),
            "best_probe_profit": float(profit[best_idx]), "max_abs_G1_cliff_mw": float(np.max(np.abs(deltas[cliff, 0])) if cliff.any() else 0.0),
            "residual_conventional_load_mw": float(group.iloc[0].solver_residual_mw), "gross_load_mw": float(group.iloc[0].gross_load_mw),
            "btm_solar_mw": float(group.iloc[0].btm_solar_mw), "wind_proxy_mw": float(group.iloc[0].wind_proxy_mw),
            "surplus_export_mw": float(group.iloc[0].surplus_export_mw), "congestion_state": "binding" if bool(group.binding_branch_status.astype(str).str.len().gt(0).any()) else "not_binding",
        })
    return pd.DataFrame(events), pd.DataFrame(curves), {"expected_curves": expected_curves, "expected_rows": expected_rows, "actual_curves": len(curves), "actual_rows": len(cells)}


def _vector_regime_count(values: np.ndarray, tolerance: float) -> int:
    regimes: list[np.ndarray] = []
    for value in values:
        if not any(np.max(np.abs(value - representative)) <= tolerance for representative in regimes):
            regimes.append(value)
    return len(regimes)


def make_summary(events: pd.DataFrame, curves: pd.DataFrame, case, completeness: dict[str, Any]) -> dict[str, Any]:
    def dist(column: str) -> dict[str, float]:
        return describe(curves[column])
    counts = curves["n_G1_dispatch_jumps"]
    event_counts = events["classification"].value_counts().to_dict() if not events.empty else {}
    aligned = int(events["crossing_in_interval"].sum()) if not events.empty else 0
    classified = int(len(events))
    if classified == 0:
        classification = "NO_MATERIAL_JUMP_STRUCTURE"
    elif aligned / classified >= .90:
        classification = "CROSSING_DOMINATED_JUMP_STRUCTURE"
    elif float((counts <= 1).mean()) >= .90:
        classification = "SPARSE_JUMP_STRUCTURE"
    elif int(event_counts.get("NETWORK_ACTIVE_SET_DRIVEN", 0) + event_counts.get("MULTIPLE_MECHANISMS", 0)) / classified >= .20:
        classification = "NETWORK_MEDIATED_JUMP_STRUCTURE"
    else:
        classification = "RICH_INTERMEDIATE_JUMP_STRUCTURE"
    return {
        "classification": classification, "cell_bank_only": True, "solver_rerun": False,
        "completeness": completeness, "tolerances": {"solver_primal_feasibility_tolerance": SOLVER_PRIMAL_TOL, "dispatch_jump_tolerance_mw": DISPATCH_TOL, "lmp_tolerance": LMP_TOL, "profit_tolerance": PROFIT_TOL},
        "curve_count": int(len(curves)), "event_count": int(len(events)),
        "distributions": {name: dist(name) for name in ("n_G1_dispatch_regimes", "n_full_dispatch_vector_regimes", "n_G1_dispatch_jumps", "n_G1_dispatch_cliffs", "n_system_dispatch_regimes", "n_structural_transitions", "n_local_profit_maxima")},
        "jump_count_fractions": {"0": float((counts == 0).mean()), "1": float((counts == 1).mean()), "2": float((counts == 2).mean()), ">=3": float((counts >= 3).mean())},
        "event_classification_counts": {str(k): int(v) for k, v in event_counts.items()},
        "crossing_alignment_counts": {"crossing_in_interval": aligned, "not_crossing_in_interval": int(classified - aligned)},
        "crossing_aligned_fraction": float(aligned / classified) if classified else 0.0,
        "interpretation": "Dispatch differences, not profit changes or active-set changes alone, define material jumps.",
    }


def renewable_diagnostic(curves: pd.DataFrame) -> dict[str, Any]:
    frame = curves.copy()
    gross = frame.gross_load_mw.replace(0, np.nan)
    frame["btm_solar_fraction"] = frame.btm_solar_mw / gross
    frame["wind_fraction"] = frame.wind_proxy_mw / gross
    frame["combined_renewable_fraction"] = (frame.btm_solar_mw + frame.wind_proxy_mw) / gross
    frame["residual_load_bin"] = pd.qcut(frame.residual_conventional_load_mw, 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
    for col, name in (("btm_solar_fraction", "btm_solar_fraction_bin"), ("wind_fraction", "wind_fraction_bin"), ("combined_renewable_fraction", "combined_renewable_fraction_bin")):
        frame[name] = _quantile_bins(frame[col])
    frame["surplus_state"] = np.where(frame.surplus_export_mw > EPS, "surplus", "ordinary")
    def agg(group):
        return {"curves": int(len(group)), "mean_jumps": float(group.n_G1_dispatch_jumps.mean()), "median_regimes": float(group.n_G1_dispatch_regimes.median()), "mean_local_maxima": float(group.n_local_profit_maxima.mean())}
    result = {name: {str(key): agg(group) for key, group in frame.groupby(name, observed=True)} for name in ("residual_load_bin", "btm_solar_fraction_bin", "wind_fraction_bin", "combined_renewable_fraction_bin", "surplus_state", "congestion_state")}
    return {"status": "PASS_RENEWABLE_JUMP_DESCRIPTIVE", "scope": "TRAIN private hidden-cell curves", "causal_claim": False, "stratification": result}


def _quantile_bins(values: pd.Series) -> pd.Series:
    try:
        return pd.qcut(values, 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
    except ValueError:
        return pd.Series(["Q1"] * len(values), index=values.index, dtype="object")


def representative_curves(cells: pd.DataFrame, curves: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    selected: list[tuple[str, pd.Series]] = []
    ordered = curves.sort_values(["timestamp_utc", "k_g3"], kind="mergesort")
    for label, subset in (("zero_jump", ordered[ordered.n_G1_dispatch_jumps == 0]), ("one_jump", ordered[ordered.n_G1_dispatch_jumps == 1]), ("two_or_more_jumps", ordered[ordered.n_G1_dispatch_jumps >= 2]), ("maximum_jump_count", ordered[ordered.n_G1_dispatch_jumps == ordered.n_G1_dispatch_jumps.max()]), ("largest_G1_cliff", ordered[ordered.max_abs_G1_cliff_mw == ordered.max_abs_G1_cliff_mw.max()])):
        if not subset.empty:
            selected.append((label, subset.iloc[0]))
    non_cross = events[events.classification.isin(["NETWORK_ACTIVE_SET_DRIVEN", "GENERATOR_BOUND_DRIVEN", "UNCLASSIFIED"])]
    if not non_cross.empty:
        row = non_cross.sort_values(["timestamp_utc", "k_g3", "k_g1_left"], kind="mergesort").iloc[0]
        selected.append(("network_driven_non_crossing_jump", ordered[(ordered.timestamp_utc == row.timestamp_utc) & (ordered.k_g3 == row.k_g3)].iloc[0]))
    rows = []
    for label, curve in selected:
        points = cells[(cells.timestamp_utc == curve.timestamp_utc) & (cells.k_g3 == curve.k_g3)].sort_values("k_g1")
        for _, point in points.iterrows():
            rows.append({"selection": label, "timestamp_utc": point.timestamp_utc, "physical_example_id": point.physical_example_id, "k_g3": point.k_g3, "k_g1": point.k_g1, "G1_dispatch_mw": point.G1_dispatch_mw, "G2_dispatch_mw": point.G2_dispatch_mw, "G3_dispatch_mw": point.G3_dispatch_mw, "G1_lmp_usd_per_mwh": point.G1_lmp_usd_per_mwh, "G1_profit": point.G1_profit, "binding_branch_status": point.binding_branch_status, "active_generator_status": active_generator_signature(point, load_frozen_case())})
    return pd.DataFrame(rows)


def main() -> int:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    cells = read_cells(); case = load_frozen_case()
    events, curves, completeness = build_audit(cells, case)
    summary = make_summary(events, curves, case, completeness)
    (REPORT_ROOT / "JUMP_EVENT_TABLE.parquet").unlink(missing_ok=True); events.to_parquet(REPORT_ROOT / "JUMP_EVENT_TABLE.parquet", index=False)
    curves.to_parquet(REPORT_ROOT / "CURVE_STRUCTURE_TABLE.parquet", index=False)
    reps = representative_curves(cells, curves, events); reps.to_parquet(REPORT_ROOT / "REPRESENTATIVE_RESPONSE_CURVES.parquet", index=False)
    crossing = {"status": "PASS_BID_CROSSING_ALIGNMENT_AUDIT", "theoretical_crossings": {str(float(k)): float(crossing_multiplier(case, float(k))) for k in ACTION_VALUES}, "event_classification_counts": summary["event_classification_counts"], "crossing_alignment_counts": summary["crossing_alignment_counts"], "crossing_aligned_fraction": summary["crossing_aligned_fraction"], "note": "MULTIPLE_MECHANISMS retains simultaneous generator-bound changes; crossing alignment is reported independently because merit-order crossing itself changes the bound signature."}
    renewable = renewable_diagnostic(curves)
    (REPORT_ROOT / "JUMP_STRUCTURE_SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORT_ROOT / "BID_CROSSING_ALIGNMENT.json").write_text(json.dumps(crossing, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORT_ROOT / "RENEWABLE_JUMP_DIAGNOSTIC.json").write_text(json.dumps(renewable, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    md = f"# Action-Space-v2 Jump Audit\n\n分类：`{summary['classification']}`\n\n本审计只读取现有 8760h × 6 × 43 private cell bank，没有重新运行 OPF，没有读取 DEV/HOLDOUT，没有解决 opponent belief，也没有训练。\n\n- 曲线：{summary['curve_count']:,}；事件：{summary['event_count']:,}\n- dispatch tolerance：`{DISPATCH_TOL:g} MW`（solver primal tolerance `{SOLVER_PRIMAL_TOL:g}`）\n- crossing-aligned fraction：{summary['crossing_aligned_fraction']:.4f}\n- jump fractions：`{json.dumps(summary['jump_count_fractions'], ensure_ascii=False)}`\n\n跳变主判据是 conventional generator dispatch 的 tolerance-aware 变化；profit/LMP 和 active/binding set 仅作辅助证据。当前结果仅用于结构审计，不直接决定 compact action space。\n"
    (REPORT_ROOT / "ACTION_SPACE_V2_JUMP_AUDIT_REPORT_CN.md").write_text(md, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
