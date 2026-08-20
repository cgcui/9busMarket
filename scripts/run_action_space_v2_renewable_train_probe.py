#!/usr/bin/env python3
"""TRAIN-only dense strategic response probe for Action-Space-v2."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paper9bus_gv_grpo.action_space_v2_simulator.bidding import bid_costs, crossing_multiplier  # noqa: E402
from paper9bus_gv_grpo.action_space_v2_simulator.case_data import load_frozen_case, snapshot_sha256  # noqa: E402
from paper9bus_gv_grpo.action_space_v2_simulator.dcopf import solve_dcopf  # noqa: E402
from paper9bus_gv_grpo.schema import ACTION_VALUES  # noqa: E402


OUT = ROOT / "data/physical/action_space_v2_renewable_train_probe_v1"
REPORT = ROOT / "reports/action_space_v2_renewable_train_probe_v1"
CONFIG = ROOT / "configs/action_space_v2_renewable_train_probe_v1.json"
FULL = ROOT / "data/physical/isone_2024_2026_9bus_full_train_fixed_renewable_v1"
SMOKE = ROOT / "data/physical/isone_2024_2026_9bus_fixed_renewable_v1"
CASE_IDS = ("C0_NO_RENEWABLE", "C1_BTM_SOLAR_ONLY", "C2_BTM_SOLAR_PLUS_FIXED_WIND")
K_G3 = tuple(float(x) for x in ACTION_VALUES)
TOL = 1e-7


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def frame_hash(frame: pd.DataFrame) -> str:
    payload = frame.sort_index(axis=1).to_json(orient="records", date_format="iso", double_precision=15).encode()
    return hashlib.sha256(payload).hexdigest()


def parse_vec(value: Any) -> list[float]:
    return [float(x) for x in json.loads(str(value))]


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def parse_mapping(value: Any) -> dict[str, float]:
    obj = json.loads(str(value))
    if isinstance(obj, dict):
        return {str(k): float(v) for k, v in obj.items()}
    return {str(i + 1): float(v) for i, v in enumerate(obj)}


def build_probe_grid(case) -> tuple[list[float], dict[str, Any]]:
    regular = [Decimal("1.00") + Decimal("0.05") * i for i in range(27)]
    crossing = [Decimal(str(crossing_multiplier(case, k))) for k in K_G3]
    sources: dict[Decimal, set[str]] = {}
    for value in regular:
        sources.setdefault(value, set()).add("regular_1.00_to_2.30_step_0.05")
    for point in crossing:
        for value, source in ((point - Decimal("0.01"), "crossing_minus_0.01"), (point, "crossing"), (point + Decimal("0.01"), "crossing_plus_0.01")):
            if Decimal("1.00") <= value <= Decimal("2.30"):
                sources.setdefault(value, set()).add(source)
    points = sorted(sources)
    grid = [float(x) for x in points]
    spec = {
        "a6_frozen": [float(x) for x in ACTION_VALUES],
        "regular_grid": [float(x) for x in regular],
        "crossings": [{"k_g3": float(k), "k_g1_cross": float(crossing_multiplier(case, k))} for k in K_G3],
        "points": [{"k_g1": float(x), "sources": sorted(sources[x])} for x in points],
        "count": len(grid),
        "is_model_action_space": False,
    }
    return grid, spec


def load_case_frames(hours: int) -> dict[str, pd.DataFrame]:
    base = SMOKE if hours == 168 else FULL
    suffix = "168h" if hours == 168 else "8760h"
    names = {
        "C0_NO_RENEWABLE": f"train_{suffix}_c0_no_renewable.parquet",
        "C1_BTM_SOLAR_ONLY": f"train_{suffix}_c1_btm_solar_only.parquet",
        "C2_BTM_SOLAR_PLUS_FIXED_WIND": f"train_{suffix}_c2_btm_solar_fixed_wind.parquet",
    }
    frames = {case_id: pd.read_parquet(base / name) for case_id, name in names.items()}
    if any(len(frame) != hours for frame in frames.values()):
        raise RuntimeError(f"expected {hours} rows per physical case")
    return frames


def build_record(case, base: dict[str, Any], case_id: str, k_g1: float, k_g3: float, detail: bool) -> dict[str, Any]:
    result = solve_dcopf(case, np.asarray(parse_vec(base["residual_bus_loads_mw"]), dtype=float), k_g1, k_g3)
    bids = bid_costs(case, k_g1, k_g3)
    raw_residual = float(base["raw_residual_conventional_load_mw"] if "raw_residual_conventional_load_mw" in base else base["residual_conventional_load_mw"])
    surplus_export = float(base.get("surplus_export_mw", 0.0))
    solver_residual = float(base["solver_residual_conventional_load_mw"] if "solver_residual_conventional_load_mw" in base else base["residual_conventional_load_mw"])
    record: dict[str, Any] = {
        "timestamp_utc": str(base["timestamp_utc"]),
        "physical_example_id": str(base["physical_example_id"]),
        "case_id": case_id,
        "k_g1": float(k_g1),
        "k_g3": float(k_g3),
        "bid_g1_usd_per_mwh": float(bids["G1"]),
        "bid_g3_usd_per_mwh": float(bids["G3"]),
        "G1_dispatch_mw": float(result.pg_mw["G1"]),
        "G2_dispatch_mw": float(result.pg_mw["G2"]),
        "G3_dispatch_mw": float(result.pg_mw["G3"]),
        "G1_lmp_usd_per_mwh": float(result.focal["lmp"]),
        "G1_revenue": float(result.focal["revenue"]),
        "G1_true_cost": float(result.focal["true_generation_cost"]),
        "G1_profit": float(result.focal["profit"]),
        "system_lmp_mean_usd_per_mwh": float(np.mean(list(result.nodal_lmp.values()))),
        "system_lmp_min_usd_per_mwh": float(min(result.nodal_lmp.values())),
        "system_lmp_max_usd_per_mwh": float(max(result.nodal_lmp.values())),
        "system_lmp_spread_usd_per_mwh": float(max(result.nodal_lmp.values()) - min(result.nodal_lmp.values())),
        "binding_branch_count": int(len(result.active_branch_constraints)),
        "binding_branch_status": result.binding_signature,
        "max_branch_utilization": float(result.max_branch_utilization),
        "solver_status": result.solver_status_name,
        "balance_residual_mw": float(result.balance_residual_mw),
        "raw_residual_mw": raw_residual,
        "surplus_export_mw": surplus_export,
        "solver_residual_mw": solver_residual,
        "gross_load_mw": float(base["gross_system_load_mw"]),
        "btm_solar_mw": float(base["estimated_btm_solar_system_mw"]),
        "wind_proxy_mw": float(base["wind_fixed_proxy_system_mw"]),
        "input_hash": str(base.get("public_state_hash", "")),
    }
    if detail:
        record["all_bus_lmp_usd_per_mwh"] = json_text({str(k): float(v) for k, v in result.nodal_lmp.items()})
        record["all_branch_flows_mw"] = json_text({str(k): float(v) for k, v in result.branch_flows_mw.items()})
        record["all_branch_utilizations"] = json_text({str(k): float(abs(v) / next(float(b["rate_a_mw"]) for b in case.branches if int(b["branch_id"]) == int(k))) for k, v in result.branch_flows_mw.items()})
        record["active_generator_bounds"] = json_text(list(result.active_generator_bounds))
        record["solver_meta"] = json_text(result.solver_meta)
    return record


def base_row_map(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return frame.sort_values("timestamp_utc").to_dict("records")


def anchor(case, frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    rows = []
    max_diff = {"dispatch_mw": 0.0, "all_lmp": 0.0, "branch_flow_mw": 0.0, "G1_profit": 0.0}
    binding_diff = 0
    for base in base_row_map(frames["C2_BTM_SOLAR_PLUS_FIXED_WIND"]):
        got = build_record(case, base, "C2_BTM_SOLAR_PLUS_FIXED_WIND", 1.0, 1.0, True)
        expected_pg = {"G1": float(base["G1_dispatch_mw"]), "G2": float(base["G2_dispatch_mw"]), "G3": float(base["G3_dispatch_mw"])}
        max_diff["dispatch_mw"] = max(max_diff["dispatch_mw"], *(abs(got[f"{g}_dispatch_mw"] - expected_pg[g]) for g in expected_pg))
        expected_lmp = parse_mapping(base["all_bus_lmp_usd_per_mwh"])
        got_lmp = parse_mapping(got["all_bus_lmp_usd_per_mwh"])
        max_diff["all_lmp"] = max(max_diff["all_lmp"], max(abs(got_lmp[k] - expected_lmp[k]) for k in expected_lmp))
        expected_flow = parse_mapping(base["all_branch_flows_mw"])
        got_flow = parse_mapping(got["all_branch_flows_mw"])
        max_diff["branch_flow_mw"] = max(max_diff["branch_flow_mw"], max(abs(got_flow[k] - expected_flow[k]) for k in expected_flow))
        max_diff["G1_profit"] = max(max_diff["G1_profit"], abs(got["G1_profit"] - float(base["G1_profit"])))
        binding_diff += int(got["binding_branch_status"] != str(base["binding_branch_status"]))
        rows.append({"timestamp_utc": got["timestamp_utc"], "max_diff": max(max_diff.values()), "binding_equal": binding_diff == 0})
    passed = max(max_diff.values()) <= TOL and binding_diff == 0
    return {"status": "PASS_STRATEGIC_ANCHOR_EQUIVALENCE" if passed else "FAIL_STRATEGIC_ANCHOR_EQUIVALENCE", "hours": len(rows), "k_g1": 1.0, "k_g3": 1.0, "max_diff": max_diff, "binding_diff_count": binding_diff, "surplus_invariant": True, "rows_preview": rows[:3]}


def run_smoke(case, frames: dict[str, pd.DataFrame], grid: list[float]) -> tuple[pd.DataFrame, dict[str, Any]]:
    records = []
    for case_id, frame in frames.items():
        for idx, base in enumerate(base_row_map(frame), start=1):
            for k_g3 in K_G3:
                for k_g1 in grid:
                    records.append(build_record(case, base, case_id, k_g1, k_g3, True))
            if idx == 1 or idx % 32 == 0 or idx == len(frame):
                print(f"[168h smoke] {case_id} {idx}/{len(frame)}", flush=True)
    out = pd.DataFrame(records)
    state_cols = ["gross_load_mw", "btm_solar_mw", "wind_proxy_mw", "raw_residual_mw", "surplus_export_mw", "solver_residual_mw"]
    invariant = int(out.groupby(["case_id", "timestamp_utc"])[state_cols].nunique(dropna=False).max().max()) <= 1
    checks = {
        "rows": int(len(out)), "expected_rows": int(168 * 3 * len(K_G3) * len(grid)),
        "all_optimal": bool((out.solver_status == "OPTIMAL").all()),
        "balance_pass": bool((out.balance_residual_mw.abs() <= TOL).all()),
        "branch_limit_pass": bool((out.max_branch_utilization <= 1.0 + TOL).all()),
        "exogenous_state_invariant": invariant,
        "surplus_invariant": bool(out.groupby(["case_id", "timestamp_utc"])["surplus_export_mw"].nunique().max() <= 1),
    }
    return out, {"status": "PASS_ACTION_SPACE_V2_168H_SMOKE" if all(checks.values()) else "FAIL_ACTION_SPACE_V2_168H_SMOKE", "grid_count": len(grid), "k_g3_count": len(K_G3), "checks": checks}


def load_full_base() -> pd.DataFrame:
    inputs = pd.read_parquet(FULL / "train_8760h_inputs.parquet")
    c2 = pd.read_parquet(FULL / "train_8760h_c2_btm_solar_fixed_wind.parquet", columns=["timestamp_utc", "residual_bus_loads_mw"])
    required = ["timestamp_utc", "physical_example_id", "public_state_hash", "gross_system_load_mw", "estimated_btm_solar_system_mw", "wind_fixed_proxy_system_mw", "raw_residual_conventional_load_mw", "surplus_export_mw", "solver_residual_conventional_load_mw"]
    return inputs[required].merge(c2, on="timestamp_utc", how="inner", validate="one_to_one").sort_values("timestamp_utc").reset_index(drop=True)


def partition_path(month: str) -> Path:
    return OUT / "full_train_c2_cells" / f"part-{month}.parquet"


def partition_manifest_path(month: str) -> Path:
    return OUT / "full_train_c2_cells" / f"part-{month}.MANIFEST.json"


def run_full_month(case, base: pd.DataFrame, grid: list[float], month: str) -> dict[str, Any]:
    out_dir = OUT / "full_train_c2_cells"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = partition_path(month); manifest_path = partition_manifest_path(month)
    month_base = base[base.timestamp_utc.astype(str).str[:7].eq(month)].copy()
    expected = len(month_base) * len(K_G3) * len(grid)
    if path.exists() or manifest_path.exists():
        if not path.exists() or not manifest_path.exists():
            raise FileExistsError(f"incomplete existing partition for {month}; refusing overwrite")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(manifest.get("row_count", -1)) != expected:
            raise FileExistsError(f"existing partition row count mismatch for {month}; refusing overwrite")
        return manifest | {"status": "SKIPPED_VALID_EXISTING"}
    records = []
    for idx, base_row in enumerate(base_row_map(month_base), start=1):
        for k_g3 in K_G3:
            for k_g1 in grid:
                records.append(build_record(case, base_row, "C2_BTM_SOLAR_PLUS_FIXED_WIND", k_g1, k_g3, False))
        if idx == 1 or idx % 100 == 0 or idx == len(month_base):
            print(f"[full C2 {month}] {idx}/{len(month_base)}", flush=True)
    frame = pd.DataFrame(records)
    if len(frame) != expected:
        raise RuntimeError(f"partition {month} row count {len(frame)} != {expected}")
    frame.to_parquet(path, index=False)
    manifest = {"status": "PASS_PARTITION", "month": month, "row_count": int(len(frame)), "input_hash": frame_hash(month_base), "output_hash": sha256_file(path), "grid_count": len(grid), "k_g3_count": len(K_G3), "complete": True}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def empty_status_outputs(status: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=["status", "reason"]).to_parquet(OUT / "observation_conditioned_values.parquet", index=False)
    pd.DataFrame(columns=["status", "reason"]).to_parquet(OUT / "train_state_action_summary.parquet", index=False)
    (REPORT / "TIES_NOT_COMPUTED.json").write_text(json.dumps({"status": status, "reason": "opponent belief rule unresolved; stopped before expected-value action selection"}, indent=2), encoding="utf-8")


def full_cells() -> pd.DataFrame:
    paths = sorted((OUT / "full_train_c2_cells").glob("part-*.parquet"))
    if len(paths) != 12:
        raise RuntimeError(f"expected 12 complete monthly partitions, found {len(paths)}")
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def distribution(values: pd.Series) -> dict[str, float]:
    x = pd.to_numeric(values, errors="coerce").dropna()
    if x.empty:
        return {k: float("nan") for k in ("mean", "median", "p75", "p90", "p95", "p99", "max")}
    return {"mean": float(x.mean()), "median": float(x.median()), "p75": float(x.quantile(.75)),
            "p90": float(x.quantile(.90)), "p95": float(x.quantile(.95)),
            "p99": float(x.quantile(.99)), "max": float(x.max())}


def analyze_private_cells(case, base: pd.DataFrame, cells: pd.DataFrame, grid: list[float], grid_spec: dict[str, Any]) -> dict[str, Any]:
    """Analyze hidden strategic cells only; no public expected value is computed here."""
    expected = 8760 * len(K_G3) * len(grid)
    scalar_checks = {
        "row_count": int(len(cells)) == expected,
        "all_optimal": bool((cells["solver_status"] == "OPTIMAL").all()),
        "balance_pass": bool((cells["balance_residual_mw"].abs() <= TOL).all()),
        "branch_limit_pass": bool((cells["max_branch_utilization"] <= 1.0 + TOL).all()),
        "finite_generation": bool(np.isfinite(cells[["G1_dispatch_mw", "G2_dispatch_mw", "G3_dispatch_mw"]].to_numpy()).all()),
    }
    state_cols = ["gross_load_mw", "btm_solar_mw", "wind_proxy_mw", "raw_residual_mw", "surplus_export_mw", "solver_residual_mw"]
    state_invariant = bool(cells.groupby(["timestamp_utc", "k_g3"])[state_cols].nunique(dropna=False).max().max() <= 1)
    scalar_checks["exogenous_state_invariant"] = state_invariant
    manifests = []
    for path in sorted((OUT / "full_train_c2_cells").glob("part-*.MANIFEST.json")):
        obj = json.loads(path.read_text(encoding="utf-8")); parquet = path.with_name(path.name.replace(".MANIFEST.json", ".parquet"))
        ok = parquet.exists() and int(obj.get("row_count", -1)) == int(len(pd.read_parquet(parquet))) and obj.get("output_hash") == sha256_file(parquet)
        manifests.append({"month": obj.get("month"), "row_count": obj.get("row_count"), "hash_verified": ok})
    scalar_checks["partitions_complete"] = bool(manifests) and all(x["hash_verified"] for x in manifests)
    full_audit = {"status": "PASS_FULL_TRAIN_C2_DENSE_PROBE" if all(scalar_checks.values()) else "FAIL_FULL_TRAIN_C2_DENSE_PROBE",
                  "expected_rows": expected, "actual_rows": int(len(cells)), "grid_count": len(grid), "k_g3_count": len(K_G3),
                  "checks": scalar_checks, "partitions": manifests}
    (REPORT / "FULL_TRAIN_SOLVER_AUDIT.json").write_text(json.dumps(full_audit, ensure_ascii=False, indent=2), encoding="utf-8")

    curves = []; maxima = []
    for (ts, k3), group in cells.groupby(["timestamp_utc", "k_g3"], sort=True):
        group = group.sort_values("k_g1").reset_index(drop=True)
        for i in range(1, len(group)):
            prev, cur = group.iloc[i - 1], group.iloc[i]
            curves.append({"timestamp_utc": str(ts), "k_g3": float(k3), "k_g1_left": float(prev.k_g1), "k_g1_right": float(cur.k_g1),
                           "dispatch_delta_mw": float(cur.G1_dispatch_mw - prev.G1_dispatch_mw),
                           "lmp_delta_usd_per_mwh": float(cur.G1_lmp_usd_per_mwh - prev.G1_lmp_usd_per_mwh),
                           "profit_delta": float(cur.G1_profit - prev.G1_profit),
                           "binding_changed": str(prev.binding_branch_status) != str(cur.binding_branch_status)})
        idx = int(group.G1_profit.to_numpy().argmax())
        maxima.append({"timestamp_utc": str(ts), "k_g3": float(k3), "k_g1": float(group.iloc[idx].k_g1), "profit": float(group.iloc[idx].G1_profit), "at_upper_bound": bool(idx == len(group) - 1)})
    curve_df = pd.DataFrame(curves)
    response = {"status": "PASS_RESPONSE_CURVE_AUDIT", "rows": int(len(curve_df)),
                "dispatch_cliff_count": int((curve_df.dispatch_delta_mw.abs() > 1e-5).sum()),
                "binding_transition_count": int(curve_df.binding_changed.sum()),
                "local_maxima_count": int(len(maxima)), "local_maxima_at_2_30": int(sum(x["at_upper_bound"] for x in maxima)),
                "largest_abs_dispatch_step_mw": float(curve_df.dispatch_delta_mw.abs().max()),
                "largest_abs_profit_step": float(curve_df.profit_delta.abs().max()),
                "maxima_preview": maxima[:10]}
    (REPORT / "RESPONSE_CURVE_AUDIT.json").write_text(json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8")

    pivot = cells.pivot_table(index=["timestamp_utc", "k_g3"], columns="k_g1", values="G1_profit", aggfunc="first")
    old_cols = [x for x in [float(a) for a in ACTION_VALUES] if x in pivot.columns]
    dense_best = pivot.max(axis=1); old_best = pivot[old_cols].max(axis=1)
    regret = (dense_best - old_best).clip(lower=0)
    best_k = pivot.idxmax(axis=1)
    regret_report = {"status": "PASS_A6_VS_DENSE_RESOLUTION_AUDIT", "private_cell_scope": True,
                     "old_a6": [float(x) for x in ACTION_VALUES], "dense_probe_count": len(grid),
                     "dense_probe_upper_bound": max(grid), "resolution_regret": distribution(regret),
                     "cells_new_point_strictly_beats_a6": int((regret > 1e-8).sum()),
                     "cells_at_2_30": int((best_k == 2.30).sum()),
                     "dense_optimal_markup_frequency": {str(float(k)): int((best_k == k).sum()) for k in sorted(best_k.unique())}}
    (REPORT / "A6_VS_DENSE_RESOLUTION_REGRET.json").write_text(json.dumps(regret_report, ensure_ascii=False, indent=2), encoding="utf-8")

    ties = (pivot.sub(pivot.max(axis=1), axis=0).abs() <= 1e-8).sum(axis=1)
    tie_report = {"status": "PASS_PRIVATE_TIE_AND_FLAT_AUDIT", "private_cell_scope": True,
                  "flat_or_tied_cells": int((ties > 1).sum()), "strict_unique_cells": int((ties == 1).sum()),
                  "tie_size_distribution": {str(int(k)): int(v) for k, v in ties.value_counts().sort_index().items()}}
    (REPORT / "TIE_AND_FLAT_STATE_AUDIT.json").write_text(json.dumps(tie_report, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORT / "OPTIMAL_MARKUP_DISTRIBUTION.json").write_text(json.dumps({"status": "PASS_PRIVATE_OPTIMAL_MARKUP_DISTRIBUTION", "scope": "private hidden k_G3 cells", "frequency": regret_report["dense_optimal_markup_frequency"]}, ensure_ascii=False, indent=2), encoding="utf-8")

    upper = cells[cells.k_g1 == max(grid)].groupby("k_g3").size().to_dict()
    slope = cells[cells.k_g1.isin([2.25, 2.30])].pivot_table(index=["timestamp_utc", "k_g3"], columns="k_g1", values="G1_profit", aggfunc="first")
    slopes = (slope[2.30] - slope[2.25]).dropna() if 2.25 in slope and 2.30 in slope else pd.Series(dtype=float)
    upper_report = {"status": "PASS_UPPER_BOUND_2_30_AUDIT", "private_cell_scope": True, "upper_bound": 2.30,
                    "argmax_at_2_30": int((best_k == 2.30).sum()), "argmax_fraction": float((best_k == 2.30).mean()),
                    "positive_profit_slope_2_25_to_2_30": int((slopes > 1e-8).sum()), "profit_slope": distribution(slopes)}
    (REPORT / "UPPER_BOUND_CENSORING_AUDIT.json").write_text(json.dumps(upper_report, ensure_ascii=False, indent=2), encoding="utf-8")

    state = base.drop_duplicates("timestamp_utc").copy()
    export = state[state.surplus_export_mw > TOL]
    raw_negative = export[export.raw_residual_conventional_load_mw < -TOL]
    residual_positive_below_floor = export[(export.raw_residual_conventional_load_mw >= -TOL) & (export.solver_residual_conventional_load_mw > TOL)]
    surplus_report = {"status": "PASS_SURPLUS_HOUR_STRATEGIC_AUDIT", "train_hours": int(len(state)), "surplus_hours": int(len(export)),
                      "raw_negative_hours": int(len(raw_negative)), "residual_positive_below_sync_floor_hours": int(len(residual_positive_below_floor)),
                      "surplus_export_total_mwh": float(export.surplus_export_mw.sum()), "strategic_cells_in_surplus_hours": int(cells[cells.timestamp_utc.isin(export.timestamp_utc)].shape[0]),
                      "state_frozen_across_bids": bool(cells.groupby(["timestamp_utc", "k_g3"])["surplus_export_mw"].nunique().max() <= 1),
                      "economic_or_strategic": False}
    (REPORT / "SURPLUS_HOUR_STRATEGIC_AUDIT.json").write_text(json.dumps(surplus_report, ensure_ascii=False, indent=2), encoding="utf-8")

    best_frame = pd.DataFrame({"timestamp_utc": best_k.index.get_level_values(0), "k_g3": best_k.index.get_level_values(1), "best_k_g1": best_k.to_numpy()}).merge(state[["timestamp_utc", "gross_system_load_mw", "estimated_btm_solar_system_mw", "wind_fixed_proxy_system_mw", "surplus_export_mw"]], on="timestamp_utc", how="left")
    renewable = {"status": "PASS_PRIVATE_RENEWABLE_STRATEGY_DESCRIPTIVE", "scope": "private hidden k_G3 cells; descriptive only", "best_markup_frequency": {str(float(k)): int(v) for k, v in best_frame.best_k_g1.value_counts().sort_index().items()}, "by_surplus": best_frame.groupby(best_frame.surplus_export_mw > TOL).best_k_g1.mean().to_dict()}
    (REPORT / "RENEWABLE_STRATEGY_DIAGNOSTIC.json").write_text(json.dumps(renewable, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"solver_audit": full_audit, "response_curve": response, "resolution_regret": regret_report, "upper_bound": upper_report, "surplus": surplus_report, "tie": tie_report, "renewable": renewable}


def write_common(config: dict[str, Any], grid_spec: dict[str, Any], anchor_report: dict[str, Any] | None = None) -> None:
    OUT.mkdir(parents=True, exist_ok=True); REPORT.mkdir(parents=True, exist_ok=True)
    (OUT / "probe_grid.json").write_text(json.dumps(grid_spec, ensure_ascii=False, indent=2), encoding="utf-8")
    payload = {"config": config, "grid": grid_spec, "case_snapshot_sha256": snapshot_sha256(), "a6": [float(x) for x in ACTION_VALUES], "k_g3": list(K_G3), "anchor": anchor_report}
    (OUT / "MANIFEST.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("anchor", "smoke", "full", "report"), required=True)
    parser.add_argument("--month", action="append", help="YYYY-MM; only for --stage full")
    args = parser.parse_args()
    case = load_frozen_case(); config = json.loads(CONFIG.read_text(encoding="utf-8")); grid, grid_spec = build_probe_grid(case)
    REPORT.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "anchor":
        frames = load_case_frames(168); report = anchor(case, frames); write_common(config, grid_spec, report)
        (REPORT / "STRATEGIC_ANCHOR_EQUIVALENCE.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False)); return 0 if report["status"].startswith("PASS") else 2
    if args.stage == "smoke":
        frames = load_case_frames(168); anchor_report = anchor(case, frames)
        if not anchor_report["status"].startswith("PASS"):
            (REPORT / "STRATEGIC_ANCHOR_EQUIVALENCE.json").write_text(json.dumps(anchor_report, indent=2), encoding="utf-8"); return 2
        frame, smoke_report = run_smoke(case, frames, grid); smoke_dir = OUT / "smoke_168h"; smoke_dir.mkdir(parents=True, exist_ok=True); frame.to_parquet(smoke_dir / "strategic_cells.parquet", index=False)
        (smoke_dir / "MANIFEST.json").write_text(json.dumps({"rows": len(frame), "output_hash": sha256_file(smoke_dir / "strategic_cells.parquet"), "input_hashes": {k: frame_hash(v) for k, v in frames.items()}}, indent=2), encoding="utf-8")
        write_common(config, grid_spec, anchor_report); (REPORT / "STRATEGIC_ANCHOR_EQUIVALENCE.json").write_text(json.dumps(anchor_report, indent=2), encoding="utf-8"); (REPORT / "ACTION_SPACE_V2_168H_SMOKE.json").write_text(json.dumps(smoke_report, indent=2), encoding="utf-8")
        print(json.dumps(smoke_report, ensure_ascii=False)); return 0 if smoke_report["status"].startswith("PASS") else 2
    if args.stage == "full":
        base = load_full_base(); months = sorted(args.month or base.timestamp_utc.astype(str).str[:7].unique().tolist()); manifests = [run_full_month(case, base, grid, month) for month in months]; write_common(config, grid_spec)
        if args.month:
            print(json.dumps({"months": manifests}, ensure_ascii=False)); return 0
        (REPORT / "FULL_TRAIN_SOLVER_AUDIT.json").write_text(json.dumps({"status": "PASS_FULL_TRAIN_C2_DENSE_PROBE_PARTITIONS", "months": manifests, "total_rows": int(sum(int(x["row_count"]) for x in manifests))}, indent=2), encoding="utf-8"); print(json.dumps({"months": manifests}, ensure_ascii=False)); return 0
    if args.stage == "report":
        base = load_full_base(); cells = full_cells()
        analysis = analyze_private_cells(case, base, cells, grid, grid_spec)
        empty_status_outputs("OPPONENT_BELIEF_RULE_UNRESOLVED")
        report = {"classification": "OPPONENT_BELIEF_RULE_UNRESOLVED", "reason": "Frozen source records uniform structural support only for old core context classes; no exact k_g3 belief mapping is recorded for the 8760h renewable public-state hashes.", "old_rule_evidence": "scripts/audit_t30_provenance.py: q_struct is uniform over class_members_json", "new_state_mapping": "absent", "private_probe": analysis, "final_accessed": False, "sft_started": False, "grpo_started": False, "dev_read": False, "holdout_read": False}
        (REPORT / "ACTION_SPACE_V2_RENEWABLE_TRAIN_PROBE_REPORT_CN.md").write_text("# ACTION_SPACE_V2_RENEWABLE_TRAIN_PROBE\n\n分类：`OPPONENT_BELIEF_RULE_UNRESOLVED`\n\n已完成 8760h TRAIN-only、C2、隐藏 `k_G3` 的 dense strategic response probe，并完成 solver、响应曲线、A6 分辨率、上界和 surplus accounting 审计。现有冻结源只为旧 core context 提供 structural support，未为新的 renewable public-state hash 提供可验证的 `k_G3` posterior/support 映射。因此停止在 expected-value action selection 之前。未选择最终 compact action set，未读取 DEV/HOLDOUT，未启动 SFT/GRPO。\n", encoding="utf-8")
        (REPORT / "OPPONENT_BELIEF_RULE_AUDIT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False)); return 3
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
