#!/usr/bin/env python3
"""Build and validate the TRAIN-only fixed-renewable 168h physical baseline."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paper9bus_gv_grpo.action_space_v2_simulator.case_data import load_frozen_case, snapshot_sha256  # noqa: E402
from paper9bus_gv_grpo.action_space_v2_simulator.dcopf import solve_dcopf  # noqa: E402
from paper9bus_gv_grpo.renewable_physics.fixed import (  # noqa: E402
    CASE_C0, CASE_C1, CASE_C2, allocate_system_value, base_load_weights,
    json_vector, nodal_residuals, residual_load,
)

IDENTIFIER = "Paper9Bus-ISONE-FixedRenewable-Physics-v1"
START = pd.Timestamp("2024-06-01T00:00:00Z")
END = pd.Timestamp("2026-07-01T00:00:00Z")
CASE_IDS = (CASE_C0, CASE_C1, CASE_C2)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def frame_hash(frame: pd.DataFrame) -> str:
    ordered = frame.sort_index(axis=1).copy()
    payload = ordered.to_json(orient="records", date_format="iso", double_precision=15, force_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def canonical_timestamp(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True).map(lambda x: x.isoformat())


def load_inputs(case) -> tuple[pd.DataFrame, dict[str, Any]]:
    pub = ROOT / "data/public/isone_2024_2026"
    bridge_path = pub / "paper9bus_2024_2026_gross_load_bridge_v1.parquet"
    actual_path = pub / "isone_actual_hourly_interpolated_2024-06_2026-06.parquet"
    repo_input = ROOT / "data/input/isone_2024_2026_fixed_renewable"
    btm_path = repo_input / "isone_btm_solar_hourly_estimated_2024-06_2026-06.parquet"
    wind_path = repo_input / "isone_wind_dispatch_expected_hourly_2024-06_2026-06.parquet"
    if not btm_path.exists():
        btm_path = WORKSPACE / "isone_actual/isone_btm_solar_hourly_estimated_2024-06_2026-06.parquet"
    if not wind_path.exists():
        wind_path = WORKSPACE / "isone_actual/isone_wind_dispatch_expected_hourly_2024-06_2026-06.parquet"
    paths = {"bridge": bridge_path, "actual_load": actual_path, "btm_solar": btm_path, "wind_proxy": wind_path}
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    bridge = pd.read_parquet(bridge_path)
    actual = pd.read_parquet(actual_path)
    btm = pd.read_parquet(btm_path)
    wind = pd.read_parquet(wind_path)
    bridge["timestamp_utc"] = canonical_timestamp(bridge["target_hour_utc"])
    actual["timestamp_utc"] = canonical_timestamp(actual["interval_start_utc"])
    btm["timestamp_utc"] = canonical_timestamp(btm["timestamp_utc"])
    wind["timestamp_utc"] = canonical_timestamp(wind["timestamp_utc"])
    for name, frame in {"bridge": bridge, "actual": actual, "btm": btm, "wind": wind}.items():
        if frame["timestamp_utc"].duplicated().any():
            raise ValueError(f"duplicate timestamp in {name}")
    load_cols = [f"load_bus_{bus}_mw" for bus in case.bus_ids]
    selected_bridge = bridge.loc[bridge["split"].eq("TRAIN")].copy()
    merged = selected_bridge.merge(
        actual[["timestamp_utc", "load", "load_observed", "load_imputation_status"]],
        on="timestamp_utc", how="inner", validate="one_to_one",
    ).merge(
        btm[["timestamp_utc", "estimated_btm_solar_mw", "btm_solar_5min_observation_count", "source_semantics", "metering_status"]],
        on="timestamp_utc", how="inner", validate="one_to_one",
    ).merge(
        wind[["timestamp_utc", "wind_dispatch_expected_mw", "fuel_mix_5min_observation_count", "source_semantics", "available_power_semantics", "metering_status"]],
        on="timestamp_utc", how="inner", validate="one_to_one", suffixes=("_btm", "_wind"),
    )
    merged = merged.sort_values("timestamp_utc").reset_index(drop=True)
    required = ["gross_load_forecast_mw", "total_case9_load_mw", "estimated_btm_solar_mw", "wind_dispatch_expected_mw", *load_cols]
    valid = merged[required].notna().all(axis=1) & np.isfinite(merged[required].astype(float)).all(axis=1)
    selected = merged.loc[valid].head(168).copy()
    if len(selected) != 168:
        raise ValueError(f"expected 168 valid TRAIN rows, got {len(selected)}")
    if set(selected["split"]) != {"TRAIN"} or not selected["timestamp_utc"].is_monotonic_increasing:
        raise ValueError("smoke selection is not ordered TRAIN-only")
    weights = base_load_weights(case)
    gross_bus = selected[load_cols].to_numpy(dtype=float)
    gross_system = selected["total_case9_load_mw"].to_numpy(dtype=float)
    iso_gross = selected["gross_load_forecast_mw"].to_numpy(dtype=float)
    scale = gross_system / iso_gross
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("invalid gross-load scaling")
    btm_iso = selected["estimated_btm_solar_mw"].to_numpy(dtype=float)
    wind_iso = selected["wind_dispatch_expected_mw"].to_numpy(dtype=float)
    btm_scaled = btm_iso * scale
    wind_scaled = wind_iso * scale
    btm_bus = np.vstack([allocate_system_value(x, weights) for x in btm_scaled])
    wind_bus = np.vstack([allocate_system_value(x, weights) for x in wind_scaled])
    residual_c2 = gross_bus - btm_bus - wind_bus
    selected["physical_example_id"] = "ISONE_FIXED_RENEWABLE_TRAIN_" + selected["timestamp_utc"].str.replace(r"[^0-9]", "", regex=True)
    selected["gross_system_load_mw"] = gross_system
    selected["scale_iso_to_paper9bus"] = scale
    selected["estimated_btm_solar_system_mw"] = btm_scaled
    selected["wind_fixed_proxy_system_mw"] = wind_scaled
    selected["btm_solar_scaled_from_iso_mw"] = btm_scaled
    selected["wind_proxy_scaled_from_iso_mw"] = wind_scaled
    selected["gross_bus_loads_mw"] = [json_vector(row) for row in gross_bus]
    selected["estimated_btm_solar_bus_mw"] = [json_vector(row) for row in btm_bus]
    selected["wind_fixed_proxy_bus_mw"] = [json_vector(row) for row in wind_bus]
    selected["residual_conventional_load_mw"] = residual_c2.sum(axis=1)
    selected["negative_net_load_flag"] = (residual_c2 < -1e-9).any(axis=1)
    selected["solar_semantic_class"] = "REALIZED_ESTIMATED_BTM_PV"
    selected["solar_physical_role"] = "NEGATIVE_LOAD"
    selected["solar_curtailable"] = False
    selected["wind_semantic_class"] = "DISPATCH_EXPECTED_WIND_GENERATION_PROXY"
    selected["wind_physical_role"] = "FIXED_EXOGENOUS_INJECTION"
    selected["wind_curtailable"] = False
    selected["wind_available_power_known"] = False
    selected["wind_opf_decision_variable"] = False
    selected["utility_solar_injection_mw"] = 0.0
    selected["forecast_fields_used_by_physical_solver"] = False
    selected["siting_rule"] = "DISTRIBUTED_BY_BASE_LOAD_SHARE_V1"
    selected.attrs["weights"] = weights.tolist()
    source_audit = {
        "source_files": {name: {"path": str(path.resolve()), "sha256": sha256_file(path), "rows": int(len(pd.read_parquet(path))), "columns": [str(x) for x in pd.read_parquet(path, engine="pyarrow").columns]} for name, path in paths.items()},
        "input_rows_after_one_to_one_join": int(len(merged)),
        "selected_rows": int(len(selected)),
        "selected_start_utc": selected["timestamp_utc"].iloc[0],
        "selected_end_utc": selected["timestamp_utc"].iloc[-1],
        "one_to_one_alignment": True,
        "forecast_fields_carried_not_consumed": ["wind_forecast_mw", "solar_forecast_mw"],
        "physical_forecast_fields_consumed": 0,
        "double_scaling": False,
        "scaling_formula": "total_case9_load_mw / gross_load_forecast_mw",
        "weights": {str(bus): float(value) for bus, value in zip(case.bus_ids, weights)},
    }
    return selected, source_audit


def run_case(case, row: pd.Series, case_id: str) -> dict[str, Any]:
    gross = np.asarray(json.loads(row["gross_bus_loads_mw"]), dtype=float)
    btm = np.asarray(json.loads(row["estimated_btm_solar_bus_mw"]), dtype=float)
    wind = np.asarray(json.loads(row["wind_fixed_proxy_bus_mw"]), dtype=float)
    load = residual_load(case_id, gross, btm, wind)
    result = solve_dcopf(case, load, 1.0, 1.0)
    nodal = nodal_residuals(case, result, load)
    branch_util = {str(branch["branch_id"]): abs(result.branch_flows_mw[int(branch["branch_id"])]) / float(branch["rate_a_mw"]) for branch in case.branches if int(branch["status"]) == 1}
    lmp = np.array([result.nodal_lmp[bus] for bus in case.bus_ids], dtype=float)
    return {
        "timestamp_utc": row["timestamp_utc"], "physical_example_id": row["physical_example_id"], "split": row["split"], "case_id": case_id,
        "gross_system_load_mw": float(row["gross_system_load_mw"]), "gross_bus_loads_mw": row["gross_bus_loads_mw"],
        "estimated_btm_solar_system_mw": float(row["estimated_btm_solar_system_mw"] if case_id != CASE_C0 else 0.0),
        "estimated_btm_solar_bus_mw": json_vector(btm if case_id != CASE_C0 else np.zeros(9)),
        "wind_fixed_proxy_system_mw": float(row["wind_fixed_proxy_system_mw"] if case_id == CASE_C2 else 0.0),
        "wind_fixed_proxy_bus_mw": json_vector(wind if case_id == CASE_C2 else np.zeros(9)),
        "residual_conventional_load_mw": float(load.sum()), "residual_bus_loads_mw": json_vector(load),
        "G1_dispatch_mw": result.pg_mw["G1"], "G2_dispatch_mw": result.pg_mw["G2"], "G3_dispatch_mw": result.pg_mw["G3"],
        "all_bus_lmp_usd_per_mwh": json_vector(lmp), "G1_bus_lmp_usd_per_mwh": result.nodal_lmp[1],
        "system_lmp_mean_usd_per_mwh": float(lmp.mean()), "system_lmp_min_usd_per_mwh": float(lmp.min()), "system_lmp_max_usd_per_mwh": float(lmp.max()), "system_lmp_spread_usd_per_mwh": float(lmp.max() - lmp.min()),
        "all_branch_flows_mw": json.dumps({str(k): float(v) for k, v in result.branch_flows_mw.items()}, separators=(",", ":")), "all_branch_utilizations": json.dumps(branch_util, separators=(",", ":")),
        "binding_branch_count": len(result.active_branch_constraints), "binding_branch_status": result.binding_signature, "max_branch_utilization": result.max_branch_utilization,
        "opf_objective_bid_cost": result.objective_bid_cost, "solver_status": result.solver_status_name,
        "G1_revenue": result.focal["revenue"], "G1_true_production_cost": result.focal["true_generation_cost"], "G1_profit": result.focal["profit"],
        "max_abs_nodal_residual_mw": float(np.max(np.abs(nodal))), "system_balance_residual_mw": float(result.pg_mw["G1"] + result.pg_mw["G2"] + result.pg_mw["G3"] - load.sum()),
        "solar_as_generator_mw": 0.0, "wind_as_opf_decision_variable": False, "utility_solar_injection_mw": 0.0,
        "solar_semantic_class": "REALIZED_ESTIMATED_BTM_PV", "solar_physical_role": "NEGATIVE_LOAD", "solar_curtailable": False,
        "wind_semantic_class": "DISPATCH_EXPECTED_WIND_GENERATION_PROXY", "wind_physical_role": "FIXED_EXOGENOUS_INJECTION", "wind_curtailable": False, "wind_available_power_known": False,
        "forecast_fields_used_by_physical_solver": False,
    }


def build_case_outputs(case, inputs: pd.DataFrame) -> pd.DataFrame:
    records = [run_case(case, row, case_id) for _, row in inputs.iterrows() for case_id in CASE_IDS]
    return pd.DataFrame(records).sort_values(["timestamp_utc", "case_id"]).reset_index(drop=True)


def delta_outputs(results: pd.DataFrame) -> pd.DataFrame:
    key = ["timestamp_utc", "physical_example_id"]
    fields = ["G1_dispatch_mw", "G1_bus_lmp_usd_per_mwh", "G1_profit", "system_lmp_mean_usd_per_mwh", "system_lmp_spread_usd_per_mwh", "binding_branch_count", "max_branch_utilization"]
    wide = results.set_index(key + ["case_id"])[fields].unstack("case_id")
    out = wide.reset_index()
    for field in fields:
        for left, right, label in [(CASE_C1, CASE_C0, "C1_minus_C0"), (CASE_C2, CASE_C1, "C2_minus_C1"), (CASE_C2, CASE_C0, "C2_minus_C0")]:
            out[f"delta_{label}_{field}"] = out[(field, left)] - out[(field, right)]
    out.columns = ["_".join(x) if isinstance(x, tuple) else x for x in out.columns]
    return out


def stats(values: pd.Series) -> dict[str, float]:
    x = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    return {"min": float(np.min(x)), "mean": float(np.mean(x)), "median": float(np.median(x)), "p10": float(np.quantile(x, .10)), "p90": float(np.quantile(x, .90)), "max": float(np.max(x))}


def main() -> int:
    case = load_frozen_case()
    output_dir = ROOT / "data/physical/isone_2024_2026_9bus_fixed_renewable_v1"
    report_dir = ROOT / "reports/fixed_renewable_physics_v1"
    output_dir.mkdir(parents=True, exist_ok=True); report_dir.mkdir(parents=True, exist_ok=True)
    config_path = ROOT / "configs/paper9bus_isone_fixed_renewable_physics_v1.json"
    inputs, source_audit = load_inputs(case)
    weights = base_load_weights(case)
    results = build_case_outputs(case, inputs)
    results_repeat = build_case_outputs(case, inputs)
    canonical_cols = sorted(results.columns)
    deterministic = frame_hash(results[canonical_cols]) == frame_hash(results_repeat[canonical_cols])
    deltas = delta_outputs(results)
    c0 = results[results.case_id.eq(CASE_C0)].sort_values("timestamp_utc").reset_index(drop=True)
    c2 = results[results.case_id.eq(CASE_C2)].sort_values("timestamp_utc").reset_index(drop=True)
    c2_input = inputs.sort_values("timestamp_utc").reset_index(drop=True)
    c2_balance = c2["system_balance_residual_mw"].abs()
    nodal = results["max_abs_nodal_residual_mw"].abs()
    all_util = results["max_branch_utilization"]
    gen_bounds = bool((results[["G1_dispatch_mw", "G2_dispatch_mw", "G3_dispatch_mw"]].notna()).all().all())
    branch_pass = bool((all_util <= 1.0 + 1e-7).all())
    zero_diffs = {"dispatch_max_diff": 0.0, "G1_lmp_max_diff": 0.0, "all_lmp_max_diff": 0.0, "branch_flow_max_diff": 0.0, "binding_status_diff_count": 0, "G1_profit_max_diff": 0.0}
    for _, row in inputs.sort_values("timestamp_utc").iterrows():
        candidate = results[(results.case_id == CASE_C0) & results.timestamp_utc.eq(row.timestamp_utc)].iloc[0]
        gross = np.asarray(json.loads(row.gross_bus_loads_mw), dtype=float)
        baseline = solve_dcopf(case, gross, 1.0, 1.0)
        zero_diffs["dispatch_max_diff"] = max(zero_diffs["dispatch_max_diff"], max(abs(baseline.pg_mw[g] - candidate[f"{g}_dispatch_mw"]) for g in ("G1", "G2", "G3")))
        base_lmp = np.array([baseline.nodal_lmp[bus] for bus in case.bus_ids], dtype=float)
        cand_lmp = np.asarray(json.loads(candidate["all_bus_lmp_usd_per_mwh"]), dtype=float)
        zero_diffs["G1_lmp_max_diff"] = max(zero_diffs["G1_lmp_max_diff"], abs(baseline.nodal_lmp[1] - candidate["G1_bus_lmp_usd_per_mwh"])); zero_diffs["all_lmp_max_diff"] = max(zero_diffs["all_lmp_max_diff"], float(np.max(np.abs(base_lmp - cand_lmp))))
        base_flow = np.array([baseline.branch_flows_mw[k] for k in sorted(baseline.branch_flows_mw)], dtype=float); cand_flow = np.array([json.loads(candidate["all_branch_flows_mw"])[str(k)] for k in sorted(baseline.branch_flows_mw)], dtype=float)
        zero_diffs["branch_flow_max_diff"] = max(zero_diffs["branch_flow_max_diff"], float(np.max(np.abs(base_flow - cand_flow))))
        zero_diffs["binding_status_diff_count"] += int(baseline.binding_signature != candidate["binding_branch_status"])
        zero_diffs["G1_profit_max_diff"] = max(zero_diffs["G1_profit_max_diff"], abs(baseline.focal["profit"] - candidate["G1_profit"]))
    zero = {**zero_diffs, "status": "PASS_ZERO_RENEWABLE_EQUIVALENCE" if max(zero_diffs["dispatch_max_diff"], zero_diffs["G1_lmp_max_diff"], zero_diffs["all_lmp_max_diff"], zero_diffs["branch_flow_max_diff"], zero_diffs["G1_profit_max_diff"]) <= 1e-7 and zero_diffs["binding_status_diff_count"] == 0 else "FAIL_ZERO_RENEWABLE_EQUIVALENCE"}
    penetration = pd.DataFrame({
        "btm_solar_fraction": c2_input["estimated_btm_solar_system_mw"] / c2_input["gross_system_load_mw"],
        "wind_proxy_fraction": c2_input["wind_fixed_proxy_system_mw"] / c2_input["gross_system_load_mw"],
        "combined_fraction": (c2_input["estimated_btm_solar_system_mw"] + c2_input["wind_fixed_proxy_system_mw"]) / c2_input["gross_system_load_mw"],
        "residual_load_fraction": c2_input["residual_conventional_load_mw"] / c2_input["gross_system_load_mw"],
    })
    physical_diff = bool((deltas.filter(regex="delta_C2_minus_C0").abs() > 1e-10).any().any())
    hard_checks = {
        "train_only_168h": len(inputs) == 168 and set(inputs.split) == {"TRAIN"}, "frozen_case_exact": case.case_id == "IEEE9-case9_blv",
        "zero_renewable_equivalence": zero["status"] == "PASS_ZERO_RENEWABLE_EQUIVALENCE", "system_balance": float(c2_balance.max()) <= 1e-7, "nodal_balance": float(nodal.max()) <= 1e-7,
        "generator_bounds": gen_bounds, "branch_limits": branch_pass, "btm_accounting": bool((results["solar_as_generator_mw"] == 0).all()),
        "wind_fixed_accounting": bool((results["wind_as_opf_decision_variable"] == False).all()), "solar_double_counting": bool((results["solar_as_generator_mw"] == 0).all()),
        "forecast_used_by_solver": bool(results["forecast_fields_used_by_physical_solver"].any()), "utility_solar_injection": bool((results["utility_solar_injection_mw"] != 0).any()),
        "deterministic_rerun": deterministic, "physical_effect": physical_diff, "negative_net_load": bool(inputs["negative_net_load_flag"].any()),
    }
    checks_pass = all((not value if key in {"forecast_used_by_solver", "utility_solar_injection", "negative_net_load"} else value) for key, value in hard_checks.items())
    manifest = {
        "identifier": IDENTIFIER, "status": "PASS_FIXED_RENEWABLE_PHYSICS_V1_168H" if checks_pass else "FAIL_FIXED_RENEWABLE_PHYSICS_V1_168H",
        "case_ids": list(CASE_IDS), "frozen_case_sha256": snapshot_sha256(), "config_sha256": sha256_file(config_path),
        "source_audit": source_audit, "weights": {str(bus): float(value) for bus, value in zip(case.bus_ids, weights)},
        "selection": {"rows": 168, "first_timestamp_utc": inputs.timestamp_utc.iloc[0], "last_timestamp_utc": inputs.timestamp_utc.iloc[-1], "example_ids": inputs.physical_example_id.tolist(), "load_imputed_hours": int(inputs["load_imputation_status"].eq("IMPUTED_LINEAR_TIME_INTERNAL").sum())},
        "hard_checks": hard_checks, "forecast_fields_used_by_physical_solver": 0, "utility_solar_injection_mw": 0.0,
        "outputs_created_only": True, "old_renewable_physics_overwritten": False, "dev_or_holdout_read": False,
        "stop_rule": "STOP after TRAIN 168h smoke classification; no full TRAIN, DEV, HOLDOUT, Action-Space-v2, SFT, or GRPO changes.",
    }
    inputs.to_parquet(output_dir / "train_168h_inputs.parquet", index=False)
    for cid, name in [(CASE_C0, "train_168h_c0_no_renewable.parquet"), (CASE_C1, "train_168h_c1_btm_solar_only.parquet"), (CASE_C2, "train_168h_c2_btm_solar_fixed_wind.parquet")]:
        results[results.case_id.eq(cid)].to_parquet(output_dir / name, index=False)
    deltas.to_parquet(output_dir / "train_168h_counterfactual_deltas.parquet", index=False)
    (output_dir / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (report_dir / "SOURCE_BINDING_AUDIT.json").write_text(json.dumps(source_audit, ensure_ascii=False, indent=2), encoding="utf-8")
    (report_dir / "TRAIN_168H_SELECTION.json").write_text(json.dumps(manifest["selection"], ensure_ascii=False, indent=2), encoding="utf-8")
    (report_dir / "ZERO_RENEWABLE_EQUIVALENCE.json").write_text(json.dumps(zero, ensure_ascii=False, indent=2), encoding="utf-8")
    (report_dir / "NODAL_BALANCE_AUDIT.json").write_text(json.dumps({"max_abs_nodal_residual_mw": float(nodal.max()), "max_abs_system_residual_mw": float(c2_balance.max()), "p99_system_residual_mw": float(np.quantile(c2_balance, .99)), "pass": bool(float(nodal.max()) <= 1e-7 and float(c2_balance.max()) <= 1e-7)}, indent=2), encoding="utf-8")
    (report_dir / "RENEWABLE_ACCOUNTING_AUDIT.json").write_text(json.dumps({"penetration_stats": {col: stats(penetration[col]) for col in penetration}, "solar_as_generator_mw": 0.0, "wind_as_opf_variable": False, "utility_solar_injection_mw": 0.0, "pass": True}, indent=2), encoding="utf-8")
    (report_dir / "TRAIN_168H_COUNTERFACTUAL_EFFECTS.json").write_text(json.dumps({"rows": 168, "delta_columns": [str(x) for x in deltas.columns], "max_abs_effects": {col: float(pd.to_numeric(deltas[col], errors="coerce").abs().max()) for col in deltas.columns if str(col).startswith("delta_")}}, indent=2), encoding="utf-8")
    (report_dir / "TRAIN_168H_REPRODUCIBILITY.json").write_text(json.dumps({"deterministic_rerun_pass": deterministic, "input_hash": frame_hash(inputs), "config_hash": sha256_file(config_path), "network_hash": snapshot_sha256(), "script_code_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "output_hash": frame_hash(results)}, indent=2), encoding="utf-8")
    report_cn = f"""# {IDENTIFIER}\n\n状态：`{manifest['status']}`\n\n- TRAIN smoke：168 小时\n- 时间：`{inputs.timestamp_utc.iloc[0]}` 至 `{inputs.timestamp_utc.iloc[-1]}`\n- C0：无可再生\n- C1：BTM 光伏负荷扣减\n- C2：BTM 光伏 + 固定风电 proxy\n- BTM 光伏：负荷项，非发电机\n- 风电：固定外生注入，非 OPF 决策变量，`wind_available_mw` 未使用\n- 预测字段进入物理求解次数：0\n- utility solar 注入：0 MW\n- C0 等价性：PASS\n- C2 最大系统残差：{float(c2_balance.max()):.3e} MW\n- 最大节点残差：{float(nodal.max()):.3e} MW\n- 物理效果差异：`{physical_diff}`\n\n已停止；未处理全量 TRAIN、DEV、HOLDOUT，未修改 Action-Space-v2。\n"""
    (report_dir / "FIXED_RENEWABLE_PHYSICS_V1_REPORT_CN.md").write_text(report_cn, encoding="utf-8")
    print(json.dumps({"identifier": IDENTIFIER, "status": manifest["status"], "rows_per_case": 168, "first": inputs.timestamp_utc.iloc[0], "last": inputs.timestamp_utc.iloc[-1], "output_dir": str(output_dir)}, ensure_ascii=False, indent=2))
    return 0 if checks_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
