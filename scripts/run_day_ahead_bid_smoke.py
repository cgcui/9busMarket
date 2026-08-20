"""TRAIN-only 24-hour day-ahead bid schedule smoke test.

The schedule is generated without hidden opponent state or future payoff. The
existing private cell bank is used only for offline market-clearing evaluation.
No OPF is rerun by this script.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paper9bus_gv_grpo.day_ahead import make_schedule  # noqa: E402


FULL = ROOT / "data/physical/isone_2024_2026_9bus_full_train_fixed_renewable_v1"
CELL_ROOT = ROOT / "data/physical/action_space_v2_renewable_train_probe_v1/full_train_c2_cells"
OUT = ROOT / "data/physical/day_ahead_bid_v1"
REPORT = ROOT / "reports/day_ahead_bid_v1"
MARKUPS = (1.30, 1.50, 1.80)


def load_inputs() -> pd.DataFrame:
    frame = pd.read_parquet(FULL / "train_8760h_inputs.parquet", columns=["timestamp_utc", "gross_system_load_mw"])
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    frame["target_date_utc"] = frame.timestamp_utc.dt.strftime("%Y-%m-%d")
    frame["hour"] = frame.timestamp_utc.dt.hour
    if len(frame) != 8760 or frame.groupby(["target_date_utc", "hour"]).size().max() != 1:
        raise RuntimeError("TRAIN input is not a complete 8760-hour UTC calendar")
    return frame.sort_values("timestamp_utc").reset_index(drop=True)


def make_schedules(inputs: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict]]:
    daily = inputs.pivot(index="target_date_utc", columns="hour", values="gross_system_load_mw").sort_index()
    q33, q67 = [float(x) for x in inputs.gross_system_load_mw.quantile([1 / 3, 2 / 3])]
    dates = list(daily.index)
    schedule_rows = []
    payloads: dict[str, dict] = {}
    for target in dates:
        target_ts = pd.Timestamp(target, tz="UTC")
        prior = (target_ts - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        if prior in daily.index:
            forecast = daily.loc[prior].to_numpy(float)
            source = "TRAIN_ONLY_PREVIOUS_UTC_DAY_SAME_HOUR_PERSISTENCE"
        else:
            forecast = inputs.groupby("hour").gross_system_load_mw.median().reindex(range(24)).to_numpy(float)
            source = "TRAIN_ONLY_HOUR_MEDIAN_WARM_START"
        markups = np.where(forecast <= q33, MARKUPS[0], np.where(forecast <= q67, MARKUPS[1], MARKUPS[2]))
        cutoff = (target_ts - pd.Timedelta(days=1)).strftime("%Y-%m-%dT10:00:00Z")
        schedule = make_schedule(target, cutoff, forecast, markups, forecast_source=source)
        payloads[target] = schedule.to_dict()
        for hour, (forecast_value, markup, price) in enumerate(zip(schedule.forecast_load_mw, schedule.bid_markup_multiplier, schedule.bid_price_usd_per_mwh)):
            schedule_rows.append({"target_date_utc": target, "hour": hour, "decision_cutoff_utc": cutoff, "forecast_load_mw": forecast_value, "bid_markup_multiplier": markup, "bid_price_usd_per_mwh": price, "forecast_source": source})
    return pd.DataFrame(schedule_rows), {"q33_load_mw": q33, "q67_load_mw": q67, "schedules": payloads}


def load_cells() -> pd.DataFrame:
    columns = ["timestamp_utc", "k_g1", "k_g3", "G1_dispatch_mw", "G1_profit", "solver_status", "balance_residual_mw", "max_branch_utilization"]
    cells = pd.concat([pd.read_parquet(path, columns=columns) for path in sorted(CELL_ROOT.glob("part-*.parquet"))], ignore_index=True)
    cells["timestamp_utc"] = pd.to_datetime(cells["timestamp_utc"], utc=True)
    cells["target_date_utc"] = cells.timestamp_utc.dt.strftime("%Y-%m-%d")
    cells["hour"] = cells.timestamp_utc.dt.hour
    return cells


def evaluate(schedule_rows: pd.DataFrame, cells: pd.DataFrame) -> dict:
    selected = schedule_rows.merge(cells, on=["target_date_utc", "hour", "k_g1"], how="left") if "k_g1" in schedule_rows else None
    # Schedule prices are public-facing; map the submitted multiplier into the
    # private bank's k_g1 column only for offline evaluation.
    schedule_rows = schedule_rows.rename(columns={"bid_markup_multiplier": "k_g1"})
    selected = schedule_rows.merge(cells, on=["target_date_utc", "hour", "k_g1"], how="left", validate="many_to_many")
    expected = 8760 * 6
    checks = {
        "schedule_rows": int(len(schedule_rows)) == 8760,
        "evaluation_rows": int(len(selected)) == expected,
        "no_missing_cells": bool(selected.G1_profit.notna().all()),
        "all_optimal": bool((selected.solver_status == "OPTIMAL").all()),
        "balance_pass": bool((selected.balance_residual_mw.abs() <= 1e-7).all()),
        "branch_limit_pass": bool((selected.max_branch_utilization <= 1.0 + 1e-7).all()),
        "every_hour_has_price": bool(schedule_rows.groupby("target_date_utc").size().eq(24).all()),
    }
    if not all(checks.values()):
        return {"status": "FAIL_DAY_AHEAD_24H_BID_SMOKE", "checks": checks}
    flat = cells[np.isclose(cells.k_g1, 1.5)].copy()
    flat["target_date_utc"] = flat.timestamp_utc.dt.strftime("%Y-%m-%d"); flat["hour"] = flat.timestamp_utc.dt.hour
    flat = flat[["target_date_utc", "hour", "k_g3", "G1_profit"]].rename(columns={"G1_profit": "flat_profit"})
    selected = selected.merge(flat, on=["target_date_utc", "hour", "k_g3"], how="left", validate="one_to_one")
    oracle = cells.groupby(["target_date_utc", "hour", "k_g3"], sort=False).G1_profit.max().rename("private_probe_best_profit").reset_index()
    selected = selected.merge(oracle, on=["target_date_utc", "hour", "k_g3"], how="left", validate="one_to_one")
    selected["flat_regret"] = selected.flat_profit - selected.G1_profit
    selected["private_probe_regret"] = selected.private_probe_best_profit - selected.G1_profit
    by_scenario = selected.groupby("k_g3").agg(hours=("G1_profit", "size"), total_profit_usd=("G1_profit", "sum"), flat_total_profit_usd=("flat_profit", "sum"), private_probe_regret_usd=("private_probe_regret", "sum"), mean_bid_price_usd_per_mwh=("bid_price_usd_per_mwh", "mean")).reset_index()
    return {"status": "PASS_DAY_AHEAD_24H_BID_SMOKE", "checks": checks, "target_days": int(schedule_rows.target_date_utc.nunique()), "hours": int(len(schedule_rows)), "hidden_k_g3_scenarios": [float(x) for x in sorted(selected.k_g3.unique())], "scenario_summary": by_scenario.to_dict("records"), "schedule_uses_hidden_state": False, "oracle_used_for_schedule": False, "oracle_used_only_for_offline_regret_diagnostic": True}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True); REPORT.mkdir(parents=True, exist_ok=True)
    inputs = load_inputs(); rows, meta = make_schedules(inputs); rows.to_parquet(OUT / "DAY_AHEAD_BID_SCHEDULES.parquet", index=False)
    first_date = str(rows.target_date_utc.iloc[0]); (OUT / f"DAY_AHEAD_SAMPLE_{first_date}.json").write_text(json.dumps(meta["schedules"][first_date], ensure_ascii=False, indent=2), encoding="utf-8")
    result = evaluate(rows, load_cells()); result["forecast_policy"] = "previous UTC day same-hour persistence; hour-wise load terciles map to k_G1={1.30,1.50,1.80}"; result["forecast_thresholds_mw"] = {"q33": meta["q33_load_mw"], "q67": meta["q67_load_mw"]}
    (REPORT / "DAY_AHEAD_24H_BID_SMOKE.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORT / "DAY_AHEAD_BID_REPORT_CN.md").write_text(f"# 日前 24 小时报价接口\n\n分类：`{result['status']}`\n\n本测试一次生成次日 24 个 G1 报价，报价单位为 USD/MWh；使用前一日同小时 TRAIN 负荷持久性作为公开预测。现有 private cell bank 仅用于离线评价，未重新求解 OPF，未读取 DEV/HOLDOUT，未训练。\n\n- 目标日：{result.get('target_days')}\n- 报价小时：{result.get('hours')}\n- 隐藏 k_G3 情景：`{result.get('hidden_k_g3_scenarios')}`\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False)); return 0 if result["status"].startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
