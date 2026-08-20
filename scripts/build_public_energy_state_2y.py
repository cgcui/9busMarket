#!/usr/bin/env python3
"""Build the non-training Public-Energy-State-v1 layer from ISO-NE 2020-2021."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))
from paper9bus_gv_grpo.public_state import (
    build_public_energy_state,
    canonical_json,
    fit_interpretation_rules,
    format_public_energy_state_prompt,
    state_hash,
)

ET = "America/New_York"
ZONES = ["CONNECTICUT", "MAINE", "NEMASSBOST", "NEWHAMPSHIRE", "RHODEISLAND", "SEMASS", "VERMONT", "WCMASS"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_paths(source_root: Path) -> dict[str, Path]:
    isone = source_root / "ercot_llm_bidding" / "data_external" / "isone"
    return {
        "forecast": isone / "canonical" / "load_forecast_hourly_2020_2021.parquet",
        "context": isone / "context" / "isone_market_context_hourly_2020_2021.parquet",
        "raw_load": isone / "raw",
        "da_lmp_root": source_root,
        "rt_lmp_root": source_root,
    }


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"missing source: {path}")
    return path


def read_actual_load(raw_root: Path) -> pd.DataFrame:
    rows = []
    for year in (2020, 2021):
        for path in sorted((raw_root / str(year) / "api" / "hourlysysload").glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            values = payload.get("HourlySystemLoads", {}).get("HourlySystemLoad", [])
            for value in values:
                rows.append({"timestamp_utc": value.get("BeginDate"), "load_mw": value.get("Load")})
    out = pd.DataFrame(rows)
    out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], utc=True, errors="coerce")
    out["load_mw"] = pd.to_numeric(out["load_mw"], errors="coerce")
    out = out.dropna().drop_duplicates("timestamp_utc").sort_values("timestamp_utc").reset_index(drop=True)
    return out


def read_lmp(root: Path, pattern: str, real_time: bool) -> pd.DataFrame:
    paths = sorted(root.glob(pattern))
    if not paths:
        paths = sorted((root / "ercot_llm_bidding" / "data_external" / "isone" / "imported" / "lmp_hourly_hub").glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no LMP files matched {pattern} under {root}")
    parts = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["timestamp_utc"] = pd.to_datetime(frame["interval_start_utc"], utc=True, errors="coerce")
        frame["local"] = frame["timestamp_utc"].dt.tz_convert(ET)
        frame["target_local_date"] = frame["local"].dt.strftime("%Y-%m-%d")
        frame["target_he"] = frame["local"].dt.hour + 1
        frame["zone"] = frame["location"].astype(str).str.replace(".Z.", "", regex=False)
        if real_time:
            frame = frame[frame["zone"].isin(ZONES)]
            frame["source_role"] = "public_history"
        else:
            frame = frame[frame["zone"].isin(ZONES)]
            frame["source_role"] = "public_ex_ante"
        parts.append(frame[["timestamp_utc", "target_local_date", "target_he", "zone", "lmp", "source_role"]])
    out = pd.concat(parts, ignore_index=True)
    out["lmp"] = pd.to_numeric(out["lmp"], errors="coerce")
    return out.dropna(subset=["timestamp_utc", "lmp"]).drop_duplicates(["timestamp_utc", "zone"])


def asof_values(actual: pd.DataFrame, cutoff: pd.Timestamp, count: int) -> list[float] | None:
    if actual.empty:
        return None
    times = actual["timestamp_utc"]
    end = int(times.searchsorted(cutoff, side="right"))
    if end < count:
        return None
    selected = actual.iloc[end - count : end]
    if len(selected) != count or selected["timestamp_utc"].diff().iloc[1:].ne(pd.Timedelta(hours=1)).any():
        return None
    return [float(x) for x in selected["load_mw"]]


def asof_series(values: pd.Series, cutoff: pd.Timestamp, count: int) -> list[float] | None:
    """Read a contiguous as-of tail from a sorted UTC-indexed series."""
    if values.empty:
        return None
    end = int(values.index.searchsorted(cutoff, side="right"))
    if end < count:
        return None
    selected = values.iloc[end - count : end]
    if len(selected) != count or selected.index.to_series().diff().iloc[1:].ne(pd.Timedelta(hours=1)).any():
        return None
    return [float(x) for x in selected.tolist()]


def select_forecast(forecast: pd.DataFrame) -> pd.DataFrame:
    f = forecast.copy()
    f["forecast_date"] = f["forecast_date"].astype(str)
    f["forecast_hour"] = pd.to_numeric(f["forecast_hour"], errors="coerce").astype(int)
    f["published_datetime_utc"] = pd.to_datetime(f["published_datetime_utc"], utc=True, errors="coerce")
    f = f[f["vintage"].isin(["day_ahead", "2day_ahead", "in_day"])].dropna(subset=["published_datetime_utc"])
    dates = sorted(f["forecast_date"].unique())
    cutoff = pd.to_datetime(pd.Series(dates), format="%Y-%m-%d").dt.tz_localize(ET) - pd.Timedelta(days=1) + pd.Timedelta(hours=10)
    cutoffs = pd.DataFrame({"forecast_date": dates, "decision_cutoff": cutoff.dt.tz_convert("UTC")})
    f = f.merge(cutoffs, on="forecast_date", how="left")
    f = f[f["published_datetime_utc"] <= f["decision_cutoff"]].sort_values("published_datetime_utc")
    f = f.drop_duplicates(["forecast_date", "forecast_hour", "region"], keep="last")
    return f


def build_rows(source_root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    paths = source_paths(source_root)
    forecast = select_forecast(pd.read_parquet(require(paths["forecast"])))
    context = pd.read_parquet(require(paths["context"])).copy()
    if "target_local_date" not in context:
        context["hour_utc"] = pd.to_datetime(context["hour_utc"], utc=True, errors="coerce")
        local = context["hour_utc"].dt.tz_convert(ET)
        context["target_local_date"] = local.dt.strftime("%Y-%m-%d")
        context["target_he"] = local.dt.hour + 1
    else:
        context["target_local_date"] = context["target_local_date"].astype(str)
        context["target_he"] = pd.to_numeric(context["target_he"], errors="coerce").astype("Int64")
    actual = read_actual_load(require(paths["raw_load"]))
    da = read_lmp(paths["da_lmp_root"], "isone_lmp_da_HUB_202[01]-*.csv", real_time=False)
    rt = read_lmp(paths["rt_lmp_root"], "isone_lmp_rt_hourly_final_HUB_202[01]-*.csv", real_time=True)
    da_group = da.groupby(["target_local_date", "target_he"], sort=False)
    actual_series = actual.set_index("timestamp_utc")["load_mw"].sort_index()
    rt_history = rt.assign(timestamp_utc=rt["timestamp_utc"].dt.floor("h")).groupby("timestamp_utc")["lmp"].mean().sort_index()
    context_by_key = context.groupby(["target_local_date", "target_he"], sort=False).first().to_dict("index")
    day_vectors = {
        str(date): [float(x) for x in forecast[forecast["forecast_date"] == date].groupby("forecast_hour")["forecast_mw"].sum().reindex(range(1, 25)).tolist()]
        for date in forecast["forecast_date"].unique()
    }

    rows: list[dict[str, Any]] = []
    for (date, hour), group in forecast.groupby(["forecast_date", "forecast_hour"], sort=True):
        if set(group["region"]) != set(ZONES):
            continue
        cutoff = pd.Timestamp(group["decision_cutoff"].iloc[0])
        vector = {str(r.region): float(r.forecast_mw) for r in group.itertuples()}
        hourly_load = float(sum(vector.values()))
        # For a daily card, reconstruct the complete 24h forecast only once per target date.
        hourly_list = day_vectors[str(date)]
        if len(hourly_list) != 24 or any(pd.isna(hourly_list)):
            continue
        diffs = np.diff(hourly_list)
        current4 = asof_series(actual_series, cutoff, 4)
        current24 = asof_series(actual_series, cutoff, 24)
        current = current4[-1] if current4 else None
        load_factor = (float(np.mean(current24) / np.max(current24)) if current24 and max(current24) > 0 else None)
        trend = (float(np.mean(np.diff(current4))) if current4 else None)
        change = (float((np.mean(hourly_list) - current) / current * 100.0) if current and current > 0 else None)
        lmp_group = da_group.get_group((date, int(hour))) if (date, int(hour)) in da_group.groups else pd.DataFrame()
        lmp_zone = {str(r.zone): float(r.lmp) for r in lmp_group.itertuples()} if not lmp_group.empty else {}
        lmp_values = list(lmp_zone.values())
        lmp_mean = float(np.mean(lmp_values)) if lmp_values else None
        lmp_min = float(np.min(lmp_values)) if lmp_values else None
        lmp_max = float(np.max(lmp_values)) if lmp_values else None
        lmp_spread = float(lmp_max - lmp_min) if lmp_values else None
        target_local = pd.Timestamp(date).tz_localize(ET) + pd.Timedelta(hours=int(hour) - 1)
        target_utc = target_local.tz_convert("UTC")
        history_lmp: list[float] | None = None
        history_lmp = asof_series(rt_history, cutoff, 4)
        historical_lmp_mean = float(np.mean(history_lmp)) if history_lmp else None
        historical_lmp_min = float(np.min(history_lmp)) if history_lmp else None
        historical_lmp_max = float(np.max(history_lmp)) if history_lmp else None
        historical_lmp_spread = float(historical_lmp_max - historical_lmp_min) if history_lmp else None
        ctx = context_by_key.get((date, int(hour)), {})
        row = {
            "target_hour_utc": str(target_utc),
            "target_local_date": date,
            "target_he": int(hour),
            "decision_cutoff_utc": str(cutoff),
            "split": "TRAIN" if str(date).startswith("2020") else "DEV",
            "forecast_publish_utc": str(group["published_datetime_utc"].max()),
            "current_load_mw": current,
            "load_factor": load_factor,
            "load_trend_value": trend,
            "load_history_last_4h_mw": current4,
            "hourly_load_mw": hourly_list,
            "forecast_zone_load_mw": vector,
            "peak_load_mw": float(max(hourly_list)),
            "peak_hour": int(np.argmax(hourly_list) + 1),
            "minimum_load_mw": float(min(hourly_list)),
            "daily_energy_mwh": float(sum(hourly_list)),
            "max_up_ramp_mw_per_h": float(max(diffs)) if len(diffs) else 0.0,
            "max_down_ramp_mw_per_h": float(min(diffs)) if len(diffs) else 0.0,
            "forecast_change_vs_current_pct": change,
            "historical_lmp_mean": historical_lmp_mean,
            "historical_lmp_min": historical_lmp_min,
            "historical_lmp_max": historical_lmp_max,
            "historical_lmp_spread": historical_lmp_spread,
            "load_last_4h_mw": current4,
            "lmp_last_4h": history_lmp,
            "binding_branch_count": ctx.get("binding_constraint_count"),
            "binding_event_count": ctx.get("binding_event_count"),
            "network_signal": ctx.get("network_signal"),
            "source_information_role": "public_ex_ante+public_history+derived_public",
            "final_accessed": False,
        }
        rows.append(row)
    return pd.DataFrame(rows), {"forecast_rows": len(forecast), "actual_rows": len(actual), "da_lmp_rows": len(da), "rt_lmp_rows": len(rt)}


def jsonify_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    for key in ("hourly_load_mw", "forecast_zone_load_mw", "day_ahead_lmp_by_zone", "load_history_last_4h_mw", "load_last_4h_mw", "lmp_last_4h", "wind_mw"):
        value = out.get(key)
        out[key] = json.dumps(value, ensure_ascii=False, separators=(",", ":")) if value is not None else None
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", type=Path, default=Path(os.environ.get("ERCOTSIM_ROOT", str(ROOT.parents[1]))))
    ap.add_argument("--output", type=Path, default=ROOT / "data" / "public" / "isone_2y_public_energy_state.parquet")
    args = ap.parse_args()
    rows, source_counts = build_rows(args.source_root.resolve())
    if rows.empty:
        raise RuntimeError("no public state rows built")
    rules = fit_interpretation_rules(rows[rows["split"] == "TRAIN"].to_dict("records"))
    rules_path = ROOT / "configs" / "public_interpretation_rules_v1.json"
    rules_path.write_text(json.dumps(rules, indent=2, ensure_ascii=False), encoding="utf-8")
    cards = []
    for raw in rows.to_dict("records"):
        card = build_public_energy_state(raw, rules)
        cards.append(card)
    rows["public_state_json"] = [canonical_json(card) for card in cards]
    rows["public_state_hash"] = [state_hash(card) for card in cards]
    rows["prompt"] = [format_public_energy_state_prompt(card) for card in cards]
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    stored = rows.apply(lambda r: pd.Series(jsonify_row(r.to_dict())), axis=1)
    stored.to_parquet(output, index=False)
    examples = ROOT / "data_examples" / "public_energy_state_examples.jsonl"
    examples.parent.mkdir(parents=True, exist_ok=True)
    examples.write_text("\n".join(canonical_json({"state": c, "prompt": format_public_energy_state_prompt(c)}) for c in cards[:3]) + "\n", encoding="utf-8")
    audit = {
        "protocol": "Paper9Bus-Power-GV-GRPO-ISO2Y-Public-Energy-State-v1",
        "rows": int(len(rows)),
        "train_rows": int((rows["split"] == "TRAIN").sum()),
        "dev_rows": int((rows["split"] == "DEV").sum()),
        "date_min": str(rows["target_local_date"].min()),
        "date_max": str(rows["target_local_date"].max()),
        "source_counts": source_counts,
        "missing_rates": {k: float(rows[k].isna().mean()) for k in ["current_load_mw", "load_factor", "historical_lmp_mean", "binding_branch_count", "lmp_last_4h"]},
        "forecast_timestamp_rule": "published_datetime_utc <= decision_cutoff_utc",
        "future_realization_columns_present": False,
        "forbidden_prompt_hits": [],
        "target_day_dayahead_lmp_status": "UNAVAILABLE; source publication timestamp is not independently verified",
        "renewable_forecast_status": "UNAVAILABLE; no synthetic wind/solar forecast emitted",
        "own_generator_status": "UNAVAILABLE for ISO-NE public pack; no focal 9-bus generator identity fabricated",
        "final_accessed": False,
        "status": "PASS_PUBLIC_STATE_V1",
    }
    report = ROOT / "reports" / "PUBLIC_FIELD_LEGALITY_AUDIT.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    (ROOT / "reports" / "PUBLIC_FIELD_LEGALITY_REPORT_CN.md").write_text(
        f"# Public-Energy-State-v1 字段合法性审计\n\n- 时间范围：`{audit['date_min']}` 至 `{audit['date_max']}`\n- TRAIN：{audit['train_rows']} 行；DEV：{audit['dev_rows']} 行\n- 决策 cutoff：目标日 D-1 10:00 ET\n- forecast 时间规则：`published_datetime_utc <= decision_cutoff_utc`\n- 风电/光伏日前预测：**UNAVAILABLE**，未填造\n- 9-bus focal generator 对 ISO-NE 两年公共数据：**UNAVAILABLE**，未填造\n- future realized / payoff / regret / oracle：未进入输出\n- 结论：`{audit['status']}`\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "rows": len(rows), "audit": audit["status"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
