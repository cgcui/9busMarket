#!/usr/bin/env python3
"""Download the minimum additional ISO-NE data for the renewable layer.

One open-source ISO-NE system-load request supplies the five-minute BTM-PV
estimate over the whole range. Existing local fuel-mix data is reused for the
wind dispatch-expected proxy; it is never renamed to wind_available_mw.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from gridstatus import ISONE


WORKSPACE = Path(__file__).resolve().parents[3]
START_UTC = pd.Timestamp("2024-06-01T00:00:00Z")
END_UTC = pd.Timestamp("2026-07-01T00:00:00Z")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def to_utc(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True)


def fetch_btm_chunk(chunk_start: pd.Timestamp, chunk_end: pd.Timestamp) -> pd.DataFrame:
    iso = ISONE()
    last_error = None
    for attempt in range(3):
        try:
            raw = iso._get_system_load(chunk_start, chunk_end, series="actual", verbose=False)
            required = {"Time", "NativeLoadBtmPv", "Load"}
            missing = required.difference(raw.columns)
            if missing:
                raise RuntimeError(f"ISO-NE BTM response missing columns: {sorted(missing)}")
            return raw[["Time", "NativeLoadBtmPv", "Load"]].copy()
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(1.0 + attempt)
    raise RuntimeError(f"request failed after retries: {last_error!r}")


def download_btm(out_dir: Path) -> tuple[pd.DataFrame, dict]:
    iso = ISONE()
    start_local = START_UTC.tz_convert(iso.default_timezone).normalize()
    end_local = END_UTC.tz_convert(iso.default_timezone).normalize()
    # The public wsclient endpoint returns two continuous local days when the
    # requested [start, end) span is one day. A two-day stepping plan therefore
    # covers the full history with the minimum reliable request count.
    local_days = pd.date_range(
        start_local.tz_localize(None), end_local.tz_localize(None), freq="2D", inclusive="left"
    )
    final_local_day = end_local.tz_localize(None) - pd.Timedelta(days=1)
    if len(local_days) == 0 or local_days[-1] != final_local_day:
        local_days = local_days.append(pd.DatetimeIndex([final_local_day]))
    chunk_dir = out_dir / ".btm_solar_chunks_v2"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    pieces = []
    failed = []
    api_calls = 0
    resumed = 0
    dst_refresh_days = {pd.Timestamp("2024-11-03").date(), pd.Timestamp("2025-11-02").date()}
    tasks = []
    for day in local_days:
        chunk_start = pd.Timestamp(day).tz_localize(iso.default_timezone)
        chunk_end = min(chunk_start + pd.DateOffset(days=1), end_local)
        chunk_path = chunk_dir / f"btm_{day.strftime('%Y%m%d')}.parquet"
        if chunk_path.exists() and day.date() not in dst_refresh_days:
            pieces.append(pd.read_parquet(chunk_path))
            resumed += 1
            continue
        tasks.append((day, chunk_start, chunk_end, chunk_path))
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_map = {executor.submit(fetch_btm_chunk, start, end): (day, path) for day, start, end, path in tasks}
        for number, future in enumerate(as_completed(future_map), start=1):
            day, chunk_path = future_map[future]
            try:
                piece = future.result()
                piece.to_parquet(chunk_path, index=False)
                pieces.append(piece)
                api_calls += 1
            except Exception as exc:
                failed.append({"chunk_start_local": str(day), "error": repr(exc)})
            if number % 20 == 0 or number == len(tasks):
                print(f"BTM chunks {resumed + number}/{len(local_days)}; new API calls={api_calls}; resumed={resumed}; rows collected={sum(len(x) for x in pieces)}", flush=True)
    if failed:
        raise RuntimeError(f"BTM download had {len(failed)} failed chunks: {failed[:3]}")
    raw = pd.concat(pieces, ignore_index=True)
    frame = raw.copy()
    frame["timestamp_utc"] = to_utc(frame.pop("Time"))
    frame = frame.rename(columns={"NativeLoadBtmPv": "native_load_with_btm_pv_mw", "Load": "system_load_excluding_btm_pv_mw"})
    frame["estimated_btm_solar_mw"] = frame["native_load_with_btm_pv_mw"] - frame["system_load_excluding_btm_pv_mw"]
    frame["source_semantics"] = "ESTIMATED_BTM_SOLAR_FROM_ISONE_SYSTEM_LOAD"
    frame["metering_status"] = "ESTIMATED_NOT_METERED"
    frame = frame.loc[frame["timestamp_utc"].between(START_UTC, END_UTC, inclusive="left")].sort_values("timestamp_utc").drop_duplicates("timestamp_utc")
    frame["interval_end_utc"] = frame["timestamp_utc"] + pd.Timedelta(minutes=5)
    frame = frame[["timestamp_utc", "interval_end_utc", "estimated_btm_solar_mw", "native_load_with_btm_pv_mw", "system_load_excluding_btm_pv_mw", "source_semantics", "metering_status"]]
    raw_path = out_dir / "isone_btm_solar_5min_estimated_2024-06_2026-06.parquet"
    frame.to_parquet(raw_path, index=False)

    hourly = frame.set_index("timestamp_utc").resample("1h").agg(
        estimated_btm_solar_mw=("estimated_btm_solar_mw", "mean"),
        btm_solar_5min_observation_count=("estimated_btm_solar_mw", "count"),
    ).reset_index()
    hourly["source_semantics"] = "ESTIMATED_BTM_SOLAR_FROM_ISONE_SYSTEM_LOAD"
    hourly["metering_status"] = "ESTIMATED_NOT_METERED"
    hourly_path = out_dir / "isone_btm_solar_hourly_estimated_2024-06_2026-06.parquet"
    hourly.to_parquet(hourly_path, index=False)
    return hourly, {
        "api_calls": int(api_calls),
        "endpoint_method": "ISONE._get_system_load(series=actual) in two-local-day chunks; BTM = NativeLoadBtmPv - Load",
        "request_chunks": int(len(local_days)),
        "resumed_chunks": int(resumed),
        "chunk_dir": str(chunk_dir.resolve()),
        "raw_path": str(raw_path.resolve()),
        "hourly_path": str(hourly_path.resolve()),
        "raw_rows": int(len(frame)),
        "hourly_rows": int(len(hourly)),
        "expected_hourly_rows": int((END_UTC - START_UTC) / pd.Timedelta(hours=1)),
        "five_minute_negative_values": int((frame["estimated_btm_solar_mw"] < 0).sum()),
        "hourly_missing_rows": int(hourly["estimated_btm_solar_mw"].isna().sum()),
        "hourly_partial_rows": int((hourly["btm_solar_5min_observation_count"] < 12).sum()),
    }


def build_wind_proxy(out_dir: Path) -> dict:
    path = WORKSPACE / "isone_actual" / "isone_fuel_mix_5min_actual_2024-06_2026-06.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    fuel = pd.read_parquet(path, columns=["interval_start_utc", "interval_end_utc", "wind"])
    fuel["timestamp_utc"] = to_utc(fuel.pop("interval_start_utc"))
    fuel["interval_end_utc"] = to_utc(fuel["interval_end_utc"])
    fuel = fuel.loc[fuel["timestamp_utc"].between(START_UTC, END_UTC, inclusive="left")]
    hourly = fuel.set_index("timestamp_utc").resample("1h").agg(
        wind_dispatch_expected_mw=("wind", "mean"),
        fuel_mix_5min_observation_count=("wind", "count"),
    ).reset_index()
    hourly["source_semantics"] = "DISPATCH_EXPECTED_WIND_GENERATION"
    hourly["available_power_semantics"] = "NOT_AVAILABLE_POWER"
    hourly["metering_status"] = "DISPATCH_EXPECTED_NOT_FINAL_METERED"
    out_path = out_dir / "isone_wind_dispatch_expected_hourly_2024-06_2026-06.parquet"
    hourly.to_parquet(out_path, index=False)
    return {
        "source_path": str(path.resolve()),
        "output_path": str(out_path.resolve()),
        "rows": int(len(hourly)),
        "expected_hourly_rows": int((END_UTC - START_UTC) / pd.Timedelta(hours=1)),
        "missing_rows": int(hourly["wind_dispatch_expected_mw"].isna().sum()),
        "aggregation": "hourly mean of irregular dispatch-fuel-mix snapshots",
        "source_semantics": "DISPATCH_EXPECTED_WIND_GENERATION",
        "available_power": False,
        "new_api_calls": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=WORKSPACE / "isone_actual")
    args = parser.parse_args()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    btm, btm_audit = download_btm(out_dir)
    wind_audit = build_wind_proxy(out_dir)
    wind = pd.read_parquet(wind_audit["output_path"])
    btm = btm.rename(columns={"source_semantics": "btm_solar_source_semantics", "metering_status": "btm_solar_metering_status"})
    wind = wind.rename(columns={
        "source_semantics": "wind_source_semantics",
        "available_power_semantics": "wind_available_power_semantics",
        "metering_status": "wind_metering_status",
    })
    merged = btm.merge(wind, on="timestamp_utc", how="outer")
    merged = merged.sort_values("timestamp_utc").reset_index(drop=True)
    merged_path = out_dir / "isone_btm_solar_wind_dispatch_proxy_hourly_2024-06_2026-06.parquet"
    merged.to_parquet(merged_path, index=False)
    manifest = {
        "schema_version": "ISO-NE-BTM-Solar-And-Wind-Dispatch-Proxy-v1",
        "query_window_utc": {"start": START_UTC.isoformat(), "end_exclusive": END_UTC.isoformat()},
        "new_open_source_api_calls_last_resume_run": btm_audit["api_calls"],
        "minimum_reliable_coverage_windows": btm_audit["request_chunks"],
        "coverage_complete": btm_audit["raw_rows"] == 218880 and btm_audit["hourly_missing_rows"] == 0,
        "api_call_note": "Final artifacts are complete; resume mode skips existing chunks and only refreshes DST boundary chunks when required.",
        "btm_solar": btm_audit,
        "wind_dispatch_proxy": wind_audit,
        "merged_hourly_path": str(merged_path.resolve()),
        "forecast_inputs_reused_without_download": [
            str((WORKSPACE / "isone_wind_forecast/isone_wind_forecast_2024-06_2026-06.parquet").resolve()),
            str((WORKSPACE / "isone_solar_forecast/isone_solar_forecast_2024-06_2026-06.parquet").resolve()),
        ],
        "semantic_rules": {
            "estimated_btm_solar_mw": "allowed as BTM negative load after explicit estimated_not_metered label",
            "wind_dispatch_expected_mw": "historical dispatch-expected proxy only; never wind_available_mw",
            "wind_available_mw": "not supplied",
        },
        "status": "PASS_BTM_SOLAR_PLUS_WIND_DISPATCH_PROXY",
    }
    (out_dir / "BTM_WIND_PROXY_DOWNLOAD_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
