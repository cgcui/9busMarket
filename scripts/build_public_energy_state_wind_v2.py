#!/usr/bin/env python3
"""Build ISO2Y Public-Energy-State-v2 with the supplied wind forecasts.

The existing v1 parquet is treated as immutable input.  This script creates a
new version and adds only wind values that are complete for the target date.
Solar is intentionally still absent, so no net-load forecast is emitted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paper9bus_gv_grpo.public_state import (  # noqa: E402
    build_public_energy_state,
    canonical_json,
    fit_interpretation_rules,
    format_public_energy_state_prompt,
    state_hash,
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def wind_cards(wind: pd.DataFrame) -> dict[str, dict[str, Any]]:
    wind = wind.copy()
    wind["target_local_date"] = wind["target_local_date"].astype(str)
    wind["target_he"] = pd.to_numeric(wind["target_he"], errors="coerce").astype("Int64")
    wind["wind_forecast_mw"] = pd.to_numeric(wind["wind_forecast_mw"], errors="coerce")
    cards: dict[str, dict[str, Any]] = {}
    for target_date, group in wind.groupby("target_local_date", sort=True):
        values = group.dropna(subset=["target_he", "wind_forecast_mw"]).drop_duplicates("target_he").set_index("target_he")["wind_forecast_mw"]
        values = values.reindex(range(1, 25))
        available = int(values.notna().sum())
        complete = available == 24
        cards[target_date] = {
            "wind_mw": [float(x) for x in values.tolist()] if complete else None,
            "wind_forecast_hours_available": available,
            "wind_forecast_status": "AVAILABLE_DATE_ONLY" if complete else "PARTIAL_DST_DATE",
            "wind_forecast_report_date": str(group["forecast_report_date"].iloc[0]),
            "wind_forecast_cutoff_eligibility": str(group["cutoff_eligibility"].iloc[0]),
        }
    return cards


def json_for_storage(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    for key in ("hourly_load_mw", "forecast_zone_load_mw", "load_history_last_4h_mw", "load_last_4h_mw", "lmp_last_4h", "wind_mw"):
        value = out.get(key)
        out[key] = json.dumps(value, ensure_ascii=False, separators=(",", ":")) if value is not None else None
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=ROOT / "data" / "public" / "isone_2y_public_energy_state.parquet")
    parser.add_argument("--wind", type=Path, default=ROOT / "data" / "public" / "isone_wind_forecast_2020_2021.parquet")
    parser.add_argument("--wind-manifest", type=Path, default=ROOT / "reports" / "WIND_FORECAST_INGESTION_AUDIT.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "public" / "isone_2y_public_energy_state_wind_v2.parquet")
    args = parser.parse_args()

    base = pd.read_parquet(args.base)
    wind = pd.read_parquet(args.wind)
    lookup = wind_cards(wind)
    rows: list[dict[str, Any]] = []
    for row in base.to_dict("records"):
        target_date = str(row["target_local_date"])
        details = lookup.get(
            target_date,
            {
                "wind_mw": None,
                "wind_forecast_hours_available": 0,
                "wind_forecast_status": "UNAVAILABLE_SOURCE_COVERAGE",
                "wind_forecast_report_date": None,
                "wind_forecast_cutoff_eligibility": "UNAVAILABLE_SOURCE_COVERAGE",
            },
        )
        row.update(details)
        rows.append(row)

    rules = fit_interpretation_rules([r for r in rows if str(r.get("split")) == "TRAIN"])
    rules["schema_version"] = "ISO2Y-Public-Energy-State-v2-WindForecast"
    rules["environment"] = "ISO-NE-2020-2021-public-pack+wind-forecast"
    rules["fixed_rules"]["wind_forecast"] = "date-only report field; not cutoff-proven until publication timestamp audit passes"

    cards = []
    for row in rows:
        card = build_public_energy_state(row, rules)
        card["schema_version"] = "ISO2Y-Public-Energy-State-v2-WindForecast"
        card["environment"] = "ISO-NE-2020-2021-public-pack+wind-forecast"
        cards.append(card)

    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    stored = pd.DataFrame([json_for_storage(row) for row in rows])
    stored["public_state_json"] = [canonical_json(card) for card in cards]
    stored["public_state_hash"] = [state_hash(card) for card in cards]
    stored["prompt"] = [format_public_energy_state_prompt(card) for card in cards]
    stored.to_parquet(output, index=False)

    manifest = json.loads(args.wind_manifest.read_text(encoding="utf-8"))
    available_rows = int(sum(row["wind_forecast_status"] == "AVAILABLE_DATE_ONLY" for row in rows))
    available_dates = sorted({str(row["target_local_date"]) for row in rows if row["wind_forecast_status"] == "AVAILABLE_DATE_ONLY"})
    audit = {
        "schema_version": "ISO2Y-Public-Energy-State-v2-WindForecast-Audit",
        "base_public_state": str(args.base.relative_to(ROOT)),
        "base_public_state_sha256": sha256_file(args.base),
        "wind_forecast_table": str(args.wind.relative_to(ROOT)),
        "wind_forecast_table_sha256": sha256_file(args.wind),
        "wind_ingestion_manifest": str(args.wind_manifest.relative_to(ROOT)),
        "wind_ingestion_status": manifest["status"],
        "rows": int(len(stored)),
        "wind_available_rows": available_rows,
        "wind_available_target_dates": len(available_dates),
        "wind_available_date_min": min(available_dates) if available_dates else None,
        "wind_available_date_max": max(available_dates) if available_dates else None,
        "wind_forecast_status": "DATE_ONLY_NOT_CUTOFF_PROVEN",
        "publication_timestamp_status": "MISSING_SOURCE_TIMESTAMP",
        "solar_forecast_status": "UNAVAILABLE",
        "net_load_forecast_status": "NOT_EMITTED_SOLAR_FORECAST_MISSING",
        "strict_training_gate": "BLOCKED_PENDING_WIND_PUBLICATION_TIMESTAMP_AUDIT",
        "v1_preserved": True,
        "future_realization_columns_present": False,
        "final_accessed": False,
        "status": "PASS_V2_DATA_ADDED_WITH_CUTOFF_LIMITATION",
    }
    rules_path = ROOT / "configs" / "public_interpretation_rules_iso2y_wind_v2.json"
    rules_path.write_text(json.dumps(rules, ensure_ascii=False, indent=2), encoding="utf-8")
    audit_path = ROOT / "reports" / "PUBLIC_FIELD_LEGALITY_AUDIT_WIND_V2.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    examples = ROOT / "data_examples" / "public_energy_state_wind_v2_examples.jsonl"
    examples.parent.mkdir(parents=True, exist_ok=True)
    examples.write_text("\n".join(canonical_json({"state": c, "prompt": format_public_energy_state_prompt(c)}) for c in cards[:3]) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "rows": len(stored), "wind_available_rows": available_rows, "status": audit["status"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
