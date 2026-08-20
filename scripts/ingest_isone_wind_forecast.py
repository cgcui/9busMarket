#!/usr/bin/env python3
"""Ingest ISO-NE seven-day wind forecasts into an auditable canonical table.

The downloaded CSVs contain a report date and a download-time "Report
generated" line, but no original publication timestamp.  This script keeps
that distinction explicit: the resulting rows are date-addressable forecasts
and are *not* treated as proven available at the D-1 10:00 ET cutoff.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import zipfile
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATE_FROM = date(2020, 1, 1)
DATE_TO = date(2021, 12, 31)
REPORT_RE = re.compile(r"seven_day_wind_power_forecast_(\d{8})\.csv$")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_date(value: str) -> date:
    return datetime.strptime(value.strip(), "%m/%d/%Y").date()


def parse_report_bytes(payload: bytes, member: str) -> dict[str, Any]:
    """Parse one CSV report without using the download timestamp as a cutoff."""
    match = REPORT_RE.search(member)
    if not match:
        raise ValueError(f"unexpected report member name: {member}")
    report_date = datetime.strptime(match.group(1), "%Y%m%d").date()
    rows = list(csv.reader(io.StringIO(payload.decode("utf-8-sig"))))
    date_row = next((row for row in rows if len(row) > 1 and row[0] == "D" and row[1] == "Date"), None)
    if date_row is None:
        raise ValueError(f"missing Date row: {member}")

    target_dates: dict[int, date] = {}
    for column, raw_date in enumerate(date_row[3:], start=3):
        if raw_date.strip():
            target_dates[column] = parse_date(raw_date)

    values: dict[str, dict[int, float]] = defaultdict(dict)
    for row in rows:
        if len(row) <= 2 or row[0] != "D" or not row[2].strip().isdigit():
            continue
        hour_ending = int(row[2].strip())
        if not 1 <= hour_ending <= 24:
            raise ValueError(f"invalid hour ending {hour_ending}: {member}")
        for column, target_date in target_dates.items():
            if column >= len(row) or not row[column].strip():
                continue
            value = float(row[column].strip())
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid wind forecast value {value}: {member}")
            values[target_date.isoformat()][hour_ending] = value

    generated_line = next(
        (row[1].strip() for row in rows if len(row) > 1 and row[0] == "C" and "Report generated" in row[1]),
        None,
    )
    semantic = json.dumps(values, sort_keys=True, separators=(",", ":"))
    return {
        "report_date": report_date,
        "values": values,
        "semantic_hash": sha256_bytes(semantic.encode("utf-8")),
        "download_generated_text": generated_line,
    }


def discover_reports(source_dir: Path) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    reports: dict[str, list[dict[str, Any]]] = defaultdict(list)
    archives: list[dict[str, Any]] = []
    archive_paths = sorted(source_dir.glob("*.zip"))
    if not archive_paths:
        raise FileNotFoundError(f"no .zip archives found under {source_dir}")
    for archive in archive_paths:
        archive_entry = {
            "archive": archive.name,
            "sha256": sha256_file(archive),
            "bytes": archive.stat().st_size,
            "report_members": [],
        }
        with zipfile.ZipFile(archive) as handle:
            for member in sorted(handle.namelist()):
                if not REPORT_RE.search(member):
                    continue
                parsed = parse_report_bytes(handle.read(member), member)
                parsed.update(
                    {
                        "archive": archive.name,
                        "member": member,
                        "member_sha256": sha256_bytes(handle.read(member)),
                    }
                )
                reports[parsed["report_date"].isoformat()].append(parsed)
                archive_entry["report_members"].append(member)
        archives.append(archive_entry)
    return reports, archives


def select_reports(reports: dict[str, list[dict[str, Any]]]) -> tuple[dict[str, dict[str, Any]], int]:
    selected: dict[str, dict[str, Any]] = {}
    duplicate_count = 0
    for report_key, copies in sorted(reports.items()):
        hashes = {copy["semantic_hash"] for copy in copies}
        if len(hashes) != 1:
            raise ValueError(f"duplicate report contains conflicting forecast values: {report_key}")
        duplicate_count += max(0, len(copies) - 1)
        selected[report_key] = sorted(copies, key=lambda item: (item["archive"], item["member"]))[0]
    return selected, duplicate_count


def build_rows(selected: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for report_key, report in sorted(selected.items()):
        report_date = report["report_date"]
        target_date = report_date + timedelta(days=1)
        if not DATE_FROM <= target_date <= DATE_TO:
            continue
        target_values = report["values"].get(target_date.isoformat(), {})
        for hour_ending, wind_mw in sorted(target_values.items()):
            rows.append(
                {
                    "target_local_date": target_date.isoformat(),
                    "target_he": int(hour_ending),
                    "wind_forecast_mw": float(wind_mw),
                    "forecast_report_date": report_date.isoformat(),
                    "forecast_horizon_day": 2,
                    "publication_timestamp_utc": None,
                    "publication_timestamp_status": "MISSING_SOURCE_TIMESTAMP",
                    "cutoff_eligibility": "NOT_PROVABLE_D1_10_ET",
                    "source_information_role": "public_ex_ante_date_only",
                    "source_archive": report["archive"],
                    "source_member": report["member"],
                    "source_member_sha256": report["member_sha256"],
                }
            )
    if not rows:
        raise RuntimeError("no target-day wind forecast rows found")
    return pd.DataFrame(rows).sort_values(["target_local_date", "target_he"]).reset_index(drop=True)


def expected_dates() -> list[date]:
    out: list[date] = []
    current = DATE_FROM
    while current <= DATE_TO:
        out.append(current)
        current += timedelta(days=1)
    return out


def make_manifest(
    source_dir: Path,
    reports: dict[str, list[dict[str, Any]]],
    selected: dict[str, dict[str, Any]],
    archives: list[dict[str, Any]],
    duplicate_count: int,
    rows: pd.DataFrame,
) -> dict[str, Any]:
    available_dates = sorted(rows["target_local_date"].unique().tolist())
    expected = {d.isoformat() for d in expected_dates()}
    available = set(available_dates)
    partial_dates = [
        d
        for d, group in rows.groupby("target_local_date")
        if len(group) != 24
    ]
    return {
        "schema_version": "ISO-NE-Wind-Forecast-Ingest-v1",
        "source_dir_name": source_dir.name,
        "source_semantics": "ISO New England seven-day wind power forecast CSV reports",
        "forecast_unit": "MW",
        "report_selection": "target date D uses report dated D-1, Day 2 column",
        "decision_cutoff": "D-1 10:00 America/New_York",
        "publication_timestamp_status": "MISSING_SOURCE_TIMESTAMP",
        "cutoff_eligibility": "NOT_PROVABLE_D1_10_ET",
        "safe_usage": [
            "Keep wind forecast as a separately provenance-tagged public field.",
            "Do not use the download-time Report generated line as the original publication time.",
            "Do not call load minus wind a net-load forecast while solar forecast is absent.",
            "Enable strict cutoff training only after an independent publication-timestamp audit.",
        ],
        "archive_count": len(archives),
        "archive_report_member_count": sum(len(a["report_members"]) for a in archives),
        "unique_report_count": len(selected),
        "duplicate_report_member_count": duplicate_count,
        "duplicate_value_conflicts": 0,
        "report_date_min": min(reports) if reports else None,
        "report_date_max": max(reports) if reports else None,
        "target_date_min": min(available_dates) if available_dates else None,
        "target_date_max": max(available_dates) if available_dates else None,
        "target_date_count": len(available_dates),
        "target_date_missing_from_2020_2021": sorted(expected - available),
        "partial_target_dates": sorted(partial_dates),
        "row_count": int(len(rows)),
        "archive_manifest": archives,
        "final_accessed": False,
        "status": "PASS_INGESTION_PROVENANCE_LIMITED",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "public" / "isone_wind_forecast_2020_2021.parquet")
    parser.add_argument("--manifest", type=Path, default=ROOT / "reports" / "WIND_FORECAST_INGESTION_AUDIT.json")
    args = parser.parse_args()

    source_dir = args.source_dir.resolve()
    reports, archives = discover_reports(source_dir)
    selected, duplicate_count = select_reports(reports)
    rows = build_rows(selected)
    manifest = make_manifest(source_dir, reports, selected, archives, duplicate_count, rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(args.output, index=False)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "manifest": str(args.manifest), "rows": len(rows), "status": manifest["status"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
