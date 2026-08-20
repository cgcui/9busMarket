#!/usr/bin/env python3
"""Audit ISO-NE solar forecast archives without fabricating empty values."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import zipfile
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORT_RE = re.compile(r"seven_day_solar_power_forecast_(\d{8})\.csv$")
DATE_FROM = date(2020, 1, 1)
DATE_TO = date(2021, 12, 31)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def count_numeric_values(payload: bytes) -> int:
    rows = csv.reader(io.StringIO(payload.decode("utf-8-sig")))
    count = 0
    for row in rows:
        if len(row) <= 3 or row[0] != "D" or not row[2].strip().isdigit():
            continue
        for value in row[3:]:
            try:
                float(value.strip())
            except (TypeError, ValueError):
                continue
            count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "SOLAR_FORECAST_INGESTION_AUDIT.json")
    args = parser.parse_args()
    source_dir = args.source_dir.resolve()
    archives = sorted(source_dir.glob("*.zip"))
    if not archives:
        raise FileNotFoundError(f"no .zip archives found under {source_dir}")

    reports: list[dict[str, Any]] = []
    archive_manifest = []
    for archive in archives:
        members = []
        with zipfile.ZipFile(archive) as handle:
            for member in sorted(handle.namelist()):
                match = REPORT_RE.search(member)
                if not match:
                    continue
                payload = handle.read(member)
                values = count_numeric_values(payload)
                report_date = datetime.strptime(match.group(1), "%Y%m%d").date()
                reports.append({"report_date": report_date.isoformat(), "archive": archive.name, "member": member, "numeric_value_count": values})
                members.append({"member": member, "numeric_value_count": values})
        archive_manifest.append({"archive": archive.name, "bytes": archive.stat().st_size, "sha256": sha256_file(archive), "report_members": members})

    counts = Counter(item["report_date"] for item in reports)
    unique_dates = {date.fromisoformat(value) for value in counts}
    missing = []
    current = DATE_FROM
    while current <= DATE_TO:
        if current not in unique_dates:
            missing.append(current.isoformat())
        current += timedelta(days=1)
    numeric_total = sum(int(item["numeric_value_count"]) for item in reports)
    audit = {
        "schema_version": "ISO-NE-Solar-Forecast-Audit-v1",
        "source_dir_name": source_dir.name,
        "source_semantics": "ISO New England seven-day solar power forecast CSV reports",
        "archive_count": len(archives),
        "report_member_count": len(reports),
        "unique_report_date_count": len(unique_dates),
        "duplicate_report_member_count": sum(max(0, value - 1) for value in counts.values()),
        "report_date_min": min(counts) if counts else None,
        "report_date_max": max(counts) if counts else None,
        "missing_report_dates_2020_2021": missing,
        "numeric_forecast_value_count": numeric_total,
        "empty_report_count": sum(int(item["numeric_value_count"]) == 0 for item in reports),
        "ingestion_decision": "DO_NOT_INGEST_EMPTY_REPORTS",
        "solar_forecast_status": "SOURCE_ARCHIVE_PRESENT_NO_NUMERIC_VALUES",
        "net_load_status": "NOT_EMITTED",
        "future_realization_columns_present": False,
        "final_accessed": False,
        "archive_manifest": archive_manifest,
        "status": "PASS_AUDIT_NO_SOLAR_VALUES",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "reports": len(reports), "numeric_values": numeric_total, "status": audit["status"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
