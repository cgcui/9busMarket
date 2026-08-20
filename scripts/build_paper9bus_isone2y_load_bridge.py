#!/usr/bin/env python3
"""Build the current gross-load-only Renewable-Aware 9-bus bridge."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paper9bus_gv_grpo.renewable_aware import (  # noqa: E402
    bridge_summary,
    build_gross_load_bridge,
    fit_gross_load_mapping,
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-state", type=Path, default=ROOT / "data" / "public" / "isone_2y_public_energy_state.parquet")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "public" / "paper9bus_isone2y_gross_load_bridge_v1.parquet")
    parser.add_argument("--mapping", type=Path, default=ROOT / "configs" / "paper9bus_isone2y_renewable_aware_load_v1.json")
    parser.add_argument("--audit", type=Path, default=ROOT / "reports" / "PAPER9BUS_ISONE2Y_RENEWABLE_AWARE_LOAD_AUDIT.json")
    args = parser.parse_args()

    public_state = pd.read_parquet(args.public_state)
    mapping, train_stats = fit_gross_load_mapping(public_state)
    bridge = build_gross_load_bridge(public_state, mapping)
    summary = bridge_summary(bridge, mapping)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    bridge.to_parquet(args.output, index=False)
    mapping_obj = {
        "schema_version": "Paper9Bus-ISONE2Y-RenewableAware-GrossLoad-v1",
        "environment": "IEEE-9 frozen network + ISO-NE 2020-2021 public gross load",
        "physical_case": "IEEE-9 case9_blv; three generators G1/G2/G3 frozen",
        "public_state_file": str(args.public_state.relative_to(ROOT)),
        "public_state_sha256": sha256_file(args.public_state),
        "mapping": mapping.as_dict(train_stats),
        "renewable_interface": {
            "wind_forecast_input": "reserved_not_used",
            "solar_forecast_input": "reserved_not_used",
            "net_load_formula": "gross_load - wind - solar, enabled only when both complete vectors are present",
            "current_mode": "DISABLED_FORECASTS_NOT_ADDED",
        },
        "outputs": {
            "bridge_file": str(args.output.relative_to(ROOT)),
            "bus_load_columns": [f"load_bus_{i}_mw" for i in range(1, 10)],
        },
        "final_accessed": False,
    }
    args.mapping.parent.mkdir(parents=True, exist_ok=True)
    args.mapping.write_text(json.dumps(mapping_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    audit = {
        "schema_version": "Paper9Bus-ISONE2Y-RenewableAware-GrossLoad-Audit-v1",
        "input_public_state": str(args.public_state.relative_to(ROOT)),
        "input_public_state_sha256": sha256_file(args.public_state),
        "output_bridge": str(args.output.relative_to(ROOT)),
        "output_bridge_sha256": sha256_file(args.output),
        "mapping_file": str(args.mapping.relative_to(ROOT)),
        "mapping": mapping_obj["mapping"],
        "summary": summary,
        "physical_generators_changed": False,
        "physical_network_changed": False,
        "wind_forecast_values_used": False,
        "solar_forecast_values_used": False,
        "net_load_forecast_used": False,
        "status": summary["status"],
        "final_accessed": False,
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "mapping": str(args.mapping), "rows": len(bridge), "status": audit["status"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
