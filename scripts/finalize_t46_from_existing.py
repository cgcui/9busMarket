#!/usr/bin/env python3
"""Rewrite the T46 parquet as the required 64-row group audit.

Uses only the already frozen T45 Grammar candidates; it does not sample or
update a model.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from paper9bus_gv_grpo.paths import CORE_ROOT
from paper9bus_gv_grpo.reward import load_context, load_payoff_tables
from paper9bus_gv_grpo.schema import load_registry
from scripts.run_public_state_v1_t45_t46 import economic_gate


def main() -> int:
    candidates = pd.read_parquet(ROOT / "runs/qwen3_1p7b_public_state_v1_sft_seed42/T45_K4_GRAMMAR_CANDIDATES.parquet").to_dict("records")
    t45 = json.loads((ROOT / "reports/T45_INTERFACE_GATE.json").read_text(encoding="utf-8"))
    contexts, tables = load_context("TRAIN"), load_payoff_tables("TRAIN")
    registry = load_registry(CORE_ROOT / "enum_registry.json")
    result, group_frame = economic_gate(candidates, contexts, tables, registry, 4, float(t45["grammar_strict_valid_rate"]))
    group_frame.to_parquet(ROOT / "reports/T46_K4_GROUP_AUDIT.parquet", index=False)
    (ROOT / "reports/T46_K4_TRUE_ECONOMIC_SIGNAL.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(json.dumps({"classification": result["classification"], "groups": len(group_frame), "action_variation_groups": result["action_variation_groups"], "true_payoff_variation_groups": result["true_payoff_variation_groups"]}, ensure_ascii=False))
    return 0 if result["classification"] == "GRPO_READY_SIGNAL_ONLY" else 3


if __name__ == "__main__":
    raise SystemExit(main())
