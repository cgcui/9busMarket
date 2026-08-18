from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import BENCHMARK_ROOT, context_file
from .schema import ACTION_VALUES, parse_core

def load_context(split: str) -> dict[str, dict]:
    frame = pd.read_parquet(context_file(split))
    return {str(r.example_id): r._asdict() for r in frame.itertuples(index=False)}

def load_payoff_tables(split: str) -> dict[tuple[str, float], dict[float, float]]:
    frame = pd.read_parquet(BENCHMARK_ROOT / "cell_bank.parquet", filters=[("split", "==", split.upper())])
    tables = {}
    for (state_id, k_g3), group in frame.groupby(["state_id", "k_g3"], sort=False):
        tables[(str(state_id), float(k_g3))] = {float(r.k_g1): float(r.profit_g1) for r in group.itertuples()}
    return tables

def expected_payoff(context: dict, tables: dict, action_id: int) -> float:
    members = [float(x) for x in json.loads(str(context["class_members_json"]))]
    values = [tables[(str(context["physical_state_id"]), k)][ACTION_VALUES[action_id]] for k in members]
    return float(np.mean(values))

def candidate_reward(raw_output: str, context: dict, tables: dict, registry: dict) -> tuple[float, dict]:
    try:
        obj = parse_core(raw_output, registry)
    except Exception as exc:
        return 0.0, {"valid": False, "reason": str(exc), "action_id": None, "regret": None}
    values = [expected_payoff(context, tables, i) for i in range(len(ACTION_VALUES))]
    action = int(obj["a"])
    best = max(values)
    reward = float(values[action])
    return reward, {"valid": True, "action_id": action, "payoff": reward, "best_payoff": best,
                    "regret": float(best - reward), "belief": obj["b"], "parsed": obj}

def group_advantages(rewards: list[float]) -> list[float]:
    x = np.asarray(rewards, dtype=np.float64)
    if len(x) == 0 or float(x.std()) < 1e-12:
        return [0.0] * len(rewards)
    return ((x - x.mean()) / (x.std() + 1e-8)).tolist()

