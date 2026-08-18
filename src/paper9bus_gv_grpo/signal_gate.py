from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .paths import CORE_ROOT
from .reward import candidate_reward, load_context, load_payoff_tables
from .schema import load_registry

def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--split", choices=["TRAIN", "DEV"], default="TRAIN")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--expected-groups", type=int, default=64)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.candidates)
    context = load_context(args.split); tables = load_payoff_tables(args.split)
    registry = load_registry(CORE_ROOT / "enum_registry.json")
    groups = {}
    for row in rows:
        groups.setdefault(str(row.get("example_id", "")), []).append(row)
    records = []
    for eid, members in sorted(groups.items()):
        if eid not in context or len(members) != args.k:
            records.append({"example_id": eid, "eligible": False, "reason": f"count_or_context:{len(members)}"})
            continue
        scored = []
        for row in members:
            reward, detail = candidate_reward(str(row.get("raw_output", "")), context[eid], tables, registry)
            scored.append({"reward": reward, **detail})
        valid = [x for x in scored if x["valid"]]
        records.append({"example_id": eid, "eligible": True, "n_valid": len(valid),
                        "strict_valid_rate": len(valid) / args.k,
                        "n_unique_action": len({x["action_id"] for x in valid}),
                        "n_unique_belief": len({tuple(round(v, 6) for v in x["belief"]) for x in valid}),
                        "reward_std": float(np.std([x["reward"] for x in valid])) if valid else 0.0,
                        "mean_regret": float(np.mean([x["regret"] for x in valid])) if valid else None,
                        "scores": scored})
    eligible = [x for x in records if x["eligible"]]
    signal = [x for x in eligible if x["reward_std"] > 1e-8]
    diversity = [x for x in eligible if x["n_unique_action"] > 1 or x["n_unique_belief"] > 1]
    strict = sum(x["n_valid"] for x in eligible) / (len(eligible) * args.k) if eligible else 0.0
    result = {"gate": "GATE_GRPO_SIGNAL", "protocol": "Paper9Bus-Power-GV-GRPO-v3", "split": args.split,
              "k": args.k, "candidate_rows": len(rows), "eligible_groups": len(eligible),
              "strict_valid_rate": strict, "groups_with_reward_variance": len(signal),
              "groups_with_action_or_belief_diversity": len(diversity), "final_accessed": False,
              "criteria": {"expected_group_count": len(eligible) == args.expected_groups,
                           "strict_valid_rate_ge_0.95": strict >= 0.95,
                           "reward_variance_exists": len(signal) > 0,
                           "action_or_belief_diversity_exists": len(diversity) > 0}}
    result["status"] = "PASS_GRPO_SIGNAL" if all(result["criteria"].values()) else "FAIL_GRPO_SIGNAL"
    (args.output / "GATE_GRPO_SIGNAL_RESULT.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.output / "GATE_GRPO_SIGNAL_REPORT.json").write_text(json.dumps({**result, "groups": records}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "PASS_GRPO_SIGNAL" else 2

if __name__ == "__main__":
    raise SystemExit(main())

