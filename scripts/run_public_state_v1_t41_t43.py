#!/usr/bin/env python3
"""T41-T43: fresh Paper9Bus-Public-State-v1 target/data gates.

This script is deliberately model-free.  It reconstructs the frozen X1
visible prompt, audits observation-conditioned targets, writes a new SFT
dataset, and performs the exact local Qwen tokenizer length audit.  It never
touches historical data, adapters, or model checkpoints.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paper9bus_gv_grpo.paths import BENCHMARK_ROOT, CORE_ROOT, context_file, split_file
from paper9bus_gv_grpo.audit_hardening import compute_conflict_metrics, prepare_create_only_directory, strict_one_to_one_merge
from paper9bus_gv_grpo.public_state import canonical_json, format_public_energy_state_prompt, state_hash
from paper9bus_gv_grpo.public_state_envs import build_paper9bus_public_state
from paper9bus_gv_grpo.reward import load_payoff_tables
from paper9bus_gv_grpo.schema import ACTION_VALUES, load_registry, parse_core


EXPECTED_KEYS = {"e", "b", "g", "cf", "i", "a", "p", "q"}
FORBIDDEN_TEXT = ("hidden", "private", "g3", "payoff", "oracle", "regret", "future")


def sha256_bytes(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_load(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def entropy(values: list[str]) -> float:
    if not values:
        return 0.0
    counts = Counter(values)
    n = len(values)
    return float(-sum((c / n) * math.log(c / n) for c in counts.values()))


def source_hash(row: dict[str, Any]) -> str:
    # Provenance uses only the frozen public observation and physical-state id;
    # hidden cell-bank columns are intentionally excluded from the artifact.
    public = {
        "physical_state_id": str(row["physical_state_id"]),
        "observation": json_load(row["observation_json"]),
        "total_load_mw": float(row["total_load_mw"]),
    }
    return state_hash(public)


def build_prompt(row: dict[str, Any], rules: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    card = build_paper9bus_public_state(row, rules, include_bus_loads=False)
    body = format_public_energy_state_prompt(card)
    prompt = (
        body
        + "\n\nReturn JSON only with exactly these compact keys: "
        + "e, b, g, cf, i, a, p, q. Use the frozen Core-ESR-v3 registry."
    )
    return prompt, card


def bayes_action(row: dict[str, Any], tables: dict[tuple[str, float], dict[float, float]]) -> tuple[int, list[float]]:
    members = [float(x) for x in json_load(row["class_members_json"])]
    if not members:
        raise RuntimeError(f"empty structural class for {row['example_id']}")
    values = []
    for action_value in ACTION_VALUES:
        values.append(float(np.mean([
            tables[(str(row["physical_state_id"]), member)][float(action_value)]
            for member in members
        ])))
    return int(np.argmax(np.asarray(values, dtype=float))), values


def load_rows(split: str, rules: dict[str, Any]) -> list[dict[str, Any]]:
    core = pd.read_parquet(split_file(split))
    ctx = pd.read_parquet(context_file(split))
    bank = pd.read_parquet(BENCHMARK_ROOT / "cell_bank.parquet")
    base = bank[(bank.split == split) & np.isclose(bank.k_g3.astype(float), 1.0) & np.isclose(bank.k_g1.astype(float), 1.0)]
    base = base.drop_duplicates("state_id").set_index(base.state_id.astype(str))
    merged = strict_one_to_one_merge(
        core,
        ctx,
        key="example_id",
        expected_rows=len(core),
        left_name=f"{split}.core",
        right_name=f"{split}.context",
        suffixes=("_core", "_ctx"),
    )
    rows = []
    for item in merged.to_dict("records"):
        state_id = str(item["physical_state_id"])
        if state_id not in base.index:
            raise RuntimeError(f"missing public k_g3=1/k_g1=1 state row: {state_id}")
        cell = base.loc[state_id]
        row = {
            "example_id": str(item["example_id"]),
            "physical_state_id": state_id,
            "split": str(item["split_core"]),
            "observation_json": item["observation_json"],
            "class_members_json": item["class_members_json"],
            "target_json": item["target_json_core"],
            "sample_weight": float(item["sample_weight_core"]),
            "total_load_mw": float(cell.total_load),
        }
        for i in range(1, 10):
            row[f"load_bus_{i}"] = float(cell[f"load_bus_{i}"])
        obs = json_load(row["observation_json"])
        if isinstance(obs, dict):
            for key, value in obs.items():
                row[key] = value
        prompt, card = build_prompt(row, rules)
        row["prompt"] = prompt
        row["public_state_json"] = json_text(card)
        row["public_state_hash"] = state_hash(card)
        row["source_state_hash"] = source_hash(row)
        row["prompt_hash"] = sha256_bytes(prompt + "\n")
        rows.append(row)
    return rows


def audit_targets(rows_by_split: dict[str, list[dict[str, Any]]], registry: dict[str, Any], tables_by_split: dict[str, dict]) -> tuple[dict, pd.DataFrame, dict]:
    all_rows = [r for rows in rows_by_split.values() for r in rows]
    plan_registry: dict[tuple[str, int], str] = {}
    plan_conflicts = []
    for row in all_rows:
        target = parse_core(row["target_json"], registry)
        key = (row["prompt_hash"], int(target["a"]))
        plan = json_text(target["p"])
        if key in plan_registry and plan_registry[key] != plan:
            plan_conflicts.append({"prompt_hash": key[0], "action_id": key[1]})
        plan_registry[key] = plan

    class_rows = []
    action_rule_conflicts = []
    class_members: dict[str, list[dict]] = defaultdict(list)
    for row in all_rows:
        target = parse_core(row["target_json"], registry)
        action_star, values = bayes_action(row, tables_by_split[row["split"]])
        if int(target["a"]) != action_star:
            action_rule_conflicts.append({"example_id": row["example_id"], "target_action": int(target["a"]), "observation_action": action_star})
        class_members[row["prompt_hash"]].append({"example_id": row["example_id"], "physical_state_id": row["physical_state_id"], "target": target, "action_star": action_star})

    for prompt_hash, members in class_members.items():
        def vals(field: str) -> list[str]:
            return [json_text(m["target"][field]) for m in members]
        unique = {field: sorted(set(vals(field))) for field in ("b", "g", "cf", "i", "a", "p", "q")}
        row = {
            "prompt_hash": prompt_hash,
            "n_physical_examples": len(members),
            "physical_state_ids_json": json_text(sorted({m["physical_state_id"] for m in members})),
            "unique_belief": len(unique["b"]),
            "unique_game": len(unique["g"]),
            "unique_counterfactual": len(unique["cf"]),
            "unique_intent": len(unique["i"]),
            "unique_action": len(unique["a"]),
            "unique_plan": len(unique["p"]),
            "unique_confidence": len(unique["q"]),
            "h_b_given_x_nats": entropy(vals("b")),
            "h_a_given_x_nats": entropy(vals("a")),
            "h_p_given_x_nats": entropy(vals("p")),
            "target_conflict": any(len(unique[x]) > 1 for x in unique),
        }
        class_rows.append(row)

    eq = pd.DataFrame(class_rows).sort_values("prompt_hash").reset_index(drop=True)
    conflict_cols = ["unique_belief", "unique_game", "unique_counterfactual", "unique_intent", "unique_action", "unique_plan", "unique_confidence"]
    audit = {
        "stage": "T41_TARGET_IDENTIFIABILITY_V1",
        "representation": "Paper9Bus-Public-State-v1 X1",
        "splits": {split: len(rows) for split, rows in rows_by_split.items()},
        "visible_prompt_classes": int(len(eq)),
        "physical_examples": int(len(all_rows)),
        "deterministic_conflict_classes": int(eq.target_conflict.sum()) if len(eq) else 0,
        "max_unique_targets": {col.replace("unique_", ""): int(eq[col].max()) if len(eq) else 0 for col in conflict_cols},
        "conditional_entropy_nats": {
            "H_b_given_x": float(np.average(eq.h_b_given_x_nats, weights=eq.n_physical_examples)) if len(eq) else 0.0,
            "H_a_given_x": float(np.average(eq.h_a_given_x_nats, weights=eq.n_physical_examples)) if len(eq) else 0.0,
            "H_p_given_x": float(np.average(eq.h_p_given_x_nats, weights=eq.n_physical_examples)) if len(eq) else 0.0,
        },
        "old_371_collision_present": bool((eq.n_physical_examples == 371).any()),
        "observation_conditioned_action_rule_conflicts": len(action_rule_conflicts),
        "visible_prompt_action_plan_conflicts": len(plan_conflicts),
        "six_action_grid_frozen": [float(x) for x in ACTION_VALUES],
        "target_contract": "Core-ESR-v3 compact JSON; six-state belief retained",
        "classification": "PASS_TARGET_IDENTIFIABILITY_V1" if len(eq) and int(eq.target_conflict.sum()) == 0 and not action_rule_conflicts and not plan_conflicts else "FAIL_TARGET_IDENTIFIABILITY_V1",
        "details": {"action_rule_conflicts_preview": action_rule_conflicts[:20], "plan_conflicts_preview": plan_conflicts[:20]},
        "final_accessed": False,
        "grpo_started": False,
    }
    return audit, eq, plan_registry


def make_target(row: dict[str, Any], registry: dict[str, Any], tables_by_split: dict[str, dict], plan_registry: dict[tuple[str, int], str]) -> dict[str, Any]:
    target = parse_core(row["target_json"], registry)
    action_star, _ = bayes_action(row, tables_by_split[row["split"]])
    target["a"] = action_star
    plan_key = (row["prompt_hash"], action_star)
    if plan_key not in plan_registry:
        raise RuntimeError(f"no deterministic plan for visible prompt/action {plan_key}")
    target["p"] = json.loads(plan_registry[plan_key])
    return target


def write_dataset(rows_by_split: dict[str, list[dict[str, Any]]], registry: dict[str, Any], tables_by_split: dict[str, dict], plan_registry: dict, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty dataset destination: {out_dir}")
    counts = {}
    for split, rows in rows_by_split.items():
        seen = set()
        output = []
        for row in rows:
            target = make_target(row, registry, tables_by_split, plan_registry)
            key = (row["prompt_hash"], json_text(target))
            if key in seen:
                continue
            seen.add(key)
            output.append({
                "example_id": row["example_id"],
                "physical_state_id": row["physical_state_id"],
                "split": split,
                "prompt": row["prompt"],
                "target_json": json_text(target),
                "sample_weight": row["sample_weight"],
                "public_state_hash": row["public_state_hash"],
                "source_state_hash": row["source_state_hash"],
                "prompt_hash": row["prompt_hash"],
                "provenance": "Paper9Bus-Public-State-v1-X1; frozen TRAIN/DEV source assignment; observation-conditioned target",
            })
        path = out_dir / f"{split.lower()}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for item in output:
                handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
        counts[split] = {"source_rows": len(rows), "written_rows": len(output), "file": path.name, "sha256": sha256_file(path)}
    return counts


def tokenizer_audit(dataset_dir: Path, tokenizer_path: Path, max_seq_length: int, completion_limit: int) -> dict:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, local_files_only=True)
    eos = tok.eos_token or ""
    rows = []
    for split in ("TRAIN", "DEV"):
        for line in (dataset_dir / f"{split.lower()}.jsonl").read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            prompt_ids = tok(item["prompt"] + "\n", add_special_tokens=True)["input_ids"]
            target_ids = tok(item["target_json"] + eos, add_special_tokens=False)["input_ids"]
            full_ids = tok(item["prompt"] + "\n" + item["target_json"] + eos, add_special_tokens=True)["input_ids"]
            rows.append({"split": split, "example_id": item["example_id"], "prompt_tokens": len(prompt_ids), "target_tokens": len(target_ids), "total_tokens": len(full_ids)})
    frame = pd.DataFrame(rows)
    def stats(column: str) -> dict[str, int | float]:
        x = frame[column].astype(int)
        return {"min": int(x.min()), "median": float(x.median()), "p95": float(x.quantile(0.95)), "p99": float(x.quantile(0.99)), "max": int(x.max())}
    malformed = []
    leakage_hits = []
    prompt_hash_mismatches = []
    split_keys: dict[str, set[tuple[str, str]]] = {}
    for split in ("TRAIN", "DEV"):
        split_keys[split] = set()
        for line in (dataset_dir / f"{split.lower()}.jsonl").read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            split_keys[split].add((str(item["prompt_hash"]), str(item["target_json"])))
            if str(item["prompt_hash"]) != sha256_bytes(str(item["prompt"]) + "\n"):
                prompt_hash_mismatches.append(item["example_id"])
            for token in ("g3", "profit", "payoff", "oracle", "regret", "future", "hidden", "private"):
                if token in str(item["prompt"]).lower():
                    leakage_hits.append({"example_id": item["example_id"], "token": token})
            try:
                parsed = parse_core(item["target_json"], load_registry(CORE_ROOT / "enum_registry.json"))
                if set(parsed) != EXPECTED_KEYS:
                    malformed.append(item["example_id"])
            except Exception:
                malformed.append(item["example_id"])
    return {"tokenizer": str(tokenizer_path.resolve()), "tokenizer_vocab_size": len(tok), "examples": int(len(frame)), "prompt_tokens": stats("prompt_tokens"), "target_tokens": stats("target_tokens"), "total_tokens": stats("total_tokens"), "max_seq_length": max_seq_length, "completion_limit": completion_limit, "completion_limit_sufficient": bool(frame.target_tokens.max() <= completion_limit), "max_seq_length_sufficient": bool(frame.total_tokens.max() <= max_seq_length), "malformed_targets": malformed, "leakage_hits": leakage_hits, "prompt_hash_mismatches": prompt_hash_mismatches, "cross_split_duplicate_prompt_target_count": len(split_keys["TRAIN"] & split_keys["DEV"]), "rows": frame.to_dict("records")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", type=Path, default=ROOT / "data/public_state_v1_sft")
    ap.add_argument("--report-dir", type=Path, default=ROOT / "reports")
    ap.add_argument("--tokenizer", type=Path, default=ROOT / "runs/qwen3_1p7b_sft_recovery_seed42/checkpoint_epoch1")
    ap.add_argument("--max-seq-length", type=int, default=768)
    ap.add_argument("--completion-limit", type=int, default=256)
    ap.add_argument("--allow-overwrite-development", action="store_true", help="explicit development-only overwrite of the dataset destination")
    args = ap.parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    rules = json.loads((ROOT / "configs/public_interpretation_rules_paper9bus_v1.json").read_text(encoding="utf-8"))
    registry = load_registry(CORE_ROOT / "enum_registry.json")
    tables_by_split = {split: load_payoff_tables(split) for split in ("TRAIN", "DEV")}
    rows_by_split = {split: load_rows(split, rules) for split in ("TRAIN", "DEV")}
    t41, eq, plan_registry = audit_targets(rows_by_split, registry, tables_by_split)
    prepare_create_only_directory(args.dataset_dir, allow_overwrite_development=args.allow_overwrite_development)
    eq.to_parquet(args.report_dir / "T41_TARGET_EQUIVALENCE_CLASSES.parquet", index=False)
    (args.report_dir / "T41_TARGET_IDENTIFIABILITY_AUDIT.json").write_text(json.dumps(t41, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    if t41["classification"] != "PASS_TARGET_IDENTIFIABILITY_V1":
        print(json.dumps(t41, ensure_ascii=False))
        return 2

    counts = write_dataset(rows_by_split, registry, tables_by_split, plan_registry, args.dataset_dir)
    manifest = {"dataset": "Paper9Bus-Public-State-v1-SFT", "representation": "Paper9Bus-Public-State-v1 X1", "source_splits": {"TRAIN": "data/core/train.parquet + train_context.parquet", "DEV": "data/core/dev.parquet + dev_context.parquet"}, "action_grid": [float(x) for x in ACTION_VALUES], "target_schema": "Power-ESR-Core-v3", "target_rule": "observation-conditioned Bayes action over frozen structural class members; plan deterministic by visible prompt and action", "deduplication": "exact prompt hash + exact target only", "counts": counts, "files": {name: {"sha256": sha256_file(args.dataset_dir / name)} for name in ("train.jsonl", "dev.jsonl")}, "final_accessed": False, "grpo_started": False}
    (args.dataset_dir / "DATASET_MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    t43 = tokenizer_audit(args.dataset_dir, args.tokenizer, args.max_seq_length, args.completion_limit)
    written_rows = [
        json.loads(line)
        for split in ("TRAIN", "DEV")
        for line in (args.dataset_dir / ("train.jsonl" if split == "TRAIN" else "dev.jsonl")).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    conflicts = compute_conflict_metrics(written_rows)
    t43.update({
        "stage": "T43_FRESH_DATASET_AUDIT",
        "t41_classification": t41["classification"],
        "split_counts": counts,
        "prompt_hash_reproducible": not t43["prompt_hash_mismatches"],
        "split_integrity": t43["cross_split_duplicate_prompt_target_count"] == 0,
        "same_input_incompatible_target_count": conflicts["same_input_incompatible_target_count"],
        "action_conflicts": conflicts["action_conflicts"],
        "plan_conflicts": conflicts["plan_conflicts"],
        "action_plan_conflicts": conflicts["action_plan_conflicts"],
        "belief_serialization_conflicts": conflicts["belief_serialization_conflicts"],
        "conflict_metric_details": conflicts["details"],
        "classification": "PASS_FRESH_DATASET_V1" if t43["completion_limit_sufficient"] and t43["max_seq_length_sufficient"] and not t43["malformed_targets"] and not t43["leakage_hits"] and not t43["prompt_hash_mismatches"] and t43["cross_split_duplicate_prompt_target_count"] == 0 and conflicts["same_input_incompatible_target_count"] == 0 and conflicts["action_conflicts"] == 0 and conflicts["plan_conflicts"] == 0 and conflicts["belief_serialization_conflicts"] == 0 else "FAIL_FRESH_DATASET_V1",
        "final_accessed": False,
        "grpo_started": False,
    })
    (args.report_dir / "T43_FRESH_DATASET_AUDIT.json").write_text(json.dumps(t43, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(json.dumps({"t41": t41["classification"], "t43": t43["classification"], "train": counts["TRAIN"], "dev": counts["DEV"], "max_total_tokens": t43["total_tokens"]["max"]}, ensure_ascii=False))
    return 0 if t43["classification"] == "PASS_FRESH_DATASET_V1" else 3


if __name__ == "__main__":
    raise SystemExit(main())
