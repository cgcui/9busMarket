#!/usr/bin/env python3
"""T45-T46: fresh checkpoint/interface selection and K=4 signal gate."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from paper9bus_gv_grpo.paths import CORE_ROOT
from paper9bus_gv_grpo.audit_hardening import validate_exact_prompt_identity
from paper9bus_gv_grpo.reward import expected_payoff, load_context, load_payoff_tables, group_advantages
from paper9bus_gv_grpo.schema import ACTION_VALUES, load_registry
from scripts.analyze_k4_interface import classify
from scripts.run_k4_interface_recovery import CoreESRGrammar, field_flags, generate as generate_paired
from scripts.run_k4_signal_gate import parse_interface, select_snapshots


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_hashes(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)).replace("\\", "/"): sha256(p) for p in sorted(root.rglob("*")) if p.is_file()}


def deterministic_batch(model, tok, prompts, max_new_tokens, batch_size=32):
    rows = []
    eos_ids = {int(tok.eos_token_id)} if isinstance(tok.eos_token_id, int) else {int(x) for x in tok.eos_token_id}
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start:start + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True).to("cuda")
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id)
        for i in range(len(chunk)):
            ids = out[i, enc.input_ids.shape[1]:].detach().cpu().tolist()
            eos_pos = next((j for j, x in enumerate(ids) if int(x) in eos_ids), None)
            actual = ids if eos_pos is None else ids[:eos_pos + 1]
            rows.append({"raw_output": tok.decode(actual[:-1] if eos_pos is not None else actual, skip_special_tokens=True).strip(), "generated_tokens": len(actual), "eos_completed": eos_pos is not None, "truncated": eos_pos is None and len(actual) >= max_new_tokens})
    return rows


def load_model(model_name: str, checkpoint: Path):
    tok = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True, local_files_only=True)
    tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = "left"
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    base = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb, device_map={"": 0}, trust_remote_code=True, torch_dtype=torch.bfloat16, local_files_only=True)
    return PeftModel.from_pretrained(base, checkpoint, local_files_only=True).eval(), tok


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def dev_summary(rows: list[dict], targets: list[dict], registry: dict) -> dict:
    audits = [parse_interface(x["raw_output"], x["eos_completed"], x["truncated"], registry) for x in rows]
    flags = [field_flags(x["raw_output"]) for x in rows]
    exact = Counter()
    action_plan = []
    cf_valid = []
    action_valid = []
    leakage = []
    categories = []
    for row, target, audit, flag in zip(rows, targets, audits, flags):
        obj = audit["parsed"]
        if obj is not None:
            for key in ("b", "g", "cf", "i", "a", "p", "q"):
                exact[key] += int(json.dumps(obj[key], sort_keys=True, separators=(",", ":")) == json.dumps(target[key], sort_keys=True, separators=(",", ":")))
            action_plan.append(int(obj["a"] == target["a"] and obj["p"] == target["p"]))
            cf_valid.append(int(obj["cf"] == target["cf"]))
            action_valid.append(int(isinstance(obj["a"], int) and obj["a"] in range(6)))
        else:
            action_plan.append(0); cf_valid.append(0); action_valid.append(0)
        leakage.append(int(audit["private_field_violation"]))
        ns = SimpleNamespace(group_index=0, candidate_index=0, example_id="dev", physical_state_id="dev", split="DEV", raw_output=row["raw_output"], generated_tokens=row["generated_tokens"], max_new_tokens=256, eos_completed=row["eos_completed"], truncated=row["truncated"], strict_valid=audit["strict_valid"], parse_success=audit["parse_success"], core_valid=audit["core_valid"], validity_reason=audit["validity_reason"])
        categories.append(classify(ns)["failure_category"])
    n = len(rows)
    return {"rows": n, "strict_valid_rate": float(np.mean([x["strict_valid"] for x in audits])), "json_valid_rate": float(np.mean([x["parse_success"] for x in audits])), "eos_rate": float(np.mean([x["eos_completed"] for x in rows])), "truncation_rate": float(np.mean([x["truncated"] for x in rows])), "belief_valid_rate": float(np.mean([x["belief_valid"] for x in flags])), "game_valid_rate": float(np.mean([x["game_valid"] for x in flags])), "counterfactual_valid_rate": float(np.mean(cf_valid)), "plan_valid_rate": float(np.mean([x["plan_valid"] for x in flags])), "action_valid_rate": float(np.mean(action_valid)), "action_plan_consistency_rate": float(np.mean(action_plan)), "hidden_leakage_rate": float(np.mean(leakage)), "exact_field_rates": {key: float(value / n) for key, value in exact.items()}, "failure_category_counts": dict(Counter(categories))}


def new_prompt_map(dataset_dir: Path) -> dict[str, dict]:
    out = {"by_example": {}, "by_physical": {}}
    for row in read_jsonl(dataset_dir / "train.jsonl"):
        out["by_example"].setdefault(str(row["example_id"]), row)
        out["by_physical"].setdefault(str(row["physical_state_id"]), []).append(row)
    return out


def run_k4(model, tok, dataset_dir: Path, temperature: float, top_p: float, completion_limit: int):
    snapshots = select_snapshots()
    prompt_map = new_prompt_map(dataset_dir)
    grammar = CoreESRGrammar(tok)
    free_rows, grammar_rows = [], []
    for gi, snap in enumerate(snapshots.itertuples(index=False)):
        item = prompt_map["by_example"].get(str(snap.example_id))
        if item is None:
            raise RuntimeError(f"missing exact fresh prompt for frozen example_id: {snap.example_id}")
        prompt_identity = validate_exact_prompt_identity(str(snap.example_id), item)
        seed = 42 * 100000 + gi * 100
        prompts = [str(item["prompt"])] * 4
        free = generate_paired(model, tok, grammar, prompts, seed=seed, max_new_tokens=completion_limit, temperature=temperature, top_p=top_p, constrained=False)
        gram = generate_paired(model, tok, grammar, prompts, seed=seed, max_new_tokens=completion_limit, temperature=temperature, top_p=top_p, constrained=True)
        common = {"group_index": gi, "example_id": str(snap.example_id), "physical_state_id": str(snap.physical_state_id), "split": "TRAIN", "sampling_batch_seed": seed, "temperature": temperature, "top_p": top_p, "max_new_tokens": completion_limit, **prompt_identity}
        for ci, (f, g) in enumerate(zip(free, gram)):
            free_rows.append({**common, "candidate_index": ci, "sampling_seed": seed + ci, **f})
            grammar_rows.append({**common, "candidate_index": ci, "sampling_seed": seed + ci, **g})
    return free_rows, grammar_rows


def economic_gate(rows: list[dict], contexts: dict, tables: dict, registry: dict, k: int, interface_strict: float) -> tuple[dict, pd.DataFrame]:
    candidates = []
    groups = []
    ordering_violations = 0
    variation_groups = 0
    payoff_variation_groups = 0
    for gi in sorted({int(x["group_index"]) for x in rows}):
        group = [x for x in rows if int(x["group_index"]) == gi]
        snap = group[0]
        ctx = contexts[str(snap["example_id"])]
        values = [expected_payoff(ctx, tables, i) for i in range(len(ACTION_VALUES))]
        best = float(max(values))
        valid = []
        for row in group:
            audit = parse_interface(row["raw_output"], row["eos_completed"], row["truncated"], registry)
            item = dict(row)
            item["strict_valid"] = bool(audit["strict_valid"])
            item["action_id"] = int(audit["parsed"]["a"]) if audit["parsed"] is not None else None
            if item["strict_valid"]:
                item["expected_profit"] = float(values[item["action_id"]])
                item["observation_regret"] = float(best - item["expected_profit"])
                valid.append(item)
            candidates.append(item)
        utilities = [float(x["expected_profit"]) for x in valid]
        regrets = [float(x["observation_regret"]) for x in valid]
        actions = [int(x["action_id"]) for x in valid]
        adv = group_advantages(utilities)
        for item, value in zip(valid, adv):
            item["group_advantage"] = float(value)
        if len(valid) >= 2:
            for i in range(len(valid)):
                for j in range(len(valid)):
                    if adv[i] > adv[j] + 1e-9 and regrets[i] > regrets[j] + 1e-8:
                        ordering_violations += 1
        action_var = len(set(actions)) > 1
        payoff_var = float(np.std(utilities)) > 1e-8
        variation_groups += int(action_var)
        payoff_variation_groups += int(payoff_var)
        groups.append({"group_index": gi, "example_id": snap["example_id"], "physical_state_id": snap["physical_state_id"], "n_candidates": len(group), "n_strict_valid": len(valid), "strict_valid_rate": len(valid) / k, "action_count": len(set(actions)), "action_entropy": float(-sum((c / len(actions)) * math.log(c / len(actions)) for c in Counter(actions).values())) if actions else 0.0, "utility_std": float(np.std(utilities)) if utilities else 0.0, "utility_range": float(max(utilities) - min(utilities)) if utilities else 0.0, "regret_std": float(np.std(regrets)) if regrets else 0.0, "action_variation": action_var, "true_payoff_variation": payoff_var, "observation_optimal_action": int(np.argmax(np.asarray(values))), "observation_optimal_profit": best})
    candidate_frame = pd.DataFrame(candidates)
    group_frame = pd.DataFrame(groups)
    result = {"stage": "T46_K4_TRUE_ECONOMIC_SIGNAL", "interface_strict_valid_rate": interface_strict, "groups": len(groups), "k": k, "strict_threshold": 0.95, "action_variation_groups": variation_groups, "true_payoff_variation_groups": payoff_variation_groups, "advantage_ordering_violation_count": ordering_violations, "within_group_advantages_only": True, "cross_state_reward_normalization": False, "forced_action_diversity": False, "oracle_or_repair": False, "classification": "GRPO_READY_SIGNAL_ONLY" if interface_strict >= 0.95 and variation_groups >= 2 and payoff_variation_groups >= 2 and ordering_violations == 0 else "FAIL_TRUE_ECONOMIC_SIGNAL", "group_audit": groups, "final_accessed": False, "grpo_started": False}
    return result, group_frame


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft-run", type=Path, default=ROOT / "runs/qwen3_1p7b_public_state_v1_sft_seed42")
    ap.add_argument("--dataset-dir", type=Path, default=ROOT / "data/public_state_v1_sft")
    ap.add_argument("--output", type=Path, default=ROOT / "reports")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--completion-limit", type=int, default=256)
    args = ap.parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("T45/T46 require CUDA BF16")
    manifest = json.loads((args.sft_run / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
    model_name = str(manifest["model"])
    registry = load_registry(CORE_ROOT / "enum_registry.json")
    dev = read_jsonl(args.dataset_dir / "dev.jsonl")
    dev_results = []
    for epoch in (1, 2, 3):
        checkpoint = args.sft_run / f"checkpoint_epoch{epoch}"
        model, tok = load_model(model_name, checkpoint)
        generated = deterministic_batch(model, tok, [str(x["prompt"]) + "\n" for x in dev], args.completion_limit)
        summary = dev_summary(generated, [json.loads(x["target_json"]) for x in dev], registry)
        summary.update({"epoch": epoch, "checkpoint": str(checkpoint.resolve()), "checkpoint_hashes": tree_hashes(checkpoint)})
        dev_results.append(summary)
        del model, tok
        gc.collect(); torch.cuda.empty_cache()
    ranked = sorted(dev_results, key=lambda x: (-x["strict_valid_rate"], x["truncation_rate"], -x["belief_valid_rate"], -x["plan_valid_rate"], -x["action_plan_consistency_rate"], -x["eos_rate"]))
    selected = ranked[0]
    selected_epoch = int(selected["epoch"])
    selected_checkpoint = args.sft_run / f"checkpoint_epoch{selected_epoch}"
    t45 = {"stage": "T45_CHECKPOINT_AND_INTERFACE_GATE", "model": model_name, "selection_hierarchy": ["maximize strict validity", "minimize truncation", "maximize belief validity", "maximize plan validity", "maximize action-plan consistency", "maximize EOS"], "dev_results": dev_results, "ranking_epochs": [int(x["epoch"]) for x in ranked], "selected_epoch": selected_epoch, "selected_checkpoint": str(selected_checkpoint.resolve()), "checkpoint_hashes": tree_hashes(selected_checkpoint), "final_accessed": False, "grpo_started": False}
    (args.output / "T45_DEV_CHECKPOINT_COMPARISON.json").write_text(json.dumps(t45, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    (args.output / "T45_CHECKPOINT_FREEZE.json").write_text(json.dumps({"stage": "T45_CHECKPOINT_FREEZE", "selected_epoch": selected_epoch, "selected_checkpoint": str(selected_checkpoint.resolve()), "checkpoint_hashes": tree_hashes(selected_checkpoint), "selection_metrics": selected, "selection_rule": t45["selection_hierarchy"], "dev_only_selection": True, "final_accessed": False, "grpo_started": False}, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    model, tok = load_model(model_name, selected_checkpoint)
    free_rows, grammar_rows = run_k4(model, tok, args.dataset_dir, args.temperature, args.top_p, args.completion_limit)
    free_frame = pd.DataFrame(free_rows); grammar_frame = pd.DataFrame(grammar_rows)
    free_frame.to_parquet(args.sft_run / "T45_K4_FREE_CANDIDATES.parquet", index=False)
    grammar_frame.to_parquet(args.sft_run / "T45_K4_GRAMMAR_CANDIDATES.parquet", index=False)
    free_rate = float(np.mean([parse_interface(x["raw_output"], x["eos_completed"], x["truncated"], registry)["strict_valid"] for x in free_rows]))
    grammar_rate = float(np.mean([parse_interface(x["raw_output"], x["eos_completed"], x["truncated"], registry)["strict_valid"] for x in grammar_rows]))
    if free_rate >= 0.95:
        mode, interface_rate, classification = "FREE", free_rate, "PASS_NATIVE_INTERFACE_V1"
    elif grammar_rate >= 0.95:
        mode, interface_rate, classification = "GRAMMAR", grammar_rate, "PASS_CONSTRAINED_INTERFACE_V1"
    else:
        mode, interface_rate, classification = "NONE", max(free_rate, grammar_rate), "FAIL_INTERFACE_V1"
    t45.update({"classification": classification, "selected_mode": mode, "same_prompts": True, "same_stochastic_config": True, "groups": 64, "k": 4, "temperature": args.temperature, "top_p": args.top_p, "completion_limit": args.completion_limit, "free_strict_valid_rate": free_rate, "grammar_strict_valid_rate": grammar_rate, "economic_gate_allowed": classification in {"PASS_NATIVE_INTERFACE_V1", "PASS_CONSTRAINED_INTERFACE_V1"}})
    (args.output / "T45_INTERFACE_GATE.json").write_text(json.dumps(t45, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    if classification == "FAIL_INTERFACE_V1":
        print(json.dumps({"stage": "T45", "classification": classification, "selected_epoch": selected_epoch, "free_strict": free_rate, "grammar_strict": grammar_rate}, ensure_ascii=False))
        return 2
    contexts, tables = load_context("TRAIN"), load_payoff_tables("TRAIN")
    selected_rows = free_rows if mode == "FREE" else grammar_rows
    t46, audit_frame = economic_gate(selected_rows, contexts, tables, registry, 4, interface_rate)
    audit_frame.to_parquet(args.output / "T46_K4_GROUP_AUDIT.parquet", index=False)
    (args.output / "T46_K4_TRUE_ECONOMIC_SIGNAL.json").write_text(json.dumps(t46, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    report = ["# Paper9Bus-Public-State-v1 T41-T46 报告", "", "## Gate 结论", "", f"- T41：`PASS_TARGET_IDENTIFIABILITY_V1`。", f"- T43：`PASS_FRESH_DATASET_V1`。", f"- T44：fresh Qwen3-1.7B，三 epoch 已完成；checkpoint 选择 `epoch{selected_epoch}`。", f"- T45：`{classification}`；FREE strict=`{free_rate:.4f}`，Grammar strict=`{grammar_rate:.4f}`。", f"- T46：`{t46['classification']}`；动作变化组=`{t46['action_variation_groups']}/64`，真实 payoff 变化组=`{t46['true_payoff_variation_groups']}/64`，advantage ordering violations=`{t46['advantage_ordering_violation_count']}`。", "", "## 停止边界", "", "- K=4 只使用 TRAIN 64 个 frozen physical states，K=4；没有强制动作多样性、oracle、post-hoc repair 或跨 state reward normalization。", "- 本阶段在 GRPO 之前停止；未启动 GRPO，未访问 FINAL，未运行 ISO2Y 训练。"]
    (args.output / "PUBLIC_STATE_V1_SFT_AND_GATE_REPORT_CN.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    del model, tok
    gc.collect(); torch.cuda.empty_cache()
    print(json.dumps({"stage": "T46", "t45": classification, "t46": t46["classification"], "selected_epoch": selected_epoch, "free_strict": free_rate, "grammar_strict": grammar_rate, "action_variation_groups": t46["action_variation_groups"], "payoff_variation_groups": t46["true_payoff_variation_groups"]}, ensure_ascii=False))
    return 0 if t46["classification"] == "GRPO_READY_SIGNAL_ONLY" else 3


if __name__ == "__main__":
    raise SystemExit(main())
