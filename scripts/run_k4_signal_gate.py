#!/usr/bin/env python3
"""Frozen Qwen SFT K=4 true-economic-signal gate.

This runner deliberately stops after the gate. It never updates the adapter and
uses only TRAIN rows plus the frozen TRAIN payoff cell bank.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paper9bus_gv_grpo.paths import CORE_ROOT, context_file, split_file
from paper9bus_gv_grpo.audit_hardening import strict_one_to_one_merge
from paper9bus_gv_grpo.reward import expected_payoff, load_context, load_payoff_tables
from paper9bus_gv_grpo.schema import ACTION_VALUES, load_registry, parse_core

EXPECTED_KEYS = {"e", "b", "g", "cf", "i", "a", "p", "q"}
EPS = 1e-8


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_text(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=float)


def entropy(action_ids: list[int]) -> float:
    if not action_ids:
        return 0.0
    counts = Counter(action_ids)
    n = len(action_ids)
    return float(-sum((c / n) * math.log(c / n) for c in counts.values()))


def parse_interface(raw: str, eos_completed: bool, truncated: bool, registry: dict) -> dict:
    parse_success = False
    private_field_violation = False
    obj = None
    parse_reason = None
    try:
        obj = json.loads(raw)
        parse_success = True
        if isinstance(obj, dict):
            private_field_violation = bool(set(obj) - EXPECTED_KEYS)
    except Exception as exc:
        parse_reason = f"json:{exc.__class__.__name__}"

    core_valid = False
    parsed = None
    if parse_success:
        try:
            parsed = parse_core(obj, registry)
            core_valid = True
        except Exception as exc:
            parse_reason = str(exc)

    reasons = []
    if not parse_success:
        reasons.append(parse_reason or "json_parse_failed")
    elif not core_valid:
        reasons.append(parse_reason or "core_invalid")
    if private_field_violation:
        reasons.append("private_field_violation")
    if not eos_completed:
        reasons.append("eos_missing")
    if truncated:
        reasons.append("truncated")

    strict_valid = bool(core_valid and eos_completed and not truncated)
    return {
        "parse_success": parse_success,
        "core_valid": core_valid,
        "strict_valid": strict_valid,
        "private_field_violation": private_field_violation,
        "eos_completed": bool(eos_completed),
        "truncated": bool(truncated),
        "validity_reason": ";".join(reasons) if reasons else None,
        "parsed": parsed,
    }


def generate_one(model, tok, prompt: str, *, seed: int, max_new_tokens: int,
                 temperature: float, top_p: float, do_sample: bool) -> dict:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    enc = tok(prompt + "\n", return_tensors="pt").to("cuda")
    with torch.inference_mode():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            **({"temperature": temperature, "top_p": top_p} if do_sample else {}),
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
        )
    generated_ids = out[0, enc.input_ids.shape[1]:].detach().cpu().tolist()
    eos_ids = tok.eos_token_id
    eos_set = {int(eos_ids)} if isinstance(eos_ids, int) else {int(x) for x in eos_ids}
    eos_completed = any(int(x) in eos_set for x in generated_ids)
    truncated = len(generated_ids) >= max_new_tokens and not eos_completed
    raw = tok.decode(generated_ids, skip_special_tokens=True).strip()
    return {
        "raw_output": raw,
        "generated_tokens": len(generated_ids),
        "eos_completed": eos_completed,
        "truncated": truncated,
    }


def generate_batch(model, tok, prompts: list[str], *, seed: int, max_new_tokens: int,
                   temperature: float, top_p: float, do_sample: bool) -> list[dict]:
    """Generate a batch with one frozen decoding mode."""
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
    with torch.inference_mode():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            **({"temperature": temperature, "top_p": top_p} if do_sample else {}),
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
        )
    eos_ids = tok.eos_token_id
    eos_set = {int(eos_ids)} if isinstance(eos_ids, int) else {int(x) for x in eos_ids}
    rows = []
    for i in range(len(prompts)):
        generated_ids = out[i, enc.input_ids.shape[1]:].detach().cpu().tolist()
        eos_completed = any(int(x) in eos_set for x in generated_ids)
        truncated = len(generated_ids) >= max_new_tokens and not eos_completed
        rows.append({
            "raw_output": tok.decode(generated_ids, skip_special_tokens=True).strip(),
            "generated_tokens": len(generated_ids),
            "eos_completed": eos_completed,
            "truncated": truncated,
        })
    return rows


def select_snapshots() -> pd.DataFrame:
    core = pd.read_parquet(split_file("TRAIN"))
    ctx = pd.read_parquet(context_file("TRAIN"))
    if set(core.split.unique()) != {"TRAIN"} or set(ctx.split.unique()) != {"TRAIN"}:
        raise RuntimeError("TRAIN snapshot selection saw a non-TRAIN split")
    merged = strict_one_to_one_merge(
        core,
        ctx[["example_id", "physical_state_id"]],
        key="example_id",
        expected_rows=len(core),
        left_name="TRAIN.core",
        right_name="TRAIN.context",
    )
    selected = merged.drop_duplicates("physical_state_id", keep="first").head(64).copy()
    if len(selected) != 64:
        raise RuntimeError(f"expected 64 unique TRAIN physical states, found {len(selected)}")
    return selected


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft-run", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--groups", type=int, default=64)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--precision", choices=["fp16", "nf4"], default="fp16")
    args = ap.parse_args()
    if args.groups != 64 or args.k != 4:
        raise ValueError("this preregistered gate requires exactly 64 TRAIN snapshots and K=4")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing frozen-policy gate on CPU")

    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.sft_run / "RUN_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model_name = str(manifest["model"])
    registry = load_registry(CORE_ROOT / "enum_registry.json")
    context = load_context("TRAIN")
    tables = load_payoff_tables("TRAIN")
    snapshots = select_snapshots()

    tok = AutoTokenizer.from_pretrained(args.sft_run / "adapter", trust_remote_code=True)
    tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = "left"
    model_kwargs = {
        "device_map": {"": 0},
        "trust_remote_code": True,
        "torch_dtype": torch.float16,
    }
    if args.precision == "nf4":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    base = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model = PeftModel.from_pretrained(base, args.sft_run / "adapter").eval()

    candidate_rows: list[dict] = []
    baseline_rows: dict[str, dict] = {}
    snapshot_records = []
    base_seed = int(manifest.get("config", {}).get("seed", 42))

    for group_index, row in enumerate(snapshots.itertuples(index=False)):
        eid = str(row.example_id)
        physical_state_id = str(row.physical_state_id)
        prompt = str(row.prompt)
        snapshot_records.append({
            "group_index": group_index,
            "example_id": eid,
            "physical_state_id": physical_state_id,
            "split": "TRAIN",
        })

        batch_seed = base_seed * 100000 + group_index * 100
        generated_batch = generate_batch(
            model, tok, [prompt] * args.k, seed=batch_seed,
            max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p,
            do_sample=True,
        )
        for candidate_index, generated in enumerate(generated_batch):
            seed = batch_seed + candidate_index
            audit = parse_interface(
                generated["raw_output"], generated["eos_completed"], generated["truncated"], registry
            )
            candidate_rows.append({
                "group_index": group_index,
                "example_id": eid,
                "physical_state_id": physical_state_id,
                "candidate_index": candidate_index,
                "split": "TRAIN",
                "sampling_seed": seed,
                "sampling_batch_seed": batch_seed,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_new_tokens": args.max_new_tokens,
                "raw_output": generated["raw_output"],
                "generated_tokens": generated["generated_tokens"],
                **{k: v for k, v in audit.items() if k != "parsed"},
                "parsed_json": json_text(audit["parsed"]) if audit["parsed"] is not None else None,
            })

    # Greedy baselines are generated in batches too; they are never used to
    # construct the stochastic candidate groups.
    snapshot_rows = list(snapshots.itertuples(index=False))
    for start in range(0, len(snapshot_rows), 8):
        chunk = snapshot_rows[start:start + 8]
        greedy_batch = generate_batch(
            model, tok, [str(row.prompt) for row in chunk],
            seed=base_seed * 200000 + start,
            max_new_tokens=args.max_new_tokens, temperature=args.temperature,
            top_p=args.top_p, do_sample=False,
        )
        for offset, (row, greedy) in enumerate(zip(chunk, greedy_batch)):
            group_index = start + offset
            eid = str(row.example_id)
            greedy_audit = parse_interface(
                greedy["raw_output"], greedy["eos_completed"], greedy["truncated"], registry
            )
            baseline_rows[eid] = {"group_index": group_index, "example_id": eid,
                                 "physical_state_id": str(row.physical_state_id),
                                 "greedy": greedy, "audit": greedy_audit}

    candidate_frame = pd.DataFrame(candidate_rows)
    group_audits: list[dict] = []
    for group_index, snap in enumerate(snapshot_records):
        eid = snap["example_id"]
        group = candidate_frame[candidate_frame.group_index == group_index].copy()
        ctx = context[eid]
        values = [expected_payoff(ctx, tables, i) for i in range(len(ACTION_VALUES))]
        best_action = int(np.argmax(values))
        best_payoff = float(values[best_action])
        valid = group[group.strict_valid == True].copy()  # noqa: E712
        # Populate candidate economic fields only for strict-valid candidates.
        action_ids = []
        utility = []
        regrets = []
        for idx in valid.index:
            action_id = int(json.loads(candidate_frame.at[idx, "parsed_json"])["a"])
            u = float(values[action_id])
            r = float(best_payoff - u)
            action_ids.append(action_id)
            candidate_frame.at[idx, "action_id"] = action_id
            candidate_frame.at[idx, "action_value"] = float(ACTION_VALUES[action_id])
            candidate_frame.at[idx, "expected_profit"] = u
            candidate_frame.at[idx, "observation_regret"] = r
            utility.append(u)
            regrets.append(r)
        utility_std = float(np.std(utility)) if utility else 0.0
        utility_range = float(max(utility) - min(utility)) if utility else 0.0
        regret_std = float(np.std(regrets)) if regrets else 0.0
        unique_utilities = len({round(x, 10) for x in utility})
        group_adv = [0.0] * len(valid)
        corr = None
        if utility and utility_std > EPS:
            mean_u = float(np.mean(utility))
            group_adv = [float((x - mean_u) / (utility_std + 1e-8)) for x in utility]
            if len(group_adv) >= 2 and np.std(group_adv) > 0 and np.std(regrets) > 0:
                corr = float(np.corrcoef(np.asarray(group_adv), -np.asarray(regrets))[0, 1])
        for idx, adv in zip(valid.index, group_adv):
            candidate_frame.at[idx, "group_advantage"] = adv

        greedy_info = baseline_rows[eid]
        greedy_audit = greedy_info["audit"]
        greedy_action = None
        greedy_profit = None
        greedy_regret = None
        if greedy_audit["strict_valid"]:
            greedy_action = int(greedy_audit["parsed"]["a"])
            greedy_profit = float(values[greedy_action])
            greedy_regret = float(best_payoff - greedy_profit)

        frequencies = dict(sorted(Counter(action_ids).items()))
        group_audits.append({
            **snap,
            "n_candidates": len(group),
            "n_strict_valid": int(len(valid)),
            "strict_valid_rate": float(len(valid) / args.k),
            "unique_action_count": len(set(action_ids)),
            "action_entropy": entropy(action_ids),
            "action_frequency_json": json_text(frequencies),
            "utility_std": utility_std,
            "utility_range": utility_range,
            "regret_std": regret_std,
            "number_unique_utility_values": unique_utilities,
            "economic_variance_signal": bool(utility_std > EPS),
            "action_variation_signal": bool(len(set(action_ids)) > 1),
            "joint_action_economic_signal": bool(len(set(action_ids)) > 1 and utility_std > EPS),
            "advantage_nonzero": bool(any(abs(x) > EPS for x in group_adv)),
            "advantage_negative_regret_corr": corr,
            "observation_optimal_action": best_action,
            "observation_optimal_profit": best_payoff,
            "always_1p50_regret": float(best_payoff - values[5]),
            "sft_greedy_action": greedy_action,
            "sft_greedy_profit": greedy_profit,
            "sft_greedy_regret": greedy_regret,
            "greedy_strict_valid": bool(greedy_audit["strict_valid"]),
        })

    group_frame = pd.DataFrame(group_audits)
    candidate_frame.to_parquet(args.output / "K4_CANDIDATES.parquet", index=False)
    group_frame.to_parquet(args.output / "K4_GROUP_ECONOMIC_AUDIT.parquet", index=False)

    valid_rate = float(candidate_frame.strict_valid.mean()) if len(candidate_frame) else 0.0
    action_groups = int(group_frame.action_variation_signal.sum())
    economic_groups = int(group_frame.economic_variance_signal.sum())
    joint_groups = int(group_frame.joint_action_economic_signal.sum())
    corr_values = group_frame.advantage_negative_regret_corr.dropna().astype(float).tolist()
    positive_corr_groups = int(sum(x > 0 for x in corr_values))
    nonzero_adv_candidates = int((candidate_frame.group_advantage.abs() > EPS).sum())
    valid_candidates = int(candidate_frame.strict_valid.sum())
    candidate_adv_fraction = float(nonzero_adv_candidates / valid_candidates) if valid_candidates else 0.0
    greedy_valid_rate = float(group_frame.greedy_strict_valid.mean()) if len(group_frame) else 0.0
    mean_always_regret = float(group_frame.always_1p50_regret.mean())
    mean_sft_greedy_regret = float(group_frame.sft_greedy_regret.dropna().mean()) if group_frame.sft_greedy_regret.notna().any() else None

    criteria = {
        "exactly_64_groups_and_256_candidates": len(group_frame) == 64 and len(candidate_frame) == 256,
        "strict_valid_rate_ge_0.95": valid_rate >= 0.95,
        "action_variation_across_multiple_states": action_groups >= 2,
        "economic_variation_across_multiple_states": economic_groups >= 2,
        "joint_action_and_economic_variation_across_multiple_states": joint_groups >= 2,
        "advantage_ranks_lower_regret_higher_across_multiple_groups": positive_corr_groups >= 2,
    }
    if not criteria["strict_valid_rate_ge_0.95"]:
        status = "FAIL_INTERFACE_VALIDITY"
    elif criteria["advantage_ranks_lower_regret_higher_across_multiple_groups"] and criteria["joint_action_and_economic_variation_across_multiple_states"]:
        status = "PASS_K4_ECONOMIC_SIGNAL"
    elif criteria["action_variation_across_multiple_states"] and not criteria["economic_variation_across_multiple_states"]:
        status = "FORMAT_SIGNAL_ONLY"
    else:
        status = "FAIL_K4_STRATEGIC_SIGNAL"

    result = {
        "gate": "K4_TRUE_ECONOMIC_SIGNAL",
        "status": status,
        "protocol": "Paper9Bus-Power-GV-GRPO-v3",
        "split": "TRAIN",
        "k": 4,
        "inference_precision": args.precision,
        "groups": len(group_frame),
        "candidate_rows": len(candidate_frame),
        "strict_valid_rate": valid_rate,
        "action_variation_groups": action_groups,
        "economic_variation_groups": economic_groups,
        "joint_action_economic_groups": joint_groups,
        "fraction_groups_with_economic_variance": float(group_frame.economic_variance_signal.mean()),
        "mean_utility_std": float(group_frame.utility_std.mean()),
        "median_utility_std": float(group_frame.utility_std.median()),
        "mean_utility_range": float(group_frame.utility_range.mean()),
        "p90_utility_range": float(group_frame.utility_range.quantile(0.90)),
        "fraction_groups_with_nonzero_advantage": float(group_frame.advantage_nonzero.mean()),
        "fraction_valid_candidates_with_nonzero_advantage": candidate_adv_fraction,
        "advantage_corr_groups": len(corr_values),
        "positive_advantage_negative_regret_corr_groups": positive_corr_groups,
        "mean_advantage_negative_regret_corr": float(np.mean(corr_values)) if corr_values else None,
        "greedy_baseline_valid_rate": greedy_valid_rate,
        "mean_always_1p50_regret": mean_always_regret,
        "mean_sft_greedy_regret": mean_sft_greedy_regret,
        "criteria": criteria,
        "final_accessed": False,
    }
    (args.output / "K4_SIGNAL_GATE_RESULT.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=float), encoding="utf-8"
    )

    gate_manifest = {
        "protocol": "Paper9Bus-Power-GV-GRPO-v3",
        "stage": "K4_TRUE_ECONOMIC_SIGNAL_GATE",
        "model": model_name,
        "sft_run": str(args.sft_run),
        "sft_manifest_sha256": sha256(manifest_path),
        "split": "TRAIN",
        "groups": 64,
        "k": 4,
        "sampling": {"do_sample": True, "temperature": args.temperature, "top_p": args.top_p,
                      "max_new_tokens": args.max_new_tokens, "inference_precision": args.precision},
        "snapshot_selection": "first row per physical_state_id in frozen TRAIN parquet, then first 64 states",
        "snapshots": snapshot_records,
        "greedy_baseline": {"do_sample": False, "same_frozen_sft_adapter": True},
        "payoff": "expected_payoff over frozen TRAIN cell bank, direct profit_g1 lookup",
        "outputs": ["K4_CANDIDATES.parquet", "K4_GROUP_ECONOMIC_AUDIT.parquet",
                    "K4_SIGNAL_GATE_RESULT.json", "K4_SIGNAL_GATE_REPORT_CN.md"],
        "grpo_started": False,
        "final_accessed": False,
    }
    (args.output / "K4_SIGNAL_GATE_MANIFEST.json").write_text(
        json.dumps(gate_manifest, indent=2, ensure_ascii=False, default=float), encoding="utf-8"
    )

    report = "# K=4 真实经济信号 Gate 报告\n\n"
    report += f"- 模型：`{model_name}`\n- 数据：TRAIN-only，64 个独立 physical state，K=4，共 256 candidates\n"
    report += f"- 状态：`{status}`\n- strict validity：{valid_rate:.4f}\n"
    report += f"- action variation groups：{action_groups}/64\n- economic variation groups：{economic_groups}/64\n"
    report += f"- joint action/economic variation groups：{joint_groups}/64\n"
    report += f"- utility std mean/median：{group_frame.utility_std.mean():.6g} / {group_frame.utility_std.median():.6g}\n"
    report += f"- utility range mean/p90：{group_frame.utility_range.mean():.6g} / {group_frame.utility_range.quantile(.9):.6g}\n"
    report += f"- nonzero advantage candidate fraction：{candidate_adv_fraction:.4f}\n"
    report += f"- positive corr(advantage, -regret) groups：{positive_corr_groups}/{len(corr_values)}\n"
    report += f"- mean ALWAYS-1.50 regret：{mean_always_regret:.6g}\n- mean frozen SFT greedy regret：{mean_sft_greedy_regret if mean_sft_greedy_regret is not None else 'N/A'}\n\n"
    report += "本次 gate 未更新 adapter，未使用 DEV/FINAL，也未启动 GRPO。经济值来自冻结 TRAIN cell bank 的 direct expected payoff，不使用历史 proxy 或 schema penalty。\n"
    (args.output / "K4_SIGNAL_GATE_REPORT_CN.md").write_text(report, encoding="utf-8")

    print(json.dumps(result, indent=2, ensure_ascii=False, default=float))
    return 0 if status == "PASS_K4_ECONOMIC_SIGNAL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
