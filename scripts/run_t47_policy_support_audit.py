#!/usr/bin/env python3
"""T47: action-policy support and sampling sanity audit.

The audit uses the frozen T45 epoch1 adapter, the exact T46 prompt/action
grammar, temperature, top-p, and TRAIN snapshot selection.  It never trains,
updates an adapter, accesses FINAL, or writes historical T41-T46 artifacts.
"""
from __future__ import annotations

import argparse
import inspect
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from paper9bus_gv_grpo.audit_hardening import validate_exact_prompt_identity
from paper9bus_gv_grpo.paths import CORE_ROOT
from paper9bus_gv_grpo.reward import expected_payoff, load_context, load_payoff_tables
from paper9bus_gv_grpo.schema import ACTION_VALUES, load_registry
from scripts.analyze_k4_interface import repetition_flags
from scripts.run_k4_interface_recovery import CoreESRGrammar, generate as generate_paired
from scripts.run_k4_signal_gate import parse_interface, select_snapshots
from scripts.run_public_state_v1_t45_t46 import load_model


ACTION_IDS = list(range(6))
ACTION_PREFIX = '{"a":'
TEMPERATURE = 0.9
TOP_P = 0.95
MAX_NEW_TOKENS = 256
K = 4
GROUPS = 64


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    return exp / exp.sum()


def entropy(probabilities: np.ndarray) -> float:
    p = probabilities[probabilities > 0]
    return float(-(p * np.log(p)).sum())


def top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Match transformers TopPLogitsWarper's sorted cumulative-probability rule."""
    filtered = logits.clone()
    sorted_logits, sorted_indices = torch.sort(filtered, descending=True)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    remove = cumulative > float(top_p)
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    indices = sorted_indices[remove]
    filtered[indices] = -float("inf")
    return filtered


def action_encoding(tok, grammar: CoreESRGrammar) -> tuple[dict, dict[int, list[int]]]:
    allowed = grammar.allowed(ACTION_PREFIX)
    allowed_details = [
        {
            "token_id": int(token_id),
            "decoded_text": grammar.token_text.get(int(token_id), tok.decode([int(token_id)], skip_special_tokens=False, clean_up_tokenization_spaces=False)),
        }
        for token_id in allowed
    ]
    actions = []
    action_tokens: dict[int, list[int]] = {}
    prefix_ids = tok.encode(ACTION_PREFIX, add_special_tokens=False)
    for action_id in ACTION_IDS:
        serialized = f'{ACTION_PREFIX}{action_id}'
        full_ids = tok.encode(serialized, add_special_tokens=False)
        continuation = full_ids[len(prefix_ids) :] if full_ids[: len(prefix_ids)] == prefix_ids else tok.encode(str(action_id), add_special_tokens=False)
        action_tokens[action_id] = [int(x) for x in continuation]
        actions.append(
            {
                "action_id": action_id,
                "markup_value": float(ACTION_VALUES[action_id]),
                "canonical_serialized_representation": serialized,
                "tokenizer_token_ids": [int(x) for x in continuation],
                "tokenizer_token_text": [tok.decode([int(x)], skip_special_tokens=False, clean_up_tokenization_spaces=False) for x in continuation],
                "number_of_tokens": len(continuation),
                "reachable_at_action_position": any(int(x) == int(continuation[0]) for x in allowed) if continuation else False,
            }
        )
    audit = {
        "current_action_field": "a",
        "action_decision_prefix": ACTION_PREFIX,
        "grammar_allowed_token_count_at_prefix": len(allowed),
        "grammar_allowed_tokens_at_prefix": allowed_details,
        "actions": actions,
        "all_six_actions_single_token": all(x["number_of_tokens"] == 1 for x in actions),
        "all_six_actions_reachable": all(x["reachable_at_action_position"] for x in actions),
        "grammar_action_support_restriction": len([x for x in actions if not x["reachable_at_action_position"]]) > 0,
    }
    return audit, action_tokens


def action_distribution(model, tok, grammar: CoreESRGrammar, prompt: str, action_tokens: dict[int, list[int]]) -> tuple[np.ndarray, np.ndarray, dict]:
    """Compute raw and T46-runtime probabilities at the action position."""
    if any(len(tokens) != 1 for tokens in action_tokens.values()):
        raise RuntimeError("T47 currently requires the verified single-token action encoding")
    encoded = tok(prompt + ACTION_PREFIX, return_tensors="pt", add_special_tokens=False).to("cuda")
    with torch.inference_mode():
        output = model(**encoded)
    logits = output.logits[0, -1, :].float().detach().cpu()
    ids = [int(action_tokens[action_id][0]) for action_id in ACTION_IDS]
    raw_logits = logits[ids].numpy().astype(np.float64)
    raw_prob = softmax(raw_logits)

    allowed = grammar.allowed(ACTION_PREFIX)
    masked = torch.full_like(logits, -float("inf"))
    masked[allowed] = logits[allowed]
    runtime_logits = top_p_filter(masked / float(TEMPERATURE), TOP_P)
    runtime_prob_full = torch.softmax(runtime_logits, dim=-1).numpy().astype(np.float64)
    runtime_prob = runtime_prob_full[ids]
    if not np.isclose(runtime_prob.sum(), 1.0, atol=1e-6):
        raise RuntimeError(f"runtime action distribution is not normalized: {runtime_prob.sum()}")

    def summary(probabilities: np.ndarray, source_logits: np.ndarray) -> dict:
        order = np.argsort(-probabilities)
        logprob = np.log(np.maximum(probabilities, 1e-300))
        return {
            "probabilities": [float(x) for x in probabilities],
            "top1_action": int(order[0]),
            "top1_probability": float(probabilities[order[0]]),
            "top2_action": int(order[1]),
            "top2_probability": float(probabilities[order[1]]),
            "top1_top2_logit_margin": float(source_logits[order[0]] - source_logits[order[1]]),
            "top1_top2_logprob_margin": float(logprob[order[0]] - logprob[order[1]]),
            "action_entropy": entropy(probabilities),
            "effective_support": float(math.exp(entropy(probabilities))),
            "non_top1_probability_mass": float(1.0 - probabilities[order[0]]),
            "support_count": int(np.sum(probabilities > 1e-12)),
        }

    return raw_prob, runtime_prob, {"raw": summary(raw_prob, raw_logits), "runtime": summary(runtime_prob, np.log(np.maximum(runtime_prob, 1e-300)))}


def economic_diagnostic(example_id: str, contexts: dict, tables: dict, runtime_prob: np.ndarray) -> dict:
    context = contexts[str(example_id)]
    values = np.asarray([expected_payoff(context, tables, action_id) for action_id in ACTION_IDS], dtype=np.float64)
    regret = values.max() - values
    action_range = float(values.max() - values.min())
    threshold = 0.10 * action_range
    return {
        "payoff_by_action": [float(x) for x in values],
        "regret_by_action": [float(x) for x in regret],
        "observation_bayes_target_action": int(np.argmax(values)),
        "expected_regret_under_pi_runtime": float(np.dot(runtime_prob, regret)),
        "near_optimal_regret_threshold": float(threshold),
        "non_top1_mass_on_near_optimal_actions": float(runtime_prob[(regret <= threshold) & (runtime_prob < runtime_prob.max() + 1e-15)].sum()),
        "non_top1_mass_on_high_regret_actions": float(runtime_prob[(regret > threshold) & (runtime_prob < runtime_prob.max() + 1e-15)].sum()),
    }


def audit_runtime_source() -> dict:
    generation_source = inspect.getsource(generate_paired)
    t45_source = inspect.getsource(__import__("scripts.run_public_state_v1_t45_t46", fromlist=["run_k4"]).run_k4)
    economic_source = inspect.getsource(__import__("scripts.run_public_state_v1_t45_t46", fromlist=["economic_gate"]).economic_gate)
    checks = {
        "do_sample_enabled": "do_sample=True" in generation_source,
        "temperature_forwarded": "temperature=temperature" in generation_source,
        "top_p_forwarded": "top_p=top_p" in generation_source,
        "grammar_uses_prefix_allowed_tokens": "prefix_allowed_tokens_fn" in generation_source,
        "grammar_path_keeps_sampling": "do_sample=True" in generation_source and "constrained" in generation_source,
        "no_beam_or_argmax_path": "num_beams" not in generation_source and "argmax" not in generation_source,
        "seed_initialized_once_per_generate_call": "seed_all(seed)" in generation_source,
        "candidate_rng_progresses_inside_batch": "generate_paired" in t45_source and "prompts = [str(item[\"prompt\"])] * 4" in t45_source,
        "parser_reads_generated_action": "parse_interface(row[\"raw_output\"]" in economic_source and "action_id" in economic_source,
        "scored_action_comes_from_parsed_action": "values[item[\"action_id\"]]" in economic_source,
        "exact_prompt_identity_guard_present": "validate_exact_prompt_identity" in t45_source,
    }
    return {"checks": checks, "all_static_checks_pass": all(checks.values())}


def audit_existing_candidates(tok, registry: dict, contexts: dict, tables: dict, dataset_by_example: dict) -> dict:
    frame = pd.read_parquet(ROOT / "runs/qwen3_1p7b_public_state_v1_sft_seed42/T45_K4_GRAMMAR_CANDIDATES.parquet")
    rows = []
    mismatches = []
    for record in frame.to_dict("records"):
        raw = str(record["raw_output"])
        raw_action = None
        try:
            obj = json.loads(raw)
            raw_action = int(obj["a"])
        except Exception:
            pass
        parsed = parse_interface(raw, bool(record["eos_completed"]), bool(record["truncated"]), registry)
        parsed_action = int(parsed["parsed"]["a"]) if parsed["parsed"] is not None else None
        scored_action = parsed_action if parsed["strict_valid"] else None
        example_id = str(record["example_id"])
        identity = validate_exact_prompt_identity(example_id, dataset_by_example[example_id])
        item = {
            "group_index": int(record["group_index"]),
            "candidate_index": int(record["candidate_index"]),
            "example_id": example_id,
            "sampling_batch_seed": int(record["sampling_batch_seed"]),
            "sampling_seed_metadata": int(record["sampling_seed"]),
            "temperature": float(record["temperature"]),
            "top_p": float(record["top_p"]),
            "raw_action": raw_action,
            "parsed_action": parsed_action,
            "scored_action": scored_action,
            "strict_valid": bool(parsed["strict_valid"]),
            **identity,
        }
        rows.append(item)
        if parsed["strict_valid"] and not (raw_action == parsed_action == scored_action):
            mismatches.append(item)
    return {
        "rows_checked": len(rows),
        "group_count": int(frame["group_index"].nunique()),
        "candidate_count_per_group": sorted(frame.groupby("group_index").size().unique().tolist()),
        "raw_parsed_scored_action_mismatches": mismatches,
        "temperature_values": sorted(frame["temperature"].unique().tolist()),
        "top_p_values": sorted(frame["top_p"].unique().tolist()),
        "candidate_audit_preview": rows[:8],
        "all_valid_action_paths_consistent": not mismatches,
    }


def full_generation_subset(model, tok, grammar: CoreESRGrammar, prompts: list[dict], registry: dict) -> list[dict]:
    results = []
    for state_index, item in enumerate(prompts):
        outputs = []
        for batch_index in range(0, 64, 4):
            seed = 47000000 + state_index * 10000 + batch_index
            generated = generate_paired(
                model,
                tok,
                grammar,
                [str(item["prompt"])] * 4,
                seed=seed,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                constrained=True,
            )
            outputs.extend(generated)
        actions = []
        strict = []
        serialized = []
        for output in outputs:
            parsed = parse_interface(output["raw_output"], output["eos_completed"], output["truncated"], registry)
            strict.append(bool(parsed["strict_valid"]))
            action = int(parsed["parsed"]["a"]) if parsed["parsed"] is not None else None
            actions.append(action)
            serialized.append(output["raw_output"])
        valid_actions = [x for x in actions if x is not None]
        results.append(
            {
                "state_index": state_index,
                "example_id": str(item["example_id"]),
                "prompt_hash": str(item["prompt_hash"]),
                "generations": 64,
                "unique_full_esr_outputs": len(set(serialized)),
                "unique_actions": len(set(valid_actions)),
                "action_frequencies": {str(a): int(valid_actions.count(a)) for a in sorted(set(valid_actions))},
                "strict_validity": float(np.mean(strict)),
            }
        )
        print(json.dumps({"T47_full_generation_state": state_index + 1, "unique_actions": results[-1]["unique_actions"], "strict_validity": results[-1]["strict_validity"]}), flush=True)
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    output_names = [
        "T47_ACTION_ENCODING_AUDIT.json",
        "T47_POLICY_SUPPORT_AUDIT.json",
        "T47_K4_VARIATION_PROBABILITY.parquet",
        "T47_SAMPLING_RUNTIME_AUDIT.json",
        "T47_MONTE_CARLO_SANITY.json",
        "T47_POLICY_SUPPORT_REPORT_CN.md",
    ]
    existing = [name for name in output_names if (args.output / name).exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite T47 outputs: {existing}")
    if not torch.cuda.is_available():
        raise RuntimeError("T47 requires CUDA and refuses CPU execution")

    checkpoint = ROOT / "runs/qwen3_1p7b_public_state_v1_sft_seed42/checkpoint_epoch1"
    freeze = read_json(ROOT / "reports/T45_CHECKPOINT_FREEZE.json")
    if int(freeze["selected_epoch"]) != 1 or Path(freeze["selected_checkpoint"]) != checkpoint:
        raise RuntimeError("T47 requires the frozen T45 epoch1 checkpoint")
    dataset = read_jsonl(ROOT / "data/public_state_v1_sft/train.jsonl")
    dataset_by_example = {str(x["example_id"]): x for x in dataset}
    snapshots = select_snapshots()
    selected = []
    for snap in snapshots.itertuples(index=False):
        item = dataset_by_example.get(str(snap.example_id))
        if item is None:
            raise RuntimeError(f"missing exact TRAIN prompt for {snap.example_id}")
        validate_exact_prompt_identity(str(snap.example_id), item)
        selected.append(item)
    if len(selected) != GROUPS:
        raise RuntimeError(f"expected exactly {GROUPS} frozen T46 states")

    manifest = read_json(SFT_DIR / "RUN_MANIFEST.json") if (SFT_DIR := ROOT / "runs/qwen3_1p7b_public_state_v1_sft_seed42") else {}
    model_name = str(manifest["model"])
    registry = load_registry(CORE_ROOT / "enum_registry.json")
    contexts = load_context("TRAIN")
    tables = load_payoff_tables("TRAIN")
    model, tok = load_model(model_name, checkpoint)
    grammar = CoreESRGrammar(tok)

    encoding, action_tokens = action_encoding(tok, grammar)
    if not encoding["all_six_actions_reachable"]:
        raise RuntimeError("grammar action support restriction detected; see encoding audit")

    rows = []
    for index, item in enumerate(selected):
        raw_prob, runtime_prob, summary = action_distribution(model, tok, grammar, str(item["prompt"]), action_tokens)
        economic = economic_diagnostic(str(item["example_id"]), contexts, tables, runtime_prob)
        p_same = float(np.sum(runtime_prob ** K))
        rows.append(
            {
                "state_index": index,
                "example_id": str(item["example_id"]),
                "physical_state_id": str(snapshots.iloc[index]["physical_state_id"]),
                "prompt_hash": str(item["prompt_hash"]),
                "grammar_allowed_action_set": json.dumps(ACTION_IDS),
                "raw_probabilities": json.dumps([float(x) for x in raw_prob]),
                "runtime_probabilities": json.dumps([float(x) for x in runtime_prob]),
                "raw_top1_action": summary["raw"]["top1_action"],
                "raw_top1_probability": summary["raw"]["top1_probability"],
                "raw_action_entropy": summary["raw"]["action_entropy"],
                "raw_effective_support": summary["raw"]["effective_support"],
                "runtime_top1_action": summary["runtime"]["top1_action"],
                "runtime_top1_probability": summary["runtime"]["top1_probability"],
                "runtime_action_entropy": summary["runtime"]["action_entropy"],
                "runtime_effective_support": summary["runtime"]["effective_support"],
                "runtime_non_top1_probability_mass": summary["runtime"]["non_top1_probability_mass"],
                "p_same_k4": p_same,
                "p_vary_k4": 1.0 - p_same,
                **economic,
            }
        )
        if (index + 1) % 8 == 0:
            print(json.dumps({"T47_probability_states": index + 1}), flush=True)

    variation = pd.DataFrame(rows)
    p_vary = variation["p_vary_k4"].to_numpy(dtype=float)
    p_same = variation["p_same_k4"].to_numpy(dtype=float)
    log_p_zero = float(np.log(np.maximum(p_same, 1e-300)).sum())
    p_zero = float(math.exp(log_p_zero)) if log_p_zero > math.log(np.finfo(float).tiny) else 0.0
    expected_varying = float(p_vary.sum())
    thresholds = {str(t): int(np.sum(p_vary > t)) for t in (0.01, 0.05, 0.10, 0.25)}
    theoretical = {
        "mean_p_vary_k4": float(np.mean(p_vary)),
        "median_p_vary_k4": float(np.median(p_vary)),
        "p10_p_vary_k4": float(np.quantile(p_vary, 0.10)),
        "p90_p_vary_k4": float(np.quantile(p_vary, 0.90)),
        "max_p_vary_k4": float(np.max(p_vary)),
        "states_p_vary_above_threshold": thresholds,
        "expected_number_of_varying_groups": expected_varying,
        "p_observe_zero_varying_groups": p_zero,
        "log_p_observe_zero_varying_groups": log_p_zero,
        "observed_t46_varying_groups": 0,
        "observed_t46_total_groups": 64,
    }

    monte_carlo_rows = []
    rng = np.random.default_rng(470470)
    for row in rows:
        probabilities = np.asarray(json.loads(row["runtime_probabilities"]), dtype=float)
        samples = rng.choice(ACTION_IDS, size=256, p=probabilities)
        empirical = np.bincount(samples, minlength=6) / 256.0
        monte_carlo_rows.append(
            {
                "state_index": int(row["state_index"]),
                "example_id": row["example_id"],
                "runtime_probabilities": [float(x) for x in probabilities],
                "empirical_probabilities": [float(x) for x in empirical],
                "total_variation_distance": float(0.5 * np.abs(empirical - probabilities).sum()),
            }
        )
    mc_tv = [x["total_variation_distance"] for x in monte_carlo_rows]

    subset = full_generation_subset(model, tok, grammar, selected[:8], registry)
    runtime_audit = audit_runtime_source()
    candidate_audit = audit_existing_candidates(tok, registry, contexts, tables, dataset_by_example)
    grammar_restriction = bool(encoding["grammar_action_support_restriction"])
    mean_support = float(variation["runtime_effective_support"].mean())
    mean_top1 = float(variation["runtime_top1_probability"].mean())
    model_concentrated = mean_support <= 1.25 and mean_top1 >= 0.95
    sampling_predicts_variation = expected_varying >= 10.0 and p_zero < 1e-6
    full_subset_varies = any(x["unique_actions"] > 1 for x in subset)
    if grammar_restriction and model_concentrated:
        classification = "MIXED_MODEL_AND_SAMPLING_COLLAPSE"
    elif grammar_restriction:
        classification = "GRAMMAR_ACTION_SUPPORT_COLLAPSE"
    elif sampling_predicts_variation and not full_subset_varies:
        classification = "SAMPLING_IMPLEMENTATION_COLLAPSE"
    elif model_concentrated and p_zero >= 0.01:
        classification = "MODEL_POLICY_SUPPORT_COLLAPSE"
    elif not model_concentrated and full_subset_varies:
        classification = "NONDEGENERATE_POLICY_SUPPORT"
    else:
        classification = "MODEL_POLICY_SUPPORT_COLLAPSE" if p_zero >= 0.01 else "SAMPLING_IMPLEMENTATION_COLLAPSE"

    action_encoding_report = {
        "stage": "T47A_ACTION_ENCODING_GRAMMAR_AUDIT",
        **encoding,
        "all_64_states_allowed_action_set": [ACTION_IDS for _ in range(GROUPS)],
        "final_accessed": False,
        "grpo_started": False,
    }
    policy_report = {
        "stage": "T47B_POLICY_SUPPORT_AUDIT",
        "checkpoint": str(checkpoint.resolve()),
        "model": model_name,
        "groups": GROUPS,
        "action_values": [float(x) for x in ACTION_VALUES],
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "action_prefix": ACTION_PREFIX,
        "mean_raw_top1_probability": float(variation["raw_top1_probability"].mean()),
        "mean_raw_action_entropy": float(variation["raw_action_entropy"].mean()),
        "mean_raw_effective_support": float(variation["raw_effective_support"].mean()),
        "mean_runtime_top1_probability": mean_top1,
        "mean_runtime_action_entropy": float(variation["runtime_action_entropy"].mean()),
        "mean_runtime_effective_support": mean_support,
        "mean_runtime_non_top1_probability_mass": float(variation["runtime_non_top1_probability_mass"].mean()),
        "theoretical_k4": theoretical,
        "economic": {
            "mean_expected_regret_under_pi_runtime": float(variation["expected_regret_under_pi_runtime"].mean()),
            "mean_non_top1_mass_on_near_optimal_actions": float(variation["non_top1_mass_on_near_optimal_actions"].mean()),
            "mean_non_top1_mass_on_high_regret_actions": float(variation["non_top1_mass_on_high_regret_actions"].mean()),
        },
        "classification": classification,
        "final_accessed": False,
        "grpo_started": False,
    }
    sampling_report = {
        "stage": "T47D_SAMPLING_RUNTIME_AUDIT",
        "static_source_audit": runtime_audit,
        "existing_t46_candidate_audit": candidate_audit,
        "grammar_action_support_restriction": grammar_restriction,
        "exact_prompt_identity_checked_for_64_states": True,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_new_tokens": MAX_NEW_TOKENS,
        "final_accessed": False,
        "grpo_started": False,
    }
    mc_report = {
        "stage": "T47_MONTE_CARLO_SANITY",
        "samples_per_state": 256,
        "states": GROUPS,
        "mean_total_variation_distance": float(np.mean(mc_tv)),
        "median_total_variation_distance": float(np.median(mc_tv)),
        "max_total_variation_distance": float(np.max(mc_tv)),
        "state_results": monte_carlo_rows,
        "full_grammar_subset": subset,
        "final_accessed": False,
        "grpo_started": False,
    }
    variation.to_parquet(args.output / "T47_K4_VARIATION_PROBABILITY.parquet", index=False)
    (args.output / "T47_ACTION_ENCODING_AUDIT.json").write_text(json.dumps(action_encoding_report, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    (args.output / "T47_POLICY_SUPPORT_AUDIT.json").write_text(json.dumps(policy_report, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    (args.output / "T47_SAMPLING_RUNTIME_AUDIT.json").write_text(json.dumps(sampling_report, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    (args.output / "T47_MONTE_CARLO_SANITY.json").write_text(json.dumps(mc_report, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    report = [
        "# T47 Action-Policy Support and Sampling Audit",
        "",
        f"- Primary classification: `{classification}`。",
        f"- checkpoint：epoch1；T45 selected Grammar；temperature=`{TEMPERATURE}`，top-p=`{TOP_P}`。",
        f"- mean runtime top1 probability：`{mean_top1:.6f}`。",
        f"- mean runtime action entropy：`{variation['runtime_action_entropy'].mean():.6f}`。",
        f"- mean runtime effective support：`{mean_support:.6f}`。",
        f"- mean P(vary,K=4)：`{theoretical['mean_p_vary_k4']:.6f}`。",
        f"- expected varying groups：`{expected_varying:.6f}/64`。",
        f"- P(observe zero varying groups)：`{p_zero:.6e}`。",
        f"- grammar action support：6/6 legal actions reachable at all 64 states。",
        f"- full-generation diagnostic subset：8 states × 64 exact Grammar generations；见 `T47_MONTE_CARLO_SANITY.json`。",
        "",
        "## 决策边界",
        "",
        f"- 未训练、未启动 GRPO、未访问 FINAL。",
        f"- 只有在本 T47 分类为 `MODEL_POLICY_SUPPORT_COLLAPSE` 时，才解锁下一项 soft-policy SFT 设计；本脚本不会自动启动下一项。",
    ]
    (args.output / "T47_POLICY_SUPPORT_REPORT_CN.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    del model, tok
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(json.dumps({"classification": classification, "mean_runtime_top1": mean_top1, "mean_runtime_entropy": float(variation["runtime_action_entropy"].mean()), "mean_effective_support": mean_support, "mean_p_vary_k4": theoretical["mean_p_vary_k4"], "expected_varying_groups": expected_varying, "p_zero": p_zero}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
