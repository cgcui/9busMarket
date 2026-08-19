#!/usr/bin/env python3
"""T23: deterministic DEV checkpoint selection plus paired 64xK4 confirmation."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))
from paper9bus_gv_grpo.paths import CORE_ROOT, split_file
from paper9bus_gv_grpo.reward import load_context, load_payoff_tables
from paper9bus_gv_grpo.schema import load_registry
from scripts.analyze_k4_interface import classify
from scripts.run_k4_signal_gate import parse_interface, select_snapshots
from scripts.run_k4_interface_recovery import CoreESRGrammar, field_flags, generate as generate_paired, summarize


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_hashes(root: Path) -> dict:
    return {str(p.relative_to(root)).replace("\\", "/"): sha256(p) for p in sorted(root.rglob("*")) if p.is_file()}


def deterministic_batch(model, tok, prompts, max_new_tokens, batch_size=32):
    rows = []
    eos_ids = {int(tok.eos_token_id)} if isinstance(tok.eos_token_id, int) else {int(x) for x in tok.eos_token_id}
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start:start + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True).to("cuda")
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id)
        for i in range(len(chunk)):
            ids = out[i, enc.input_ids.shape[1]:].detach().cpu().tolist()
            eos_pos = next((j for j, x in enumerate(ids) if int(x) in eos_ids), None)
            actual = ids if eos_pos is None else ids[:eos_pos + 1]
            rows.append({"raw_output": tok.decode(actual[:-1] if eos_pos is not None else actual, skip_special_tokens=True).strip(),
                         "generated_tokens": len(actual), "eos_completed": eos_pos is not None,
                         "truncated": eos_pos is None and len(actual) >= max_new_tokens})
    return rows


def dev_summary(rows, registry):
    flags = [field_flags(x["raw_output"]) for x in rows]
    audits = [parse_interface(x["raw_output"], x["eos_completed"], x["truncated"], registry) for x in rows]
    complete_objs = []
    extra = []
    leakage = []
    categories = []
    for row, audit in zip(rows, audits):
        obj = None
        try: obj = json.loads(row["raw_output"])
        except Exception: pass
        complete_objs.append(obj)
        leakage.append(bool(audit["private_field_violation"]))
        extra.append(bool(audit["private_field_violation"]))
        ns = SimpleNamespace(group_index=0, candidate_index=0, example_id="dev", physical_state_id="dev", split="DEV",
                             raw_output=row["raw_output"], generated_tokens=row["generated_tokens"], max_new_tokens=256,
                             eos_completed=row["eos_completed"], truncated=row["truncated"], strict_valid=audit["strict_valid"],
                             parse_success=audit["parse_success"], core_valid=audit["core_valid"], validity_reason=audit["validity_reason"])
        c = classify(ns); categories.append(c["failure_category"])
        if obj is not None:
            try:
                text = row["raw_output"]; start = text.find("{"); _, end = json.JSONDecoder().raw_decode(text[start:]); end += start
                extra[-1] = bool(text[:start].strip() or text[end:].strip())
            except Exception: pass
    action_valid = []
    for obj in complete_objs:
        action_valid.append(isinstance(obj, dict) and isinstance(obj.get("a"), (int, float)) and int(obj["a"]) in range(6))
    return {"rows": len(rows), "strict_valid_rate": float(np.mean([x["strict_valid"] for x in audits])),
            "json_valid_rate": float(np.mean([x["parse_success"] for x in audits])), "eos_rate": float(np.mean([x["eos_completed"] for x in rows])),
            "truncation_rate": float(np.mean([x["truncated"] for x in rows])), "plan_valid_rate": float(np.mean([x["plan_valid"] for x in flags])),
            "belief_valid_rate": float(np.mean([x["belief_valid"] for x in flags])), "game_valid_rate": float(np.mean([x["game_valid"] for x in flags])),
            "action_valid_rate": float(np.mean(action_valid)), "hidden_leakage_rate": float(np.mean(leakage)),
            "prefix_suffix_extra_text_rate": float(np.mean(extra)), "failure_category_counts": dict(Counter(categories))}


def load_inference_model(model_name, checkpoint):
    tok = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True); tok.pad_token = tok.pad_token or tok.eos_token; tok.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(model_name, device_map={"": 0}, trust_remote_code=True, torch_dtype=torch.float16)
    return PeftModel.from_pretrained(base, checkpoint).eval(), tok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recovery-run", type=Path, required=True)
    ap.add_argument("--protocol", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)
    args = ap.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8")); model_name = protocol["base_model"]
    completion_limit = int(protocol["training"]["completion_limit"])
    registry = load_registry(CORE_ROOT / "enum_registry.json")
    dev = pd.read_parquet(split_file("DEV"))
    checkpoint_paths = [args.recovery_run / f"checkpoint_epoch{i}" for i in (1, 2, 3)]
    dev_results = []
    for epoch, checkpoint in zip((1, 2, 3), checkpoint_paths):
        if not checkpoint.exists(): raise RuntimeError(f"missing checkpoint_epoch{epoch}")
        model, tok = load_inference_model(model_name, checkpoint)
        rows = deterministic_batch(model, tok, [str(x) + "\n" for x in dev.prompt.tolist()], completion_limit)
        summary = dev_summary(rows, registry); summary.update({"epoch": epoch, "checkpoint": str(checkpoint), "checkpoint_hashes": tree_hashes(checkpoint)})
        dev_results.append(summary)
        del model; gc.collect(); torch.cuda.empty_cache()
    ranked = sorted(dev_results, key=lambda x: (-x["strict_valid_rate"], x["truncation_rate"], -x["eos_rate"], -x["plan_valid_rate"], -(x["belief_valid_rate"] + x["game_valid_rate"])))
    selected = ranked[0]; selected_epoch = int(selected["epoch"]); selected_checkpoint = checkpoint_paths[selected_epoch - 1]
    comparison = {"stage": "T23_DEV_CHECKPOINT_COMPARISON", "model": model_name, "split": "DEV", "completion_limit": completion_limit,
                  "selection_hierarchy": ["maximize strict validity", "minimize truncation", "maximize EOS", "maximize plan validity", "maximize belief/game validity"],
                  "results": dev_results, "ranking_epochs": [int(x["epoch"]) for x in ranked], "selected_epoch": selected_epoch, "final_accessed": False, "grpo_started": False}
    (args.output / "SFT_RECOVERY_CHECKPOINT_COMPARISON.json").write_text(json.dumps(comparison, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    freeze = {"stage": "T23_BEST_CHECKPOINT_FREEZE", "selected_epoch": selected_epoch, "selected_checkpoint": str(selected_checkpoint.resolve()),
              "model": model_name, "checkpoint_hashes": tree_hashes(selected_checkpoint), "selection_metrics": selected,
              "selection_rule": comparison["selection_hierarchy"], "dev_only_selection": True, "final_accessed": False, "grpo_started": False}
    (args.output / "SFT_RECOVERY_BEST_CHECKPOINT_FREEZE.json").write_text(json.dumps(freeze, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

    # K4 confirmation on the exact previous 64 TRAIN physical states.
    contexts, tables = load_context("TRAIN"), load_payoff_tables("TRAIN")
    snapshots = select_snapshots()
    model, tok = load_inference_model(model_name, selected_checkpoint); grammar = CoreESRGrammar(tok)
    free_rows, grammar_rows = [], []
    for gi, snap in enumerate(snapshots.itertuples(index=False)):
        seed = 42 * 100000 + gi * 100
        prompts = [str(snap.prompt)] * 4
        free = generate_paired(model, tok, grammar, prompts, seed=seed, max_new_tokens=completion_limit, temperature=args.temperature, top_p=args.top_p, constrained=False)
        gram = generate_paired(model, tok, grammar, prompts, seed=seed, max_new_tokens=completion_limit, temperature=args.temperature, top_p=args.top_p, constrained=True)
        for ci, (f, g) in enumerate(zip(free, gram)):
            common = {"group_index": gi, "candidate_index": ci, "example_id": str(snap.example_id), "physical_state_id": str(snap.physical_state_id), "split": "TRAIN", "sampling_batch_seed": seed, "sampling_seed": seed + ci, "temperature": args.temperature, "top_p": args.top_p, "max_new_tokens": completion_limit}
            free_rows.append({**common, **f}); grammar_rows.append({**common, **g})
    free_summary = summarize(free_rows, contexts, tables, registry, 4); grammar_summary = summarize(grammar_rows, contexts, tables, registry, 4)
    pd.DataFrame(free_rows).to_parquet(args.output / "T23_K4_FREE_CANDIDATES.parquet", index=False); pd.DataFrame(grammar_rows).to_parquet(args.output / "T23_K4_GRAMMAR_CANDIDATES.parquet", index=False)
    if free_summary["strict_valid_rate"] >= 0.95:
        classification = "PASS_NATIVE_INTERFACE"
    elif grammar_summary["strict_valid_rate"] >= 0.95:
        classification = "PASS_CONSTRAINED_INTERFACE"
    else:
        classification = "FAIL_INTERFACE_RECOVERY"
    confirmation = {"stage": "T23_K4_INTERFACE_CONFIRMATION", "classification": classification, "selected_epoch": selected_epoch, "selected_checkpoint": str(selected_checkpoint.resolve()),
                    "split": "TRAIN", "groups": 64, "k": 4, "completion_limit": completion_limit, "temperature": args.temperature, "top_p": args.top_p,
                    "free": free_summary, "grammar": grammar_summary, "same_prompts": True, "same_seeds": True, "forced_action_diversity": False,
                    "only_decoding_constraint_changed": True, "threshold": 0.95, "economic_gate_allowed": classification in {"PASS_NATIVE_INTERFACE", "PASS_CONSTRAINED_INTERFACE"},
                    "grpo_started": False, "final_accessed": False}
    (args.output / "K4_INTERFACE_CONFIRMATION.json").write_text(json.dumps(confirmation, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    report = ["# T23 Recovery Checkpoint 与 K4 接口确认", "", f"- DEV 选择：`epoch{selected_epoch}`", f"- T23 classification：`{classification}`", f"- completion_limit：`{completion_limit}`", "", "## DEV checkpoint selection", "", "| epoch | strict | truncation | EOS | plan | belief | game |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for x in dev_results: report.append(f"| {x['epoch']} | {x['strict_valid_rate']:.4f} | {x['truncation_rate']:.4f} | {x['eos_rate']:.4f} | {x['plan_valid_rate']:.4f} | {x['belief_valid_rate']:.4f} | {x['game_valid_rate']:.4f} |")
    report += ["", "## K4 interface confirmation", "", f"- FREE strict validity：`{free_summary['strict_valid_rate']:.4f}`", f"- Grammar strict validity：`{grammar_summary['strict_valid_rate']:.4f}`", f"- FREE action/economic variation：`{free_summary['action_variation_groups']}/64` / `{free_summary['economic_variation_groups']}/64`", f"- Grammar action/economic variation：`{grammar_summary['action_variation_groups']}/64` / `{grammar_summary['economic_variation_groups']}/64`", "", "如果 classification 为 FAIL_INTERFACE_RECOVERY，按 DAG 在 T23 停止，不运行 T24/T25/T26。", "", "本阶段未使用 FINAL，未启动 GRPO。"]
    (args.output / "K4_INTERFACE_CONFIRMATION_REPORT_CN.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"selected_epoch": selected_epoch, "classification": classification, "dev": [(x["epoch"], x["strict_valid_rate"]) for x in dev_results], "free_strict": free_summary["strict_valid_rate"], "grammar_strict": grammar_summary["strict_valid_rate"]}, ensure_ascii=False))
    return 0 if classification != "FAIL_INTERFACE_RECOVERY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
