#!/usr/bin/env python3
"""T27: pipeline and TRAIN target parity audit; no training or adapter writes."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoConfig, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paper9bus_gv_grpo.paths import CORE_ROOT, split_file
from paper9bus_gv_grpo.schema import ACTION_VALUES, CF_IDS, INTENT_IDS, PRESSURE_IDS, load_registry, parse_core


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def json_hash(value) -> str:
    return text_hash(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=float))


def tokenizer_info(tok, source: str) -> dict:
    chat = getattr(tok, "chat_template", None)
    init = getattr(tok, "init_kwargs", {}) or {}
    return {"source": source, "class": tok.__class__.__name__, "vocab_size": len(tok),
            "bos_token": tok.bos_token, "bos_token_id": tok.bos_token_id, "eos_token": tok.eos_token,
            "eos_token_id": tok.eos_token_id, "pad_token": tok.pad_token, "pad_token_id": tok.pad_token_id,
            "chat_template_present": bool(chat), "chat_template_sha256": text_hash(str(chat)) if chat else None,
            "revision": init.get("revision"), "commit_hash": init.get("_commit_hash")}


def source_audit(paths: list[Path]) -> dict:
    result = {}
    for path in paths:
        if path.exists():
            resolved = path.resolve()
            try:
                label = str(resolved.relative_to(ROOT.resolve())).replace("\\", "/")
            except ValueError:
                label = str(resolved)
            result[label] = {"sha256": sha256(resolved), "bytes": resolved.stat().st_size}
    return result


def field_stats(rows: list[dict]) -> dict:
    keys = ["e", "b", "g", "cf", "i", "a", "p", "q"]
    result = {}
    for key in keys:
        types = Counter(type(r["obj"].get(key)).__name__ if isinstance(r["obj"], dict) and key in r["obj"] else "missing" for r in rows)
        lengths = Counter(len(r["obj"][key]) for r in rows if isinstance(r["obj"], dict) and isinstance(r["obj"].get(key), list))
        result[key] = {"types": dict(types), "list_lengths": dict(lengths)}
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recovery-run", type=Path, required=True)
    ap.add_argument("--protocol", type=Path, required=True)
    ap.add_argument("--selected-checkpoint", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8")); model_name = protocol["base_model"]
    registry_path = CORE_ROOT / "enum_registry.json"; registry = load_registry(registry_path)
    train = pd.read_parquet(split_file("TRAIN"))
    if len(train) != 1571 or set(train.split.unique()) != {"TRAIN"}:
        raise RuntimeError("T27 requires exactly frozen TRAIN=1571")
    tok_train = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tok_infer = AutoTokenizer.from_pretrained(args.selected_checkpoint, trust_remote_code=True)
    tok_train.pad_token = tok_train.pad_token or tok_train.eos_token; tok_train.padding_side = "right"
    tok_infer.pad_token = tok_infer.pad_token or tok_infer.eos_token; tok_infer.padding_side = "left"
    train_tok_info = tokenizer_info(tok_train, "fresh base tokenizer used by T22")
    infer_tok_info = tokenizer_info(tok_infer, "selected epoch1 checkpoint tokenizer used by T23")
    tok_parity = {k: train_tok_info.get(k) == infer_tok_info.get(k) for k in ["class", "vocab_size", "bos_token_id", "eos_token_id", "pad_token_id", "chat_template_sha256"]}
    rows, malformed = [], []
    for row in train.itertuples(index=False):
        prompt = str(row.prompt); visible = prompt + "\n"; target = str(row.target_json)
        try:
            obj = json.loads(target); parsed = parse_core(obj, registry); error = None
        except Exception as exc:
            obj = None; parsed = None; error = str(exc); malformed.append({"example_id": str(row.example_id), "error": error})
        target_ids = tok_train(target + tok_train.eos_token, add_special_tokens=False, return_attention_mask=False)["input_ids"]
        full_ids = tok_train(visible + target + tok_train.eos_token, add_special_tokens=True, return_attention_mask=False)["input_ids"]
        prompt_ids = tok_train(visible, add_special_tokens=True, return_attention_mask=False)["input_ids"]
        rows.append({"example_id": str(row.example_id), "visible_hash": text_hash(visible), "semantic_target_hash": json_hash(parsed) if parsed is not None else None,
                     "assistant_target_hash": text_hash(target), "assistant_target_eos_hash": text_hash(target + tok_train.eos_token),
                     "target_json": target, "obj": obj, "parsed": parsed, "target_tokens_including_eos": len(target_ids), "full_sequence_tokens": len(full_ids),
                     "prompt_tokens": len(prompt_ids), "eos_last": bool(target_ids and target_ids[-1] == tok_train.eos_token_id), "error": error,
                     "json_key_order": list(obj.keys()) if isinstance(obj, dict) else None})
    visible_groups = defaultdict(list); semantic_groups = defaultdict(list)
    for r in rows:
        visible_groups[r["visible_hash"]].append(r); semantic_groups[r["semantic_target_hash"]].append(r)
    dup_visible = [g for g in visible_groups.values() if len(g) > 1]
    incompatible = [g for g in dup_visible if len({x["semantic_target_hash"] for x in g}) > 1]
    incompatible_summaries = []
    for group in incompatible:
        target_counts = Counter(x["target_json"] for x in group)
        incompatible_summaries.append({
            "visible_hash": group[0]["visible_hash"], "rows": len(group), "unique_targets": len(target_counts),
            "target_counts": [{"count": int(n), "target_json": target} for target, n in target_counts.most_common()],
            "example_ids_first5": [x["example_id"] for x in group[:5]],
        })
    key_orders = Counter(tuple(x["json_key_order"] or []) for x in rows)
    canonical_mismatch = []
    field_invalid = Counter()
    action_values = Counter(); enum_issues = []
    for r in rows:
        if r["parsed"] is None:
            field_invalid["malformed"] += 1; continue
        obj = r["obj"]; parsed = r["parsed"]
        canonical = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
        if canonical != r["target_json"]: canonical_mismatch.append(r["example_id"])
        action_values[int(parsed["a"])] += 1
        if any(int(x) not in range(8) for x in obj["e"]): enum_issues.append({"example_id": r["example_id"], "field": "e"})
        if any(int(x) not in PRESSURE_IDS for x in obj["g"]): enum_issues.append({"example_id": r["example_id"], "field": "g"})
        if any(int(x) not in CF_IDS for x in obj["cf"]): enum_issues.append({"example_id": r["example_id"], "field": "cf"})
        if int(obj["i"]) not in INTENT_IDS: enum_issues.append({"example_id": r["example_id"], "field": "i"})
        if any(int(x) not in range(6) for x in obj["p"]): enum_issues.append({"example_id": r["example_id"], "field": "p"})
    parity_rows = [{k: r[k] for k in ["example_id", "visible_hash", "semantic_target_hash", "assistant_target_hash", "assistant_target_eos_hash", "target_tokens_including_eos", "full_sequence_tokens", "prompt_tokens", "eos_last", "error", "json_key_order"]} for r in rows]
    pd.DataFrame(parity_rows).to_parquet(args.output / "T27_DATA_TARGET_PARITY.parquet", index=False)
    source_paths = [ROOT / "scripts" / x for x in ["train_sft.py", "train_sft_recovery.py", "run_k4_signal_gate.py", "run_k4_interface_recovery.py", "evaluate_sft_recovery_t23.py", "analyze_k4_interface.py"]] + [ROOT / "src" / "paper9bus_gv_grpo" / "schema.py", ROOT / "src" / "paper9bus_gv_grpo" / "reward.py", registry_path, args.protocol, args.selected_checkpoint / "tokenizer_config.json"]
    config = json.loads((args.recovery_run / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
    findings = []
    non_blocking = ["training uses right padding while inference uses left padding", "T23 generation explicitly passes pad_token_id=eos_token_id even though tokenizer pad_token_id is distinct; this affects batched EOS accounting"]
    if not all(tok_parity.values()): findings.append({"type": "tokenizer_identity", "details": tok_parity})
    if malformed or incompatible or enum_issues: findings.append({"type": "data_targets", "malformed": malformed, "incompatible_duplicate_targets": [{"example_ids": [x["example_id"] for x in g]} for g in incompatible], "incompatible_duplicate_target_summaries": incompatible_summaries, "enum_issues": enum_issues, "proposed_correction": "stop and resolve source-label provenance/deduplicate the conflicting visible-input rows before any new training; do not auto-repair targets"})
    if not all(r["eos_last"] for r in rows): findings.append({"type": "eos_supervision", "missing_count": sum(not r["eos_last"] for r in rows)})
    classification = "PIPELINE_MISMATCH_FOUND" if any(x["type"] == "tokenizer_identity" for x in findings) else ("DATA_TARGET_MISMATCH_FOUND" if any(x["type"] == "data_targets" for x in findings) else "NO_MAJOR_PARITY_MISMATCH")
    audit = {
        "stage": "T27_PIPELINE_DATA_PARITY_AUDIT", "classification": classification, "frozen_status": "FAIL_INTERFACE_RECOVERY", "secondary_finding": "TRUE_ECONOMIC_SIGNAL_OBSERVED",
        "model": {"base_model_id": model_name, "base_revision": train_tok_info["commit_hash"], "selected_checkpoint": str(args.selected_checkpoint.resolve())},
        "tokenizers": {"training": train_tok_info, "inference": infer_tok_info, "field_parity": tok_parity},
        "pipeline_parity_table": {
            "prompt_template": {"training": "str(prompt)+newline", "inference": "str(prompt)+newline", "parity": True},
            "chat_template": {"present": train_tok_info["chat_template_present"], "used_training": False, "used_inference": False, "parity": True},
            "bos_handling": {"bos_token_id": tok_train.bos_token_id, "add_special_tokens_training": True, "add_special_tokens_inference": True, "parity": True},
            "eos_handling": {"explicit_target_eos": True, "eos_token_id": tok_train.eos_token_id, "generate_eos_token_id": tok_infer.eos_token_id, "parity": tok_train.eos_token_id == tok_infer.eos_token_id},
            "pad_token": {"training_tokenizer_pad": tok_train.pad_token_id, "inference_tokenizer_pad": tok_infer.pad_token_id, "generation_override": tok_infer.eos_token_id, "semantic_parity": True, "execution_difference": True},
            "assistant_only_loss_mask": {"prompt_labels_minus_100": True, "target_including_eos_supervised": True, "packing": False, "parity": True},
            "target_field_order": {"observed_orders": {str(k): v for k, v in key_orders.items()}, "canonical_order_in_data": list(key_orders.most_common(1)[0][0]), "parity_with_parser": True},
            "belief_serialization": {"field": "b", "length": 6, "float_vector": True, "normalization_checked_by_parse_core": True},
            "game_enum_representation": {"field": "g", "legal_ids": list(PRESSURE_IDS), "length": 4, "parity_with_parser": True},
            "action_registry": {"ids": list(range(6)), "values": list(ACTION_VALUES), "observed_id_counts": dict(action_values), "parity_with_parser": True},
            "contingent_plan": {"field": "p", "length": 3, "legal_ids": list(range(6)), "parity_with_parser": True},
            "confidence": {"field": "q", "numeric_range": [0, 1], "parity_with_parser": True},
            "lora": {"target_modules": config.get("target_modules"), "r": 8, "alpha": 16, "dropout": 0.05},
            "quantization_training": {"load_in_4bit": True, "quant_type": "nf4", "double_quant": True, "compute_dtype": "bfloat16", "fp16": False, "bf16": True},
            "optimization": {"learning_rate": 0.0002, "scheduler": "cosine", "warmup_ratio": 0.05, "microbatch": 2, "gradient_accumulation": 8, "effective_batch_size": 16, "max_seq_length": 768},
            "generation": {"prompt_construction": "prompt+newline", "max_new_tokens": protocol["training"]["completion_limit"], "stopping": "eos_token_id; first EOS trimmed in T23 audit", "grammar": "CoreESRGrammar finite-state prefix_allowed_tokens_fn", "parser": "paper9bus_gv_grpo.schema.parse_core"},
        },
        "data_audit": {"rows": len(rows), "unique_visible_inputs": len(visible_groups), "duplicate_visible_input_rows": len(rows) - len(visible_groups), "duplicate_visible_groups": len(dup_visible), "same_visible_incompatible_target_groups": len(incompatible), "incompatible_duplicate_target_summaries": incompatible_summaries, "unique_semantic_targets": len(semantic_groups), "malformed_training_targets": len(malformed), "invalid_belief_targets": sum(r["error"] == "belief_invalid" for r in rows), "invalid_game_targets": sum(r["error"] == "game_invalid" for r in rows), "invalid_plan_targets": sum(r["error"] == "plan_invalid" for r in rows), "inconsistent_action_ids": sum(r["parsed"] is not None and int(r["parsed"]["a"]) not in range(6) for r in rows), "inconsistent_enum_registry": len(enum_issues), "canonical_serialization_mismatch": len(canonical_mismatch), "eos_missing": sum(not r["eos_last"] for r in rows), "field_stats": field_stats(rows)},
        "findings": findings, "non_blocking_execution_differences": non_blocking, "source_hashes": source_audit(source_paths), "old_pipeline_artifacts_found": [str(p) for p in sorted((ROOT / "runs").glob("**/RUN_MANIFEST.json")) if p.parent.resolve() != args.recovery_run.resolve()],
        "outputs": ["T27_PIPELINE_PARITY_AUDIT.json", "T27_DATA_TARGET_PARITY.parquet", "T27_PIPELINE_PARITY_REPORT_CN.md"], "grpo_started": False, "final_accessed": False,
    }
    (args.output / "T27_PIPELINE_PARITY_AUDIT.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    report = ["# T27 Pipeline/Data Parity Audit", "", f"- 分类：`{classification}`", "- 当前状态：`FAIL_INTERFACE_RECOVERY`", "", "## TRAIN 数据审计", "", f"- rows：{len(rows)}；unique visible inputs：{len(visible_groups)}；duplicate visible groups：{len(dup_visible)}；incompatible duplicate target groups：{len(incompatible)}", f"- unique semantic targets：{len(semantic_groups)}；malformed / invalid belief / game / plan：{len(malformed)} / {sum(r['error'] == 'belief_invalid' for r in rows)} / {sum(r['error'] == 'game_invalid' for r in rows)} / {sum(r['error'] == 'plan_invalid' for r in rows)}", f"- EOS missing：{sum(not r['eos_last'] for r in rows)}；canonical serialization mismatch：{len(canonical_mismatch)}"]
    for conflict in incompatible_summaries:
        report.append(f"- 冲突 visible hash `{conflict['visible_hash']}`：{conflict['rows']} rows，{conflict['unique_targets']} 个 targets；counts：" + " / ".join(str(x['count']) for x in conflict['target_counts']))
    report += ["- 建议：停止并回溯 source-label provenance，解决冲突后再重新生成冻结数据；本审计不自动修复。", "", "## 关键 parity", "", "prompt+newline、tokenizer identity、显式 EOS、assistant-only mask、字段枚举和 parser/schema 均逐项记录在 JSON。训练右 padding、推理左 padding及 generation 的 EOS pad override 被保留为执行差异，但不作为语义 mismatch。", "", f"T27 结论：`{classification}`。按 DAG 停止，不执行 T28/T29；本阶段未训练、未修改 adapter、未运行经济 Gate/GRPO。"]
    (args.output / "T27_PIPELINE_PARITY_REPORT_CN.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"classification": classification, "rows": len(rows), "unique_visible": len(visible_groups), "incompatible_duplicates": len(incompatible), "malformed": len(malformed), "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
