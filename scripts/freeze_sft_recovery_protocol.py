#!/usr/bin/env python3
"""T21: freeze recovery SFT protocol from frozen TRAIN targets only."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paper9bus_gv_grpo.paths import CORE_ROOT, split_file
from paper9bus_gv_grpo.schema import load_registry, parse_core


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pct(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q, method="linear"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft-run", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.sft_run / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(args.sft_run / "adapter", trust_remote_code=True)
    registry = load_registry(CORE_ROOT / "enum_registry.json")
    frame = pd.read_parquet(split_file("TRAIN"))
    if set(frame.split.unique()) != {"TRAIN"} or len(frame) != 1571:
        raise RuntimeError(f"T21 requires exactly frozen TRAIN=1571, got {len(frame)} / {set(frame.split.unique())}")
    target_tokens, full_tokens = [], []
    invalid_targets, eos_missing, full_over_limit = [], [], []
    rows = []
    for row in frame.itertuples(index=False):
        target_json = str(row.target_json)
        try:
            parse_core(target_json, registry)
        except Exception as exc:
            invalid_targets.append({"example_id": str(row.example_id), "error": str(exc)})
        prompt = str(row.prompt) + "\n"
        target_with_eos = target_json + tok.eos_token
        p_ids = tok(prompt, add_special_tokens=True, return_attention_mask=False)["input_ids"]
        t_ids = tok(target_with_eos, add_special_tokens=False, return_attention_mask=False)["input_ids"]
        f_ids = tok(prompt + target_with_eos, add_special_tokens=True, return_attention_mask=False)["input_ids"]
        target_tokens.append(len(t_ids)); full_tokens.append(len(f_ids))
        has_eos = bool(t_ids and t_ids[-1] == tok.eos_token_id)
        if not has_eos:
            eos_missing.append({"example_id": str(row.example_id), "last_token": t_ids[-1] if t_ids else None})
        if len(f_ids) > 768:
            full_over_limit.append({"example_id": str(row.example_id), "full_tokens": len(f_ids)})
        rows.append({"example_id": str(row.example_id), "target_tokens_including_eos": len(t_ids), "full_sequence_tokens": len(f_ids), "eos_supervised": has_eos})
    max_target = max(target_tokens)
    raw_limit = 64 * math.ceil((max_target + 64) / 64)
    completion_limit = min(768, max(256, raw_limit))
    stats = {
        "count": len(target_tokens), "min": int(min(target_tokens)), "median": float(statistics.median(target_tokens)),
        "p90": pct(target_tokens, 90), "p95": pct(target_tokens, 95), "p99": pct(target_tokens, 99),
        "max": int(max_target), "fraction_gt_128": float(np.mean(np.asarray(target_tokens) > 128)),
        "fraction_gt_256": float(np.mean(np.asarray(target_tokens) > 256)), "fraction_gt_384": float(np.mean(np.asarray(target_tokens) > 384)),
        "fraction_gt_512": float(np.mean(np.asarray(target_tokens) > 512)), "full_sequence_max": int(max(full_tokens)),
    }
    audit = {
        "protocol": "Paper9Bus-Power-GV-GRPO-v3", "stage": "T21_SFT_RECOVERY_TOKEN_AUDIT",
        "model": manifest["model"], "tokenizer_source": str((args.sft_run / "adapter").resolve()),
        "tokenizer_class": tok.__class__.__name__, "tokenizer_vocab_size": len(tok), "eos_token": tok.eos_token, "eos_token_id": tok.eos_token_id,
        "split": "TRAIN", "train_examples": len(frame), "target_token_definition": "target_json + tokenizer.eos_token, add_special_tokens=False",
        "statistics": stats, "valid_core_target_count": len(frame) - len(invalid_targets), "invalid_core_targets": invalid_targets,
        "eos_present_in_target_count": len(frame) - len(eos_missing), "eos_missing": eos_missing,
        "target_truncated_count_at_max_seq_length_768": len(full_over_limit), "target_truncated": full_over_limit,
        "assistant_only_supervision": True, "eos_in_supervised_target": True,
        "template": "prompt = str(row.prompt) + newline; target = target_json + eos; prompt labels -100; target including EOS supervised",
        "source_manifest_sha256": sha256(args.sft_run / "RUN_MANIFEST.json"),
    }
    (args.output / "SFT_TARGET_TOKEN_AUDIT.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    protocol = {
        "protocol": "Paper9Bus-Power-GV-GRPO-v3", "stage": "T21_FROZEN_SFT_RECOVERY_PROTOCOL",
        "status": "FROZEN", "prior_status": "FAIL_SEMANTIC_INTERFACE_SFT", "secondary_finding": "TRUE_ECONOMIC_SIGNAL_OBSERVED",
        "base_model": "Qwen/Qwen3-1.7B", "initialization": "fresh base model; existing 1-epoch adapter is not used",
        "data": {"split": "TRAIN", "examples": len(frame), "synthetic_repair_examples": False, "dev_used": False, "final_used": False},
        "training": {"epochs_exact": 3, "max_seq_length": 768, "completion_limit": completion_limit, "completion_limit_formula": "clamp(64*ceil((max_train_target_tokens+64)/64), 256, 768)", "max_train_target_tokens": max_target,
                      "quantization": "NF4 4-bit", "compute_dtype": "bfloat16", "fp16": False, "bf16": True, "lora": True, "assistant_only_loss": True, "explicit_eos_supervision": True},
        "generation": {"previous_failed_cap": 128, "previous_cap_status": "interface execution limitation; not used to choose recovery limit", "recovery_max_new_tokens": completion_limit},
        "checkpoint_layout": ["checkpoint_epoch1", "checkpoint_epoch2", "checkpoint_epoch3"], "grpo_started": False,
        "protected": ["current adapter", "existing K4 gate outputs", "existing interface-recovery outputs", "frozen TRAIN/DEV/FINAL", "frozen payoff cell bank"],
        "token_audit": "SFT_TARGET_TOKEN_AUDIT.json",
    }
    (args.output / "SFT_RECOVERY_PROTOCOL.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8")
    report = ["# T21 SFT Recovery Protocol 冻结报告", "", "- 当前冻结状态：`FAIL_SEMANTIC_INTERFACE_SFT`", "- 次级结论：`TRUE_ECONOMIC_SIGNAL_OBSERVED`", "- 数据：冻结 TRAIN 1571 条，未使用 DEV/FINAL", "", "## Token audit", "", f"- target tokens (含 EOS) min/median/p90/p95/p99/max：{stats['min']} / {stats['median']} / {stats['p90']:.1f} / {stats['p95']:.1f} / {stats['p99']:.1f} / {stats['max']}", f"- >128 / >256 / >384 / >512：{stats['fraction_gt_128']:.4f} / {stats['fraction_gt_256']:.4f} / {stats['fraction_gt_384']:.4f} / {stats['fraction_gt_512']:.4f}", f"- Core target valid：{len(invalid_targets) == 0}；EOS complete：{len(eos_missing) == 0}；max_seq_length=768 不截断：{len(full_over_limit) == 0}", "", "## Frozen settings", "", f"- completion_limit：`{completion_limit}`，由 TRAIN max target token 自动推导", "- max_seq_length：`768`", "- fresh base：`Qwen/Qwen3-1.7B`", "- NF4 + BF16，FP16 off，exactly 3 epochs", "", "T21 只冻结协议，不启动训练。"]
    (args.output / "SFT_RECOVERY_PROTOCOL_CN.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"status": "FROZEN", "train": len(frame), "max_target_tokens": max_target, "completion_limit": completion_limit, "invalid_targets": len(invalid_targets), "eos_missing": len(eos_missing), "over_768": len(full_over_limit)}, ensure_ascii=False))
    return 0 if not invalid_targets and not eos_missing and not full_over_limit else 2


if __name__ == "__main__":
    raise SystemExit(main())
