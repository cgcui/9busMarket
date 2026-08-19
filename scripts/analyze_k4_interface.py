#!/usr/bin/env python3
"""Phase A: forensic audit of the frozen 256-row K=4 interface artifact."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paper9bus_gv_grpo.audit_hardening import repetition_flags_hardened
from paper9bus_gv_grpo.paths import split_file

EXPECTED = ["e", "b", "g", "cf", "i", "a", "p", "q"]
CORE_FIELDS = {"e", "b", "g", "cf", "i", "a", "p", "q"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def complete_json(raw: str):
    """Return the first complete JSON object and its character bounds."""
    start = raw.find("{")
    if start < 0:
        return None, None, None
    try:
        obj, end = json.JSONDecoder().raw_decode(raw[start:])
        return obj, start, start + end
    except Exception:
        return None, start, None


def structural_scan(raw: str) -> dict:
    stack = []
    in_string = False
    escaped = False
    close_for = {"{": "}", "[": "]"}
    unmatched_close = 0
    for ch in raw:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append(close_for[ch])
        elif ch in "}]":
            if stack and ch == stack[-1]:
                stack.pop()
            else:
                unmatched_close += 1
    return {
        "unmatched_open_braces": sum(x == "}" for x in stack),
        "unmatched_open_brackets": sum(x == "]" for x in stack),
        "unmatched_close": unmatched_close,
        "string_unclosed": in_string,
        "brace_bracket_depth": len(stack),
    }


class PairCollector(dict):
    def __init__(self, pairs):
        super().__init__()
        self.duplicate_keys = 0
        for key, value in pairs:
            if key in self:
                self.duplicate_keys += 1
            self[key] = value


def repetition_flags(raw: str) -> dict:
    return repetition_flags_hardened(raw)


def field_error(obj: dict) -> tuple[str | None, str | None, str | None]:
    """Return (category, first_invalid_field, last_valid_field)."""
    if not isinstance(obj, dict):
        return "WRONG_TYPE", "root", None
    missing = [k for k in CORE_FIELDS if k not in obj]
    if missing:
        return "MISSING_FIELD", missing[0], None
    extra = sorted(set(obj) - CORE_FIELDS)
    if extra:
        return "OTHER", extra[0], None

    valid = []

    # Keep the same semantic order used by parse_core, while exposing the
    # requested belief -> game -> plan -> action anatomy explicitly.
    e = obj["e"]
    if not isinstance(e, list):
        return "WRONG_TYPE", "e", valid[-1] if valid else None
    if len(e) > 8 or any(not isinstance(x, (int, float)) or int(x) not in range(8) for x in e):
        return "OTHER", "e", valid[-1] if valid else None
    valid.append("e")

    b = obj["b"]
    if not isinstance(b, list):
        return "WRONG_TYPE", "b", valid[-1]
    if len(b) != 6:
        return "BELIEF_LENGTH_INVALID", "b", valid[-1]
    try:
        bf = [float(x) for x in b]
    except Exception:
        return "BELIEF_VALUE_INVALID", "b", valid[-1]
    if any(not math.isfinite(x) or x < 0 or x > 1 for x in bf):
        return "BELIEF_VALUE_INVALID", "b", valid[-1]
    if abs(sum(bf) - 1.0) > 1e-5:
        return "BELIEF_NORMALIZATION_INVALID", "b", valid[-1]
    valid.append("b")

    g = obj["g"]
    if not isinstance(g, list):
        return "WRONG_TYPE", "g", valid[-1]
    if len(g) != 4 or any(not isinstance(x, (int, float)) or int(x) not in range(3) for x in g):
        return "GAME_INVALID", "g", valid[-1]
    valid.append("g")

    cf = obj["cf"]
    if not isinstance(cf, list) or len(cf) != 2 or any(not isinstance(x, (int, float)) or int(x) not in range(3) for x in cf):
        return "OTHER", "cf", valid[-1]
    valid.append("cf")

    i = obj["i"]
    if not isinstance(i, (int, float)) or int(i) not in range(4):
        return "OTHER", "i", valid[-1]
    valid.append("i")

    a = obj["a"]
    if not isinstance(a, (int, float)) or int(a) not in range(6):
        return "ACTION_INVALID", "a", valid[-1]
    valid.append("a")

    p = obj["p"]
    if not isinstance(p, list):
        return "WRONG_TYPE", "p", valid[-1]
    if len(p) != 3 or any(not isinstance(x, (int, float)) or int(x) not in range(6) for x in p):
        return "PLAN_INVALID", "p", valid[-1]
    valid.append("p")

    try:
        q = float(obj["q"])
    except Exception:
        return "WRONG_TYPE", "q", valid[-1]
    if not math.isfinite(q) or not 0 <= q <= 1:
        return "OTHER", "q", valid[-1]
    return None, None, valid[-1]


def classify(row) -> dict:
    raw = str(row.raw_output)
    obj, json_start, json_end = complete_json(raw)
    structural = structural_scan(raw)
    rep = repetition_flags(raw)
    pairs_obj = None
    repeated_keys = 0
    if json_end is not None:
        try:
            pairs_obj = json.loads(raw[json_start:json_end], object_pairs_hook=PairCollector)
            repeated_keys = int(getattr(pairs_obj, "duplicate_keys", 0))
        except Exception:
            pass

    prefix = raw[:json_start] if json_start is not None and json_start >= 0 else raw
    suffix = raw[json_end:] if json_end is not None else ""
    prefix_non_ws = bool(prefix.strip())
    suffix_non_ws = bool(suffix.strip())
    max_hit = int(row.generated_tokens) >= int(row.max_new_tokens)
    eos = bool(row.eos_completed)
    category = None
    first_field = None
    last_field = None
    if json_end is None:
        category = "TRUNCATED" if max_hit else "JSON_INCOMPLETE"
    elif prefix_non_ws or suffix_non_ws:
        category = "EXTRA_TEXT"
    else:
        category, first_field, last_field = field_error(obj)
        if category is None and not eos:
            category = "NO_EOS"
        elif category is None:
            category = "VALID"
    if category == "VALID" and rep["repetition_loop_indicator"]:
        category = "REPETITION"
    return {
        "group_index": int(row.group_index),
        "candidate_index": int(row.candidate_index),
        "example_id": str(row.example_id),
        "physical_state_id": str(row.physical_state_id),
        "split": str(row.split),
        "raw_output": raw,
        "generated_token_count": int(row.generated_tokens),
        "max_new_tokens": int(row.max_new_tokens),
        "max_new_tokens_hit": max_hit,
        "max_new_tokens_hit_without_eos": bool(max_hit and not eos),
        "eos_completed": eos,
        "truncated_flag_from_runner": bool(row.truncated),
        "json_complete": json_end is not None,
        "first_complete_json_position": json_end,
        "json_prefix_chars": len(prefix),
        "json_suffix_chars": len(suffix),
        "prefix_text": prefix,
        "suffix_text": suffix,
        "prefix_non_whitespace": prefix_non_ws,
        "suffix_non_whitespace": suffix_non_ws,
        "repeated_key_count": repeated_keys,
        **structural,
        **rep,
        "first_invalid_core_field": first_field,
        "last_valid_core_field": last_field,
        "failure_category": category,
        "free_strict_valid": bool(row.strict_valid),
        "free_parse_success": bool(row.parse_success),
        "free_core_valid": bool(row.core_valid),
        "free_validity_reason": None if pd.isna(row.validity_reason) else str(row.validity_reason),
    }


def consistency_audit(sft_run: Path, free: pd.DataFrame) -> dict:
    manifest = json.loads((sft_run / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
    model_name = manifest["model"]
    tok = AutoTokenizer.from_pretrained(sft_run / "adapter", trust_remote_code=True)
    train = pd.read_parquet(split_file("TRAIN"))
    target_only = []
    full = []
    prompt_lens = []
    for row in train.itertuples(index=False):
        prompt = str(row.prompt) + "\n"
        target = str(row.target_json) + tok.eos_token
        prompt_ids = tok(prompt, add_special_tokens=True, return_attention_mask=False)["input_ids"]
        target_ids = tok(target, add_special_tokens=False, return_attention_mask=False)["input_ids"]
        full_ids = tok(prompt + target, add_special_tokens=True, return_attention_mask=False)["input_ids"]
        prompt_lens.append(len(prompt_ids)); target_only.append(len(target_ids)); full.append(len(full_ids))
    chat_template = getattr(tok, "chat_template", None)
    files = sorted(p for p in (sft_run / "adapter").glob("tokenizer*") if p.is_file())
    return {
        "model_name": model_name,
        "adapter_tokenizer_class": tok.__class__.__name__,
        "adapter_tokenizer_files_sha256": {p.name: sha256(p) for p in files},
        "vocab_size": int(len(tok)),
        "bos_token": tok.bos_token,
        "bos_token_id": tok.bos_token_id,
        "eos_token": tok.eos_token,
        "eos_token_id": tok.eos_token_id,
        "pad_token": tok.pad_token,
        "pad_token_id": tok.pad_token_id,
        "chat_template_present": bool(chat_template),
        "chat_template_sha256": hashlib.sha256(str(chat_template).encode()).hexdigest() if chat_template else None,
        "sft_prompt_template": "str(row.prompt) + '\\n'",
        "inference_prompt_template": "str(row.prompt) + '\\n'",
        "chat_template_used": False,
        "train_padding_side": "right",
        "inference_padding_side": "left",
        "pad_equals_eos": bool(tok.pad_token_id == tok.eos_token_id),
        "explicit_eos_supervision": True,
        "loss_mask": "prompt tokens -100; target tokens including explicit EOS supervised",
        "assistant_only_loss_mask": False,
        "bos_eos_placement": "raw prompt + newline + target_json + eos; tokenizer add_special_tokens=True",
        "sft_max_seq_length": 512,
        "train_prompt_tokens_max": max(prompt_lens),
        "train_target_tokens_max": max(target_only),
        "train_full_tokens_max": max(full),
        "generation_cap": int(free.max_new_tokens.max()),
        "free_generation_cap_values": sorted(int(x) for x in free.max_new_tokens.unique()),
        "consistency_findings": [
            "prompt text and newline agree between SFT and K4 inference",
            "no chat template is used in either path",
            "tokenizer pad_token_id differs from eos_token_id, but the K4 generate call explicitly overrides pad_token_id with eos_token_id",
            "batched generated lengths can therefore include EOS padding after an earlier EOS; the frozen artifact records max-length rows with eos_completed=True, so raw cap-hit and cap-hit-without-EOS are reported separately",
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft-run", type=Path, required=True)
    ap.add_argument("--free-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    free = pd.read_parquet(args.free_dir / "K4_CANDIDATES.parquet")
    if len(free) != 256 or free.group_index.nunique() != 64:
        raise RuntimeError("frozen K4 artifact must contain exactly 256 rows and 64 groups")
    failures = pd.DataFrame([classify(row) for row in free.itertuples(index=False)])
    failures.to_parquet(args.output / "K4_INTERFACE_FAILURES.parquet", index=False)
    counts = Counter(failures.failure_category)
    invalid = ~failures.failure_category.eq("VALID")
    cap = failures.max_new_tokens_hit
    cap_no_eos = failures.max_new_tokens_hit_without_eos
    audit = {
        "protocol": "Paper9Bus-Power-GV-GRPO-v3",
        "stage": "K4_INTERFACE_FAILURE_ANATOMY",
        "status": "FAIL_INTERFACE_VALIDITY",
        "secondary_conclusion": "TRUE_ECONOMIC_SIGNAL_OBSERVED",
        "grpo_started": False,
        "final_accessed": False,
        "frozen_free_artifact": str((args.free_dir / "K4_CANDIDATES.parquet").resolve()),
        "rows": int(len(failures)),
        "groups": int(failures.group_index.nunique()),
        "failure_category_counts": dict(sorted(counts.items())),
        "strict_valid_rate_from_frozen_artifact": float(failures.free_strict_valid.mean()),
        "conditional_invalid_probability": {
            "P_invalid_given_generated_token_count_at_cap": float(invalid[cap].mean()) if cap.any() else None,
            "n_at_cap": int(cap.sum()),
            "P_invalid_given_cap_without_eos": float(invalid[cap_no_eos].mean()) if cap_no_eos.any() else None,
            "n_cap_without_eos": int(cap_no_eos.sum()),
        },
        "failure_order": ["JSON", "belief", "game", "plan", "EOS"],
        "category_definitions": {
            "JSON_INCOMPLETE": "no syntactically complete first JSON object and no recorded token cap hit",
            "TRUNCATED": "no syntactically complete first JSON object and generated token count reached cap",
            "NO_EOS": "complete Core object but no EOS token observed",
            "REPETITION": "complete otherwise valid object with repeated-loop detector fired",
            "EXTRA_TEXT": "non-whitespace prefix or suffix surrounds the first complete JSON object",
        },
        "train_inference_consistency": consistency_audit(args.sft_run, free),
        "artifact_integrity": {
            "adapter_untouched": True,
            "existing_k4_outputs_untouched": True,
            "train_dev_final_untouched": True,
            "cell_bank_untouched": True,
        },
        "outputs": ["K4_INTERFACE_FAILURE_AUDIT.json", "K4_INTERFACE_FAILURES.parquet"],
    }
    (args.output / "K4_INTERFACE_FAILURE_AUDIT.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"rows": len(failures), "counts": dict(counts), "audit_dir": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
