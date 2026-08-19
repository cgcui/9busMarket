#!/usr/bin/env python3
"""T31: teacher-forced field audit for the frozen epoch1 recovery adapter."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paper9bus_gv_grpo.paths import split_file, CORE_ROOT
from paper9bus_gv_grpo.schema import load_registry, parse_core

FIELDS = ["a", "b", "cf", "e", "g", "i", "p", "q"]
ACTION_IDS = list(range(6))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_hashes(root: Path) -> dict:
    return {str(p.relative_to(root)).replace("\\", "/"): sha256_file(p) for p in sorted(root.rglob("*")) if p.is_file()}


def json_equal(a, b) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return bool(np.isclose(float(a), float(b), atol=1e-7, rtol=1e-7))
    return a == b


def value_spans(target: str) -> dict[str, tuple[int, int]]:
    spans = {}
    decoder = json.JSONDecoder()
    for match in re.finditer(r'"([a-z]+)"\s*:', target):
        key = match.group(1)
        if key not in FIELDS:
            continue
        start = match.end()
        while start < len(target) and target[start].isspace():
            start += 1
        _, end = decoder.raw_decode(target, start)
        spans[key] = (start, end)
    return spans


def parse_decoded_value(text: str):
    cleaned = text.strip()
    try:
        return json.loads(cleaned)
    except Exception:
        # The decoder can omit a leading/trailing space but should not invent
        # structure.  A numeric scalar is handled explicitly for robustness.
        try:
            return float(cleaned) if "." in cleaned else int(cleaned)
        except Exception:
            return None


def field_token_indices(tok, target_with_eos: str, target_ids: list[int]) -> dict[str, list[int]]:
    enc = tok(target_with_eos, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    spans = value_spans(target_with_eos)
    out = {key: [] for key in FIELDS}
    for i, (start, end) in enumerate(offsets):
        if i >= len(target_ids) or end <= start:
            continue
        for key, (lo, hi) in spans.items():
            if start < hi and end > lo:
                out[key].append(i)
                break
    return out


def teacher_forced(model, tok, dev: pd.DataFrame, batch_size: int) -> tuple[pd.DataFrame, dict]:
    rows = dev.to_dict("records")
    prepared = []
    for row in rows:
        prompt = str(row["prompt"]) + "\n"
        target = str(row["target_json"])
        target_with_eos = target + tok.eos_token
        prompt_ids = tok(prompt, add_special_tokens=True, return_attention_mask=False)["input_ids"]
        target_ids = tok(target_with_eos, add_special_tokens=False, return_attention_mask=False)["input_ids"]
        full_ids = tok(prompt + target_with_eos, add_special_tokens=True, return_attention_mask=False)["input_ids"]
        if full_ids[len(prompt_ids):len(prompt_ids) + len(target_ids)] != target_ids:
            raise RuntimeError(f"prompt/target token boundary mismatch for {row['example_id']}")
        prepared.append({**row, "prompt": prompt, "target": target, "prompt_ids": prompt_ids, "target_ids": target_ids,
                         "full_ids": full_ids, "field_token_indices": field_token_indices(tok, target_with_eos, target_ids)})

    field_acc = defaultdict(list)
    field_prob = defaultdict(list)
    result_rows = []
    for start in range(0, len(prepared), batch_size):
        chunk = prepared[start:start + batch_size]
        max_len = max(len(x["full_ids"]) for x in chunk)
        pad = int(tok.pad_token_id)
        input_ids = torch.tensor([x["full_ids"] + [pad] * (max_len - len(x["full_ids"])) for x in chunk], device="cuda")
        attention = torch.tensor([[1] * len(x["full_ids"]) + [0] * (max_len - len(x["full_ids"])) for x in chunk], device="cuda")
        with torch.inference_mode():
            logits = model(input_ids=input_ids, attention_mask=attention).logits
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        top_ids = logits.argmax(dim=-1).detach().cpu().numpy()
        for bi, item in enumerate(chunk):
            prompt_len = len(item["prompt_ids"])
            t_ids = item["target_ids"]
            token_records = []
            for j, gold_id in enumerate(t_ids):
                pos = prompt_len + j
                if pos <= 0 or pos >= logits.shape[1]:
                    continue
                lp = float(log_probs[bi, pos - 1, int(gold_id)].detach().cpu())
                pred = int(top_ids[bi, pos - 1])
                token_records.append({"j": j, "gold": int(gold_id), "pred": pred, "logprob": lp,
                                      "prob": float(math.exp(max(-80.0, lp))), "correct": pred == int(gold_id)})

            target_obj = json.loads(item["target"])
            field_metrics = {}
            greedy_values = {}
            field_indices = item["field_token_indices"]
            for field in FIELDS:
                indices = field_indices.get(field, [])
                recs = [token_records[j] for j in indices if j < len(token_records)]
                gold_value = target_obj.get(field)
                decoded = tok.decode([x["pred"] for x in recs], skip_special_tokens=True).strip() if recs else ""
                pred_value = parse_decoded_value(decoded)
                greedy_values[field] = pred_value
                exact = pred_value is not None and json_equal(pred_value, gold_value)
                acc = float(np.mean([x["correct"] for x in recs])) if recs else 0.0
                mean_lp = float(np.mean([x["logprob"] for x in recs])) if recs else float("nan")
                min_lp = float(np.min([x["logprob"] for x in recs])) if recs else float("nan")
                field_metrics[field] = {"token_count": len(recs), "token_accuracy": acc, "mean_gold_logprob": mean_lp,
                                        "min_gold_logprob": min_lp, "greedy_decoded": decoded, "greedy_value": pred_value,
                                        "greedy_exact": bool(exact)}
                field_acc[field].append(acc); field_prob[field].append(mean_lp); 

            b_gold = target_obj.get("b", [])
            b_pred = greedy_values.get("b") if isinstance(greedy_values.get("b"), list) else None
            first_b_error = None
            if b_pred is None:
                first_b_error = "parse"
            else:
                for idx in range(max(len(b_gold), len(b_pred))):
                    if idx >= len(b_pred) or idx >= len(b_gold) or not json_equal(b_pred[idx], b_gold[idx]):
                        first_b_error = idx
                        break
            # The first action token is also used for a legal six-way action
            # probe under the gold prefix.
            action_distribution = {}
            action_position = field_indices.get("a", [None])[0]
            if action_position is not None:
                pos = prompt_len + action_position
                if pos > 0:
                    for action_id in ACTION_IDS:
                        ids = tok(str(action_id), add_special_tokens=False, return_attention_mask=False)["input_ids"]
                        if len(ids) == 1:
                            action_distribution[str(action_id)] = float(torch.softmax(logits[bi, pos - 1].float(), dim=-1)[int(ids[0])].detach().cpu())
            result_rows.append({"example_id": str(item["example_id"]), "physical_state_id": str(item.get("physical_state_id", "")),
                                "target_json": item["target"], "teacher_forced_fields": json.dumps(field_metrics, ensure_ascii=False, default=float),
                                "teacher_forced_greedy_values": json.dumps(greedy_values, ensure_ascii=False, default=float),
                                "belief_first_incorrect_position": first_b_error, "action_legal_distribution": json.dumps(action_distribution, sort_keys=True),
                                "teacher_forced_action_top1": int(max(action_distribution, key=action_distribution.get)) if action_distribution else None,
                                "target_action": int(target_obj["a"]), "target_belief": json.dumps(target_obj["b"], separators=(",", ":")),
                                "target_game": json.dumps(target_obj["g"], separators=(",", ":")), "target_plan": json.dumps(target_obj["p"], separators=(",", ":"))})
        del logits, log_probs, input_ids, attention
        torch.cuda.empty_cache()

    aggregate = {}
    for field in FIELDS:
        vals = [json.loads(x["teacher_forced_fields"])[field] for x in result_rows]
        aggregate[field] = {
            "rows": len(vals), "greedy_exact_rate": float(np.mean([x["greedy_exact"] for x in vals])),
            "token_accuracy_mean": float(np.mean([x["token_accuracy"] for x in vals])),
            "mean_gold_logprob": float(np.mean([x["mean_gold_logprob"] for x in vals])),
            "min_gold_logprob_p05": float(np.nanquantile([x["min_gold_logprob"] for x in vals], 0.05)),
            "token_count_mean": float(np.mean([x["token_count"] for x in vals])),
        }
    return pd.DataFrame(result_rows), aggregate


def free_generate(model, tok, prompts: list[str], batch_size: int, max_new_tokens: int) -> list[dict]:
    tok.padding_side = "left"
    out_rows = []
    eos_ids = {int(tok.eos_token_id)} if isinstance(tok.eos_token_id, int) else {int(x) for x in tok.eos_token_id}
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start:start + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True).to("cuda")
        with torch.inference_mode():
            generated = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                       pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id)
        width = enc.input_ids.shape[1]
        for i in range(len(chunk)):
            ids = generated[i, width:].detach().cpu().tolist()
            eos_pos = next((j for j, x in enumerate(ids) if int(x) in eos_ids), None)
            actual = ids if eos_pos is None else ids[:eos_pos + 1]
            raw = tok.decode(actual[:-1] if eos_pos is not None else actual, skip_special_tokens=True).strip()
            out_rows.append({"raw_output": raw, "generated_tokens": len(actual), "eos_completed": eos_pos is not None,
                             "truncated": eos_pos is None and len(actual) >= max_new_tokens})
    return out_rows


def first_free_divergence(raw: str, gold: dict) -> str:
    try:
        obj = json.loads(raw)
    except Exception:
        # Attribute a malformed sequence to the first canonical key absent
        # from the emitted text; this is a structural cascade label.
        keys = [x for x in FIELDS if f'"{x}"' not in raw]
        return keys[0] if keys else "structure"
    if not isinstance(obj, dict):
        return "structure"
    for field in FIELDS:
        if field not in obj or not json_equal(obj[field], gold.get(field)):
            return field
    return "none"


def free_field_rates(free_rows: list[dict], dev: pd.DataFrame, registry) -> tuple[dict, list[dict]]:
    rates = {field: [] for field in FIELDS}
    details = []
    for generated, gold_row in zip(free_rows, dev.to_dict("records")):
        gold = json.loads(str(gold_row["target_json"]))
        raw = generated["raw_output"]
        try:
            obj = json.loads(raw)
        except Exception:
            obj = None
        for field in FIELDS:
            rates[field].append(bool(isinstance(obj, dict) and field in obj and json_equal(obj[field], gold.get(field))))
        parse_success = False
        core_valid = False
        if isinstance(obj, dict):
            try:
                parse_core(obj, registry); parse_success = True; core_valid = True
            except Exception:
                parse_success = True
        details.append({"example_id": str(gold_row["example_id"]), "free_raw_output": raw, "free_eos_completed": generated["eos_completed"],
                        "free_truncated": generated["truncated"], "free_json_parse": parse_success, "free_core_valid": core_valid,
                        "free_first_divergence_field": first_free_divergence(raw, gold)})
    summary = {field: {"valid_rate": float(np.mean(vals)) for field, vals in rates.items()}
               for field in FIELDS}
    summary["strict_core_valid_rate"] = float(np.mean([x["free_core_valid"] for x in details]))
    summary["json_parse_rate"] = float(np.mean([x["free_json_parse"] for x in details]))
    summary["eos_rate"] = float(np.mean([x["free_eos_completed"] for x in details]))
    summary["truncation_rate"] = float(np.mean([x["free_truncated"] for x in details]))
    summary["first_divergence_counts"] = dict(Counter(x["free_first_divergence_field"] for x in details))
    return summary, details


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    dev = pd.read_parquet(split_file("DEV"))
    if len(dev) != 533:
        raise RuntimeError(f"expected frozen DEV size 533, got {len(dev)}")
    tok = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    tok.pad_token = tok.pad_token or tok.eos_token; tok.padding_side = "right"
    base = AutoModelForCausalLM.from_pretrained(args.model, device_map={"": 0}, trust_remote_code=True, torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(base, args.checkpoint).eval()
    tf_rows, tf_summary = teacher_forced(model, tok, dev, args.batch_size)
    free_rows = free_generate(model, tok, [str(x) + "\n" for x in dev.prompt.tolist()], args.batch_size, args.max_new_tokens)
    free_summary, free_details = free_field_rates(free_rows, dev, load_registry(CORE_ROOT / "enum_registry.json"))
    details = pd.DataFrame(free_details)
    result = tf_rows.merge(details, on="example_id", how="left")
    belief_tf = tf_summary["b"]["greedy_exact_rate"]
    belief_free = free_summary["b"]["valid_rate"]
    if belief_tf >= 0.95 and belief_free < 0.95:
        classification = "AUTOREGRESSIVE_CASCADE_DOMINANT"
    elif belief_tf < 0.80:
        classification = "SEMANTIC_SUPERVISION_FAILURE"
    else:
        classification = "MIXED_FAILURE"
    summary = {
        "stage": "T31_TEACHER_FORCED_FIELD_AUDIT",
        "classification": classification,
        "checkpoint": str(args.checkpoint.resolve()), "model": args.model, "split": "DEV", "rows": len(dev),
        "teacher_forced": tf_summary, "free_autoregressive": free_summary,
        "classification_rule": {"autoregressive_cascade_dominant": "teacher-forced belief greedy exact >= 0.95 and free belief validity < 0.95",
                                 "semantic_supervision_failure": "teacher-forced belief greedy exact < 0.80", "otherwise": "MIXED_FAILURE"},
        "selected_checkpoint_hashes": tree_hashes(args.checkpoint),
        "inference": {"torch_dtype": "float16", "max_new_tokens": args.max_new_tokens, "do_sample": False, "generation_pad_override": "eos_token_id"},
        "frozen_prior_status": "FAIL_INTERFACE_RECOVERY", "t30_classification": "VISIBLE_REPRESENTATION_COLLISION",
        "final_accessed": False, "grpo_started": False,
    }
    result.to_parquet(args.output / "T31_FIELD_ERROR_CASCADE.parquet", index=False)
    (args.output / "T31_TEACHER_FORCED_FIELD_AUDIT.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    report = ["# T31 Teacher-forced 字段审计", "", f"- checkpoint：`epoch1`", f"- DEV：`{len(dev)}` 条", f"- 结论：`{classification}`", "", "## Teacher-forced", "", "| field | greedy exact | token accuracy | mean gold logprob |", "|---|---:|---:|---:|"]
    for field in FIELDS:
        x = tf_summary[field]; report.append(f"| `{field}` | {x['greedy_exact_rate']:.4f} | {x['token_accuracy_mean']:.4f} | {x['mean_gold_logprob']:.4f} |")
    report += ["", "## Free autoregressive 对照", "", f"- strict Core valid：`{free_summary['strict_core_valid_rate']:.4f}`", f"- JSON parse：`{free_summary['json_parse_rate']:.4f}`", f"- EOS：`{free_summary['eos_rate']:.4f}`", f"- belief validity：`{belief_free:.4f}`", f"- first divergence：`{free_summary['first_divergence_counts']}`", "", "本阶段仅做推理审计；未训练、未修改 adapter/data、未运行经济 Gate/GRPO、未访问 FINAL。", "逐行结果见 `T31_FIELD_ERROR_CASCADE.parquet`。"]
    (args.output / "T31_TEACHER_FORCED_REPORT_CN.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"classification": classification, "belief_teacher_forced_exact": belief_tf, "belief_free_valid": belief_free, "free_strict": free_summary["strict_core_valid_rate"]}, ensure_ascii=False))
    del model, base; gc.collect(); torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
