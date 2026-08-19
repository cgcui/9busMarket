#!/usr/bin/env python3
"""Correct JSON field-boundary decoration in already-computed T31 evidence."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

FIELDS = ["a", "b", "cf", "e", "g", "i", "p", "q"]


def normalize(text: str) -> str:
    s = str(text).strip()
    if s.startswith('":'):
        s = s[2:]
    if s.endswith(',"'):
        s = s[:-2]
    return s.strip()


def parse_value(text: str):
    try:
        return json.loads(normalize(text))
    except Exception:
        return None


def eq(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return bool(np.isclose(float(a), float(b), atol=1e-7, rtol=1e-7))
    return a == b


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--output", type=Path, required=True); args = ap.parse_args()
    parquet = args.output / "T31_FIELD_ERROR_CASCADE.parquet"
    summary_path = args.output / "T31_TEACHER_FORCED_FIELD_AUDIT.json"
    d = pd.read_parquet(parquet)
    for idx, row in d.iterrows():
        target = json.loads(row.target_json)
        metrics = json.loads(row.teacher_forced_fields)
        greedy = json.loads(row.teacher_forced_greedy_values)
        for field in FIELDS:
            decoded = metrics[field].get("greedy_decoded", "")
            value = parse_value(decoded)
            greedy[field] = value
            metrics[field]["normalized_greedy_decoded"] = normalize(decoded)
            metrics[field]["greedy_value"] = value
            metrics[field]["greedy_exact"] = bool(value is not None and eq(value, target.get(field)))
        b = greedy.get("b") if isinstance(greedy.get("b"), list) else None
        first = "parse" if b is None else None
        if b is not None:
            gold = target.get("b", [])
            for pos in range(max(len(b), len(gold))):
                if pos >= len(b) or pos >= len(gold) or not eq(b[pos], gold[pos]):
                    first = pos; break
        d.at[idx, "teacher_forced_fields"] = json.dumps(metrics, ensure_ascii=False, default=float)
        d.at[idx, "teacher_forced_greedy_values"] = json.dumps(greedy, ensure_ascii=False, default=float)
        d.at[idx, "belief_first_incorrect_position"] = first
    tf = {}
    for field in FIELDS:
        vals = [json.loads(x)[field] for x in d.teacher_forced_fields]
        tf[field] = {"rows": len(vals), "greedy_exact_rate": float(np.mean([x["greedy_exact"] for x in vals])),
                      "token_accuracy_mean": float(np.mean([x["token_accuracy"] for x in vals])),
                      "mean_gold_logprob": float(np.mean([x["mean_gold_logprob"] for x in vals])),
                      "min_gold_logprob_p05": float(np.nanquantile([x["min_gold_logprob"] for x in vals], 0.05)),
                      "token_count_mean": float(np.mean([x["token_count"] for x in vals]))}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    belief_tf = tf["b"]["greedy_exact_rate"]
    belief_free = summary["free_autoregressive"]["b"]["valid_rate"]
    classification = "AUTOREGRESSIVE_CASCADE_DOMINANT" if belief_tf >= .95 and belief_free < .95 else ("SEMANTIC_SUPERVISION_FAILURE" if belief_tf < .80 else "MIXED_FAILURE")
    summary["teacher_forced"] = tf
    summary["classification"] = classification
    summary["audit_normalization"] = "normalized Qwen offset-boundary decoration on array field decoded strings; token metrics unchanged"
    d.to_parquet(parquet, index=False)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    report = ["# T31 Teacher-forced 字段审计", "", "- checkpoint：`epoch1`", "- DEV：`533` 条", f"- 结论：`{classification}`", "", "## Teacher-forced", "", "| field | greedy exact | token accuracy | mean gold logprob |", "|---|---:|---:|---:|"]
    for field in FIELDS:
        x = tf[field]; report.append(f"| `{field}` | {x['greedy_exact_rate']:.4f} | {x['token_accuracy_mean']:.4f} | {x['mean_gold_logprob']:.4f} |")
    free = summary["free_autoregressive"]
    report += ["", "## Free autoregressive 对照", "", f"- strict Core valid：`{free['strict_core_valid_rate']:.4f}`", f"- JSON parse：`{free['json_parse_rate']:.4f}`", f"- EOS：`{free['eos_rate']:.4f}`", f"- belief validity：`{belief_free:.4f}`", f"- first divergence：`{free['first_divergence_counts']}`", "", "本阶段仅做推理审计；未训练、未修改 adapter/data、未运行经济 Gate/GRPO、未访问 FINAL。", "逐行结果见 `T31_FIELD_ERROR_CASCADE.parquet`。"]
    (args.output / "T31_TEACHER_FORCED_REPORT_CN.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"classification": classification, "belief_teacher_forced_exact": belief_tf, "belief_free_valid": belief_free}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
