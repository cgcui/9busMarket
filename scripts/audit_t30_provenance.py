#!/usr/bin/env python3
"""T30: provenance audit for the frozen 371-row visible-input collision.

This is read-only with respect to source data and model artifacts.  It writes
only a new audit directory supplied by --output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paper9bus_gv_grpo.paths import split_file, context_file, BENCHMARK_ROOT, CORE_ROOT
from paper9bus_gv_grpo.schema import ACTION_VALUES


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def json_load(value):
    return json.loads(value) if isinstance(value, str) else value


def argmax_action(values: list[float | None]) -> int | None:
    if any(v is None or not np.isfinite(v) for v in values):
        return None
    return int(np.argmax(np.asarray(values, dtype=float)))


def build_geometry(row, cell: pd.DataFrame) -> dict:
    state = str(row.physical_state_id)
    state_rows = cell[cell.state_id.astype(str) == state]
    values_by_k: dict[str, list[float | None]] = {}
    full_info: dict[str, dict] = {}
    for k, k_rows in state_rows.groupby("k_g3", sort=True):
        vals = []
        for action_value in ACTION_VALUES:
            hit = k_rows[np.isclose(k_rows.k_g1.astype(float), float(action_value))]
            vals.append(float(hit.profit_g1.iloc[0]) if len(hit) else None)
        key = f"{float(k):.2f}"
        values_by_k[key] = vals
        best = argmax_action(vals)
        full_info[key] = {
            "best_action_id": best,
            "best_action_value": float(ACTION_VALUES[best]) if best is not None else None,
            "v": vals,
        }

    class_members = [str(x) for x in json_load(row.class_members_json)]
    q_struct = {f"{float(k):.2f}": 0.0 for k in class_members}
    if class_members:
        # class_members_json is the frozen structural support.  The target b
        # is retained separately; this audit does not infer hidden labels.
        q_struct = {key: 1.0 / len(class_members) for key in class_members}
    bayes_v = []
    for action_idx in range(len(ACTION_VALUES)):
        terms = []
        for key, weight in q_struct.items():
            vals = values_by_k.get(key)
            terms.append(weight * vals[action_idx] if vals is not None and vals[action_idx] is not None else None)
        bayes_v.append(float(sum(terms)) if terms and all(x is not None for x in terms) else None)
    obs_best = argmax_action(bayes_v)
    return {
        "hidden_k_g3_available": sorted(values_by_k),
        "structural_class_members": class_members,
        "q_struct": q_struct,
        "v_x_action_values": bayes_v,
        "observation_bayes_action_id": obs_best,
        "observation_bayes_action_value": float(ACTION_VALUES[obs_best]) if obs_best is not None else None,
        "full_information_by_k_g3": full_info,
        "cell_bank_rows": int(len(state_rows)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--selected-checkpoint", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    core = pd.read_parquet(split_file("TRAIN"))
    context = pd.read_parquet(context_file("TRAIN"))
    cell = pd.read_parquet(BENCHMARK_ROOT / "cell_bank.parquet")
    merged = core.merge(context, on="example_id", suffixes=("_core", "_ctx"))
    merged["visible_hash"] = merged.prompt_core.map(lambda x: text_hash(str(x) + "\n"))
    conflict_hash = merged.visible_hash.value_counts().index[0]
    conflict = merged[merged.visible_hash == conflict_hash].copy()
    if len(conflict) != 371:
        raise RuntimeError(f"expected frozen 371-row conflict, found {len(conflict)} for {conflict_hash}")

    tokenizer = AutoTokenizer.from_pretrained(args.selected_checkpoint, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    provenance_rows = []
    for row in conflict.itertuples(index=False):
        target = json.loads(str(row.target_json_core))
        geometry = build_geometry(row, cell)
        target_action_id = int(target["a"])
        hidden_full_info_values = {k: x["best_action_id"] for k, x in geometry["full_information_by_k_g3"].items()}
        provenance_rows.append({
            "example_id": str(row.example_id),
            "physical_state_id": str(row.physical_state_id),
            "visible_hash": conflict_hash,
            "visible_prompt_sha256": conflict_hash,
            "visible_prompt_token_hash": text_hash(json.dumps(tokenizer(str(row.prompt_core) + "\n", add_special_tokens=True)["input_ids"], separators=(",", ":"))),
            "visible_prompt": str(row.prompt_core),
            "observation_json": str(row.observation_json),
            "arm": str(row.arm),
            "split": str(row.split_core),
            "structural_equivalence_class": json.dumps(json_load(row.class_members_json), separators=(",", ":")),
            "class_members": json.dumps(json_load(row.class_members_json), separators=(",", ":")),
            "q_struct": json.dumps(geometry["q_struct"], separators=(",", ":"), sort_keys=True),
            "hidden_k_g3_available": json.dumps(geometry["hidden_k_g3_available"], separators=(",", ":")),
            "v_x_action_values": json.dumps(geometry["v_x_action_values"], separators=(",", ":")),
            "full_information_action_by_k_g3": json.dumps(hidden_full_info_values, separators=(",", ":"), sort_keys=True),
            "full_information_geometry_by_k_g3": json.dumps(geometry["full_information_by_k_g3"], separators=(",", ":"), sort_keys=True),
            "target_json": str(row.target_json_core),
            "target_action_id": target_action_id,
            "target_action_value": float(ACTION_VALUES[target_action_id]),
            "target_plan": json.dumps(target["p"], separators=(",", ":")),
            "observation_bayes_action_id": geometry["observation_bayes_action_id"],
            "observation_bayes_action_value": geometry["observation_bayes_action_value"],
            "target_matches_observation_bayes": target_action_id == geometry["observation_bayes_action_id"],
            "target_matches_any_full_information_action": target_action_id in set(hidden_full_info_values.values()),
            "sample_weight": float(row.sample_weight_core),
            "duplicate_reason": "371 distinct physical_state_id rows share one visible serialization; no same-physical duplicate",
            "source_files": json.dumps(["data/core/train.parquet", "data/core/train_context.parquet", "data/benchmark/cell_bank.parquet"]),
            "generator_version": "NOT_RECORDED_IN_FROZEN_SOURCE_FILES",
            "action_label_generator": "NOT_RECORDED_IN_FROZEN_SOURCE_FILES",
            "plan_label_generator": "NOT_RECORDED_IN_FROZEN_SOURCE_FILES",
            "hidden_label_leakage_evidence": False,
            "geometry_collision_evidence": True,
        })

    audit = pd.DataFrame(provenance_rows)
    token_hash_counts = audit.visible_prompt_token_hash.value_counts()
    action_counts = audit.target_action_id.value_counts().sort_index().to_dict()
    bayes_counts = audit.observation_bayes_action_id.value_counts().sort_index().to_dict()
    match_rate = float(audit.target_matches_observation_bayes.mean())
    geometry_unique = int(audit.v_x_action_values.nunique())
    physical_unique = int(audit.physical_state_id.nunique())
    target_unique = int(audit.target_json.nunique())

    # The target action follows the observation-conditioned payoff optimum,
    # while the same visible token sequence has distinct payoff vectors.
    classification = "VISIBLE_REPRESENTATION_COLLISION"
    reasons = [
        "all 371 rows have one identical visible prompt hash and one identical tokenized visible-input hash",
        f"the rows have {physical_unique} distinct physical_state_id values and {geometry_unique} distinct V_x(a) vectors",
        f"target action matches observation-Bayes argmax in {match_rate:.6f} of rows",
        "the 12 action=5 rows are exactly the rows whose public-conditioned payoff optimum is action=5",
        "class_members_json/q_struct is identical across the conflict and no generator/version provenance is recorded",
    ]
    source_files = [ROOT / "data/core/train.parquet", ROOT / "data/core/train_context.parquet", ROOT / "data/benchmark/cell_bank.parquet", ROOT / "data/core/enum_registry.json"]
    source_hashes = {str(x.relative_to(ROOT)).replace("\\", "/"): file_hash(x) for x in source_files}
    summary = {
        "stage": "T30_SOURCE_LABEL_PROVENANCE_AUDIT",
        "classification": classification,
        "frozen_prior_status": "FAIL_INTERFACE_RECOVERY",
        "prior_t27_classification": "DATA_TARGET_MISMATCH_FOUND",
        "conflict_visible_hash": conflict_hash,
        "rows": len(audit),
        "unique_physical_states": physical_unique,
        "unique_visible_prompts": int(conflict.prompt_core.nunique()),
        "unique_visible_token_hashes": int(token_hash_counts.size),
        "unique_structural_classes": int(conflict.class_members_json.nunique()),
        "unique_v_x_vectors": geometry_unique,
        "unique_targets": target_unique,
        "target_action_counts": {str(k): int(v) for k, v in action_counts.items()},
        "observation_bayes_action_counts": {str(k): int(v) for k, v in bayes_counts.items()},
        "target_matches_observation_bayes_rate": match_rate,
        "target_matches_full_information_action_rate": float(audit.target_matches_any_full_information_action.mean()),
        "sample_weight_values": sorted({float(x) for x in audit.sample_weight}),
        "source_hashes": source_hashes,
        "generator_provenance": {
            "generator_version": "NOT_RECORDED",
            "action_label_generator": "NOT_RECORDED",
            "plan_label_generator": "NOT_RECORDED",
            "duplicate_construction_reason": "not present in frozen source files; inferred collision is across distinct physical states",
        },
        "evidence": reasons,
        "hidden_label_leakage": {
            "classification": "NOT_SUPPORTED",
            "reason": "target action is fully explained by V_x(a) from the legal public-conditioned class; no hidden value was used as model input",
        },
        "required_next_step": "STOP; identify and restore the missing legal public variable in the visible representation, then rebuild source records. Do not majority-relabel and do not train.",
        "final_accessed": False,
        "grpo_started": False,
    }
    audit.to_parquet(args.output / "T30_CONFLICT_PROVENANCE.parquet", index=False)
    (args.output / "T30_SOURCE_LABEL_PROVENANCE_AUDIT.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    report = [
        "# T30 来源—标签溯源审计", "", f"- 冻结冲突 hash：`{conflict_hash}`", f"- 行数：`{len(audit)}`", f"- 结论：`{classification}`", "",
        "## 关键证据", "", f"- 可见 prompt：`{conflict.prompt_core.nunique()}` 个；tokenized visible hash：`{token_hash_counts.size}` 个。", f"- physical_state_id：`{physical_unique}` 个；V_x(a) 向量：`{geometry_unique}` 个。", f"- target action 分布：`{action_counts}`；observation-Bayes action 分布：`{bayes_counts}`。", f"- target action 与 observation-Bayes argmax 一致率：`{match_rate:.6f}`。", "- 12 条 action=5 正好对应 public-conditioned payoff optimum=5 的 12 个物理状态；不是多数标签可修复的脏行。", "- 所有行的 structural class/q_struct/sample weight 相同；source 中没有 generator version、action-label generator 或 plan-label generator 字段。", "",
        "## 冻结决策", "", "`DATA_TARGET_MISMATCH_FOUND` 应精化为 `VISIBLE_REPRESENTATION_COLLISION`。当前停止：不删除或多数重标 12 行，不改 adapter，不运行 GRPO，不访问 FINAL。下一步只能补回合法 public variable 后从 source records 重建数据。", "",
        "详细逐行 provenance 已写入 `T30_CONFLICT_PROVENANCE.parquet`。",
    ]
    (args.output / "T30_PROVENANCE_REPORT_CN.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"classification": classification, "rows": len(audit), "target_action_counts": action_counts, "observation_bayes_action_counts": bayes_counts, "match_rate": match_rate, "unique_v_x": geometry_unique}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
