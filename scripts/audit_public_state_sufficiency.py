#!/usr/bin/env python3
"""TRAIN-only public-feature ablation for the frozen 9-bus Core dataset."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))
from paper9bus_gv_grpo.public_state import canonical_json, state_hash


def entropy(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0
    counts = series.value_counts(normalize=True).to_numpy(dtype=float)
    return float(-(counts * np.log(np.maximum(counts, 1e-12))).sum())


def conditional_entropy(frame: pd.DataFrame, feature_hash: str, target: str) -> float:
    total = len(frame)
    return float(sum(len(group) / total * entropy(group[target]) for _, group in frame.groupby(feature_hash, sort=False))) if total else 0.0


def make_frame() -> pd.DataFrame:
    context = pd.read_parquet(ROOT / "data" / "core" / "train_context.parquet")
    targets = pd.read_parquet(ROOT / "data" / "core" / "train.parquet")[["example_id", "target_json"]]
    context = context.merge(targets, on="example_id", suffixes=("_long", ""))
    bank = pd.read_parquet(ROOT / "data" / "benchmark" / "cell_bank.parquet")
    base = bank[(np.isclose(bank["k_g3"].astype(float), 1.0)) & (np.isclose(bank["k_g1"].astype(float), 1.0))].copy()
    columns = ["state_id", "total_load"] + [f"load_bus_{i}" for i in range(1, 10)]
    base = base[columns].drop_duplicates("state_id")
    frame = context.merge(base, left_on="physical_state_id", right_on="state_id", how="left", validate="many_to_one")
    frame["observation"] = frame["observation_json"].map(json.loads)
    frame["target"] = frame["target_json"].map(json.loads)
    frame["action_target"] = frame["target"].map(lambda x: str(int(x["a"])))
    frame["belief_target"] = frame["target"].map(lambda x: canonical_json(x["b"]))
    frame["plan_target"] = frame["target"].map(lambda x: canonical_json(x["p"]))
    frame["physical_state_id"] = frame["physical_state_id"].astype(str)
    if frame["total_load"].isna().any():
        raise RuntimeError("cell bank lookup incomplete")
    return frame


def feature_key(row: pd.Series, representation: str) -> dict:
    # X0 already contains the public focal dispatch/LMP and network features.
    key = {"public_observation": row["observation"]}
    if representation in {"X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"}:
        key["current_energy_state.total_load_mw"] = round(float(row["total_load"]), 6)
    if representation in {"X2", "X3", "X4", "X5", "X6", "X7", "X8"}:
        key["current_energy_state.bus_loads_mw"] = [round(float(row[f"load_bus_{i}"]), 6) for i in range(1, 10)]
    return key


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=ROOT / "reports")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = make_frame()
    specs = {
        "X0": "existing model-visible public observation",
        "X1": "X0 + current total load MW",
        "X2": "X1 + current 9-bus load distribution",
        "X3": "X2 + own-generator physical state (already present in X0; no new legal field)",
        "X4": "X3 + market price structure (already present in X0; no new legal field)",
        "X5": "X4 + network state (already present in X0; no new legal field)",
        "X6": "X5 + recent public history (unavailable in the frozen Paper9Bus cells)",
        "X7": "X6 + genuine day-ahead forecast (unavailable for Paper9Bus; ISO-NE data is not joined to labels)",
        "X8": "X7 + renewable/net-load forecast (unavailable and not fabricated)",
    }
    class_rows = []
    summary_rows = []
    for representation, description in specs.items():
        keys = frame.apply(lambda row: feature_key(row, representation), axis=1)
        frame[f"{representation}_hash"] = keys.map(state_hash)
        grouped = frame.groupby(f"{representation}_hash", sort=False)
        decision_conflicts = int((grouped["action_target"].nunique() > 1).sum())
        belief_conflicts = int((grouped["belief_target"].nunique() > 1).sum())
        plan_conflicts = int((grouped["plan_target"].nunique() > 1).sum())
        physical_collisions = int((grouped["physical_state_id"].nunique() > 1).sum())
        for h, group in grouped:
            if group["physical_state_id"].nunique() > 1 or group["action_target"].nunique() > 1:
                class_rows.append({
                    "representation": representation,
                    "feature_hash": h,
                    "example_count": int(len(group)),
                    "physical_state_count": int(group["physical_state_id"].nunique()),
                    "physical_state_ids": canonical_json(sorted(group["physical_state_id"].unique().tolist())),
                    "unique_belief_targets": int(group["belief_target"].nunique()),
                    "unique_action_targets": int(group["action_target"].nunique()),
                    "unique_plan_targets": int(group["plan_target"].nunique()),
                })
        summary_rows.append({
            "representation": representation,
            "description": description,
            "visible_classes": int(frame[f"{representation}_hash"].nunique()),
            "physical_collision_classes": physical_collisions,
            "decision_conflicting_classes": decision_conflicts,
            "belief_conflicting_classes": belief_conflicts,
            "plan_conflicting_classes": plan_conflicts,
            "unique_belief_targets": int(frame.groupby(f"{representation}_hash")["belief_target"].nunique().gt(0).sum()),
            "unique_action_targets": int(frame.groupby(f"{representation}_hash")["action_target"].nunique().gt(0).sum()),
            "unique_plan_targets": int(frame.groupby(f"{representation}_hash")["plan_target"].nunique().gt(0).sum()),
            "H_action_given_X_nats": conditional_entropy(frame, f"{representation}_hash", "action_target"),
            "H_belief_given_X_nats": conditional_entropy(frame, f"{representation}_hash", "belief_target"),
            "H_plan_given_X_nats": conditional_entropy(frame, f"{representation}_hash", "plan_target"),
            "train_only": True,
            "final_accessed": False,
        })
    ablation = pd.DataFrame(summary_rows)
    collisions = pd.DataFrame(class_rows)
    ablation.to_parquet(args.output_dir / "PUBLIC_STATE_FEATURE_ABLATION.parquet", index=False)
    collisions.to_parquet(args.output_dir / "PUBLIC_STATE_COLLISION_CLASSES.parquet", index=False)
    baseline = ablation[ablation.representation == "X0"].iloc[0]
    selected = next((r for r in summary_rows if r["decision_conflicting_classes"] == 0), summary_rows[-1])
    report = {
        "protocol": "Paper9Bus-Power-GV-GRPO-Public-Energy-State-v1",
        "audit_split": "TRAIN",
        "train_examples": int(len(frame)),
        "baseline": baseline.to_dict(),
        "selected_representation": selected["representation"],
        "selected_feature_set": selected["description"],
        "previous_371_state_collision": {
            "baseline_physical_state_count": int(collisions[(collisions.representation == "X0")]["physical_state_count"].max()) if not collisions[collisions.representation == "X0"].empty else 0,
            "baseline_decision_conflicting_classes": int(baseline["decision_conflicting_classes"]),
            "selected_decision_conflicting_classes": int(selected["decision_conflicting_classes"]),
        },
        "iso2y_join_policy": "ISO-NE 2020-2021 public chronology is audited separately and is not artificially joined to Paper9Bus strategic labels",
        "classification": "LEGAL_PUBLIC_REPRESENTATION_SUFFICIENT" if selected["decision_conflicting_classes"] == 0 else "PUBLIC_OBSERVATION_STILL_INSUFFICIENT",
        "llm_called": False,
        "sft_run": False,
        "grpo_run": False,
        "final_accessed": False,
    }
    (args.output_dir / "PUBLIC_STATE_FEATURE_SUFFICIENCY.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    cn = f"""# Public-Energy-State-v1 特征充分性审计

本审计只使用冻结 9-bus `TRAIN`（{len(frame)} 条），不使用 DEV/FINAL，不训练 SFT，不运行 GRPO。

## 结果

- 原始 X0 的 decision-conflicting visible class：**{int(baseline['decision_conflicting_classes'])}**
- 原始 collision class 中的 physical state 数：**{report['previous_371_state_collision']['baseline_physical_state_count']}**
- 选择表示：**{selected['representation']}**（{selected['description']}）
- 选择后 action conflict：**{int(selected['decision_conflicting_classes'])}**
- 选择后 belief conflict：**{int(selected['belief_conflicting_classes'])}**
- 选择后 plan conflict：**{int(selected['plan_conflicting_classes'])}**
- 结论：`{report['classification']}`

X1 只增加 cell bank 中合法的当前总负荷 MW；没有把 ISO-NE 两年未来真实负荷、payoff、regret 或 oracle action 写进 9-bus prompt。ISO-NE 两年数据单独产出 Public-Energy-State-v1 时序层，只有决策 cutoff 之前发布的 forecast 和历史数据进入状态卡。
"""
    (args.output_dir / "PUBLIC_STATE_CARD_REPORT_CN.md").write_text(cn, encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
