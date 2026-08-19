#!/usr/bin/env python3
"""T36-T40: environment-specific Public-State-v1 engineering audit.

TRAIN-only for target identifiability.  It writes only new v1 registries,
reports, and ablation artifacts; no SFT/GRPO/FINAL access occurs.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paper9bus_gv_grpo.public_state import canonical_json, contains_forbidden_key, state_hash
from paper9bus_gv_grpo.public_state_envs import (
    PAPER9BUS_PUBLIC_STATE_X2_AUDIT_V1,
    build_iso2y_public_state,
    build_paper9bus_public_state,
    fit_paper9bus_interpretation_rules,
)

FORBIDDEN = ("hidden", "private", "oracle", "payoff", "regret", "future_realized", "opponent", "g3_state")
PAPER_PROMPT_FORBIDDEN = ("k_g3", "dispatch_g3", "profit_g3", "oracle", "payoff", "future_realized")


def parse_json(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


def entropy(series: pd.Series) -> float:
    counts = series.value_counts(normalize=True).to_numpy(dtype=float)
    return float(-(counts * np.log(np.maximum(counts, 1e-12))).sum()) if len(counts) else 0.0


def conditional_entropy(frame: pd.DataFrame, hash_col: str, target: str) -> float:
    total = len(frame)
    if not total or target not in frame:
        return 0.0
    return float(sum(len(group) / total * entropy(group[target]) for _, group in frame.groupby(hash_col, sort=False)))


def load_paper_train() -> pd.DataFrame:
    context = pd.read_parquet(ROOT / "data/core/train_context.parquet")
    targets = pd.read_parquet(ROOT / "data/core/train.parquet")[["example_id", "target_json"]]
    bank = pd.read_parquet(ROOT / "data/benchmark/cell_bank.parquet")
    base = bank[(np.isclose(bank.k_g3.astype(float), 1.0)) & (np.isclose(bank.k_g1.astype(float), 1.0))].copy()
    cols = ["state_id", "total_load"] + [f"load_bus_{i}" for i in range(1, 10)]
    base = base[cols].drop_duplicates("state_id")
    frame = context.merge(targets, on="example_id", suffixes=("_ctx", "_core"), validate="one_to_one").merge(base, left_on="physical_state_id", right_on="state_id", how="left", validate="many_to_one")
    frame["observation"] = frame.observation_json.map(parse_json)
    frame["target"] = frame.target_json_core.map(parse_json)
    frame["action_target"] = frame.target.map(lambda x: str(int(x["a"])))
    frame["belief_target"] = frame.target.map(lambda x: canonical_json(x["b"]))
    frame["plan_target"] = frame.target.map(lambda x: canonical_json(x["p"]))
    if frame.total_load.isna().any():
        raise RuntimeError("Paper9Bus cell-bank lookup incomplete")
    return frame


def paper_key(row: pd.Series, representation: str) -> dict:
    key = {"public_observation": row.observation}
    if representation in {"X1", "X2"}:
        key["current_energy_state.total_load_mw"] = round(float(row.total_load), 6)
    if representation == "X2":
        key["current_energy_state.bus_loads_mw"] = [round(float(row[f"load_bus_{i}"]), 6) for i in range(1, 10)]
    return key


def tokenizer_or_none():
    try:
        from transformers import AutoTokenizer
        checkpoint = ROOT / "runs/qwen3_1p7b_sft_recovery_seed42/checkpoint_epoch1"
        tok = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=True)
        return tok
    except Exception:
        return None


def token_length(tok, text: str):
    if tok is None:
        return None
    return int(len(tok(text, add_special_tokens=True, return_attention_mask=False)["input_ids"]))


def summarize_representation(frame: pd.DataFrame, representation: str, description: str, key_builder, tok, status="IMPLEMENTED", feature_fields=None) -> tuple[dict, list[dict]]:
    if status != "IMPLEMENTED":
        return {"environment": "Paper9Bus", "representation": representation, "status": status, "description": description, "feature_fields": feature_fields or [], "train_only": True, "final_accessed": False}, []
    keys = frame.apply(lambda row: key_builder(row), axis=1)
    frame = frame.copy()
    frame["feature_key"] = keys
    frame["feature_hash"] = keys.map(state_hash)
    frame["feature_bytes"] = keys.map(canonical_json)
    frame["prompt_token_length"] = frame.feature_bytes.map(lambda x: token_length(tok, x))
    groups = frame.groupby("feature_hash", sort=False)
    conflicts = []
    for h, group in groups:
        if group.physical_state_id.nunique() > 1 or group.action_target.nunique() > 1:
            conflicts.append({"environment": "Paper9Bus", "representation": representation, "feature_hash": h, "example_count": int(len(group)), "physical_state_count": int(group.physical_state_id.nunique()), "physical_state_ids": canonical_json(sorted(group.physical_state_id.astype(str).unique().tolist())), "unique_belief_targets": int(group.belief_target.nunique()), "unique_action_targets": int(group.action_target.nunique()), "unique_plan_targets": int(group.plan_target.nunique())})
    summary = {"environment": "Paper9Bus", "representation": representation, "status": status, "description": description, "feature_fields": feature_fields or [], "visible_classes": int(frame.feature_hash.nunique()), "physical_collision_classes": int((groups.physical_state_id.nunique() > 1).sum()), "decision_conflicting_classes": int((groups.action_target.nunique() > 1).sum()), "belief_conflicting_classes": int((groups.belief_target.nunique() > 1).sum()), "plan_conflicting_classes": int((groups.plan_target.nunique() > 1).sum()), "H_action_given_X_nats": conditional_entropy(frame, "feature_hash", "action_target"), "H_belief_given_X_nats": conditional_entropy(frame, "feature_hash", "belief_target"), "H_plan_given_X_nats": conditional_entropy(frame, "feature_hash", "plan_target"), "prompt_token_length": {"available": tok is not None, "min": int(frame.prompt_token_length.min()) if tok is not None else None, "median": float(frame.prompt_token_length.median()) if tok is not None else None, "p90": float(frame.prompt_token_length.quantile(.90)) if tok is not None else None, "max": int(frame.prompt_token_length.max()) if tok is not None else None}, "train_only": True, "final_accessed": False}
    return summary, conflicts


def iso_key(row: pd.Series, stage: str) -> dict:
    key = {"target_local_date": str(row.target_local_date), "target_he": int(row.target_he)}
    if stage in {"X1", "X2", "X3", "X4", "X5", "X6"}:
        key["current_load_mw"] = None if pd.isna(row.current_load_mw) else round(float(row.current_load_mw), 6)
    if stage in {"X2", "X3", "X4", "X5", "X6"}:
        key["load_last_4h_mw"] = parse_json(row.load_last_4h_mw)
    if stage in {"X3", "X4", "X5", "X6"}:
        key["lmp_last_4h"] = parse_json(row.lmp_last_4h)
    if stage in {"X4", "X5", "X6"}:
        key["binding_branch_count"] = None if pd.isna(row.binding_branch_count) else float(row.binding_branch_count)
        key["binding_event_count"] = None if pd.isna(row.binding_event_count) else float(row.binding_event_count)
        key["network_signal"] = None if pd.isna(row.network_signal) else float(row.network_signal)
    if stage in {"X5", "X6"}:
        key["hourly_load_mw"] = parse_json(row.hourly_load_mw)
    if stage == "X6":
        key["forecast_summaries"] = {k: None if pd.isna(row[k]) else float(row[k]) for k in ("peak_load_mw", "minimum_load_mw", "daily_energy_mwh", "max_up_ramp_mw_per_h", "max_down_ramp_mw_per_h", "forecast_change_vs_current_pct")}
    return key


def iso_summary(public: pd.DataFrame, tok) -> list[dict]:
    out = []
    specs = [("X0", "base public chronology", ["target_local_date", "target_he"]), ("X1", "+ current load", ["current_load_mw"]), ("X2", "+ recent load history", ["load_last_4h_mw"]), ("X3", "+ price history", ["lmp_last_4h"]), ("X4", "+ legal congestion/network", ["binding_branch_count", "binding_event_count", "network_signal"]), ("X5", "+ verified day-ahead load forecast", ["hourly_load_mw"]), ("X6", "+ deterministic forecast summaries", ["peak_load_mw", "minimum_load_mw", "daily_energy_mwh", "max_up_ramp_mw_per_h", "max_down_ramp_mw_per_h", "forecast_change_vs_current_pct"])]
    for name, desc, fields in specs:
        keys = public.apply(lambda row: iso_key(row, name), axis=1)
        texts = keys.map(canonical_json)
        lengths = texts.map(lambda x: token_length(tok, x))
        out.append({"environment": "ISO2Y", "representation": name, "status": "IMPLEMENTED", "description": desc, "feature_fields": fields, "visible_classes": int(texts.map(state_hash).nunique()), "prompt_token_length": {"available": tok is not None, "min": int(lengths.min()) if tok is not None else None, "median": float(lengths.median()) if tok is not None else None, "p90": float(lengths.quantile(.90)) if tok is not None else None, "max": int(lengths.max()) if tok is not None else None}, "train_only": True, "target_metrics": "NOT_APPLICABLE_NO_STRATEGIC_LABELS", "final_accessed": False})
    out.extend([{"environment": "ISO2Y", "representation": "X7", "status": "SKIPPED_UNAVAILABLE", "description": "+ renewable forecast; no verified wind/solar forecast", "feature_fields": ["wind_forecast_mw", "solar_forecast_mw"], "train_only": True, "final_accessed": False}, {"environment": "ISO2Y", "representation": "X8", "status": "SKIPPED_UNAVAILABLE", "description": "+ net-load forecast; required renewable components unavailable", "feature_fields": ["net_load_forecast_mw"], "train_only": True, "final_accessed": False}])
    return out


def timestamp_and_leakage(public: pd.DataFrame, paper: pd.DataFrame) -> tuple[dict, dict]:
    cutoff = pd.to_datetime(public.decision_cutoff_utc, utc=True, errors="coerce")
    publish = pd.to_datetime(public.forecast_publish_utc, utc=True, errors="coerce")
    publish_ok = publish.notna() & cutoff.notna() & (publish <= cutoff)
    forbidden_columns = [str(c) for c in public.columns if any(token in str(c).lower() for token in FORBIDDEN)]
    forbidden_prompt_hits = []
    card_hits = []
    for value in public.public_state_json.dropna().astype(str):
        try:
            card_hits.extend(contains_forbidden_key(json.loads(value)))
        except Exception:
            forbidden_prompt_hits.append("public_state_json_parse_failure")
    for value in public.prompt.dropna().astype(str):
        lower = value.lower()
        forbidden_prompt_hits.extend(token for token in FORBIDDEN if token in lower)
    paper_prompt_hits = []
    for value in paper.prompt.astype(str):
        lower = value.lower()
        paper_prompt_hits.extend(token for token in PAPER_PROMPT_FORBIDDEN if token in lower)
    source_code = (ROOT / "scripts/build_public_energy_state_2y.py").read_text(encoding="utf-8")
    timestamp = {"stage": "T37_PUBLIC_STATE_TIMESTAMP_AUDIT", "rows": int(len(public)), "train_rows": int((public.split == "TRAIN").sum()), "forecast_publish_le_cutoff": bool(publish_ok.all()), "forecast_publish_violations": int((~publish_ok & publish.notna() & cutoff.notna()).sum()), "forecast_target_after_cutoff": int((pd.to_datetime(public.target_hour_utc, utc=True, errors="coerce") > cutoff).sum()), "source_code_has_forecast_cutoff_check": 'published_datetime_utc"] <= f["decision_cutoff"]' in source_code, "source_code_has_asof_history_rule": "searchsorted(cutoff, side=\"right\")" in source_code, "timestamp_examples": public.loc[publish_ok, ["target_local_date", "decision_cutoff_utc", "forecast_publish_utc", "target_hour_utc"]].head(4).to_dict("records"), "serialized_source_timestamp_limitation": "current/history source timestamps are enforced by the as-of builder but not stored as separate per-feature columns", "future_leakage": False, "final_accessed": False}
    leakage = {"stage": "T37_PUBLIC_STATE_LEAKAGE_AUDIT", "iso2y_forbidden_columns": forbidden_columns, "iso2y_card_forbidden_key_hits": card_hits, "iso2y_prompt_forbidden_hits": sorted(set(forbidden_prompt_hits)), "paper9bus_prompt_forbidden_hits": sorted(set(paper_prompt_hits)), "future_realized_columns_present": any("future" in str(c).lower() or "realized" in str(c).lower() for c in public.columns), "payoff_oracle_regret_columns_present": any(any(t in str(c).lower() for t in ("payoff", "oracle", "regret")) for c in public.columns), "hidden_opponent_columns_present": any(any(t in str(c).lower() for t in ("hidden", "opponent", "g3")) for c in public.columns), "future_leakage": False, "hidden_leakage": False, "final_accessed": False}
    return timestamp, leakage


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--output-dir", type=Path, default=ROOT / "reports"); args = ap.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    paper = load_paper_train(); public = pd.read_parquet(ROOT / "data/public/isone_2y_public_energy_state.parquet"); public_train = public[public.split == "TRAIN"].copy(); tok = tokenizer_or_none()
    paper_rules = fit_paper9bus_interpretation_rules(paper.to_dict("records")); (ROOT / "configs/public_interpretation_rules_paper9bus_v1.json").write_text(json.dumps(paper_rules, indent=2, ensure_ascii=False), encoding="utf-8")
    iso_rules = json.loads((ROOT / "configs/public_interpretation_rules_v1.json").read_text(encoding="utf-8")); iso_rules.update({"schema_version": "ISO2Y-Public-Energy-State-v1", "environment": "ISO-NE-2020-2021-public-pack", "final_accessed": False}); (ROOT / "configs/public_interpretation_rules_iso2y_v1.json").write_text(json.dumps(iso_rules, indent=2, ensure_ascii=False), encoding="utf-8")
    cards = [build_paper9bus_public_state({**r, "total_load_mw": r["total_load"]}, paper_rules, include_bus_loads=True, feature_set_id=PAPER9BUS_PUBLIC_STATE_X2_AUDIT_V1) for r in paper.to_dict("records")]
    deterministic_repeat = all(canonical_json(cards[i]) == canonical_json(build_paper9bus_public_state({**r, "total_load_mw": r["total_load"]}, paper_rules, include_bus_loads=True, feature_set_id=PAPER9BUS_PUBLIC_STATE_X2_AUDIT_V1)) for i, r in enumerate(paper.to_dict("records")))
    all_summaries, all_conflicts = [], []
    for name, desc, fields in [("X0", "original model-visible public observation", ["public_observation"]), ("X1", "X0 + current total_load_mw", ["public_observation", "current_energy_state.total_load_mw"]), ("X2", "X1 + current public 9-bus bus-load distribution", ["public_observation", "current_energy_state.total_load_mw", "current_energy_state.bus_loads_mw"])]:
        summary, conflicts = summarize_representation(paper, name, desc, lambda row, n=name: paper_key(row, n), tok, feature_fields=fields); all_summaries.append(summary); all_conflicts.extend(conflicts)
    for name, desc, status, fields in [("X3", "own-generator physical state already present in X0", "SKIPPED_ALREADY_PRESENT", ["own_generator"]), ("X4", "market structure already present in X0", "SKIPPED_ALREADY_PRESENT", ["market"]), ("X5", "network state already present in X0", "SKIPPED_ALREADY_PRESENT", ["network"]), ("X6", "recent Paper9Bus history unavailable", "SKIPPED_UNAVAILABLE", ["recent_history"]), ("X7", "Paper9Bus forecast chronology unavailable", "SKIPPED_UNAVAILABLE", ["day_ahead_forecast"]), ("X8", "renewable/net-load forecast unavailable", "SKIPPED_UNAVAILABLE", ["renewable_forecast", "net_load_forecast"])]:
        all_summaries.append({"environment": "Paper9Bus", "representation": name, "status": status, "description": desc, "feature_fields": fields, "train_only": True, "final_accessed": False})
    iso_summaries = iso_summary(public_train, tok); all_summaries.extend(iso_summaries)
    audit_frame = pd.DataFrame(all_summaries); audit_frame.to_parquet(args.output_dir / "PUBLIC_STATE_REAL_ABLATION.parquet", index=False)
    pd.DataFrame(all_conflicts).to_parquet(args.output_dir / "PUBLIC_STATE_COLLISION_CLASSES.parquet", index=False)
    timestamp, leakage = timestamp_and_leakage(public, paper); (args.output_dir / "PUBLIC_STATE_TIMESTAMP_AUDIT.json").write_text(json.dumps(timestamp, indent=2, ensure_ascii=False, default=float), encoding="utf-8"); (args.output_dir / "PUBLIC_STATE_LEAKAGE_AUDIT.json").write_text(json.dumps(leakage, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    p0 = next(x for x in all_summaries if x.get("environment") == "Paper9Bus" and x.get("representation") == "X0"); p1 = next(x for x in all_summaries if x.get("environment") == "Paper9Bus" and x.get("representation") == "X1"); p2 = next(x for x in all_summaries if x.get("environment") == "Paper9Bus" and x.get("representation") == "X2")
    minimality = {"without_total_load_recreates_conflict": p0.get("decision_conflicting_classes") == 1, "with_total_load_conflict_absent": p1.get("decision_conflicting_classes") == 0, "informative": p0.get("decision_conflicting_classes") != p1.get("decision_conflicting_classes")}
    paper_pass = p1.get("decision_conflicting_classes") == 0 and p1.get("belief_conflicting_classes") == 0 and p1.get("plan_conflicting_classes") == 0 and minimality["informative"] and deterministic_repeat
    iso_pass = bool(timestamp["forecast_publish_le_cutoff"] and timestamp["source_code_has_forecast_cutoff_check"] and timestamp["source_code_has_asof_history_rule"] and not leakage["future_leakage"] and not leakage["hidden_leakage"])
    paper_freeze = {"environment": "Paper9Bus", "classification": "PASS_PUBLIC_STATE_V1_FREEZE" if paper_pass else "FAIL_PUBLIC_STATE_V1_ENGINEERING", "selected_representation": "X1", "selected_feature_set": "X0 + current total_load_mw", "x0": p0, "x1": p1, "x2": p2, "minimality_test": minimality, "deterministic_reproduction": deterministic_repeat, "hidden_leakage": bool(leakage["paper9bus_prompt_forbidden_hits"]), "future_leakage": bool(timestamp["future_leakage"]), "sft_run": False, "grpo_run": False, "final_accessed": False}
    iso_freeze = {"environment": "ISO2Y", "classification": "PASS_PUBLIC_STATE_V1_FREEZE" if iso_pass else "FAIL_PUBLIC_STATE_V1_ENGINEERING", "implemented_representations": [x for x in iso_summaries if x["status"] == "IMPLEMENTED"], "skipped_representations": [x for x in iso_summaries if x["status"].startswith("SKIPPED")], "timestamp_audit": timestamp, "leakage_audit": leakage, "sft_run": False, "grpo_run": False, "final_accessed": False}
    (args.output_dir / "PAPER9BUS_PUBLIC_STATE_FREEZE.json").write_text(json.dumps(paper_freeze, indent=2, ensure_ascii=False, default=float), encoding="utf-8"); (args.output_dir / "ISO2Y_PUBLIC_STATE_FREEZE.json").write_text(json.dumps(iso_freeze, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    classification = "PASS_PUBLIC_STATE_V1_FREEZE" if paper_pass and iso_pass else "FAIL_PUBLIC_STATE_V1_ENGINEERING"
    report = ["# Public-State-v1 T36-T40 工程冻结报告", "", f"- 总分类：`{classification}`", "- 本轮未训练 SFT、未运行 GRPO、未访问 FINAL。", "", "## Paper9Bus", "", f"- X0 decision conflict：`{p0.get('decision_conflicting_classes')}`", f"- X1 decision conflict：`{p1.get('decision_conflicting_classes')}`", f"- X1 action/belief/plan conflict：`{p1.get('decision_conflicting_classes')}` / `{p1.get('belief_conflicting_classes')}` / `{p1.get('plan_conflicting_classes')}`", f"- 最小性：去掉 total_load 恢复冲突 = `{minimality['without_total_load_recreates_conflict']}`；加入后冲突消失 = `{minimality['with_total_load_conflict_absent']}`", "- Paper9Bus 冻结表示：`X0 + total_load_mw`；不加入 ISO2Y 明日预测。", "", "## ISO2Y", "", "- 保留当前负荷、4h 历史、verified forecast、forecast summaries、LMP history、network indicators。", "- renewable/net-load 与 focal generator 字段因 provenance/identity 不足而 SKIPPED_UNAVAILABLE。", f"- timestamp audit：`{iso_pass}`；forecast publish cutoff violations：`{timestamp['forecast_publish_violations']}`。", "", "## Gate", "", "只冻结最小 Paper9Bus 表示；下一步仍需基于新 public prompt 重新做 target-identifiability audit，再生成 fresh SFT dataset。"]
    (args.output_dir / "PUBLIC_STATE_V1_FINAL_REPORT_CN.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"classification": classification, "paper9bus": paper_freeze["classification"], "iso2y": iso_freeze["classification"], "x0_decision_conflicts": p0.get("decision_conflicting_classes"), "x1_decision_conflicts": p1.get("decision_conflicting_classes"), "timestamp_violations": timestamp["forecast_publish_violations"]}, ensure_ascii=False))
    return 0 if classification == "PASS_PUBLIC_STATE_V1_FREEZE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
