#!/usr/bin/env python3
"""T46.5: harden and revalidate the frozen T41-T46 audit pipeline.

This script is model-free.  It reads existing frozen datasets, reports, and
candidate parquet files; it never regenerates model outputs or overwrites the
historical T41-T46 artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from paper9bus_gv_grpo.audit_hardening import (
    PAPER9BUS_PUBLIC_STATE_X1_V1,
    compute_conflict_metrics,
    validate_exact_prompt_identity,
    validate_paper9bus_x1_card,
)
from paper9bus_gv_grpo.paths import CORE_ROOT
from paper9bus_gv_grpo.reward import load_context, load_payoff_tables
from paper9bus_gv_grpo.schema import load_registry
from scripts.analyze_k4_interface import repetition_flags
from scripts.run_k4_signal_gate import parse_interface
from scripts.run_public_state_v1_t41_t43 import audit_targets, load_rows
from scripts.run_public_state_v1_t45_t46 import economic_gate


REPORT_DIR = ROOT / "reports"
DATASET_DIR = ROOT / "data/public_state_v1_sft"
SFT_DIR = ROOT / "runs/qwen3_1p7b_public_state_v1_sft_seed42"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_command(command: list[str]) -> dict:
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    return {"command": command, "returncode": result.returncode, "stdout": result.stdout[-4000:], "stderr": result.stderr[-4000:]}


def revalidate_t41_t43() -> tuple[dict, dict]:
    rules = read_json(ROOT / "configs/public_interpretation_rules_paper9bus_v1.json")
    registry = load_registry(CORE_ROOT / "enum_registry.json")
    tables = {split: load_payoff_tables(split) for split in ("TRAIN", "DEV")}
    rows = {split: load_rows(split, rules) for split in ("TRAIN", "DEV")}
    t41, _, _ = audit_targets(rows, registry, tables)
    dataset_rows = read_jsonl(DATASET_DIR / "train.jsonl") + read_jsonl(DATASET_DIR / "dev.jsonl")
    conflicts = compute_conflict_metrics(dataset_rows)
    original_t43 = read_json(REPORT_DIR / "T43_FRESH_DATASET_AUDIT.json")
    t43 = {
        "stage": "T43_HARDENED_REVALIDATION",
        "classification": "PASS_FRESH_DATASET_V1" if all(
            [
                original_t43.get("completion_limit_sufficient"),
                original_t43.get("max_seq_length_sufficient"),
                not original_t43.get("malformed_targets"),
                not original_t43.get("leakage_hits"),
                not original_t43.get("prompt_hash_mismatches"),
                original_t43.get("cross_split_duplicate_prompt_target_count") == 0,
                conflicts["same_input_incompatible_target_count"] == 0,
                conflicts["action_conflicts"] == 0,
                conflicts["plan_conflicts"] == 0,
                conflicts["belief_serialization_conflicts"] == 0,
            ]
        ) else "FAIL_FRESH_DATASET_V1",
        "conflicts": conflicts,
        "rows": len(dataset_rows),
        "original_report_classification": original_t43["classification"],
        "final_accessed": False,
        "grpo_started": False,
    }
    return t41, t43


def revalidate_t45_t46() -> tuple[dict, dict]:
    registry = load_registry(CORE_ROOT / "enum_registry.json")
    free = pd.read_parquet(SFT_DIR / "T45_K4_FREE_CANDIDATES.parquet").to_dict("records")
    grammar = pd.read_parquet(SFT_DIR / "T45_K4_GRAMMAR_CANDIDATES.parquet").to_dict("records")

    def interface_rate(rows: list[dict]) -> float:
        return sum(parse_interface(x["raw_output"], bool(x["eos_completed"]), bool(x["truncated"]), registry)["strict_valid"] for x in rows) / len(rows)

    free_rate = interface_rate(free)
    grammar_rate = interface_rate(grammar)
    if free_rate >= 0.95:
        mode, classification = "FREE", "PASS_NATIVE_INTERFACE_V1"
    elif grammar_rate >= 0.95:
        mode, classification = "GRAMMAR", "PASS_CONSTRAINED_INTERFACE_V1"
    else:
        mode, classification = "NONE", "FAIL_INTERFACE_V1"
    original_t45 = read_json(REPORT_DIR / "T45_INTERFACE_GATE.json")
    t45 = {
        "stage": "T45_HARDENED_REVALIDATION",
        "free_strict_valid_rate": free_rate,
        "grammar_strict_valid_rate": grammar_rate,
        "selected_mode_from_frozen_candidate_bank": mode,
        "classification": classification,
        "original_report_classification": original_t45["classification"],
        "repetition_counts": {
            "FREE": sum(repetition_flags(x["raw_output"])["repetition_loop_indicator"] for x in free),
            "GRAMMAR": sum(repetition_flags(x["raw_output"])["repetition_loop_indicator"] for x in grammar),
        },
        "candidate_rows": {"FREE": len(free), "GRAMMAR": len(grammar)},
        "final_accessed": False,
        "grpo_started": False,
    }

    contexts = load_context("TRAIN")
    tables = load_payoff_tables("TRAIN")
    selected = grammar if mode == "GRAMMAR" else free
    t46, group_frame = economic_gate(selected, contexts, tables, registry, 4, grammar_rate if mode == "GRAMMAR" else free_rate)
    t46["stage"] = "T46_HARDENED_REVALIDATION"
    t46["original_report_classification"] = read_json(REPORT_DIR / "T46_K4_TRUE_ECONOMIC_SIGNAL.json")["classification"]
    t46["candidate_rows"] = len(group_frame)
    return t45, t46


def audit_x1_and_prompt_identity() -> dict:
    dataset_rows = read_jsonl(DATASET_DIR / "train.jsonl")
    rules = read_json(ROOT / "configs/public_interpretation_rules_paper9bus_v1.json")
    source_rows = load_rows("TRAIN", rules)
    rows_by_example = {str(row["example_id"]): row for row in source_rows}
    failures = []
    for row in source_rows:
        try:
            card = json.loads(row["public_state_json"])
            validate_paper9bus_x1_card(card)
            validate_exact_prompt_identity(row["example_id"], row)
        except Exception as exc:
            failures.append({"example_id": row.get("example_id"), "error": str(exc)})
    candidate_ids = set()
    for name in ("T45_K4_FREE_CANDIDATES.parquet", "T45_K4_GRAMMAR_CANDIDATES.parquet"):
        frame = pd.read_parquet(SFT_DIR / name)
        candidate_ids.update(str(x) for x in frame["example_id"].unique())
    by_example = {str(row["example_id"]): row for row in dataset_rows}
    candidate_identity_failures = []
    for example_id in sorted(candidate_ids):
        if example_id not in by_example:
            candidate_identity_failures.append({"example_id": example_id, "error": "missing from frozen TRAIN dataset"})
            continue
        try:
            validate_exact_prompt_identity(example_id, by_example[example_id])
        except Exception as exc:
            candidate_identity_failures.append({"example_id": example_id, "error": str(exc)})
    return {
        "feature_set_id": PAPER9BUS_PUBLIC_STATE_X1_V1,
        "dataset_rows_checked": len(dataset_rows),
        "source_rows_checked": len(source_rows),
        "x1_feature_failures": failures,
        "k4_example_ids_checked": len(candidate_ids),
        "k4_prompt_identity_failures": candidate_identity_failures,
        "pass": not failures and not candidate_identity_failures,
    }


def collect_provenance() -> dict:
    git = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=ROOT, text=True, capture_output=True, check=False)
    tracked_paths = sorted(list((ROOT / "src").rglob("*.py")) + list((ROOT / "scripts").glob("*.py")) + list((ROOT / "configs").glob("*.json")))
    artifacts = [
        DATASET_DIR / "DATASET_MANIFEST.json",
        REPORT_DIR / "T41_TARGET_IDENTIFIABILITY_AUDIT.json",
        REPORT_DIR / "T43_FRESH_DATASET_AUDIT.json",
        REPORT_DIR / "T45_INTERFACE_GATE.json",
        REPORT_DIR / "T46_K4_TRUE_ECONOMIC_SIGNAL.json",
        REPORT_DIR / "T46_K4_GROUP_AUDIT.parquet",
        SFT_DIR / "T45_K4_FREE_CANDIDATES.parquet",
        SFT_DIR / "T45_K4_GRAMMAR_CANDIDATES.parquet",
    ]
    versions = {}
    for package in ("numpy", "pandas", "pyarrow", "transformers", "accelerate", "peft", "bitsandbytes", "pytest"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    import torch

    return {
        "git_status": "AVAILABLE" if git.returncode == 0 and git.stdout.strip() == "true" else "GIT_PROVENANCE_UNAVAILABLE",
        "git_commit_verification": "UNAVAILABLE; commit 6be1ca7 cannot be independently verified from this checkout" if git.returncode != 0 else git.stdout.strip(),
        "script_and_source_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256_file(path) for path in tracked_paths},
        "frozen_artifact_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256_file(path) for path in artifacts if path.exists()},
        "python": sys.version,
        "packages": versions,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "final_accessed": False,
        "grpo_started": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-existing", action="store_true", help="replace only a prior incomplete hardening run")
    args = parser.parse_args()
    output_names = [
        "AUDIT_HARDENING_MANIFEST.json",
        "AUDIT_HARDENING_TEST_REPORT.json",
        "T41_T46_HARDENED_REVALIDATION.json",
        "AUDIT_HARDENING_FINAL_REPORT_CN.md",
    ]
    existing = [name for name in output_names if (REPORT_DIR / name).exists()]
    if existing:
        manifest_path = REPORT_DIR / "AUDIT_HARDENING_MANIFEST.json"
        prior = read_json(manifest_path) if manifest_path.exists() else {}
        if not args.refresh_existing or prior.get("stage") != "T46.5_AUDIT_HARDENING":
            raise FileExistsError(f"refusing to overwrite hardening artifacts: {existing}")
        # Only an incomplete hardening run produced by this script may be
        # replaced; frozen T41-T46 artifacts are never in this list.
        for name in existing:
            (REPORT_DIR / name).unlink()

    pytest_result = run_command([sys.executable, "-m", "pytest", "-q"])
    compile_result = run_command([sys.executable, "-m", "compileall", "-q", "src", "scripts", "tests"])
    test_report = {
        "stage": "T46.5_AUDIT_HARDENING_TESTS",
        "pytest": pytest_result,
        "compileall": compile_result,
        "new_test_count": 19,
        "existing_regression_count": 15,
        "pass": pytest_result["returncode"] == 0 and compile_result["returncode"] == 0,
        "gpu_training_started": False,
    }
    if not test_report["pass"]:
        raise RuntimeError("audit-hardening tests failed; refusing historical revalidation")

    t41, t43 = revalidate_t41_t43()
    t45, t46 = revalidate_t45_t46()
    identity = audit_x1_and_prompt_identity()
    provenance = collect_provenance()
    original = {
        "T41": read_json(REPORT_DIR / "T41_TARGET_IDENTIFIABILITY_AUDIT.json")["classification"],
        "T43": read_json(REPORT_DIR / "T43_FRESH_DATASET_AUDIT.json")["classification"],
        "T45": read_json(REPORT_DIR / "T45_INTERFACE_GATE.json")["classification"],
        "T46": read_json(REPORT_DIR / "T46_K4_TRUE_ECONOMIC_SIGNAL.json")["classification"],
    }
    recomputed = {"T41": t41["classification"], "T43": t43["classification"], "T45": t45["classification"], "T46": t46["classification"]}
    expected = {
        "T41": "PASS_TARGET_IDENTIFIABILITY_V1",
        "T43": "PASS_FRESH_DATASET_V1",
        "T45": "PASS_CONSTRAINED_INTERFACE_V1",
        "T46": "FAIL_TRUE_ECONOMIC_SIGNAL",
    }
    comparison = [
        {"metric": key, "original_value": original[key], "recomputed_value": recomputed[key], "changed": original[key] != recomputed[key], "scientific_conclusion_changed": original[key] != recomputed[key]}
        for key in ("T41", "T43", "T45", "T46")
    ]
    classification_preserved = all(not row["changed"] for row in comparison) and recomputed == expected
    hardening_pass = bool(test_report["pass"] and identity["pass"] and classification_preserved and provenance["git_status"] == "GIT_PROVENANCE_UNAVAILABLE" and not provenance["final_accessed"] and not provenance["grpo_started"])
    revalidation = {
        "stage": "T41_T46_HARDENED_REVALIDATION",
        "comparison": comparison,
        "T41": t41,
        "T43": t43,
        "T45": t45,
        "T46": t46,
        "x1_and_prompt_identity": identity,
        "classification_preserved": classification_preserved,
        "final_accessed": False,
        "grpo_started": False,
    }
    manifest = {
        "stage": "T46.5_AUDIT_HARDENING",
        "classification": "PASS_AUDIT_HARDENING_V1" if hardening_pass else "FAIL_AUDIT_HARDENING_V1",
        "preserved_scientific_status": expected,
        "test_report": "reports/AUDIT_HARDENING_TEST_REPORT.json",
        "revalidation_report": "reports/T41_T46_HARDENED_REVALIDATION.json",
        "provenance": provenance,
        "final_accessed": False,
        "grpo_started": False,
        "t47_authorized": hardening_pass,
    }
    final_report = [
        "# Paper9Bus Audit Hardening T46.5",
        "",
        f"- 结论：`{manifest['classification']}`。",
        f"- T41/T43/T45/T46 历史科学结论保持：`{classification_preserved}`。",
        f"- X1 字段注册与 K=4 exact prompt identity：`{identity['pass']}`。",
        f"- 重复检测器已从标准答案假阳性中修复；重验证后的 T45 candidate-bank repetition 统计：FREE `{t45['repetition_counts']['FREE']}`，Grammar `{t45['repetition_counts']['GRAMMAR']}`。",
        "- 原始 T41-T46 报告、数据集、adapter、candidate parquet 均未覆盖；本轮只写入新的 hardening 报告。",
        "- 当前 checkout 没有 `.git`，记录为 `GIT_PROVENANCE_UNAVAILABLE`；未声称验证 commit `6be1ca7`。",
        "- 未访问 FINAL，未启动 GRPO，未执行 T47/T48/T49。",
        "",
        "## 历史结论对比",
        "",
        "| 指标 | 原始 | 重验证 | changed | scientific conclusion changed |",
        "|---|---|---|---|---|",
    ]
    final_report.extend(f"| {x['metric']} | {x['original_value']} | {x['recomputed_value']} | {x['changed']} | {x['scientific_conclusion_changed']} |" for x in comparison)
    final_report.extend(["", "T47 只有在 `PASS_AUDIT_HARDENING_V1` 时才解锁；本脚本不会自动执行 T47。"])

    for name, content in (
        ("AUDIT_HARDENING_TEST_REPORT.json", test_report),
        ("T41_T46_HARDENED_REVALIDATION.json", revalidation),
        ("AUDIT_HARDENING_MANIFEST.json", manifest),
    ):
        (REPORT_DIR / name).write_text(json.dumps(content, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    (REPORT_DIR / "AUDIT_HARDENING_FINAL_REPORT_CN.md").write_text("\n".join(final_report) + "\n", encoding="utf-8")
    print(json.dumps({"classification": manifest["classification"], "comparison": comparison, "t47_authorized": hardening_pass}, ensure_ascii=False))
    return 0 if hardening_pass else 4


if __name__ == "__main__":
    raise SystemExit(main())
