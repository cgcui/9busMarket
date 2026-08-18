from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from .paths import BENCHMARK_ROOT, CORE_ROOT, PROJECT_ROOT

REQUIRED_CORE = {"example_id", "prompt", "target_json", "sample_weight", "split"}
REQUIRED_CONTEXT = {"example_id", "physical_state_id", "class_members_json", "split", "final_accessed"}

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main() -> int:
    train = pd.read_parquet(CORE_ROOT / "train.parquet")
    dev = pd.read_parquet(CORE_ROOT / "dev.parquet")
    train_ctx = pd.read_parquet(CORE_ROOT / "train_context.parquet")
    dev_ctx = pd.read_parquet(CORE_ROOT / "dev_context.parquet")
    cell = pd.read_parquet(BENCHMARK_ROOT / "cell_bank.parquet")
    errors = []
    for name, frame, required in [("train", train, REQUIRED_CORE), ("dev", dev, REQUIRED_CORE),
                                  ("train_context", train_ctx, REQUIRED_CONTEXT), ("dev_context", dev_ctx, REQUIRED_CONTEXT)]:
        missing = required - set(frame.columns)
        if missing: errors.append(f"{name}:missing:{sorted(missing)}")
        if "final_accessed" in frame and bool(frame.final_accessed.any()): errors.append(f"{name}:final_accessed")
    if set(train.split.unique()) != {"TRAIN"} or set(dev.split.unique()) != {"DEV"}:
        errors.append("split_labels_invalid")
    if set(train.example_id.astype(str)) & set(dev.example_id.astype(str)):
        errors.append("train_dev_example_overlap")
    if set(cell.split.unique()) != {"TRAIN", "DEV"}:
        errors.append("cell_bank_split_invalid")
    if not (cell.solver_status_name == "OPTIMAL").all(): errors.append("non_optimal_cell")
    if errors:
        raise SystemExit("PREPARE_FAIL: " + "; ".join(errors))
    manifest = {"protocol": "Paper9Bus-Power-GV-GRPO-v3", "benchmark": "Paper9Bus-3Gen-C3",
                "train_examples": len(train), "dev_examples": len(dev), "cell_rows": len(cell),
                "final_accessed": False, "files": {str(p.relative_to(PROJECT_ROOT)): sha256(p)
                for p in [CORE_ROOT / "train.parquet", CORE_ROOT / "dev.parquet", BENCHMARK_ROOT / "cell_bank.parquet"]}}
    out = PROJECT_ROOT / "data_manifest.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

