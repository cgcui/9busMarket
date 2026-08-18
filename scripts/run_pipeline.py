#!/usr/bin/env python3
"""Run prepare -> audit -> SFT -> DEV -> sampling -> gate -> GRPO."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd):
    print("+", " ".join(map(str, cmd)), flush=True)
    return subprocess.run(cmd, cwd=ROOT, check=False).returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft-config", type=Path, default=ROOT / "configs" / "sft_smoke_qwen3.yaml")
    ap.add_argument("--grpo-config", type=Path, default=ROOT / "configs" / "gv_grpo_smoke.yaml")
    ap.add_argument("--run-root", type=Path, default=ROOT / "runs" / "pipeline_smoke")
    ap.add_argument("--groups", type=int, default=4)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--skip-sft", action="store_true", help="reuse an existing run-root/sft adapter")
    args = ap.parse_args()

    py = sys.executable
    run_root = args.run_root.resolve()
    sft = run_root / "sft"
    run_root.mkdir(parents=True, exist_ok=True)

    for cmd in ([py, "-m", "paper9bus_gv_grpo.prepare"], [py, "-m", "paper9bus_gv_grpo.audit"]):
        if run(cmd):
            return 1
    if not args.skip_sft:
        if run([py, "scripts/train_sft.py", "--config", str(args.sft_config), "--output", str(sft)]):
            return 1
    elif not (sft / "RUN_MANIFEST.json").exists() or not (sft / "adapter").exists():
        raise FileNotFoundError("--skip-sft requires an existing SFT run with RUN_MANIFEST.json and adapter/")

    if run([py, "scripts/evaluate_sft.py", "--run-dir", str(sft), "--split", "DEV", "--n", str(args.groups), "--output", str(sft / "dev_eval")]):
        return 1
    candidates = sft / f"train_candidates_k{args.k}.jsonl"
    if run([py, "scripts/sample_candidates.py", "--run-dir", str(sft), "--split", "TRAIN", "--groups", str(args.groups), "--k", str(args.k), "--output", str(candidates)]):
        return 1
    gate_dir = sft / "signal_gate"
    gate_code = run([py, "-m", "paper9bus_gv_grpo.signal_gate", "--candidates", str(candidates), "--split", "TRAIN", "--k", str(args.k), "--expected-groups", str(args.groups), "--output", str(gate_dir)])
    if gate_code:
        print("Signal gate did not pass; GRPO is correctly blocked.", flush=True)
        return gate_code
    return run([py, "scripts/train_gv_grpo.py", "--config", str(args.grpo_config), "--run-dir", str(sft), "--candidates", str(candidates), "--gate-report", str(gate_dir / "GATE_GRPO_SIGNAL_RESULT.json"), "--output", str(run_root / "gv_grpo")])


if __name__ == "__main__":
    raise SystemExit(main())
