#!/usr/bin/env python3
"""Deterministic DEV evaluation for a saved SFT or GV-GRPO adapter."""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paper9bus_gv_grpo.paths import split_file
from paper9bus_gv_grpo.schema import load_registry, parse_core

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--adapter", type=Path)
    ap.add_argument("--split", choices=["TRAIN", "DEV"], default="DEV")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing model evaluation")
    manifest = json.loads((args.run_dir / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
    adapter = args.adapter or (args.run_dir / "adapter")
    tok = AutoTokenizer.from_pretrained(adapter, trust_remote_code=True)
    tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = "left"
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    base = AutoModelForCausalLM.from_pretrained(manifest["model"], quantization_config=bnb, device_map={"": 0}, trust_remote_code=True, torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(base, adapter).eval()
    registry = load_registry(ROOT / "data" / "core" / "enum_registry.json")
    frame = pd.read_parquet(split_file(args.split)).head(args.n)
    records = []
    for row in frame.itertuples(index=False):
        prompt = str(row.prompt) + "\n"
        enc = tok(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id)
        text = tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True).strip()
        valid = False; reason = None; obj = None
        try:
            obj = parse_core(text, registry); valid = True
        except Exception as exc:
            reason = str(exc)
        target = json.loads(str(row.target_json))
        rec = {"example_id": str(row.example_id), "strict_valid": valid, "raw_output": text, "invalid_reason": reason}
        if valid:
            pred = np.asarray(obj["b"], dtype=float); gold = np.asarray(target["b"], dtype=float)
            rec.update({"belief_brier": float(np.mean((pred - gold) ** 2)), "belief_nll": float(-np.sum(gold * np.log(np.maximum(pred, 1e-12)))), "action_accuracy": int(obj["a"] == target["a"]), "counterfactual_accuracy": float(np.mean(np.asarray(obj["cf"]) == np.asarray(target["cf"]))), "plan_accuracy": float(np.mean(np.asarray(obj["p"]) == np.asarray(target["p"])))})
        records.append(rec)
    valid = [r for r in records if r["strict_valid"]]
    summary = {"protocol": "Paper9Bus-Power-GV-GRPO-v3", "stage": "MODEL_EVALUATION", "split": args.split, "n": len(records), "strict_valid_rate": len(valid) / len(records) if records else 0.0, "belief_brier": float(np.mean([r["belief_brier"] for r in valid])) if valid else None, "belief_nll": float(np.mean([r["belief_nll"] for r in valid])) if valid else None, "action_accuracy": float(np.mean([r["action_accuracy"] for r in valid])) if valid else None, "counterfactual_accuracy": float(np.mean([r["counterfactual_accuracy"] for r in valid])) if valid else None, "plan_accuracy": float(np.mean([r["plan_accuracy"] for r in valid])) if valid else None, "final_accessed": False}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "EVALUATION_SUMMARY.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(records).to_json(args.output / "EVALUATION_RECORDS.jsonl", orient="records", lines=True, force_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()

