#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, random, sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from paper9bus_gv_grpo.paths import split_file

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run-dir", type=Path, required=True); ap.add_argument("--split", choices=["TRAIN", "DEV"], default="TRAIN"); ap.add_argument("--groups", type=int, default=64); ap.add_argument("--k", type=int, default=4); ap.add_argument("--output", type=Path, required=True); ap.add_argument("--temperature", type=float, default=.9); ap.add_argument("--top-p", type=float, default=.95); ap.add_argument("--max-new-tokens", type=int, default=384); args = ap.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable; refusing candidate sampling")
    manifest = json.loads((args.run_dir / "RUN_MANIFEST.json").read_text(encoding="utf-8")); model_name = manifest["model"]
    tok = AutoTokenizer.from_pretrained(args.run_dir / "adapter", trust_remote_code=True); tok.pad_token = tok.pad_token or tok.eos_token; tok.padding_side = "left"
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    base = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb, device_map={"": 0}, trust_remote_code=True, torch_dtype=torch.float16); model = PeftModel.from_pretrained(base, args.run_dir / "adapter").eval()
    frame = pd.read_parquet(split_file(args.split)).head(args.groups); records = []
    for gi, row in enumerate(frame.itertuples(index=False)):
        prompt = str(row.prompt) + "\n"; enc = tok(prompt, return_tensors="pt").to("cuda")
        for ci in range(args.k):
            seed = int(manifest.get("seed", 42)) * 100000 + gi * 100 + ci; random.seed(seed); torch.manual_seed(seed)
            with torch.no_grad(): out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=True, temperature=args.temperature, top_p=args.top_p, pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id)
            text = tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True).strip()
            records.append({"example_id": str(row.example_id), "group_index": gi, "candidate_index": ci, "split": args.split, "raw_output": text, "source_run": str(args.run_dir)})
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in records) + "\n", encoding="utf-8")
    print(json.dumps({"split": args.split, "groups": len(frame), "k": args.k, "rows": len(records), "output": str(args.output), "final_accessed": False}, indent=2, ensure_ascii=False))

if __name__ == "__main__": main()

