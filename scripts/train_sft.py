#!/usr/bin/env python3
"""QLoRA SFT for compact Core-ESR targets."""
from __future__ import annotations

import argparse, json, math, random, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch, yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments, set_seed
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paper9bus_gv_grpo.paths import split_file

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

def examples(tok, path: Path, max_len: int):
    out = []
    for row in pd.read_parquet(path).itertuples(index=False):
        prompt = str(row.prompt) + "\n"
        target = str(row.target_json) + tok.eos_token
        p = tok(prompt, add_special_tokens=True, truncation=True, max_length=max_len, return_attention_mask=False)["input_ids"]
        ids = tok(prompt + target, add_special_tokens=True, truncation=True, max_length=max_len, return_attention_mask=False)["input_ids"]
        if len(ids) <= len(p) or ids[-1] != tok.eos_token_id:
            raise ValueError(f"target truncated or EOS missing: {row.example_id}")
        out.append({"input_ids": ids, "labels": [-100] * len(p) + ids[len(p):], "sample_weight": float(row.sample_weight)})
    return out

class Collator:
    def __init__(self, tok): self.tok = tok
    def __call__(self, features):
        n = max(len(x["input_ids"]) for x in features); pad = self.tok.pad_token_id
        return {"input_ids": torch.tensor([x["input_ids"] + [pad] * (n-len(x["input_ids"])) for x in features]),
                "attention_mask": torch.tensor([[1] * len(x["input_ids"]) + [0] * (n-len(x["input_ids"])) for x in features]),
                "labels": torch.tensor([x["labels"] + [-100] * (n-len(x["labels"])) for x in features]),
                "sample_weight": torch.tensor([x["sample_weight"] for x in features], dtype=torch.float32)}

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        weights = inputs.pop("sample_weight"); labels = inputs["labels"]; outputs = model(**inputs)
        logits = outputs.logits[..., :-1, :].contiguous(); gold = labels[..., 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), gold.view(-1), ignore_index=-100, reduction="none").view(gold.shape)
        mask = gold.ne(-100); per = (loss * mask).sum(1) / mask.sum(1).clamp_min(1)
        value = (per * weights).sum() / weights.sum().clamp_min(1e-8)
        return (value, outputs) if return_outputs else value

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--resume-from-checkpoint", type=Path)
    ap.add_argument("--max-steps", type=int)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable; refusing SFT on CPU")
    set_seed(int(cfg["seed"])); random.seed(int(cfg["seed"])); np.random.seed(int(cfg["seed"]))
    tok = AutoTokenizer.from_pretrained(cfg["model_name"], trust_remote_code=True); tok.pad_token = tok.pad_token or tok.eos_token; tok.padding_side = "right"
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type=cfg["bnb_4bit_quant_type"], bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(cfg["model_name"], quantization_config=bnb, device_map={"": 0}, trust_remote_code=True, torch_dtype=torch.float16)
    found = [x for x in TARGET_MODULES if x in {n.split(".")[-1] for n, _ in model.named_modules()}]
    if not found: raise RuntimeError("no supported LoRA target module found")
    model = prepare_model_for_kbit_training(model); model = get_peft_model(model, LoraConfig(r=int(cfg["lora_r"]), lora_alpha=int(cfg["lora_alpha"]), lora_dropout=float(cfg["lora_dropout"]), bias="none", task_type="CAUSAL_LM", target_modules=found)); model.config.use_cache = False; model.gradient_checkpointing_enable(); model.enable_input_require_grads()
    args.output.mkdir(parents=True, exist_ok=True); (args.output / "checkpoints").mkdir(exist_ok=True)
    train = examples(tok, split_file("TRAIN"), int(cfg["max_seq_length"]))
    max_steps = int(args.max_steps if args.max_steps is not None else cfg["max_steps"])
    batches_per_epoch = max(1, math.ceil(len(train) / (int(cfg["per_device_train_batch_size"]) * int(cfg["gradient_accumulation_steps"]))))
    estimated_steps = max_steps if max_steps > 0 else math.ceil(batches_per_epoch * float(cfg["num_train_epochs"]))
    warmup_steps = max(0, round(estimated_steps * float(cfg.get("warmup_ratio", 0.0))))
    targs = TrainingArguments(output_dir=str(args.output / "checkpoints"), per_device_train_batch_size=int(cfg["per_device_train_batch_size"]), gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]), learning_rate=float(cfg["learning_rate"]), lr_scheduler_type=cfg["lr_scheduler_type"], warmup_steps=warmup_steps, max_grad_norm=float(cfg["max_grad_norm"]), fp16=True, bf16=False, num_train_epochs=float(cfg["num_train_epochs"]), max_steps=max_steps, save_strategy=cfg["save_strategy"], save_steps=int(cfg.get("save_steps", 100)), save_total_limit=cfg.get("save_total_limit"), logging_steps=int(cfg["logging_steps"]), report_to=[], remove_unused_columns=False, seed=int(cfg["seed"]), data_seed=int(cfg["seed"]))
    trainer = WeightedTrainer(model=model, args=targs, train_dataset=train, data_collator=Collator(tok)); result = trainer.train(resume_from_checkpoint=str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None); trainer.save_model(str(args.output / "adapter")); tok.save_pretrained(str(args.output / "adapter"))
    manifest = {"protocol": "Paper9Bus-Power-GV-GRPO-v3", "stage": "ESR_SFT", "model": cfg["model_name"], "config": {**cfg, "resolved_max_steps": max_steps}, "seed": int(cfg["seed"]), "train_examples": len(train), "target_modules": found, "final_accessed": False, "result": result.metrics, "resume_from_checkpoint": str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None}
    (args.output / "RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False, default=float))

if __name__ == "__main__": main()
