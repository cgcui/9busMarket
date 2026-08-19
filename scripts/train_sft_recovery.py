#!/usr/bin/env python3
"""T22: fresh, exactly three-epoch BF16/NF4 Core-ESR recovery SFT."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer,
                          TrainerCallback, TrainingArguments, set_seed)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paper9bus_gv_grpo.paths import split_file

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_hashes(root: Path) -> dict:
    return {str(p.relative_to(root)).replace("\\", "/"): sha256_file(p) for p in sorted(root.rglob("*")) if p.is_file()}


def build_examples(tok, path: Path, max_len: int):
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
        return {
            "input_ids": torch.tensor([x["input_ids"] + [pad] * (n - len(x["input_ids"])) for x in features]),
            "attention_mask": torch.tensor([[1] * len(x["input_ids"]) + [0] * (n - len(x["input_ids"])) for x in features]),
            "labels": torch.tensor([x["labels"] + [-100] * (n - len(x["labels"])) for x in features]),
            "sample_weight": torch.tensor([x["sample_weight"] for x in features], dtype=torch.float32),
        }


class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        weights = inputs.pop("sample_weight"); labels = inputs["labels"]; outputs = model(**inputs)
        logits = outputs.logits[..., :-1, :].contiguous(); gold = labels[..., 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), gold.view(-1), ignore_index=-100, reduction="none").view(gold.shape)
        mask = gold.ne(-100); per = (loss * mask).sum(1) / mask.sum(1).clamp_min(1)
        value = (per * weights).sum() / weights.sum().clamp_min(1e-8)
        return (value, outputs) if return_outputs else value


class RecoveryCallback(TrainerCallback):
    def __init__(self, root: Path, tok):
        self.root = root; self.tok = tok; self.logs = []; self.checkpoints = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            self.logs.append({"step": int(state.global_step), "epoch": state.epoch, **{k: (float(v) if isinstance(v, (int, float)) else v) for k, v in logs.items()}})

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = int(round(float(state.epoch)))
        path = self.root / f"checkpoint_epoch{epoch}"
        path.mkdir(parents=True, exist_ok=True)
        model = kwargs["model"]
        model.save_pretrained(path, safe_serialization=True)
        self.tok.save_pretrained(path)
        metadata = {"epoch": epoch, "global_step": int(state.global_step), "log_count": len(self.logs)}
        (path / "CHECKPOINT_METADATA.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        self.checkpoints.append({"epoch": epoch, "path": str(path), "global_step": int(state.global_step)})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    seed = 42
    set_seed(seed); random.seed(seed); np.random.seed(seed)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("T22 requires CUDA BF16 support; refusing fallback to FP16/CPU")
    start = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    model_name = protocol["base_model"]
    max_len = int(protocol["training"]["max_seq_length"])
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tok.pad_token = tok.pad_token or tok.eos_token; tok.padding_side = "right"
    train = build_examples(tok, split_file("TRAIN"), max_len)
    if len(train) != 1571:
        raise RuntimeError(f"expected 1571 TRAIN examples, got {len(train)}")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    base = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb, device_map={"": 0}, trust_remote_code=True, torch_dtype=torch.bfloat16)
    found = [x for x in TARGET_MODULES if x in {n.split(".")[-1] for n, _ in base.named_modules()}]
    if not found: raise RuntimeError("no supported LoRA target module found")
    model = prepare_model_for_kbit_training(base)
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM", target_modules=found))
    model.config.use_cache = False; model.gradient_checkpointing_enable(); model.enable_input_require_grads()
    callback = RecoveryCallback(args.output, tok)
    train_args = TrainingArguments(
        output_dir=str(args.output / "trainer_state"), per_device_train_batch_size=2, gradient_accumulation_steps=8,
        learning_rate=2e-4, lr_scheduler_type="cosine", warmup_ratio=0.05, max_grad_norm=1.0,
        bf16=True, fp16=False, num_train_epochs=3, max_steps=-1, save_strategy="no", logging_steps=10,
        report_to=[], remove_unused_columns=False, seed=seed, dataloader_num_workers=0,
    )
    trainer = WeightedTrainer(model=model, args=train_args, train_dataset=train, data_collator=Collator(tok), callbacks=[callback])
    result = trainer.train()
    runtime = time.perf_counter() - start
    if torch.cuda.is_available(): torch.cuda.synchronize()
    peak_vram = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
    metrics = {"train_result": result.metrics, "logs": callback.logs, "checkpoints": callback.checkpoints,
               "runtime_seconds": runtime, "peak_vram_bytes": peak_vram, "peak_vram_gib": peak_vram / 2**30 if peak_vram else None,
               "nan_or_inf": False, "optimizer_steps": int(trainer.state.global_step), "epochs": 3,
               "model": model_name, "train_examples": len(train), "bf16": True, "fp16": False, "max_seq_length": max_len,
               "completion_limit": int(protocol["training"]["completion_limit"]), "target_modules": found}
    flat_numbers = [v for item in callback.logs for v in item.values() if isinstance(v, float)]
    metrics["nan_or_inf"] = any(not math.isfinite(v) for v in flat_numbers)
    (args.output / "TRAINING_METRICS.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    manifest = {"protocol": "Paper9Bus-Power-GV-GRPO-v3", "stage": "T22_FRESH_3_EPOCH_SFT_RECOVERY", "model": model_name,
                "initialization": "fresh base; no existing adapter loaded", "config": protocol["training"], "seed": seed,
                "train_examples": len(train), "optimizer_steps": int(trainer.state.global_step), "checkpoints": callback.checkpoints,
                "metrics_file": "TRAINING_METRICS.json", "metrics_sha256": sha256_file(args.output / "TRAINING_METRICS.json"),
                "checkpoint_hashes": {f"epoch{e['epoch']}": tree_hashes(Path(e["path"])) for e in callback.checkpoints},
                "runtime_seconds": runtime, "peak_vram_bytes": peak_vram, "nan_or_inf": metrics["nan_or_inf"],
                "final_accessed": False, "grpo_started": False}
    (args.output / "RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    report = ["# T22 Fresh 3-Epoch SFT Recovery", "", "- 初始化：`Qwen/Qwen3-1.7B` fresh base，未加载旧 adapter", "- 数据：TRAIN 1571 条", "- 配置：NF4 + BF16，FP16 off，LoRA，exactly 3 epochs", f"- optimizer steps：`{trainer.state.global_step}`", f"- runtime：`{runtime:.1f}s`", f"- peak VRAM：`{peak_vram / 2**30:.2f} GiB`" if peak_vram else "- peak VRAM：N/A", f"- NaN/Inf：`{metrics['nan_or_inf']}`", "", "已保存 `checkpoint_epoch1`、`checkpoint_epoch2`、`checkpoint_epoch3`。本阶段未运行 GRPO、未访问 DEV/FINAL。"]
    (args.output / "RECOVERY_TRAINING_REPORT_CN.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"stage": "T22", "optimizer_steps": trainer.state.global_step, "checkpoints": callback.checkpoints, "runtime_seconds": runtime, "peak_vram_gib": metrics["peak_vram_gib"], "nan_or_inf": metrics["nan_or_inf"]}, ensure_ascii=False, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
