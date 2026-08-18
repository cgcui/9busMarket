#!/usr/bin/env python3
"""Compact simulator-grounded GV-GRPO update over sampled candidate groups."""
from __future__ import annotations

import argparse, json, random, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch, yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from paper9bus_gv_grpo.paths import CORE_ROOT
from paper9bus_gv_grpo.reward import candidate_reward, group_advantages, load_context, load_payoff_tables
from paper9bus_gv_grpo.schema import load_registry

def sequence_logprob(model, tok, prompts, outputs, max_len):
    texts = [p + "\n" + o for p, o in zip(prompts, outputs)]
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to("cuda")
    prompt_lens = [len(tok(p + "\n", add_special_tokens=True, truncation=True, max_length=max_len)["input_ids"]) for p in prompts]
    labels = enc.input_ids.clone()
    for i, n in enumerate(prompt_lens): labels[i, :min(n, labels.shape[1])] = -100
    out = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask); logits = out.logits[:, :-1].contiguous(); gold = labels[:, 1:].contiguous()
    token_lp = -torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), gold.reshape(-1), ignore_index=-100, reduction="none").reshape(gold.shape)
    mask = gold.ne(-100); return (token_lp * mask).sum(1) / mask.sum(1).clamp_min(1)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", type=Path, required=True); ap.add_argument("--run-dir", type=Path, required=True); ap.add_argument("--candidates", type=Path, required=True); ap.add_argument("--gate-report", type=Path, required=True); ap.add_argument("--output", type=Path, required=True); ap.add_argument("--checkpoint-every", type=int, default=25); args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")); gate = json.loads(args.gate_report.read_text(encoding="utf-8"))
    if gate.get("status") != "PASS_GRPO_SIGNAL": raise RuntimeError("GATE_GRPO_SIGNAL is not PASS; GRPO training is blocked")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable; refusing GV-GRPO on CPU")
    records = [json.loads(x) for x in args.candidates.read_text(encoding="utf-8").splitlines() if x.strip()]; registry = load_registry(CORE_ROOT / "enum_registry.json"); split = records[0].get("split", "TRAIN")
    context = load_context(split); tables = load_payoff_tables(split); grouped = {}
    for row in records:
        reward, detail = candidate_reward(str(row["raw_output"]), context[str(row["example_id"])], tables, registry)
        if not detail["valid"]: raise RuntimeError(f"invalid candidate reached training: {row['example_id']} {detail['reason']}")
        grouped.setdefault(str(row["example_id"]), []).append({**row, "reward": reward, "detail": detail})
    if any(len(v) != int(cfg["group_size"]) for v in grouped.values()): raise RuntimeError("candidate groups do not match group_size")
    manifest = json.loads((args.run_dir / "RUN_MANIFEST.json").read_text(encoding="utf-8")); model_name = manifest["model"]
    tok = AutoTokenizer.from_pretrained(args.run_dir / "adapter", trust_remote_code=True); tok.pad_token = tok.pad_token or tok.eos_token; tok.padding_side = "right"
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    base = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb, device_map={"": 0}, trust_remote_code=True, torch_dtype=torch.float16); model = PeftModel.from_pretrained(base, args.run_dir / "adapter", is_trainable=True).train(); model.config.use_cache = False; model.gradient_checkpointing_enable(); model.enable_input_require_grads()
    params = [p for p in model.parameters() if p.requires_grad]; optimizer = torch.optim.AdamW(params, lr=float(cfg["learning_rate"])); groups = list(grouped.values()); random.seed(int(cfg["seed"])); random.shuffle(groups); history = []
    # Freeze old-policy scores before the first optimizer update.  Computing
    # old_lp from the already-updated model would make PPO's ratio nearly 1.
    old_logprobs = {}
    model.eval()
    with torch.no_grad():
        for group in groups:
            prompts = [str(context[str(x["example_id"])] ["prompt"]) for x in group]; outputs = [str(x["raw_output"]) for x in group]
            old_logprobs[str(group[0]["example_id"])] = sequence_logprob(model, tok, prompts, outputs, int(cfg["max_seq_length"])).detach().cpu()
    model.train()
    for step in range(int(cfg["optimizer_steps"])):
        group = groups[step % len(groups)]; rewards = [float(x["reward"]) for x in group]; adv = torch.tensor(group_advantages(rewards), dtype=torch.float32, device="cuda")
        prompts = [str(context[str(x["example_id"])] ["prompt"]) for x in group]; outputs = [str(x["raw_output"]) for x in group]
        old_lp = old_logprobs[str(group[0]["example_id"])].to("cuda")
        optimizer.zero_grad(set_to_none=True); new_lp = sequence_logprob(model, tok, prompts, outputs, int(cfg["max_seq_length"]))
        ratio = torch.exp(new_lp - old_lp); clipped = torch.clamp(ratio, 1 - float(cfg["clip_eps"]), 1 + float(cfg["clip_eps"]))
        policy_loss = -torch.minimum(ratio * adv, clipped * adv).mean(); kl = (new_lp - old_lp).mean(); loss = policy_loss + float(cfg["kl_coef"]) * kl.square(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, float(cfg["max_grad_norm"])); optimizer.step()
        history.append({"step": step + 1, "loss": float(loss.detach().cpu()), "policy_loss": float(policy_loss.detach().cpu()), "kl_proxy": float(kl.detach().cpu()), "mean_reward": float(np.mean(rewards)), "mean_regret": float(np.mean([x["detail"]["regret"] for x in group]))})
        if (step + 1) % int(args.checkpoint_every) == 0 or step + 1 == int(cfg["optimizer_steps"]):
            ck = args.output / "checkpoints" / f"checkpoint-{step + 1}"; ck.mkdir(parents=True, exist_ok=True); model.eval(); model.save_pretrained(ck); tok.save_pretrained(ck); model.train()
            (ck / "STEP_METRICS.json").write_text(json.dumps(history[-1], indent=2, ensure_ascii=False), encoding="utf-8")
        if (step + 1) % 25 == 0: print(json.dumps(history[-1]), flush=True)
    args.output.mkdir(parents=True, exist_ok=True); model.eval(); model.save_pretrained(args.output / "adapter"); tok.save_pretrained(args.output / "adapter")
    report = {"protocol": "Paper9Bus-Power-GV-GRPO-v3", "stage": "GV_GRPO", "optimizer_steps": len(history), "group_size": int(cfg["group_size"]), "horizon": int(cfg["horizon"]), "gamma": float(cfg["gamma"]), "clip_eps": float(cfg["clip_eps"]), "kl_coef": float(cfg["kl_coef"]), "source_candidates": str(args.candidates), "gate_status": gate["status"], "old_policy_frozen_before_updates": True, "final_accessed": False, "history": history}
    (args.output / "GRPO_TRAINING_METRICS.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"); print(json.dumps({k: v for k, v in report.items() if k != "history"}, indent=2, ensure_ascii=False))

if __name__ == "__main__": main()
