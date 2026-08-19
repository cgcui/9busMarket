#!/usr/bin/env python3
"""Phase B: paired FREE vs canonical Core-ESR grammar-constrained decoding."""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from paper9bus_gv_grpo.paths import CORE_ROOT
from paper9bus_gv_grpo.reward import expected_payoff, load_context, load_payoff_tables
from paper9bus_gv_grpo.schema import ACTION_VALUES, load_registry
from scripts.analyze_k4_interface import classify
from scripts.run_k4_signal_gate import parse_interface, select_snapshots


def json_text(x):
    return json.dumps(x, ensure_ascii=False, separators=(",", ":"), default=float)


class CoreESRGrammar:
    """Canonical Core-ESR JSON grammar.

    This is a decoder constraint, not a post-generation repair: every token is
    admitted only when the resulting decoded prefix remains in the language.
    It fixes field order, list cardinality, enum domains, JSON punctuation and
    numeric lexical form. Semantic belief normalization remains intentionally
    outside the grammar and is audited as an SFT/interface property.
    """

    def __init__(self, tok):
        self.tok = tok
        self.eos_ids = {int(tok.eos_token_id)} if isinstance(tok.eos_token_id, int) else {int(x) for x in tok.eos_token_id}
        self.token_text = {}
        for tid in range(len(tok)):
            if tid in self.eos_ids:
                continue
            text = tok.decode([tid], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            if not text or "\ufffd" in text or any(ord(c) > 127 for c in text):
                continue
            self.token_text[tid] = text
        self.pieces = []
        self._build_pieces()
        self.final_state = (len(self.pieces), 0)
        self.states = self._enumerate_states()
        # Precompute the exact token mask once per finite grammar state.
        self.allowed_by_state = {state: self._allowed_for_state(state) for state in self.states}
        self.cache = {}

    def _build_pieces(self):
        def lit(x): self.pieces.append(("lit", x))
        def enum(x): self.pieces.append(("enum", x))
        def num(): self.pieces.append(("num", None))
        def enum_list(n, chars):
            enum(chars)
            for _ in range(n - 1): lit(","); enum(chars)
        lit('{"a":'); enum("012345"); lit(',"b":[')
        num()
        for _ in range(5): lit(","); num()
        lit('],"cf":['); enum_list(2, "012"); lit('],"e":['); enum_list(8, "01234567")
        lit('],"g":['); enum_list(4, "012"); lit('],"i":'); enum("0123")
        lit(',"p":['); enum_list(3, "012345"); lit('],"q":'); num(); lit("}")

    def _advance(self, idx):
        nxt = idx + 1
        return (nxt, "start") if nxt < len(self.pieces) and self.pieces[nxt][0] == "num" else (nxt, 0)

    def _step(self, state, ch):
        idx, sub = state
        if idx >= len(self.pieces):
            return None
        kind, value = self.pieces[idx]
        if kind == "lit":
            if ch != value[sub]: return None
            return (idx, sub + 1) if sub + 1 < len(value) else self._advance(idx)
        if kind == "enum":
            return self._advance(idx) if ch in value else None
        phase = sub
        if phase == "start":
            if not ch.isdigit(): return None
            return (idx, "zero" if ch == "0" else "int")
        if phase == "zero":
            if ch == ".": return (idx, "frac_start")
            return self._step(self._advance(idx), ch)
        if phase == "int":
            if ch.isdigit(): return (idx, "int")
            if ch == ".": return (idx, "frac_start")
            return self._step(self._advance(idx), ch)
        if phase == "frac_start":
            return (idx, "frac") if ch.isdigit() else None
        if phase == "frac":
            if ch.isdigit(): return (idx, "frac")
            return self._step(self._advance(idx), ch)
        raise RuntimeError(f"unknown numeric phase {phase}")

    def _enumerate_states(self):
        charset = '{}[],:.0123456789"abcefgipq'
        seen = { (0, 0) }
        todo = [(0, 0)]
        while todo:
            state = todo.pop()
            for ch in charset:
                nxt = self._step(state, ch)
                if nxt is not None and nxt not in seen:
                    seen.add(nxt); todo.append(nxt)
        return sorted(seen, key=str)

    def _allowed_for_state(self, state):
        allowed = []
        for tid, text in self.token_text.items():
            cur = state
            for ch in text:
                cur = self._step(cur, ch)
                if cur is None:
                    break
            if cur is not None:
                allowed.append(tid)
        if state == self.final_state:
            allowed.extend(self.eos_ids)
        return sorted(set(allowed))

    def _state_for_prefix(self, text):
        cur = (0, 0)
        for ch in text:
            cur = self._step(cur, ch)
            if cur is None:
                raise RuntimeError(f"Core-ESR grammar dead-end at prefix: {text!r}")
        return cur

    def allowed(self, prefix: str) -> list[int]:
        if prefix in self.cache:
            return self.cache[prefix]
        state = self._state_for_prefix(prefix)
        allowed = self.allowed_by_state.get(state, [])
        if not allowed:
            raise RuntimeError(f"Core-ESR grammar dead-end at prefix: {prefix!r}")
        self.cache[prefix] = allowed
        return self.cache[prefix]

    def callback(self, input_ids, prompt_len: int, batch_id: int):
        # transformers passes the current sentence as a 1-D tensor here;
        # batch_id is retained in the signature for the HF callback contract.
        prefix = self.tok.decode(input_ids[prompt_len:].tolist(), skip_special_tokens=True)
        return self.allowed(prefix)


def seed_all(seed: int):
    random.seed(seed); np.random.seed(seed % (2**32 - 1)); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate(model, tok, grammar, prompts, *, seed, max_new_tokens, temperature, top_p, constrained):
    seed_all(seed)
    enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
    kwargs = dict(max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature, top_p=top_p,
                  pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id)
    if constrained:
        prompt_len = enc.input_ids.shape[1]
        kwargs["prefix_allowed_tokens_fn"] = lambda batch_id, ids: grammar.callback(ids, prompt_len, batch_id)
    with torch.inference_mode():
        out = model.generate(**enc, **kwargs)
    rows = []
    eos_ids = grammar.eos_ids
    for i in range(len(prompts)):
        ids = out[i, enc.input_ids.shape[1]:].detach().cpu().tolist()
        eos_pos = next((j for j, x in enumerate(ids) if int(x) in eos_ids), None)
        actual = ids if eos_pos is None else ids[:eos_pos + 1]
        rows.append({
            "raw_output": tok.decode(actual[:-1] if eos_pos is not None else actual, skip_special_tokens=True).strip(),
            "generated_tokens": len(actual),
            "eos_completed": eos_pos is not None,
            "truncated": eos_pos is None and len(actual) >= max_new_tokens,
        })
    return rows


def field_flags(raw: str) -> dict:
    try:
        obj = json.loads(raw)
    except Exception:
        return {"belief_valid": False, "game_valid": False, "plan_valid": False}
    flags = {"belief_valid": False, "game_valid": False, "plan_valid": False}
    b = obj.get("b") if isinstance(obj, dict) else None
    if isinstance(b, list) and len(b) == 6:
        try:
            x = [float(v) for v in b]
            flags["belief_valid"] = all(np.isfinite(x)) and all(0 <= v <= 1 for v in x) and abs(sum(x) - 1) <= 1e-5
        except Exception:
            pass
    g = obj.get("g") if isinstance(obj, dict) else None
    flags["game_valid"] = isinstance(g, list) and len(g) == 4 and all(isinstance(v, (int, float)) and int(v) in range(3) for v in g)
    p = obj.get("p") if isinstance(obj, dict) else None
    flags["plan_valid"] = isinstance(p, list) and len(p) == 3 and all(isinstance(v, (int, float)) and int(v) in range(6) for v in p)
    return flags


def summarize(rows: list[dict], contexts, tables, registry, k: int) -> dict:
    frame = pd.DataFrame(rows)
    flags = [field_flags(x["raw_output"]) for x in rows]
    for key in ("belief_valid", "game_valid", "plan_valid"):
        frame[key] = [x[key] for x in flags]
    audits = []
    for row in rows:
        a = parse_interface(row["raw_output"], row["eos_completed"], row["truncated"], registry)
        audits.append(a)
    frame["strict_valid"] = [x["strict_valid"] for x in audits]
    frame["action_id"] = [x["parsed"]["a"] if x["parsed"] is not None else np.nan for x in audits]
    group_rows = []
    for gi in sorted(frame.group_index.unique()):
        g = frame[frame.group_index == gi]
        valid = g[g.strict_valid]
        snap = g.iloc[0]
        values = [expected_payoff(contexts[str(snap.example_id)], tables, i) for i in range(len(ACTION_VALUES))]
        utilities = [float(values[int(x)]) for x in valid.action_id.dropna()]
        actions = [int(x) for x in valid.action_id.dropna()]
        group_rows.append({
            "group_index": int(gi), "n_strict_valid": int(len(valid)),
            "unique_action_count": len(set(actions)), "action_variation": len(set(actions)) > 1,
            "utility_std": float(np.std(utilities)) if utilities else 0.0,
            "utility_range": float(max(utilities) - min(utilities)) if utilities else 0.0,
            "economic_variation": bool(np.std(utilities) > 1e-8),
        })
    gf = pd.DataFrame(group_rows)
    return {
        "rows": int(len(frame)), "groups": int(frame.group_index.nunique()),
        "strict_valid_rate": float(frame.strict_valid.mean()),
        "eos_rate": float(frame.eos_completed.mean()), "truncated_rate": float(frame.truncated.mean()),
        "belief_valid_rate": float(frame.belief_valid.mean()), "game_valid_rate": float(frame.game_valid.mean()),
        "plan_valid_rate": float(frame.plan_valid.mean()),
        "action_variation_groups": int(gf.action_variation.sum()),
        "economic_variation_groups": int(gf.economic_variation.sum()),
        "mean_utility_std": float(gf.utility_std.mean()),
        "mean_utility_range": float(gf.utility_range.mean()),
        "group_audit": gf.to_dict("records"),
        "rows_preview": frame[["group_index", "candidate_index", "raw_output", "strict_valid", "eos_completed", "truncated", "belief_valid", "game_valid", "plan_valid"]].to_dict("records"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft-run", type=Path, required=True)
    ap.add_argument("--free-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    free = pd.read_parquet(args.free_dir / "K4_CANDIDATES.parquet")
    if len(free) != 256 or free.group_index.nunique() != 64:
        raise RuntimeError("expected frozen 64 x K4 FREE artifact")
    manifest = json.loads((args.sft_run / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
    model_name = str(manifest["model"])
    registry = load_registry(CORE_ROOT / "enum_registry.json")
    contexts, tables = load_context("TRAIN"), load_payoff_tables("TRAIN")
    snapshots = select_snapshots()
    tok = AutoTokenizer.from_pretrained(args.sft_run / "adapter", trust_remote_code=True)
    tok.pad_token = tok.pad_token or tok.eos_token; tok.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(model_name, device_map={"": 0}, trust_remote_code=True, torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(base, args.sft_run / "adapter").eval()
    grammar = CoreESRGrammar(tok)
    grammar_rows = []
    for gi, snap in enumerate(snapshots.itertuples(index=False)):
        seed = int(manifest.get("config", {}).get("seed", 42)) * 100000 + gi * 100
        generated = generate(model, tok, grammar, [str(snap.prompt)] * 4, seed=seed,
                             max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                             top_p=args.top_p, constrained=True)
        for ci, out in enumerate(generated):
            grammar_rows.append({"group_index": gi, "candidate_index": ci, "example_id": str(snap.example_id),
                                 "physical_state_id": str(snap.physical_state_id), "split": "TRAIN",
                                 "sampling_batch_seed": seed, "sampling_seed": seed + ci,
                                 "temperature": args.temperature, "top_p": args.top_p,
                                 "max_new_tokens": args.max_new_tokens, **out})
    grammar_frame = pd.DataFrame(grammar_rows)
    grammar_frame.to_parquet(args.output / "K4_GRAMMAR_CANDIDATES.parquet", index=False)
    grammar_summary = summarize(grammar_rows, contexts, tables, registry, 4)
    free_rows = []
    for row in free.itertuples(index=False):
        free_rows.append({"group_index": int(row.group_index), "candidate_index": int(row.candidate_index),
                          "example_id": str(row.example_id), "raw_output": str(row.raw_output),
                          "generated_tokens": int(row.generated_tokens), "eos_completed": bool(row.eos_completed),
                          "truncated": bool(row.truncated)})
    free_summary = summarize(free_rows, contexts, tables, registry, 4)
    result = {
        "protocol": "Paper9Bus-Power-GV-GRPO-v3", "stage": "K4_INTERFACE_RECOVERY",
        "status_before_recovery": "FAIL_INTERFACE_VALIDITY",
        "free": {"source": str((args.free_dir / "K4_CANDIDATES.parquet").resolve()), "decoding": "existing frozen FREE artifact", **free_summary},
        "grammar": {"decoding": "CoreESRGrammar prefix_allowed_tokens_fn; no post-generation repair", **grammar_summary},
        "paired_design": {"same_adapter": True, "same_prompts": True, "same_train_snapshots": True, "groups": 64, "k": 4,
                          "same_sampling_seeds": True, "temperature": args.temperature, "top_p": args.top_p,
                          "max_new_tokens": args.max_new_tokens, "only_decoding_constraint_changed": True,
                          "forced_action_diversity": False, "payoff": "direct expected_profit from frozen TRAIN cell bank"},
        "decision_rule": {
            "pass": "grammar strict_valid_rate >= 0.95 and economic variation is preserved",
            "semantic_failure": "otherwise, if semantic field errors remain after constrained decoding, FAIL_SEMANTIC_INTERFACE_SFT",
            "current": "pending metric evaluation",
        },
        "grpo_started": False, "final_accessed": False,
        "outputs": ["K4_GRAMMAR_CANDIDATES.parquet", "K4_FREE_VS_GRAMMAR_AUDIT.json", "K4_INTERFACE_RECOVERY_REPORT_CN.md"],
    }
    grammar_valid = grammar_summary["strict_valid_rate"] >= 0.95
    econ_preserved = grammar_summary["economic_variation_groups"] >= 2
    if grammar_valid and econ_preserved:
        result["decision_rule"]["current"] = "PASS_INTERFACE_VIA_CONSTRAINED_DECODING"
    elif (grammar_summary["belief_valid_rate"] < 0.95 or grammar_summary["game_valid_rate"] < 0.95 or grammar_summary["plan_valid_rate"] < 0.95):
        result["decision_rule"]["current"] = "FAIL_SEMANTIC_INTERFACE_SFT"
    else:
        result["decision_rule"]["current"] = "FAIL_INTERFACE_VALIDITY"
    (args.output / "K4_FREE_VS_GRAMMAR_AUDIT.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    report = ["# K4 接口恢复诊断报告", "", "## 结论", "", f"- 冻结原结论：`FAIL_INTERFACE_VALIDITY`。", f"- FREE strict validity：`{free_summary['strict_valid_rate']:.4f}`。", f"- Core-ESR Grammar strict validity：`{grammar_summary['strict_valid_rate']:.4f}`。", f"- 恢复判定：`{result['decision_rule']['current']}`。", "", "## 配对设计", "", "同一 Qwen3-1.7B SFT adapter、同一 TRAIN 64 physical states、K=4、同一 seed/temperature/top-p，仅增加 Core-ESR 前缀语法约束；没有后处理修复，也没有强制动作多样性。经济值来自冻结 TRAIN cell bank 的 direct expected_profit。", "", "## 指标对照", "", "| 指标 | FREE | Core-ESR Grammar |", "|---|---:|---:|", f"| strict validity | {free_summary['strict_valid_rate']:.4f} | {grammar_summary['strict_valid_rate']:.4f} |", f"| EOS rate | {free_summary['eos_rate']:.4f} | {grammar_summary['eos_rate']:.4f} |", f"| truncation rate | {free_summary['truncated_rate']:.4f} | {grammar_summary['truncated_rate']:.4f} |", f"| belief validity | {free_summary['belief_valid_rate']:.4f} | {grammar_summary['belief_valid_rate']:.4f} |", f"| game validity | {free_summary['game_valid_rate']:.4f} | {grammar_summary['game_valid_rate']:.4f} |", f"| plan validity | {free_summary['plan_valid_rate']:.4f} | {grammar_summary['plan_valid_rate']:.4f} |", f"| action-variation groups | {free_summary['action_variation_groups']}/64 | {grammar_summary['action_variation_groups']}/64 |", f"| economic-variation groups | {free_summary['economic_variation_groups']}/64 | {grammar_summary['economic_variation_groups']}/64 |", "", "本诊断不启动 GRPO，不访问 DEV/FINAL，不修改 adapter、已有 K4 结果、数据分区或 cell bank。"]
    (args.output / "K4_INTERFACE_RECOVERY_REPORT_CN.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"decision": result["decision_rule"]["current"], "free_valid": free_summary["strict_valid_rate"], "grammar_valid": grammar_summary["strict_valid_rate"], "grammar_cache": len(grammar.cache)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
