from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .paths import CORE_ROOT, PROJECT_ROOT
from .prepare import main as prepare_main

FORBIDDEN_PROMPT_TERMS = ("k_g3", "dispatch_g3", "profit_g3", "oracle", "payoff", "FINAL", "hidden")

def main() -> int:
    prepare_main()
    train = pd.read_parquet(CORE_ROOT / "train.parquet")
    dev = pd.read_parquet(CORE_ROOT / "dev.parquet")
    prompts = "\n".join(train.prompt.tolist() + dev.prompt.tolist()).lower()
    hits = [term for term in FORBIDDEN_PROMPT_TERMS if term.lower() in prompts]
    report = {"protocol": "Paper9Bus-Power-GV-GRPO-v3", "train_examples": len(train), "dev_examples": len(dev),
              "prompt_forbidden_hits": hits, "final_accessed": False, "status": "PASS" if not hits else "FAIL"}
    out = PROJECT_ROOT / "audit_report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if not hits else 1

if __name__ == "__main__":
    raise SystemExit(main())

