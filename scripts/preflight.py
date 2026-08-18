#!/usr/bin/env python3
"""Check installation and the frozen data boundary before a run."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

REQUIRED = ("numpy", "pandas", "pyarrow", "yaml", "transformers", "accelerate", "peft")

def main() -> int:
    missing = []
    versions = {}
    for name in REQUIRED:
        try:
            mod = importlib.import_module(name)
            versions[name] = getattr(mod, "__version__", "imported")
        except Exception as exc:
            missing.append(f"{name}: {exc}")
    print(f"python: {sys.version.split()[0]}")
    print(f"project_root: {ROOT}")
    print(f"dependencies: {versions}")
    try:
        import torch
        print(f"torch: {torch.__version__}")
        print(f"cuda_available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"gpu: {torch.cuda.get_device_name(0)}")
    except Exception as exc:
        missing.append(f"torch: {exc}")
    if missing:
        print("missing_or_invalid:")
        for item in missing:
            print(f"  - {item}")
        return 1
    print("preflight: PASS")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

