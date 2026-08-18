from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
CORE_ROOT = DATA_ROOT / "core"
BENCHMARK_ROOT = DATA_ROOT / "benchmark"
RUNS_ROOT = PROJECT_ROOT / "runs"

def split_file(split: str) -> Path:
    split = split.upper()
    if split not in {"TRAIN", "DEV"}:
        raise ValueError("split must be TRAIN or DEV")
    return CORE_ROOT / ("train.parquet" if split == "TRAIN" else "dev.parquet")

def context_file(split: str) -> Path:
    split = split.upper()
    if split not in {"TRAIN", "DEV"}:
        raise ValueError("split must be TRAIN or DEV")
    return CORE_ROOT / ("train_context.parquet" if split == "TRAIN" else "dev_context.parquet")

