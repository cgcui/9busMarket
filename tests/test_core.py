import json
from pathlib import Path

from paper9bus_gv_grpo.paths import CORE_ROOT
from paper9bus_gv_grpo.schema import parse_core
from paper9bus_gv_grpo.reward import group_advantages

def test_core_schema_rejects_private_field():
    obj = {"e": [], "b": [1, 0, 0, 0, 0, 0], "g": [0, 0, 0, 0], "cf": [0, 0], "i": 0, "a": 0, "p": [0, 0, 0], "q": .5, "profit": 1}
    try:
        parse_core(obj, {})
    except ValueError as exc:
        assert "keys_mismatch" in str(exc)
    else:
        raise AssertionError("private field was accepted")

def test_group_advantages_ties_are_zero():
    assert group_advantages([1, 1, 1, 1]) == [0, 0, 0, 0]

def test_frozen_core_data_exists():
    assert (CORE_ROOT / "train.parquet").exists()
    assert (CORE_ROOT / "dev.parquet").exists()
    registry = json.loads((CORE_ROOT / "enum_registry.json").read_text(encoding="utf-8"))
    assert registry["final_accessed"] is False

