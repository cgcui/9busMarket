import json
import importlib.util
from pathlib import Path

from paper9bus_gv_grpo.action_space_v2_simulator.bidding import crossing_multiplier
from paper9bus_gv_grpo.action_space_v2_simulator.case_data import load_frozen_case


def _probe_module():
    path = Path(__file__).resolve().parents[1] / "scripts/run_action_space_v2_renewable_train_probe.py"
    spec = importlib.util.spec_from_file_location("action_space_v2_renewable_probe", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_probe_grid_contains_regular_crossing_and_upper_bound_points():
    grid, spec = _probe_module().build_probe_grid(load_frozen_case())
    assert len(grid) == 43
    assert 1.00 in grid and 1.50 in grid and 2.30 in grid
    for k in (1.00, 1.05, 1.10, 1.20, 1.30, 1.50):
        cross = crossing_multiplier(load_frozen_case(), k)
        assert any(abs(x - (cross - 0.01)) < 1e-9 for x in grid)
        assert any(abs(x - cross) < 1e-9 for x in grid)
        assert any(abs(x - (cross + 0.01)) < 1e-9 for x in grid)
    assert spec["is_model_action_space"] is False


def test_probe_outputs_are_frozen_before_expected_value_selection():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/action_space_v2_renewable_train_probe_v1.json").read_text())
    assert config["train_only"] is True
    assert config["stop_rules"]["compact_action_selection"] is False
    assert config["stop_rules"]["sft_started"] is False
    assert config["stop_rules"]["grpo_started"] is False
