import json

import pandas as pd
import pytest

from paper9bus_gv_grpo.audit_hardening import (
    PAPER9BUS_PUBLIC_STATE_X1_V1,
    canonical_prompt_hash,
    compute_conflict_metrics,
    prepare_create_only_directory,
    repetition_flags_hardened,
    strict_one_to_one_merge,
    validate_exact_prompt_identity,
    validate_paper9bus_x1_card,
)
from paper9bus_gv_grpo.public_state_envs import build_paper9bus_public_state, fit_paper9bus_interpretation_rules
from paper9bus_gv_grpo.schema import ACTION_VALUES, load_registry


ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


def target(action=0, belief=None, plan=None):
    return {
        "e": [],
        "b": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0] if belief is None else belief,
        "g": [0, 0, 0, 0],
        "cf": [0, 0],
        "i": 0,
        "a": action,
        "p": [0, 0, 0] if plan is None else plan,
        "q": 0.5,
    }


def row(example_id, prompt="x", value=None):
    return {
        "example_id": example_id,
        "prompt": prompt,
        "prompt_hash": canonical_prompt_hash(prompt),
        "target_json": json.dumps(target() if value is None else value),
    }


def test_canonical_gold_target_is_not_repetition():
    gold = json.dumps(target(belief=[1 / 6] * 6), separators=(",", ":"))
    assert repetition_flags_hardened(gold)["repetition_loop_indicator"] is False


def test_actual_generation_loop_is_repetition():
    loop = "temperature is high\ntemperature is high\ntemperature is high"
    assert repetition_flags_hardened(loop)["repetition_loop_indicator"] is True


def test_conflict_metrics_are_computed_from_data():
    metrics = compute_conflict_metrics([row("a"), row("b")])
    assert metrics["same_input_incompatible_target_count"] == 0
    assert metrics["action_conflicts"] == 0
    assert metrics["plan_conflicts"] == 0


def test_injected_action_conflict_is_detected():
    metrics = compute_conflict_metrics([row("a"), row("b", value=target(action=1))])
    assert metrics["same_input_incompatible_target_count"] == 1
    assert metrics["action_conflicts"] == 1


def test_injected_belief_conflict_is_detected():
    metrics = compute_conflict_metrics([row("a"), row("b", value=target(belief=[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]))])
    assert metrics["belief_serialization_conflicts"] == 1


def test_injected_plan_conflict_is_detected():
    metrics = compute_conflict_metrics([row("a"), row("b", value=target(plan=[1, 1, 1]))])
    assert metrics["plan_conflicts"] == 1


@pytest.mark.parametrize("left_dup,right_dup", [(True, False), (False, True)])
def test_duplicate_side_fails(left_dup, right_dup):
    left = pd.DataFrame({"id": [1, 1] if left_dup else [1, 2]})
    right = pd.DataFrame({"id": [1, 2] if not right_dup else [1, 1]})
    with pytest.raises(ValueError, match="duplicate"):
        strict_one_to_one_merge(left, right, key="id")


def test_missing_key_and_many_to_many_fail():
    with pytest.raises(ValueError, match="key mismatch"):
        strict_one_to_one_merge(pd.DataFrame({"id": [1, 2]}), pd.DataFrame({"id": [1, 3]}), key="id")
    with pytest.raises(ValueError, match="duplicate"):
        strict_one_to_one_merge(pd.DataFrame({"id": [1, 2]}), pd.DataFrame({"id": [1, 1]}), key="id")


def test_existing_frozen_directory_is_not_deleted(tmp_path):
    destination = tmp_path / "frozen"
    destination.mkdir()
    sentinel = destination / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        prepare_create_only_directory(destination)
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_rebuild_requires_new_destination(tmp_path):
    destination = tmp_path / "new"
    prepare_create_only_directory(destination)
    with pytest.raises(FileExistsError):
        prepare_create_only_directory(destination)


def test_x1_contains_total_load_and_does_not_contain_bus_loads():
    row_data = {"total_load_mw": 100.0, **{f"load_bus_{i}": float(i) for i in range(1, 10)}, "observation_json": json.dumps({"g1_dispatch": 10.0, "g1_lmp": 20.0, "system_lmp_mean": 20.0, "system_lmp_min": 10.0, "system_lmp_max": 30.0, "system_lmp_spread": 20.0, "binding_branch_count": 1, "max_branch_utilization": 0.9})}
    rules = fit_paper9bus_interpretation_rules([row_data])
    card = build_paper9bus_public_state(row_data, rules)
    assert card["current_energy_state"]["total_load_mw"] == 100.0
    assert "bus_loads_mw" not in card["current_energy_state"]
    assert validate_paper9bus_x1_card(card)["feature_set_id"] == PAPER9BUS_PUBLIC_STATE_X1_V1


def test_x1_exact_feature_registry_match():
    card = {"schema_version": "Paper9Bus-Public-State-v1", "environment": "Paper9Bus-3Gen-C3", "current_energy_state": {"total_load_mw": 1.0}}
    assert validate_paper9bus_x1_card(card)["exact"] is True
    card["current_energy_state"]["bus_loads_mw"] = [1.0] * 9
    with pytest.raises(ValueError, match="registry"):
        validate_paper9bus_x1_card(card)
    config = json.loads((ROOT / "configs/public_feature_registry_paper9bus_x1_v1.json").read_text(encoding="utf-8"))
    assert set(config["included_fields"]) == set(validate_paper9bus_x1_card({"schema_version": "Paper9Bus-Public-State-v1", "environment": "Paper9Bus-3Gen-C3", "current_energy_state": {"total_load_mw": 1.0}})["registry_fields"])


def test_missing_example_id_fails():
    with pytest.raises(ValueError, match="example_id"):
        validate_exact_prompt_identity("wanted", {"example_id": "other", "prompt": "x", "prompt_hash": "bad"})


def test_same_physical_state_wrong_prompt_is_rejected():
    with pytest.raises(ValueError, match="prompt_hash"):
        validate_exact_prompt_identity("wanted", {"example_id": "wanted", "prompt": "x", "prompt_hash": "wrong"})


def test_exact_prompt_hash_required():
    item = {"example_id": "wanted", "prompt": "x", "prompt_hash": canonical_prompt_hash("x")}
    identity = validate_exact_prompt_identity("wanted", item)
    assert identity["requested_example_id"] == identity["matched_example_id"] == "wanted"


def test_t41_target_identifiability_recomputation():
    from scripts.run_public_state_v1_t41_t43 import audit_targets

    registry = load_registry(ROOT / "data/core/enum_registry.json")
    value_table = {float(value): float(-index) for index, value in enumerate(ACTION_VALUES)}
    synthetic = {"example_id": "x", "physical_state_id": "s", "split": "TRAIN", "prompt_hash": "p", "class_members_json": "[1.0]", "target_json": json.dumps(target(action=0)), "observation_json": "{}"}
    audit, _, _ = audit_targets({"TRAIN": [synthetic]}, registry, {"TRAIN": {("s", 1.0): value_table}})
    assert audit["classification"] == "PASS_TARGET_IDENTIFIABILITY_V1"


def _economic_rows(actions, group_index=0, example_id="x"):
    rows = []
    for index, action in enumerate(actions):
        rows.append({"group_index": group_index, "candidate_index": index, "example_id": example_id, "physical_state_id": "s", "split": "TRAIN", "raw_output": json.dumps(target(action=action)), "eos_completed": True, "truncated": False})
    return rows


def test_t46_group_economic_audit_and_hard_stop():
    from scripts.run_public_state_v1_t45_t46 import economic_gate

    registry = load_registry(ROOT / "data/core/enum_registry.json")
    values = {float(value): float(index) for index, value in enumerate(ACTION_VALUES)}
    contexts = {"x": {"physical_state_id": "s", "class_members_json": "[1.0]"}}
    tables = {("s", 1.0): values}
    varied = _economic_rows([0, 1, 2, 3], group_index=0, example_id="x") + _economic_rows([1, 2, 3, 4], group_index=1, example_id="y")
    contexts["y"] = contexts["x"]
    passed, _ = economic_gate(varied, contexts, tables, registry, 4, 1.0)
    failed, _ = economic_gate(_economic_rows([0, 0, 0, 0], group_index=0, example_id="x") + _economic_rows([1, 1, 1, 1], group_index=1, example_id="y"), contexts, tables, registry, 4, 1.0)
    assert passed["classification"] == "GRPO_READY_SIGNAL_ONLY"
    assert failed["classification"] == "FAIL_TRUE_ECONOMIC_SIGNAL"


def test_t45_interface_result_aggregation():
    from scripts.run_public_state_v1_t45_t46 import dev_summary

    registry = load_registry(ROOT / "data/core/enum_registry.json")
    generated = {"raw_output": json.dumps(target()), "generated_tokens": 20, "eos_completed": True, "truncated": False}
    summary = dev_summary([generated], [target()], registry)
    assert summary["rows"] == 1
    assert summary["strict_valid_rate"] == 1.0
