import json

import pytest

from paper9bus_gv_grpo.public_state import (
    build_public_energy_state,
    canonical_json,
    contains_forbidden_key,
    fit_interpretation_rules,
    format_public_energy_state_prompt,
    state_hash,
)


def sample_row():
    return {
        "current_load_mw": 100.0,
        "load_factor": 0.8,
        "load_trend_value": 2.0,
        "load_history_last_4h_mw": [94.0, 96.0, 98.0, 100.0],
        "hourly_load_mw": [100.0, 110.0, 120.0, 115.0],
        "forecast_zone_load_mw": {"A": 40.0, "B": 60.0},
        "peak_load_mw": 120.0,
        "peak_hour": 3,
        "minimum_load_mw": 100.0,
        "daily_energy_mwh": 445.0,
        "max_up_ramp_mw_per_h": 20.0,
        "max_down_ramp_mw_per_h": -5.0,
        "forecast_change_vs_current_pct": 10.0,
        "historical_lmp_mean": 35.0,
        "historical_lmp_min": 30.0,
        "historical_lmp_max": 40.0,
        "historical_lmp_spread": 10.0,
        "load_last_4h_mw": [94.0, 96.0, 98.0, 100.0],
        "lmp_last_4h": [20.0, 21.0, 22.0, 23.0],
        "binding_branch_count": 1,
        "binding_event_count": 1,
        "network_signal": 1,
    }


def test_deterministic_state_card_serialization():
    rows = [sample_row(), {**sample_row(), "current_load_mw": 110.0}]
    rules = fit_interpretation_rules(rows)
    a = build_public_energy_state(rows[0], rules)
    b = build_public_energy_state(rows[0], rules)
    assert canonical_json(a) == canonical_json(b)
    assert state_hash(a) == state_hash(b)
    assert format_public_energy_state_prompt(a) == format_public_energy_state_prompt(b)


def test_train_threshold_reproducibility():
    rules_a = fit_interpretation_rules([sample_row(), {**sample_row(), "current_load_mw": 200.0}])
    rules_b = fit_interpretation_rules([sample_row(), {**sample_row(), "current_load_mw": 200.0}])
    assert canonical_json(rules_a) == canonical_json(rules_b)
    assert rules_a["fit_split"] == "TRAIN"


def test_hidden_fields_cannot_enter_public_state_card():
    rules = fit_interpretation_rules([sample_row()])
    with pytest.raises(ValueError):
        build_public_energy_state({**sample_row(), "payoff": 10.0}, rules)
    assert contains_forbidden_key({"hidden_opponent_state": 1}) == ["hidden_opponent_state"]


def test_future_realization_cannot_enter_forecast_features():
    rules = fit_interpretation_rules([sample_row()])
    with pytest.raises(ValueError):
        build_public_energy_state({**sample_row(), "future_realized_load_mw": 999.0}, rules)


def test_net_load_is_only_derived_when_all_components_exist():
    row = {**sample_row(), "wind_mw": [10.0, 12.0, 14.0, 16.0], "solar_mw": [1.0, 2.0, 3.0, 4.0]}
    rules = fit_interpretation_rules([row])
    card = build_public_energy_state(row, rules)
    assert "renewable_forecast" in card
    assert card["renewable_forecast"]["net_load_mw"] == [89.0, 96.0, 103.0, 95.0]
    assert card["renewable_forecast"]["net_load_peak_mw"] == 103.0
    incomplete = {**sample_row(), "wind_mw": [10.0, 12.0, 14.0, 16.0]}
    assert "net_load_mw" not in build_public_energy_state(incomplete, rules).get("renewable_forecast", {})


def test_history_chronology_and_split_integrity_are_explicit():
    row = sample_row()
    rules = fit_interpretation_rules([row])
    card = build_public_energy_state(row, rules)
    assert card["recent_history"]["load_last_4h_mw"][-1] == 100.0
    assert "final_accessed" not in card


def test_prompt_hash_changes_when_public_value_changes():
    rules = fit_interpretation_rules([sample_row(), {**sample_row(), "current_load_mw": 200.0}])
    a = build_public_energy_state(sample_row(), rules)
    b = build_public_energy_state({**sample_row(), "current_load_mw": 101.0}, rules)
    assert state_hash(a) != state_hash(b)
