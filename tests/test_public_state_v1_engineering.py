import json
from pathlib import Path

import pandas as pd

from paper9bus_gv_grpo.public_state import canonical_json, contains_forbidden_key
from paper9bus_gv_grpo.public_state_envs import (
    build_iso2y_public_state,
    build_paper9bus_public_state,
    fit_paper9bus_interpretation_rules,
)

ROOT = Path(__file__).resolve().parents[1]


def paper_sample():
    return {
        "total_load_mw": 315.2,
        "load_bus_1": 10.0,
        "load_bus_2": 20.0,
        "load_bus_3": 30.0,
        "load_bus_4": 40.0,
        "load_bus_5": 50.0,
        "load_bus_6": 60.0,
        "load_bus_7": 40.0,
        "load_bus_8": 35.0,
        "load_bus_9": 30.2,
        "observation_json": json.dumps({"g1_dispatch": 87.4, "g1_lmp": 34.6, "system_lmp_mean": 32.8, "system_lmp_min": 29.1, "system_lmp_max": 36.7, "system_lmp_spread": 7.6, "binding_branch_count": 2, "max_branch_utilization": 0.96}),
    }


def test_environment_registries_are_split_and_statused():
    paper = json.loads((ROOT / "configs/public_feature_registry_paper9bus_v1.json").read_text(encoding="utf-8"))
    iso = json.loads((ROOT / "configs/public_feature_registry_iso2y_v1.json").read_text(encoding="utf-8"))
    assert paper["environment"] != iso["environment"]
    allowed = {"LEGAL_PUBLIC_DIRECT", "LEGAL_PUBLIC_DERIVED", "HIDDEN_PRIVATE", "ORACLE_DERIVED", "ADMIN_ONLY", "UNAVAILABLE"}
    assert all(field["status"] in allowed for field in paper["fields"].values())
    assert all(field["status"] in allowed for field in iso["fields"].values())


def test_paper9bus_minimal_builder_is_deterministic_and_has_no_forecast():
    row = paper_sample()
    rules = fit_paper9bus_interpretation_rules([row])
    a = build_paper9bus_public_state(row, rules)
    b = build_paper9bus_public_state(row, rules)
    assert canonical_json(a) == canonical_json(b)
    assert "day_ahead_load_forecast" not in a
    assert a["current_energy_state"]["total_load_mw"] == 315.2
    assert not contains_forbidden_key(a)


def test_iso2y_builder_omits_unavailable_renewable_and_generator_fields():
    row = {"current_load_mw": 100.0, "load_last_4h_mw": "[95,97,99,100]", "hourly_load_mw": "[100,101]", "forecast_zone_load_mw": "{}", "peak_load_mw": 101.0, "peak_hour": 2, "minimum_load_mw": 100.0, "daily_energy_mwh": 201.0, "max_up_ramp_mw_per_h": 1.0, "max_down_ramp_mw_per_h": 0.0, "forecast_change_vs_current_pct": 0.5, "historical_lmp_mean": 30.0, "historical_lmp_min": 29.0, "historical_lmp_max": 31.0, "historical_lmp_spread": 2.0, "lmp_last_4h": "[29,30,31,30]", "binding_branch_count": 0, "final_accessed": False}
    rules = json.loads((ROOT / "configs/public_interpretation_rules_v1.json").read_text(encoding="utf-8"))
    card = build_iso2y_public_state(row, rules)
    assert "renewable_forecast" not in card
    assert "own_generator" not in card
    assert not contains_forbidden_key(card)


def test_iso2y_forecast_publish_cutoff_is_zero_violations():
    frame = pd.read_parquet(ROOT / "data/public/isone_2y_public_energy_state.parquet")
    cutoff = pd.to_datetime(frame["decision_cutoff_utc"], utc=True)
    published = pd.to_datetime(frame["forecast_publish_utc"], utc=True)
    assert bool((published <= cutoff).all())
    assert bool(frame["final_accessed"].eq(False).all())


def test_paper9bus_x1_minimality_is_measurable():
    context = pd.read_parquet(ROOT / "data/core/train_context.parquet")
    bank = pd.read_parquet(ROOT / "data/benchmark/cell_bank.parquet")
    base = bank[(bank.k_g3 == 1.0) & (bank.k_g1 == 1.0)][["state_id", "total_load"]].drop_duplicates("state_id")
    frame = context.merge(base, left_on="physical_state_id", right_on="state_id", validate="many_to_one")
    frame["obs"] = frame.observation_json.map(json.loads).map(canonical_json)
    frame["x1"] = frame.apply(lambda r: canonical_json({"public_observation": json.loads(r.observation_json), "total_load_mw": round(float(r.total_load), 6)}), axis=1)
    targets = pd.read_parquet(ROOT / "data/core/train.parquet")[["example_id", "target_json"]].rename(columns={"target_json": "core_target_json"})
    frame = frame.merge(targets, on="example_id").assign(action=lambda x: x.core_target_json.map(lambda v: str(json.loads(v)["a"])))
    assert int(frame.groupby("obs").action.nunique().gt(1).sum()) == 1
    assert int(frame.groupby("x1").action.nunique().gt(1).sum()) == 0
