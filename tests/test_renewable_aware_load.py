import json

import numpy as np
import pandas as pd

from paper9bus_gv_grpo.renewable_aware import (
    FROZEN_CASE9_LOAD_MW,
    build_gross_load_bridge,
    fit_gross_load_mapping,
)


def test_gross_load_mapping_is_train_only_and_monotone():
    frame = pd.DataFrame(
        [
            {"split": "TRAIN", "target_he": 1, "hourly_load_mw": json.dumps([100.0])},
            {"split": "TRAIN", "target_he": 1, "hourly_load_mw": json.dumps([200.0])},
            {"split": "DEV", "target_he": 1, "hourly_load_mw": json.dumps([1000.0])},
        ]
    )
    mapping, stats = fit_gross_load_mapping(frame)
    assert stats["rows"] == 2
    assert mapping.reference_train_median_mw == 150.0
    assert mapping.alpha(200.0) > mapping.alpha(100.0)
    assert mapping.alpha(1000.0) == 1.5


def test_bridge_preserves_frozen_ieee9_load_pattern_and_disables_renewables():
    frame = pd.DataFrame(
        [
            {
                "split": "TRAIN",
                "target_he": 1,
                "hourly_load_mw": json.dumps([150.0]),
                "target_local_date": "2020-01-01",
                "target_hour_utc": "2020-01-01 05:00:00+00:00",
                "public_state_hash": "abc",
            }
        ]
    )
    mapping, _ = fit_gross_load_mapping(frame)
    bridge = build_gross_load_bridge(frame, mapping)
    row = bridge.iloc[0]
    loads = np.asarray([row[f"load_bus_{i}_mw"] for i in range(1, 10)])
    assert np.allclose(loads / loads.sum(), FROZEN_CASE9_LOAD_MW / FROZEN_CASE9_LOAD_MW.sum())
    assert row.renewable_input_mode == "DISABLED_FORECASTS_NOT_ADDED"
    assert not bool(row.wind_forecast_used)
    assert not bool(row.solar_forecast_used)
    assert not bool(row.net_load_forecast_used)
