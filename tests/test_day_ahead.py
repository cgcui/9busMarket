import json

import pytest

from paper9bus_gv_grpo.day_ahead import make_schedule, schedule_from_json


def test_day_ahead_schedule_has_24_direct_prices():
    schedule = make_schedule("2024-06-02", "2024-06-01T10:00:00Z", [300.0] * 24, [1.30, 1.50, 1.80] * 8, forecast_source="test")
    assert len(schedule.bid_price_usd_per_mwh) == 24
    assert schedule.bid_price_usd_per_mwh[:3] == (26.0, 30.0, 36.0)
    assert schedule.to_dict()["horizon_hours"] == list(range(24))


def test_day_ahead_round_trip_and_rejects_wrong_length():
    schedule = make_schedule("2024-06-02", "2024-06-01T10:00:00Z", [300.0] * 24, [1.50] * 24, forecast_source="test")
    restored = schedule_from_json(json.dumps(schedule.to_dict()))
    assert restored == schedule
    with pytest.raises(ValueError, match="exactly 24"):
        make_schedule("2024-06-02", "2024-06-01T10:00:00Z", [300.0] * 23, [1.50] * 24, forecast_source="test")
