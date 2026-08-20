"""Validated 24-hour day-ahead bid schedule interface."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from typing import Any, Sequence

import numpy as np


HORIZON_HOURS = 24
G1_TRUE_MC_USD_PER_MWH = 20.0
DEFAULT_MARKUP_GRID = (1.30, 1.50, 1.80)


@dataclass(frozen=True)
class DayAheadBidSchedule:
    target_date_utc: str
    decision_cutoff_utc: str
    bid_price_usd_per_mwh: tuple[float, ...]
    bid_markup_multiplier: tuple[float, ...]
    forecast_load_mw: tuple[float, ...]
    forecast_source: str
    generator_id: str = "G1"
    schema_version: str = "Paper9Bus-DayAhead-Bid-v1"

    def to_dict(self) -> dict[str, Any]:
        validate_schedule(self)
        return {
            "schema_version": self.schema_version,
            "generator_id": self.generator_id,
            "target_date_utc": self.target_date_utc,
            "decision_cutoff_utc": self.decision_cutoff_utc,
            "horizon_hours": list(range(24)),
            "bid_price_usd_per_mwh": list(self.bid_price_usd_per_mwh),
            "bid_markup_multiplier": list(self.bid_markup_multiplier),
            "forecast_load_mw": list(self.forecast_load_mw),
            "forecast_source": self.forecast_source,
            "price_definition": "G1 true marginal cost (20 USD/MWh) multiplied by submitted markup",
        }


def _finite_vector(values: Sequence[float], name: str) -> tuple[float, ...]:
    vector = tuple(float(x) for x in values)
    if len(vector) != HORIZON_HOURS or not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain exactly 24 finite values")
    return vector


def validate_schedule(schedule: DayAheadBidSchedule) -> None:
    if schedule.schema_version != "Paper9Bus-DayAhead-Bid-v1":
        raise ValueError("unsupported day-ahead bid schema")
    if schedule.generator_id != "G1":
        raise ValueError("this frozen benchmark exposes only G1 as focal bidder")
    _finite_vector(schedule.bid_price_usd_per_mwh, "bid_price_usd_per_mwh")
    _finite_vector(schedule.bid_markup_multiplier, "bid_markup_multiplier")
    _finite_vector(schedule.forecast_load_mw, "forecast_load_mw")
    for price, markup in zip(schedule.bid_price_usd_per_mwh, schedule.bid_markup_multiplier):
        if markup <= 0 or not np.isclose(price, G1_TRUE_MC_USD_PER_MWH * markup, atol=1e-8):
            raise ValueError("each bid price must equal 20 USD/MWh times its positive markup")
    if not isinstance(schedule.target_date_utc, str) or not schedule.target_date_utc:
        raise ValueError("target_date_utc is required")
    if not isinstance(schedule.decision_cutoff_utc, str) or not schedule.decision_cutoff_utc:
        raise ValueError("decision_cutoff_utc is required")


def make_schedule(
    target_date_utc: str,
    decision_cutoff_utc: str,
    forecast_load_mw: Sequence[float],
    markups: Sequence[float],
    *,
    forecast_source: str,
) -> DayAheadBidSchedule:
    forecast = _finite_vector(forecast_load_mw, "forecast_load_mw")
    multipliers = _finite_vector(markups, "bid_markup_multiplier")
    if any(x <= 0 for x in multipliers):
        raise ValueError("bid markups must be positive")
    schedule = DayAheadBidSchedule(
        target_date_utc=str(target_date_utc), decision_cutoff_utc=str(decision_cutoff_utc),
        bid_price_usd_per_mwh=tuple(G1_TRUE_MC_USD_PER_MWH * x for x in multipliers),
        bid_markup_multiplier=multipliers, forecast_load_mw=forecast, forecast_source=str(forecast_source),
    )
    validate_schedule(schedule)
    return schedule


def schedule_from_json(payload: str | dict[str, Any]) -> DayAheadBidSchedule:
    obj = json.loads(payload) if isinstance(payload, str) else dict(payload)
    schedule = DayAheadBidSchedule(
        target_date_utc=str(obj["target_date_utc"]), decision_cutoff_utc=str(obj["decision_cutoff_utc"]),
        bid_price_usd_per_mwh=tuple(obj["bid_price_usd_per_mwh"]), bid_markup_multiplier=tuple(obj["bid_markup_multiplier"]),
        forecast_load_mw=tuple(obj["forecast_load_mw"]), forecast_source=str(obj.get("forecast_source", "unspecified")),
        generator_id=str(obj.get("generator_id", "G1")), schema_version=str(obj.get("schema_version", "")),
    )
    validate_schedule(schedule)
    return schedule
