from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..paths import PROJECT_ROOT


SNAPSHOT = PROJECT_ROOT / "data/action_space_v2_simulator/case_snapshot/frozen_case9_blv.json"


@dataclass(frozen=True)
class FrozenCase:
    case_id: str
    base_mva: float
    reference_bus: int
    buses: tuple[dict, ...]
    generators: tuple[dict, ...]
    branches: tuple[dict, ...]
    load_base_mw: np.ndarray
    source_sha256: dict[str, str]

    @property
    def bus_ids(self) -> tuple[int, ...]:
        return tuple(int(x["bus_i"]) for x in self.buses)

    @property
    def generator_ids(self) -> tuple[str, ...]:
        return tuple(str(x["generator_id"]) for x in self.generators)

    @property
    def branch_ids(self) -> tuple[int, ...]:
        return tuple(int(x["branch_id"]) for x in self.branches)

    def generator(self, generator_id: str) -> dict:
        return next(x for x in self.generators if x["generator_id"] == generator_id)


def snapshot_sha256(path: Path = SNAPSHOT) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_frozen_case(path: Path = SNAPSHOT) -> FrozenCase:
    obj = json.loads(path.read_text(encoding="utf-8"))
    buses = tuple(obj["buses"])
    generators = tuple(obj["generators"])
    branches = tuple(obj["branches"])
    bus_ids = [int(x["bus_i"]) for x in buses]
    if bus_ids != list(range(1, 10)):
        raise ValueError(f"unexpected IEEE-9 buses: {bus_ids}")
    if int(obj["reference_bus"]) not in bus_ids:
        raise ValueError("reference bus is not in the frozen case")
    if [x["generator_id"] for x in generators] != ["G1", "G2", "G3"]:
        raise ValueError("generator identity/order mismatch")
    if [int(x["branch_id"]) for x in branches] != list(range(1, 10)):
        raise ValueError("branch identity/order mismatch")
    base = np.zeros(9, dtype=np.float64)
    for row in buses:
        base[int(row["bus_i"]) - 1] = float(row["Pd"])
    expected = np.array([0, 0, 0, 0, 90, 0, 100, 0, 125], dtype=np.float64)
    if not np.array_equal(base, expected):
        raise ValueError("snapshot load vector does not match frozen case")
    if not np.isclose(float(obj["baseMVA"]), 100.0):
        raise ValueError("baseMVA mismatch")
    return FrozenCase(str(obj["case_id"]), float(obj["baseMVA"]), int(obj["reference_bus"]), buses,
                      generators, branches, base, dict(obj["provenance"]["source_files"]))


def scaled_load(load_base_mw: np.ndarray, alpha: float) -> np.ndarray:
    load = np.asarray(load_base_mw, dtype=np.float64) * float(alpha)
    if np.any(load < 0) or not np.all(np.isfinite(load)):
        raise ValueError("invalid load state")
    return load
