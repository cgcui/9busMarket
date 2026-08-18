from __future__ import annotations

import json
import math
from typing import Any, Mapping

STATES = ("1.00", "1.05", "1.10", "1.20", "1.30", "1.50")
ACTION_VALUES = (1.00, 1.05, 1.10, 1.20, 1.30, 1.50)
PRESSURE_IDS = (0, 1, 2)
CF_IDS = (0, 1, 2)
INTENT_IDS = (0, 1, 2, 3)

def load_registry(path):
    return json.loads(path.read_text(encoding="utf-8"))

def parse_core(text_or_obj: str | Mapping[str, Any], registry: Mapping[str, Any]) -> dict[str, Any]:
    obj = json.loads(text_or_obj) if isinstance(text_or_obj, str) else dict(text_or_obj)
    if set(obj) != {"e", "b", "g", "cf", "i", "a", "p", "q"}:
        raise ValueError("keys_mismatch")
    if not isinstance(obj["e"], list) or len(obj["e"]) > 8 or any(int(x) not in range(8) for x in obj["e"]):
        raise ValueError("evidence_invalid")
    b = [float(x) for x in obj["b"]]
    if len(b) != 6 or any(not math.isfinite(x) or x < 0 or x > 1 for x in b) or abs(sum(b) - 1.0) > 1e-5:
        raise ValueError("belief_invalid")
    if len(obj["g"]) != 4 or any(int(x) not in PRESSURE_IDS for x in obj["g"]):
        raise ValueError("game_invalid")
    if len(obj["cf"]) != 2 or any(int(x) not in CF_IDS for x in obj["cf"]):
        raise ValueError("counterfactual_invalid")
    if int(obj["i"]) not in INTENT_IDS or int(obj["a"]) not in range(6):
        raise ValueError("intent_action_invalid")
    if len(obj["p"]) != 3 or any(int(x) not in range(6) for x in obj["p"]):
        raise ValueError("plan_invalid")
    q = float(obj["q"])
    if not math.isfinite(q) or not 0 <= q <= 1:
        raise ValueError("confidence_invalid")
    return {"e": [int(x) for x in obj["e"]], "b": b, "g": [int(x) for x in obj["g"]],
            "cf": [int(x) for x in obj["cf"]], "i": int(obj["i"]), "a": int(obj["a"]),
            "p": [int(x) for x in obj["p"]], "q": q}

def action_value(action_id: int) -> float:
    return ACTION_VALUES[int(action_id)]

