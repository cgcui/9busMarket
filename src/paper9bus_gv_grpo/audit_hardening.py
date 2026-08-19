"""Pure, freeze-safe helpers for the Paper9Bus audit pipeline."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd


PAPER9BUS_PUBLIC_STATE_X1_V1 = "PAPER9BUS_PUBLIC_STATE_X1_V1"
PAPER9BUS_PUBLIC_STATE_X2_AUDIT_V1 = "PAPER9BUS_PUBLIC_STATE_X2_AUDIT_V1"

# These are the possible semantic leaves of the frozen X1 card.  The
# runtime validator permits unavailable optional observations to be absent,
# but rejects every field outside this registry, especially bus loads.
PAPER9BUS_X1_FEATURE_REGISTRY = frozenset(
    {
        "schema_version",
        "environment",
        "current_energy_state.total_load_mw",
        "own_generator.dispatch_mw",
        "own_generator.own_lmp",
        "market.system_lmp_mean",
        "market.system_lmp_min",
        "market.system_lmp_max",
        "market.lmp_spread",
        "network.binding_branch_count",
        "network.max_branch_utilization",
        "public_interpretation.load_level",
        "public_interpretation.price_dispersion",
        "public_interpretation.network_stress",
        "public_interpretation.congestion_status",
    }
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_prompt_hash(prompt: str) -> str:
    # Frozen T41-T43 artifacts hash the training prompt plus its separator.
    return hashlib.sha256((str(prompt) + "\n").encode("utf-8")).hexdigest()


def _as_target(row: Mapping[str, Any]) -> Any:
    value = row.get("target_json", row.get("target"))
    if isinstance(value, str):
        return json.loads(value)
    return value


def _target_signature(value: Any) -> str:
    return canonical_json(value)


def compute_conflict_metrics(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute target conflicts conditioned on the exact visible prompt hash."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        prompt = row.get("prompt")
        prompt_hash = str(row.get("prompt_hash") or canonical_prompt_hash(str(prompt)))
        if prompt is not None and canonical_prompt_hash(str(prompt)) != prompt_hash:
            raise ValueError(f"prompt_hash mismatch for {row.get('example_id', '<unknown>')}")
        target = _as_target(row)
        if not isinstance(target, Mapping):
            raise ValueError(f"target must be an object for {row.get('example_id', '<unknown>')}")
        groups[prompt_hash].append({"row": row, "target": target})

    def conflicts(selector) -> list[dict[str, Any]]:
        out = []
        for prompt_hash, members in groups.items():
            signatures = {_target_signature(selector(x["target"])) for x in members}
            if len(signatures) > 1:
                out.append(
                    {
                        "prompt_hash": prompt_hash,
                        "example_ids": [str(x["row"].get("example_id", "")) for x in members],
                        "unique_values": len(signatures),
                    }
                )
        return out

    full = conflicts(lambda target: target)
    actions = conflicts(lambda target: target.get("a"))
    plans = conflicts(lambda target: target.get("p"))
    action_plans = conflicts(lambda target: {"a": target.get("a"), "p": target.get("p")})
    beliefs = conflicts(lambda target: target.get("b"))
    return {
        "visible_input_groups": len(groups),
        "same_input_incompatible_target_count": len(full),
        "action_conflicts": len(actions),
        "plan_conflicts": len(plans),
        "action_plan_conflicts": len(action_plans),
        "belief_serialization_conflicts": len(beliefs),
        "details": {
            "same_input_incompatible_target_preview": full[:20],
            "action_conflicts_preview": actions[:20],
            "plan_conflicts_preview": plans[:20],
            "action_plan_conflicts_preview": action_plans[:20],
            "belief_conflicts_preview": beliefs[:20],
        },
    }


def strict_one_to_one_merge(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    key: str,
    expected_rows: int | None = None,
    left_name: str = "left",
    right_name: str = "right",
    suffixes: tuple[str, str] = ("_left", "_right"),
) -> pd.DataFrame:
    """Merge two frozen tables only when their key sets align exactly."""
    for name, frame in ((left_name, left), (right_name, right)):
        if key not in frame.columns:
            raise ValueError(f"{name} is missing merge key {key!r}")
        if frame[key].isna().any():
            raise ValueError(f"{name} contains null merge keys")
        if frame[key].duplicated().any():
            raise ValueError(f"{name} contains duplicate merge keys")
    left_keys = set(left[key].tolist())
    right_keys = set(right[key].tolist())
    missing_right = left_keys - right_keys
    missing_left = right_keys - left_keys
    if missing_right or missing_left:
        raise ValueError(
            f"merge key mismatch: missing_from_{right_name}={len(missing_right)}, "
            f"missing_from_{left_name}={len(missing_left)}"
        )
    merged = left.merge(right, on=key, how="inner", validate="one_to_one", suffixes=suffixes)
    expected = len(left) if expected_rows is None else int(expected_rows)
    if len(merged) != expected or len(merged) != len(left) or len(merged) != len(right):
        raise ValueError(f"unexpected merge row count: left={len(left)}, right={len(right)}, merged={len(merged)}, expected={expected}")
    if merged[key].duplicated().any():
        raise ValueError("merged result contains duplicate keys")
    return merged


def prepare_create_only_directory(path: Path, *, allow_overwrite_development: bool = False) -> None:
    """Prepare a destination without touching existing evidence by default."""
    path = Path(path)
    if path.exists():
        if not allow_overwrite_development:
            raise FileExistsError(f"refusing to overwrite existing frozen destination: {path}")
        if not path.is_dir():
            raise FileExistsError(f"destination exists and is not a directory: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=False)


def _leaf_paths(value: Any, prefix: str = "") -> set[str]:
    if isinstance(value, Mapping):
        paths: set[str] = set()
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            paths.update(_leaf_paths(child, child_prefix))
        return paths
    return {prefix}


def validate_paper9bus_x1_card(card: Mapping[str, Any]) -> dict[str, Any]:
    """Enforce the exact frozen X1 feature registry at runtime."""
    if card.get("schema_version") != "Paper9Bus-Public-State-v1":
        raise ValueError("unexpected Paper9Bus public-state schema")
    if card.get("environment") != "Paper9Bus-3Gen-C3":
        raise ValueError("unexpected Paper9Bus environment")
    if "total_load_mw" not in card.get("current_energy_state", {}):
        raise ValueError("X1 requires current_energy_state.total_load_mw")
    actual = _leaf_paths(card)
    unexpected = sorted(actual - PAPER9BUS_X1_FEATURE_REGISTRY)
    if unexpected:
        raise ValueError(f"X1 feature registry violation: {unexpected}")
    return {
        "feature_set_id": PAPER9BUS_PUBLIC_STATE_X1_V1,
        "included_fields": sorted(actual),
        "registry_fields": sorted(PAPER9BUS_X1_FEATURE_REGISTRY),
        "exact": True,
    }


def validate_exact_prompt_identity(
    requested_example_id: str,
    item: Mapping[str, Any],
    *,
    expected_prompt_hash: str | None = None,
) -> dict[str, str]:
    """Require example id and prompt hash identity for K=4 reuse."""
    requested = str(requested_example_id)
    matched = str(item.get("example_id", ""))
    if matched != requested:
        raise ValueError(f"example_id mismatch: requested={requested}, matched={matched}")
    prompt = str(item.get("prompt", ""))
    prompt_hash = str(item.get("prompt_hash", ""))
    if not prompt_hash or canonical_prompt_hash(prompt) != prompt_hash:
        raise ValueError(f"prompt_hash mismatch for example_id={requested}")
    if expected_prompt_hash is not None and prompt_hash != str(expected_prompt_hash):
        raise ValueError(f"unexpected prompt hash for example_id={requested}")
    return {
        "requested_example_id": requested,
        "matched_example_id": matched,
        "prompt_hash": prompt_hash,
    }


def repetition_flags_hardened(raw: str) -> dict[str, bool]:
    """Detect generation loops without treating valid JSON value repetition as a loop."""
    text = str(raw).strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, (dict, list)):
            return {
                "repetition_loop_indicator": False,
                "repeated_chunk_indicator": False,
                "repeated_line_indicator": False,
                "repeated_ngram_indicator": False,
            }
    except (TypeError, ValueError, json.JSONDecodeError):
        pass

    compact = re.sub(r"\s+", " ", text).strip()
    repeated_chunk = bool(re.search(r"(.{24,}?)(?:\1){2,}", compact))
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    repeated_line = any(lines[i] == lines[i + 1] == lines[i + 2] for i in range(max(0, len(lines) - 2)))
    tokens = re.findall(r"\S+", text)
    repeated_ngram = False
    for n in range(3, 9):
        for i in range(0, max(0, len(tokens) - 3 * n + 1)):
            block = tokens[i : i + n]
            if tokens[i + n : i + 2 * n] == block and tokens[i + 2 * n : i + 3 * n] == block:
                repeated_ngram = True
                break
        if repeated_ngram:
            break
    return {
        "repetition_loop_indicator": bool(repeated_chunk or repeated_line or repeated_ngram),
        "repeated_chunk_indicator": repeated_chunk,
        "repeated_line_indicator": repeated_line,
        "repeated_ngram_indicator": repeated_ngram,
    }
