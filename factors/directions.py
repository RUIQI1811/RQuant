"""Configuration-driven factor direction without changing source formulas."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping, Sequence

import yaml

from factors.gtja191 import normalize_gtja_name


VALID_FACTOR_DIRECTIONS = frozenset((-1, 1))


def _normalize_direction(value: object, *, factor: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"factor direction for {factor} must be -1 or 1")
    try:
        direction = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"factor direction for {factor} must be -1 or 1") from exc
    if direction not in VALID_FACTOR_DIRECTIONS or str(value).strip() not in {
        "-1",
        "1",
        "-1.0",
        "1.0",
    }:
        raise ValueError(f"factor direction for {factor} must be -1 or 1")
    return direction


def load_factor_directions(
    path: str | Path,
    factors: Sequence[str],
    *,
    normalize_name: Callable[[object], str],
) -> dict[str, int]:
    """Load default and per-factor direction multipliers from lifecycle YAML.

    The preferred representation is a top-level ``directions`` mapping. A
    structured entry under ``factors`` may also contain ``direction``. Defining
    conflicting values in both places is rejected instead of silently choosing
    one.
    """

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"factor config not found: {config_path}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"factor config root must be a mapping: {config_path}")

    default = _normalize_direction(
        payload.get("default_direction", 1),
        factor="default_direction",
    )
    configured = payload.get("directions", {}) or {}
    if not isinstance(configured, dict):
        raise ValueError(f"factor config directions must be a mapping: {config_path}")
    entries = payload.get("factors", {}) or {}
    if not isinstance(entries, dict):
        raise ValueError(f"factor config factors must be a mapping: {config_path}")

    explicit: dict[str, int] = {}
    for raw_name, raw_direction in configured.items():
        name = normalize_name(raw_name)
        if name in explicit:
            raise ValueError(f"duplicate factor direction after normalization: {name}")
        explicit[name] = _normalize_direction(raw_direction, factor=name)

    structured: dict[str, int] = {}
    for raw_name, entry in entries.items():
        if not isinstance(entry, Mapping) or "direction" not in entry:
            continue
        name = normalize_name(raw_name)
        structured[name] = _normalize_direction(entry["direction"], factor=name)
        if name in explicit and explicit[name] != structured[name]:
            raise ValueError(f"conflicting factor directions configured for {name}")

    normalized_factors = tuple(dict.fromkeys(normalize_name(name) for name in factors))
    return {
        name: explicit.get(name, structured.get(name, default))
        for name in normalized_factors
    }


def load_gtja_factor_directions(
    path: str | Path,
    factors: Sequence[str | int],
) -> dict[str, int]:
    """Load GTJA191 direction multipliers keyed by normalized factor name."""

    return load_factor_directions(
        path,
        tuple(str(name) for name in factors),
        normalize_name=normalize_gtja_name,
    )
