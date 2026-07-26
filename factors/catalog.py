"""Configuration-backed lifecycle status for built-in research factors."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import yaml

from factors.alpha101 import ALPHA101_NAMES, normalize_alpha_name


FACTOR_STATUSES = ("active", "watch", "disabled")


@dataclass(frozen=True)
class FactorCatalog:
    """Assign factors to active, watch, or disabled research tiers."""

    default_status: str = "active"
    statuses: Mapping[str, object] | None = None
    factor_details: Mapping[str, Mapping[str, object]] = field(
        init=False,
        default_factory=dict,
    )

    def __post_init__(self) -> None:
        if self.default_status not in FACTOR_STATUSES:
            raise ValueError(
                f"default_status must be one of {', '.join(FACTOR_STATUSES)}"
            )
        normalized: dict[str, str] = {}
        details: dict[str, dict[str, object]] = {}
        for raw_name, raw_entry in (self.statuses or {}).items():
            name = normalize_alpha_name(raw_name)
            raw_status = _status_from_entry(name, raw_entry)
            status = str(raw_status).strip().lower()
            if status not in FACTOR_STATUSES:
                raise ValueError(
                    f"invalid status for {name}: {raw_status!r}; "
                    f"expected one of {', '.join(FACTOR_STATUSES)}"
                )
            normalized[name] = status
            if isinstance(raw_entry, Mapping):
                details[name] = {
                    str(key): value
                    for key, value in raw_entry.items()
                    if str(key) != "status"
                }
        object.__setattr__(self, "statuses", normalized)
        object.__setattr__(self, "factor_details", details)

    def status_for(self, factor: str | int) -> str:
        """Return the configured status, falling back to the catalog default."""

        return self.statuses.get(normalize_alpha_name(factor), self.default_status)

    def select(
        self,
        factors: Sequence[str],
        *,
        include_statuses: Sequence[str] = ("active", "watch"),
    ) -> tuple[str, ...]:
        """Filter and order factors by lifecycle tier, then original order."""

        requested_statuses = tuple(str(value).strip().lower() for value in include_statuses)
        invalid = set(requested_statuses).difference(FACTOR_STATUSES)
        if invalid:
            raise ValueError(f"unknown factor statuses: {', '.join(sorted(invalid))}")
        normalized = tuple(dict.fromkeys(normalize_alpha_name(name) for name in factors))
        return tuple(
            name
            for status in requested_statuses
            for name in normalized
            if self.status_for(name) == status
        )

    def status_map(self, factors: Sequence[str] = ALPHA101_NAMES) -> dict[str, str]:
        """Return an auditable status mapping for the requested factors."""

        return {normalize_alpha_name(name): self.status_for(name) for name in factors}

    def category_for(self, factor: str | int) -> str:
        """Return the configured research category or an explicit fallback."""
        name = normalize_alpha_name(factor)
        category = self.factor_details.get(name, {}).get("category", "unclassified")
        return str(category).strip() or "unclassified"

    def category_map(self, factors: Sequence[str] = ALPHA101_NAMES) -> dict[str, str]:
        return {normalize_alpha_name(name): self.category_for(name) for name in factors}


def load_factor_catalog(path: str | Path) -> FactorCatalog:
    """Load factor lifecycle settings from YAML and validate all factor names."""

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"factor config not found: {config_path}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("factor config root must be a mapping")
    raw_statuses = payload.get("factors", {}) or {}
    if not isinstance(raw_statuses, dict):
        raise ValueError("factor config 'factors' must be a mapping")
    raw_categories = payload.get("categories", {}) or {}
    if not isinstance(raw_categories, dict):
        raise ValueError("factor config 'categories' must be a mapping")
    catalog_entries: dict[str, object] = {}
    for name, entry in raw_statuses.items():
        if name not in raw_categories:
            catalog_entries[name] = entry
        elif isinstance(entry, Mapping):
            catalog_entries[name] = {**entry, "category": raw_categories[name]}
        else:
            catalog_entries[name] = {
                "status": entry,
                "category": raw_categories[name],
            }
    catalog = FactorCatalog(
        default_status=str(payload.get("default_status", "active")).strip().lower(),
        statuses=catalog_entries,
    )
    unknown = set(catalog.statuses).difference(ALPHA101_NAMES)
    unknown.update(
        normalize_alpha_name(name)
        for name in raw_categories
        if normalize_alpha_name(name) not in ALPHA101_NAMES
    )
    if unknown:
        raise ValueError(f"unknown Alpha101 factors in config: {', '.join(sorted(unknown))}")
    return catalog


def _status_from_entry(factor_name: str, entry: object) -> object:
    if not isinstance(entry, Mapping):
        return entry
    if "status" in entry:
        return entry["status"]
    raise ValueError(
        f"structured config for {factor_name} must contain status"
    )
