"""Resumable sequential batch evaluation for the GTJA191 factor family."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from reports.alpha101_batch import (
    Alpha101BatchConfig,
    Alpha101BatchRunner,
    _atomic_write_json,
    _atomic_write_csv,
    build_leaderboard,
    write_long_only_profitability_reports,
    _read_json,
    build_run_fingerprint,
)
from factors.catalog import FACTOR_STATUSES
from factors.directions import VALID_FACTOR_DIRECTIONS
from factors.gtja191 import (
    GTJA191,
    GTJA191DataError,
    GTJA191FormulaError,
    GTJA191Panels,
    GTJA191_NAMES,
    normalize_gtja_name,
)


GTJA191BatchConfig = Alpha101BatchConfig


class _DirectedGTJA191:
    """Apply a research direction after evaluating the original GTJA formula."""

    def __init__(self, calculator: GTJA191, directions: Mapping[str, int]) -> None:
        self._calculator = calculator
        self._directions = directions

    def calculate(self, name: str | int) -> pd.DataFrame:
        factor = normalize_gtja_name(name)
        return self._calculator.calculate(factor) * self._directions.get(factor, 1)


@dataclass(frozen=True)
class GTJA191BatchResult:
    output_dir: Path
    status: pd.DataFrame
    leaderboard: pd.DataFrame

    @property
    def failed_factors(self) -> tuple[str, ...]:
        if self.status.empty:
            return ()
        failed = self.status["status"].isin(("failed", "missing_input", "formula_error"))
        return tuple(self.status.loc[failed, "factor"].astype(str))


def _expand_tokens(tokens: Sequence[str]) -> set[str]:
    selected: set[str] = set()
    for token in tokens:
        for raw in str(token).split(","):
            value = raw.strip().lower()
            if not value:
                continue
            if value == "all":
                selected.update(GTJA191_NAMES)
                continue
            if value.isdigit():
                selected.add(normalize_gtja_name(int(value)))
                continue
            if "-" in value and not value.startswith("gtja-"):
                left, right = value.split("-", 1)
                start = int(left.removeprefix("gtja_").removeprefix("gtja"))
                end = int(right.removeprefix("gtja_").removeprefix("gtja"))
                if start > end:
                    raise ValueError(f"invalid descending factor range: {value}")
                selected.update(normalize_gtja_name(number) for number in range(start, end + 1))
            else:
                selected.add(normalize_gtja_name(value))
    return selected


def parse_gtja_selection(
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Parse names, comma lists, and inclusive numeric ranges in registry order."""

    selected = _expand_tokens(include or ("all",))
    selected.difference_update(_expand_tokens(exclude or ()))
    return tuple(name for name in GTJA191_NAMES if name in selected)


def filter_gtja_selection_from_start(
    factors: Sequence[str],
    start_factor: str | int | None,
) -> tuple[str, ...]:
    """Keep selected GTJA factors whose registry position is at or after start_factor."""

    normalized = tuple(normalize_gtja_name(name) for name in factors)
    if start_factor is None:
        return normalized
    start_value = int(start_factor) if str(start_factor).strip().isdigit() else start_factor
    start_name = normalize_gtja_name(start_value)
    start_index = GTJA191_NAMES.index(start_name)
    registry_positions = {name: position for position, name in enumerate(GTJA191_NAMES)}
    return tuple(
        name
        for name in normalized
        if registry_positions[name] >= start_index
    )


class GTJA191BatchRunner(Alpha101BatchRunner):
    """Reuse proven checkpoint/report mechanics with the independent GTJA registry."""

    def __init__(
        self,
        panels: GTJA191Panels,
        *,
        factors: Sequence[str],
        output_dir: str | Path,
        config: GTJA191BatchConfig | None = None,
        data_signature: str = "unspecified-data",
        implementation_signature: str = "unspecified-implementation",
        factor_statuses: Mapping[str, str] | None = None,
        factor_categories: Mapping[str, str] | None = None,
        factor_directions: Mapping[str, int] | None = None,
    ) -> None:
        normalized = tuple(normalize_gtja_name(name) for name in factors)
        if not normalized:
            raise ValueError("at least one GTJA191 factor must be selected")
        self.panels = panels
        self.factors = tuple(dict.fromkeys(normalized))
        self.output_dir = Path(output_dir)
        self.config = config or GTJA191BatchConfig()
        self.data_signature = data_signature
        supplied_directions = factor_directions or {}
        self.leaderboard_factor_directions = {
            name: int(supplied_directions.get(name, 1))
            for name in GTJA191_NAMES
        }
        self.factor_directions = {
            name: self.leaderboard_factor_directions[name]
            for name in self.factors
        }
        invalid_directions = set(self.leaderboard_factor_directions.values()).difference(
            VALID_FACTOR_DIRECTIONS
        )
        if invalid_directions:
            raise ValueError("GTJA191 factor directions must be -1 or 1")
        direction_signature = ",".join(
            f"{name}:{self.leaderboard_factor_directions[name]}"
            for name in GTJA191_NAMES
        )
        self.implementation_signature = (
            f"{implementation_signature}:factor-directions={direction_signature}"
        )
        supplied = factor_statuses or {}
        self.leaderboard_factor_statuses = {
            name: str(supplied.get(name, "active")).strip().lower()
            for name in GTJA191_NAMES
        }
        self.factor_statuses = {
            name: str(supplied.get(name, "active")).strip().lower()
            for name in self.factors
        }
        supplied_categories = factor_categories or {}
        self.leaderboard_factor_categories = {
            name: str(supplied_categories.get(name, "unclassified")).strip()
            or "unclassified"
            for name in GTJA191_NAMES
        }
        self.factor_categories = {
            name: self.leaderboard_factor_categories[name]
            for name in self.factors
        }
        invalid = set(self.leaderboard_factor_statuses.values()).difference(FACTOR_STATUSES)
        if invalid:
            raise ValueError(f"unknown factor statuses: {', '.join(sorted(invalid))}")
        self.fingerprint = build_run_fingerprint(
            self.config,
            data_signature=data_signature,
            implementation_signature=self.implementation_signature,
        )
        self.calculator = _DirectedGTJA191(GTJA191(panels), self.factor_directions)

    def _status_row(
        self,
        factor: str,
        status: str,
        duration: float,
        *,
        row_count: object,
        message: str = "",
    ) -> dict[str, object]:
        row = super()._status_row(
            factor,
            status,
            duration,
            row_count=row_count,
            message=message,
        )
        row["factor_direction"] = self.factor_directions[factor]
        return row

    def _manifest(self, *, status: str, base_rows: int) -> dict[str, object]:
        manifest = super()._manifest(status=status, base_rows=base_rows)
        manifest["factor_directions"] = self.factor_directions
        return manifest

    def run(self) -> GTJA191BatchResult:
        result = super().run()
        status = result.status.copy()
        if not status.empty:
            messages = status["message"].fillna("").astype(str)
            status.loc[
                status["status"].eq("failed") & messages.str.startswith(GTJA191DataError.__name__),
                "status",
            ] = "missing_input"
            status.loc[
                status["status"].eq("failed") & messages.str.startswith(GTJA191FormulaError.__name__),
                "status",
            ] = "formula_error"
            status.to_csv(self.output_dir / "batch_status.csv", index=False)
            manifest_path = self.output_dir / "batch_manifest.json"
            manifest = _read_json(manifest_path)
            manifest["failed_count"] = int(status["status"].isin(("failed", "formula_error")).sum())
            manifest["missing_input_count"] = int(status["status"].eq("missing_input").sum())
            _atomic_write_json(manifest_path, manifest)
        leaderboard = build_leaderboard(
            self.output_dir,
            GTJA191_NAMES,
            fingerprint=self.fingerprint,
            factor_statuses=self.leaderboard_factor_statuses,
            factor_categories=self.leaderboard_factor_categories,
        )
        _atomic_write_csv(self.output_dir / "leaderboard.csv", leaderboard)
        write_long_only_profitability_reports(self.output_dir, leaderboard)
        return GTJA191BatchResult(self.output_dir, status, leaderboard)
