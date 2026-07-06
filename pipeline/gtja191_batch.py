"""Resumable sequential batch evaluation for the GTJA191 factor family."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from .alpha101_batch import (
    Alpha101BatchConfig,
    Alpha101BatchRunner,
    _atomic_write_json,
    _read_json,
    build_run_fingerprint,
)
from .factors.catalog import FACTOR_STATUSES
from .factors.gtja191 import (
    GTJA191,
    GTJA191DataError,
    GTJA191FormulaError,
    GTJA191Panels,
    GTJA191_NAMES,
    normalize_gtja_name,
)


GTJA191BatchConfig = Alpha101BatchConfig


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
    ) -> None:
        normalized = tuple(normalize_gtja_name(name) for name in factors)
        if not normalized:
            raise ValueError("at least one GTJA191 factor must be selected")
        self.panels = panels
        self.factors = tuple(dict.fromkeys(normalized))
        self.output_dir = Path(output_dir)
        self.config = config or GTJA191BatchConfig()
        self.data_signature = data_signature
        self.implementation_signature = implementation_signature
        supplied = factor_statuses or {}
        self.factor_statuses = {
            name: str(supplied.get(name, "active")).strip().lower()
            for name in self.factors
        }
        invalid = set(self.factor_statuses.values()).difference(FACTOR_STATUSES)
        if invalid:
            raise ValueError(f"unknown factor statuses: {', '.join(sorted(invalid))}")
        self.fingerprint = build_run_fingerprint(
            self.config,
            data_signature=data_signature,
            implementation_signature=implementation_signature,
        )
        self.calculator = GTJA191(panels)

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
        return GTJA191BatchResult(self.output_dir, status, result.leaderboard)
