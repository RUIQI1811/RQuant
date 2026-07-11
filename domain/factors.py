"""Typed factor-evaluation result while retaining named table access."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class FactorEvaluationResult(Mapping[str, pd.DataFrame]):
    factor_name: str
    tables: dict[str, pd.DataFrame]

    def __post_init__(self) -> None:
        object.__setattr__(self, "tables", dict(self.tables))
        if "summary" not in self.tables:
            raise ValueError("factor evaluation requires a summary table")

    @property
    def summary(self) -> pd.DataFrame:
        return self.tables["summary"]

    def __getitem__(self, key: str) -> pd.DataFrame:
        return self.tables[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.tables)

    def __len__(self) -> int:
        return len(self.tables)
