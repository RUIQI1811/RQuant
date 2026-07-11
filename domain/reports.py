"""Typed report workflow results."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Iterator, Mapping
from typing import Any


@dataclass(frozen=True)
class SignalReturnResult:
    total_signals: int
    horizons: tuple[int, ...]
    buy_mode: str
    metrics: dict[str, Any]
    rows: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class ResearchReportResult:
    validation_status: str
    summary: dict[str, Any]
    source_fingerprints: dict[str, str | None] = field(default_factory=dict)


@dataclass(frozen=True)
class SystemDoctorResult(Mapping[str, Any]):
    report: dict[str, Any]

    @property
    def ok(self) -> bool:
        return bool(self.report.get("ok"))

    @property
    def status(self) -> str:
        return str(self.report.get("status", "unknown"))

    def __getitem__(self, key: str) -> Any:
        return self.report[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.report)

    def __len__(self) -> int:
        return len(self.report)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.report)
