"""Typed market-data workflow outcomes."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .artifacts import WorkflowStatus


@dataclass(frozen=True)
class FetchResult(Mapping[str, Any]):
    start: str
    end: str
    symbol_count: int
    output_dir: Path
    manifest_path: Path
    outcomes: dict[str, int]
    config_path: Path | None = None
    log_path: Path | None = None
    workers: int | None = None
    max_requests_per_minute: int | None = None
    submitted_count: int = 0
    resumed_count: int = 0
    completed: dict[str, str] = field(default_factory=dict)
    failed_codes: tuple[str, ...] = ()
    ok: bool = True

    @property
    def status(self) -> WorkflowStatus:
        return WorkflowStatus.COMPLETE if self.ok else WorkflowStatus.PARTIAL

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "symbol_count": self.symbol_count,
            "output_dir": self.output_dir,
            "outcomes": dict(self.outcomes),
            "config_path": self.config_path,
            "log_path": self.log_path,
            "workers": self.workers,
            "max_requests_per_minute": self.max_requests_per_minute,
            "submitted_count": self.submitted_count,
            "resumed_count": self.resumed_count,
            "completed": dict(self.completed),
            "failed_codes": list(self.failed_codes),
            "manifest_path": self.manifest_path,
            "ok": self.ok,
            "status": self.status,
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())
