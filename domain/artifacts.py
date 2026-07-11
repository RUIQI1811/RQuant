"""Typed workflow results and artifact references."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, MutableMapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Generic, Mapping, TypeVar


T = TypeVar("T")


class WorkflowStatus(str, Enum):
    RUNNING = "running"
    COMPLETE = "complete"
    PARTIAL = "partial"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


@dataclass(frozen=True)
class ArtifactRef:
    path: Path
    kind: str = "artifact"
    schema_version: int | None = None

    @classmethod
    def from_path(cls, path: str | Path, *, kind: str = "artifact") -> "ArtifactRef":
        return cls(Path(path), kind=kind)

    @property
    def exists(self) -> bool:
        return self.path.exists()

    @property
    def sha256(self) -> str | None:
        if not self.path.is_file():
            return None
        digest = hashlib.sha256()
        with self.path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "kind": self.kind,
            "schema_version": self.schema_version,
            "exists": self.exists,
            "sha256": self.sha256,
        }


@dataclass
class WorkflowResult(MutableMapping[str, Any], Generic[T]):
    result: T
    status: WorkflowStatus = WorkflowStatus.COMPLETE
    artifacts: dict[str, ArtifactRef] = field(default_factory=dict)
    values: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.status, WorkflowStatus):
            self.status = WorkflowStatus(self.status)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "WorkflowResult[Any]":
        result = values.get("result")
        workflow = cls(result=result)
        for key, value in values.items():
            if key != "result":
                workflow[key] = value
        return workflow

    def __getitem__(self, key: str) -> Any:
        if key == "result":
            return self.result
        if key in self.artifacts:
            return self.artifacts[key].path
        return self.values[key]

    def __setitem__(self, key: str, value: Any) -> None:
        if key == "result":
            self.result = value
        elif isinstance(value, ArtifactRef):
            self.artifacts[key] = value
            self.values.pop(key, None)
        elif isinstance(value, Path):
            self.artifacts[key] = ArtifactRef.from_path(value, kind=key)
            self.values.pop(key, None)
        else:
            self.values[key] = value
            self.artifacts.pop(key, None)

    def __delitem__(self, key: str) -> None:
        if key == "result":
            raise KeyError("result is required")
        if key in self.artifacts:
            del self.artifacts[key]
        else:
            del self.values[key]

    def __iter__(self) -> Iterator[str]:
        yield "result"
        yield from self.artifacts
        yield from self.values

    def __len__(self) -> int:
        return 1 + len(self.artifacts) + len(self.values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "artifacts": {key: ref.to_dict() for key, ref in self.artifacts.items()},
            "warnings": list(self.warnings),
        }
