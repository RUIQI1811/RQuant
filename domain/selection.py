"""Custom-strategy selection records composed around the canonical Signal."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterator

from .signals import Signal
from .values import Symbol, TradingDate


@dataclass
class Candidate:
    code: str
    date: str
    strategy: str
    close: float
    turnover_n: float
    brick_growth: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.code = str(Symbol(self.code))
        self.date = str(TradingDate(self.date))
        self.strategy = str(self.strategy).strip()

    @property
    def symbol(self) -> str:
        return self.code

    @property
    def source(self) -> str:
        return self.strategy

    def to_signal(self, *, weight: float | None = None) -> Signal:
        metadata = dict(self.extra or {})
        metadata.update(
            {
                "close": self.close,
                "turnover_n": self.turnover_n,
                "strategy": self.strategy,
            }
        )
        if self.brick_growth is not None:
            metadata["brick_growth"] = self.brick_growth
        return Signal(
            date=self.date,
            symbol=self.code,
            signal_type="buy",
            source=self.strategy,
            score=self.brick_growth,
            weight=weight,
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if not data["extra"]:
            data.pop("extra")
        if data["brick_growth"] is None:
            data.pop("brick_growth")
        return data


@dataclass
class CandidateRun:
    run_date: str
    pick_date: str
    candidates: list[Candidate] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.run_date = str(TradingDate(self.run_date))
        self.pick_date = str(TradingDate(self.pick_date))
        self.candidates = [
            item if isinstance(item, Candidate) else Candidate(**item)
            for item in self.candidates
        ]

    @property
    def signals(self) -> tuple[Signal, ...]:
        return tuple(candidate.to_signal() for candidate in self.candidates)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_date": self.run_date,
            "pick_date": self.pick_date,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CandidateRun":
        return cls(
            run_date=data["run_date"],
            pick_date=data["pick_date"],
            candidates=[
                Candidate(
                    **{
                        key: value
                        for key, value in candidate.items()
                        if key in Candidate.__dataclass_fields__
                    }
                )
                for candidate in data.get("candidates", [])
            ],
            meta=data.get("meta", {}),
        )


@dataclass(frozen=True)
class SelectionResult:
    pick_date: TradingDate
    candidates: tuple[Candidate, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.pick_date, TradingDate):
            object.__setattr__(self, "pick_date", TradingDate(self.pick_date))
        object.__setattr__(
            self,
            "candidates",
            tuple(
                item if isinstance(item, Candidate) else Candidate(**item)
                for item in self.candidates
            ),
        )

    @property
    def signals(self) -> tuple[Signal, ...]:
        return tuple(candidate.to_signal() for candidate in self.candidates)

    def __iter__(self) -> Iterator[Any]:
        """Preserve historical ``pick_date, candidates = run_preselect()`` unpacking."""
        yield self.pick_date.value
        yield list(self.candidates)
