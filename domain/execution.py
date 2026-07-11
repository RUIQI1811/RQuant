"""Typed execution and backtest records with legacy mapping compatibility."""

from __future__ import annotations

from collections.abc import Iterator, MutableMapping
from dataclasses import asdict, dataclass, field
from typing import Any, Generic, Mapping, TypeVar

from .signals import Signal
from .values import SourceId, Symbol, TradingDate


class RecordMapping(Mapping[str, Any]):
    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)


@dataclass(frozen=True)
class OrderIntent:
    signal: Signal
    execution_date: str
    side: str = "buy"
    cohort_id: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "execution_date", str(TradingDate(self.execution_date)))
        if self.side not in {"buy", "sell"}:
            raise ValueError(f"unsupported order side: {self.side!r}")

    @property
    def symbol(self) -> str:
        return self.signal.symbol


@dataclass(frozen=True)
class Fill(RecordMapping):
    date: str
    symbol: str
    side: str
    price: float
    shares: int
    cash_delta: float
    signal_date: str
    source: str = ""
    cohort_id: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "date", str(TradingDate(self.date)))
        object.__setattr__(self, "symbol", str(Symbol(self.symbol)))
        object.__setattr__(self, "signal_date", str(TradingDate(self.signal_date)))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["code"] = data.pop("symbol")
        return data


@dataclass(frozen=True)
class OrderResult(RecordMapping):
    date: str
    symbol: str | None
    side: str
    status: str
    reason: str
    signal_date: str
    price: float | str = ""
    shares: int = 0
    cash_delta: float = 0.0
    cohort_id: int | None = None
    source: str = ""
    score: float | None = None
    weight: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "date", str(TradingDate(self.date)))
        object.__setattr__(self, "signal_date", str(TradingDate(self.signal_date)))
        if self.symbol:
            object.__setattr__(self, "symbol", str(Symbol(self.symbol)))
        if self.side not in {"buy", "sell"}:
            raise ValueError(f"unsupported order side: {self.side!r}")
        if self.status not in {"filled", "blocked", "skipped"}:
            raise ValueError(f"unsupported order status: {self.status!r}")
        if self.shares < 0:
            raise ValueError("order shares cannot be negative")
        object.__setattr__(self, "source", str(SourceId(self.source)))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "OrderResult":
        return cls(
            date=row["date"],
            symbol=row.get("symbol") or row.get("code") or None,
            side=str(row["side"]),
            status=str(row["status"]),
            reason=str(row.get("reason", "")),
            signal_date=row.get("signal_date") or row["date"],
            price=row.get("price", ""),
            shares=int(row.get("shares", 0)),
            cash_delta=float(row.get("cash_delta", 0.0)),
            cohort_id=row.get("cohort_id"),
            source=str(row.get("source", "")),
            score=row.get("score"),
            weight=row.get("weight"),
            metadata=dict(row.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "cohort_id": self.cohort_id,
            "code": self.symbol or "",
            "side": self.side,
            "status": self.status,
            "reason": self.reason,
            "signal_date": self.signal_date,
            "price": self.price,
            "shares": self.shares,
            "cash_delta": self.cash_delta,
            "source": self.source,
            "score": self.score,
            "weight": self.weight,
            "metadata": dict(self.metadata),
        }

    def to_legacy_dict(self) -> dict[str, Any]:
        data = {
            "date": self.date,
            "code": self.symbol or "",
            "side": self.side,
            "status": self.status,
            "reason": self.reason,
            "signal_date": self.signal_date,
            "price": self.price,
            "shares": self.shares,
            "cash_delta": self.cash_delta,
        }
        if self.cohort_id is not None:
            data["cohort_id"] = self.cohort_id
        return data

    @property
    def fill(self) -> Fill | None:
        if (
            self.status != "filled"
            or not self.symbol
            or not isinstance(self.price, (int, float))
            or self.shares <= 0
        ):
            return None
        return Fill(
            date=self.date,
            symbol=self.symbol,
            side=self.side,
            price=float(self.price),
            shares=self.shares,
            cash_delta=self.cash_delta,
            signal_date=self.signal_date,
            source=self.source,
            cohort_id=self.cohort_id,
        )


@dataclass
class Position:
    symbol: str
    shares: int
    entry_date: str
    entry_price: float
    entry_value: float
    signal_date: str
    hold_days: int
    source: str = ""
    score: float | None = None
    weight: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.symbol = str(Symbol(self.symbol))
        self.entry_date = str(TradingDate(self.entry_date))
        self.signal_date = str(TradingDate(self.signal_date))
        if self.shares <= 0:
            raise ValueError("position shares must be positive")
        self.source = str(SourceId(self.source))
        self.metadata = dict(self.metadata or {})

    @property
    def code(self) -> str:
        return self.symbol


@dataclass(frozen=True)
class PositionSnapshot(RecordMapping):
    symbol: str
    shares: int
    entry_date: str
    entry_price: float
    entry_value: float
    signal_date: str
    market_value: float
    cohort_id: int | None = None
    source: str = ""
    score: float | None = None
    weight: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", str(Symbol(self.symbol)))
        object.__setattr__(self, "entry_date", str(TradingDate(self.entry_date)))
        object.__setattr__(self, "signal_date", str(TradingDate(self.signal_date)))

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "PositionSnapshot":
        return cls(
            symbol=row.get("symbol") or row["code"],
            shares=int(row["shares"]),
            entry_date=row["entry_date"],
            entry_price=float(row["entry_price"]),
            entry_value=float(row["entry_value"]),
            signal_date=row["signal_date"],
            market_value=float(row["market_value"]),
            cohort_id=row.get("cohort_id"),
            source=str(row.get("source", "")),
            score=row.get("score"),
            weight=row.get("weight"),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["code"] = data.pop("symbol")
        return data


@dataclass(frozen=True)
class Trade(RecordMapping):
    signal_date: str
    strategy: str
    buy_mode: str
    hold_days: int
    symbol: str | None = None
    shares: int = 0
    entry_date: str | None = None
    exit_date: str | None = None
    entry_price: float | None = None
    exit_price: float | None = None
    entry_value: float | None = None
    exit_value: float | None = None
    return_value: float | None = None
    pnl: float | None = None
    cohort_id: int | None = None
    stock_count: int | None = None
    start_cash: float | None = None
    basket_return: float | None = None
    end_cash: float | None = None
    details: tuple[dict[str, Any], ...] = ()
    source: str = ""
    score: float | None = None
    weight: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "signal_date", str(TradingDate(self.signal_date)))
        if self.symbol:
            object.__setattr__(self, "symbol", str(Symbol(self.symbol)))
        if self.entry_date:
            object.__setattr__(self, "entry_date", str(TradingDate(self.entry_date)))
        if self.exit_date:
            object.__setattr__(self, "exit_date", str(TradingDate(self.exit_date)))
        object.__setattr__(self, "details", tuple(self.details or ()))
        object.__setattr__(self, "source", str(SourceId(self.source)))

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "Trade":
        return cls(
            signal_date=str(row["signal_date"]),
            strategy=str(row.get("strategy", "")),
            buy_mode=str(row.get("buy_mode", "")),
            hold_days=int(row.get("hold_days", 0)),
            symbol=row.get("symbol") or row.get("code"),
            shares=int(row.get("shares", 0)),
            entry_date=row.get("entry_date"),
            exit_date=row.get("exit_date"),
            entry_price=row.get("entry_price"),
            exit_price=row.get("exit_price"),
            entry_value=row.get("entry_value"),
            exit_value=row.get("exit_value"),
            return_value=row.get("return"),
            pnl=row.get("pnl"),
            cohort_id=row.get("cohort_id"),
            stock_count=row.get("stock_count"),
            start_cash=row.get("start_cash"),
            basket_return=row.get("basket_return"),
            end_cash=row.get("end_cash"),
            details=tuple(row.get("details") or ()),
            source=str(row.get("source", "")),
            score=row.get("score"),
            weight=row.get("weight"),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["code"] = data.pop("symbol") or ""
        data["return"] = data.pop("return_value")
        data["details"] = list(self.details)
        return data


@dataclass(frozen=True)
class EquityPoint(RecordMapping):
    date: str
    cash: float
    total_return: float
    market_value: float = 0.0
    total_value: float | None = None
    position_count: int = 0
    active_cohort_count: int | None = None

    def __post_init__(self) -> None:
        if self.date:
            object.__setattr__(self, "date", str(TradingDate(self.date)))

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "EquityPoint":
        return cls(
            date=str(row["date"]),
            cash=float(row["cash"]),
            total_return=float(row["total_return"]),
            market_value=float(row.get("market_value", 0.0)),
            total_value=row.get("total_value", row.get("cash")),
            position_count=int(row.get("position_count", 0)),
            active_cohort_count=row.get("active_cohort_count"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BacktestSummary(MutableMapping[str, Any]):
    def __init__(self, values: Mapping[str, Any] | None = None, **kwargs: Any) -> None:
        self._values = dict(values or {})
        self._values.update(kwargs)

    def __getitem__(self, key: str) -> Any:
        return self._values[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._values[key] = value

    def __delitem__(self, key: str) -> None:
        del self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def to_dict(self) -> dict[str, Any]:
        return dict(self._values)


@dataclass(frozen=True)
class BacktestResult:
    initial_cash: float
    final_cash: float
    total_return: float
    trades: list[Trade]
    orders: list[OrderResult]
    positions: list[PositionSnapshot]
    equity_curve: list[EquityPoint]
    summary: BacktestSummary

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "trades",
            [item if isinstance(item, Trade) else Trade.from_dict(item) for item in self.trades],
        )
        object.__setattr__(
            self,
            "orders",
            [
                item if isinstance(item, OrderResult) else OrderResult.from_dict(item)
                for item in self.orders
            ],
        )
        object.__setattr__(
            self,
            "positions",
            [
                item if isinstance(item, PositionSnapshot) else PositionSnapshot.from_dict(item)
                for item in self.positions
            ],
        )
        object.__setattr__(
            self,
            "equity_curve",
            [
                item if isinstance(item, EquityPoint) else EquityPoint.from_dict(item)
                for item in self.equity_curve
            ],
        )
        if not isinstance(self.summary, BacktestSummary):
            object.__setattr__(self, "summary", BacktestSummary(self.summary))

    @property
    def fills(self) -> tuple[Fill, ...]:
        return tuple(fill for order in self.orders if (fill := order.fill) is not None)


PortfolioBacktestResult = BacktestResult
