from __future__ import annotations

import csv
import datetime as dt
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from domain.artifacts import WorkflowResult
from domain.execution import (
    BacktestResult as PortfolioBacktestResult,
    BacktestSummary,
    EquityPoint,
    OrderIntent,
    OrderResult,
    Position,
    PositionSnapshot,
    Trade,
)
from domain.signals import Signal, SignalBook
from backtest.performance import annualized_return, yearly_return_rows

from reports.signal_returns import (
    BUY_MODE_NEXT_OPEN,
    BUY_MODE_SIGNAL_CLOSE,
    DEFAULT_HORIZONS,
    VALID_BUY_MODES,
    build_enabled_selectors,
    filter_picks_by_date,
    filter_selectors_by_strategy,
    format_percent,
)
from market.data import StockPoolConfig, build_stock_pool_by_date, clean_market_data
from market.preparation import MarketDataPreparer, SelectorPickPrecomputer
from strategies.preselect import _calc_warmup, _resolve_cfg_path, load_config, load_raw_data


DEFAULT_OUTPUT_DIR = Path("data") / "portfolio_backtest"
ANNUALIZATION_DAYS = 252


@dataclass(frozen=True)
class FeeModel:
    commission_rate: float
    stamp_tax_rate: float = 0.0005
    transfer_fee_rate: float = 0.00001

    @classmethod
    def from_commission_wan(cls, commission_wan: float) -> "FeeModel":
        return cls(commission_rate=float(commission_wan) / 10000.0)

    @property
    def buy_cost_rate(self) -> float:
        return self.commission_rate + self.transfer_fee_rate

    @property
    def sell_cost_rate(self) -> float:
        return self.commission_rate + self.stamp_tax_rate + self.transfer_fee_rate


@dataclass(frozen=True)
class PortfolioSettings:
    initial_cash: float
    strategy: str
    buy_mode: str
    hold_days: int
    fee_model: FeeModel
    max_positions: int = 10
    position_pct: float = 0.1
    lot_size: int = 100


@dataclass
class CohortSleeve:
    """One independently funded slot in a staggered holding-period portfolio."""

    cohort_id: int
    cash: float
    positions: Dict[str, Position]


def _date_str(value: object) -> str:
    return pd.to_datetime(value).strftime("%Y-%m-%d")


def _coerce_signal_book(
    picks_by_date: Dict[pd.Timestamp, List[str]] | SignalBook,
    *,
    source: str,
) -> SignalBook:
    if isinstance(picks_by_date, SignalBook):
        return picks_by_date
    signals: list[Signal] = []
    for date, values in picks_by_date.items():
        for value in values:
            if isinstance(value, Signal):
                signals.append(value)
            else:
                signals.append(
                    Signal(date=_date_str(date), symbol=str(value), source=source)
                )
    return SignalBook(signals)


def _row_on_or_none(df: pd.DataFrame, date: pd.Timestamp) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    date = pd.to_datetime(date)
    if date not in df.index:
        return None
    return df.loc[date]


def _execution_price(row: pd.Series, buy_mode: str, *, side: str) -> float:
    if buy_mode == BUY_MODE_SIGNAL_CLOSE:
        return float(row["close"])
    if buy_mode == BUY_MODE_NEXT_OPEN:
        return float(row["open"])
    raise ValueError(f"buy_mode must be one of {sorted(VALID_BUY_MODES)}")


def _can_buy(row: Optional[pd.Series]) -> tuple[bool, str]:
    if row is None:
        return False, "missing_bar"
    if not bool(row.get("is_tradeable", True)):
        return False, "suspended"
    if bool(row.get("is_limit_up", False)):
        return False, "limit_up"
    return True, ""


def _can_sell(row: Optional[pd.Series]) -> tuple[bool, str]:
    if row is None:
        return False, "missing_bar"
    if not bool(row.get("is_tradeable", True)):
        return False, "suspended"
    if bool(row.get("is_limit_down", False)):
        return False, "limit_down"
    return True, ""


def _position_market_value(position: Position, prepared: Dict[str, pd.DataFrame], date: pd.Timestamp) -> float:
    df = prepared.get(position.code)
    close = _latest_close_on_or_before(df, date)
    if close is None:
        return position.entry_value
    return close * position.shares


def _latest_close_on_or_before(
    df: Optional[pd.DataFrame],
    date: pd.Timestamp,
) -> Optional[float]:
    if df is None or df.empty:
        return None
    index = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.DatetimeIndex(df.index)
    position = int(index.searchsorted(pd.to_datetime(date), side="right")) - 1
    if position < 0:
        return None
    return float(df.iloc[position]["close"])


def _trading_bars_held(
    date_positions: Dict[pd.Timestamp, int],
    entry_date: pd.Timestamp,
    current_date: pd.Timestamp,
) -> int:
    return max(
        0,
        int(date_positions[pd.to_datetime(current_date)])
        - int(date_positions[pd.to_datetime(entry_date)]),
    )


def _portfolio_value(cash: float, positions: Dict[str, Position], prepared: Dict[str, pd.DataFrame], date: pd.Timestamp) -> float:
    return float(cash) + sum(_position_market_value(pos, prepared, date) for pos in positions.values())


def _shares_for_cash(*, cash_budget: float, price: float, fee_model: FeeModel, lot_size: int) -> int:
    if cash_budget <= 0 or price <= 0 or lot_size <= 0:
        return 0
    gross_share_budget = cash_budget / (price * (1.0 + fee_model.buy_cost_rate))
    lots = int(gross_share_budget // lot_size)
    return lots * lot_size


def _build_all_trade_dates(
    prepared: Dict[str, pd.DataFrame],
    *,
    start_date: Optional[str],
    end_date: Optional[str],
) -> list[pd.Timestamp]:
    start_ts = pd.to_datetime(start_date) if start_date else None
    end_ts = pd.to_datetime(end_date) if end_date else None
    dates = sorted({pd.to_datetime(d) for df in prepared.values() for d in df.index})
    if start_ts is not None:
        dates = [d for d in dates if d >= start_ts]
    if end_ts is not None:
        dates = [d for d in dates if d <= end_ts]
    return dates


def calculate_trade_return(
    *,
    entry_price: float,
    exit_price: float,
    fee_model: FeeModel,
) -> float:
    if entry_price <= 0 or not math.isfinite(entry_price) or not math.isfinite(exit_price):
        return float("nan")
    net_entry_value = 1.0 / (1.0 + fee_model.buy_cost_rate)
    exit_value = net_entry_value * (exit_price / entry_price)
    return exit_value * (1.0 - fee_model.sell_cost_rate) - 1.0


def build_equity_curve_rows(
    *,
    initial_cash: float,
    trades: List[dict],
    start_date: Optional[str],
) -> List[dict]:
    rows = [
        {
            "date": start_date or "",
            "cash": float(initial_cash),
            "total_return": 0.0,
        }
    ]
    for trade in trades:
        cash = float(trade["end_cash"])
        rows.append(
            {
                "date": trade["signal_date"],
                "cash": cash,
                "total_return": cash / initial_cash - 1.0 if initial_cash else float("nan"),
            }
        )
    return rows


def calculate_risk_metrics(
    *,
    equity_rows: List[dict],
    trades: List[dict],
    annualization_days: int = ANNUALIZATION_DAYS,
    initial_cash: float | None = None,
) -> dict:
    max_drawdown = 0.0
    peak = None
    for row in equity_rows:
        equity_value = float(row.get("total_value", row.get("cash", 0.0)))
        peak = equity_value if peak is None else max(peak, equity_value)
        if peak and peak > 0:
            drawdown = equity_value / peak - 1.0
            max_drawdown = max(max_drawdown, abs(min(drawdown, 0.0)))

    equity_values = [
        float(row.get("total_value", row.get("cash", 0.0)))
        for row in equity_rows
        if math.isfinite(float(row.get("total_value", row.get("cash", 0.0))))
    ]
    if len(equity_values) > 1:
        returns = pd.Series(equity_values, dtype="float64").pct_change().dropna()
    else:
        returns = pd.Series(
            [
                float(trade.get("basket_return", trade.get("return")))
                for trade in trades
                if trade.get("basket_return", trade.get("return")) is not None
                and math.isfinite(float(trade.get("basket_return", trade.get("return"))))
            ],
            dtype="float64",
        )
    if returns.empty:
        annualized_return_mean = None
        annualized_volatility = None
        sharpe_ratio = None
    else:
        mean_return = float(returns.mean())
        annualized_return_mean = mean_return * annualization_days
        volatility = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
        annualized_volatility = volatility * float(np.sqrt(annualization_days))
        sharpe_ratio = (
            annualized_return_mean / annualized_volatility
            if annualized_volatility and annualized_volatility > 0
            else None
        )

    baseline = float(initial_cash) if initial_cash is not None else (
        equity_values[0] if equity_values else 0.0
    )
    yearly_rows = yearly_return_rows(equity_rows, initial_cash=baseline)
    yearly_annualized = [
        float(row["annualized_return"])
        for row in yearly_rows
        if row["annualized_return"] is not None
        and math.isfinite(float(row["annualized_return"]))
    ]
    overall_total_return = (
        equity_values[-1] / baseline - 1.0
        if equity_values and baseline > 0
        else None
    )

    return {
        "max_drawdown": max_drawdown,
        "annualized_return_mean": annualized_return_mean,
        "overall_annualized_return": (
            annualized_return(
                overall_total_return,
                sum(int(row["trading_days"]) for row in yearly_rows),
            )
            if overall_total_return is not None
            else None
        ),
        "average_yearly_annualized_return": (
            sum(yearly_annualized) / len(yearly_annualized)
            if yearly_annualized
            else None
        ),
        "year_count": len(yearly_rows),
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe_ratio,
    }


def _price_points(
    df: pd.DataFrame,
    signal_pos: int,
    *,
    buy_mode: str,
    hold_days: int,
) -> Optional[tuple[str, float, str, float]]:
    if buy_mode == BUY_MODE_SIGNAL_CLOSE:
        entry_pos = signal_pos
        exit_pos = signal_pos + hold_days
        price_col = "close"
    elif buy_mode == BUY_MODE_NEXT_OPEN:
        entry_pos = signal_pos + 1
        exit_pos = entry_pos + hold_days
        price_col = "open"
    else:
        raise ValueError(f"buy_mode must be one of {sorted(VALID_BUY_MODES)}")

    if entry_pos >= len(df) or exit_pos >= len(df):
        return None

    entry_date = pd.to_datetime(df.iloc[entry_pos]["date"]).strftime("%Y-%m-%d")
    exit_date = pd.to_datetime(df.iloc[exit_pos]["date"]).strftime("%Y-%m-%d")
    entry_price = float(df.iloc[entry_pos][price_col])
    exit_price = float(df.iloc[exit_pos][price_col])
    return entry_date, entry_price, exit_date, exit_price


def run_portfolio_from_prepared(
    *,
    prepared: Dict[str, pd.DataFrame],
    picks_by_date: Dict[pd.Timestamp, List[str]] | SignalBook,
    settings: PortfolioSettings,
) -> PortfolioBacktestResult:
    if settings.hold_days <= 0:
        raise ValueError("hold_days must be positive")
    if settings.buy_mode not in VALID_BUY_MODES:
        raise ValueError(f"buy_mode must be one of {sorted(VALID_BUY_MODES)}")

    signal_book = _coerce_signal_book(picks_by_date, source=settings.strategy)
    cash = float(settings.initial_cash)
    trades: List[dict] = []

    for signal_date in sorted(signal_book):
        signal_ts = pd.to_datetime(signal_date)
        stock_returns: List[float] = []
        details: List[dict] = []

        for signal in sorted(signal_book.signals_for(signal_date), key=lambda item: item.symbol):
            code = signal.symbol
            df = prepared.get(code)
            if df is None or df.empty:
                continue
            positions = pd.DatetimeIndex(df.index).get_indexer([signal_ts])
            signal_pos = int(positions[0])
            if signal_pos < 0:
                continue
            points = _price_points(
                df,
                signal_pos,
                buy_mode=settings.buy_mode,
                hold_days=settings.hold_days,
            )
            if points is None:
                continue
            entry_date, entry_price, exit_date, exit_price = points
            stock_return = calculate_trade_return(
                entry_price=entry_price,
                exit_price=exit_price,
                fee_model=settings.fee_model,
            )
            if not math.isfinite(stock_return):
                continue
            stock_returns.append(stock_return)
            details.append(
                {
                    "code": code,
                    "entry_date": entry_date,
                    "entry_price": entry_price,
                    "exit_date": exit_date,
                    "exit_price": exit_price,
                    "return": stock_return,
                    "source": signal.source,
                    "score": signal.score,
                    "weight": signal.weight,
                    "metadata": signal.metadata,
                }
            )

        if not stock_returns:
            continue

        start_cash = cash
        basket_return = sum(stock_returns) / len(stock_returns)
        cash = cash * (1.0 + basket_return)
        trades.append(
            {
                "signal_date": signal_ts.strftime("%Y-%m-%d"),
                "strategy": settings.strategy,
                "buy_mode": settings.buy_mode,
                "hold_days": settings.hold_days,
                "stock_count": len(stock_returns),
                "start_cash": start_cash,
                "basket_return": basket_return,
                "end_cash": cash,
                "details": details,
                "source": settings.strategy,
            }
        )

    total_return = cash / settings.initial_cash - 1.0 if settings.initial_cash else float("nan")
    equity_rows = build_equity_curve_rows(
        initial_cash=settings.initial_cash,
        trades=trades,
        start_date=None,
    )
    risk_metrics = calculate_risk_metrics(
        equity_rows=equity_rows,
        trades=trades,
        initial_cash=settings.initial_cash,
    )
    summary = {
        "run_date": dt.date.today().isoformat(),
        "initial_cash": settings.initial_cash,
        "final_cash": cash,
        "total_return": total_return,
        "strategy": settings.strategy,
        "buy_mode": settings.buy_mode,
        "hold_days": settings.hold_days,
        "trade_count": len(trades),
        "commission_rate": settings.fee_model.commission_rate,
        "stamp_tax_rate": settings.fee_model.stamp_tax_rate,
        "transfer_fee_rate": settings.fee_model.transfer_fee_rate,
        **risk_metrics,
    }
    return PortfolioBacktestResult(
        initial_cash=settings.initial_cash,
        final_cash=cash,
        total_return=total_return,
        trades=trades,
        orders=[],
        positions=[],
        equity_curve=equity_rows,
        summary=summary,
    )


def run_realistic_portfolio_from_prepared(
    *,
    prepared: Dict[str, pd.DataFrame],
    picks_by_date: Dict[pd.Timestamp, List[str]] | SignalBook,
    settings: PortfolioSettings,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> PortfolioBacktestResult:
    """Run a cash/position based portfolio simulation.

    Signals are generated after the signal bar and become buy candidates on the
    next available trading day. Sells happen after hold_days bars from entry.
    The engine enforces cash, whole lots, limit-up no-buy, limit-down no-sell,
    suspension no-trade, and T+1 no same-day sell.
    """
    if settings.hold_days <= 0:
        raise ValueError("hold_days must be positive")
    if settings.max_positions <= 0:
        raise ValueError("max_positions must be positive")
    if not (0 < settings.position_pct <= 1):
        raise ValueError("position_pct must be in (0, 1]")
    if settings.buy_mode not in VALID_BUY_MODES:
        raise ValueError(f"buy_mode must be one of {sorted(VALID_BUY_MODES)}")

    dates = _build_all_trade_dates(prepared, start_date=start_date, end_date=end_date)
    if not dates:
        summary = {
            "run_date": dt.date.today().isoformat(),
            "initial_cash": settings.initial_cash,
            "final_cash": settings.initial_cash,
            "total_return": 0.0,
            "strategy": settings.strategy,
            "buy_mode": settings.buy_mode,
            "hold_days": settings.hold_days,
            "max_positions": settings.max_positions,
            "position_pct": settings.position_pct,
            "lot_size": settings.lot_size,
            "trade_count": 0,
            "order_count": 0,
            "realized_trade_count": 0,
            "open_position_count": 0,
            "commission_rate": settings.fee_model.commission_rate,
            "stamp_tax_rate": settings.fee_model.stamp_tax_rate,
            "transfer_fee_rate": settings.fee_model.transfer_fee_rate,
            "max_drawdown": 0.0,
            "annualized_return_mean": None,
            "overall_annualized_return": None,
            "average_yearly_annualized_return": None,
            "year_count": 0,
            "annualized_volatility": None,
            "sharpe_ratio": None,
        }
        return PortfolioBacktestResult(
            initial_cash=settings.initial_cash,
            final_cash=settings.initial_cash,
            total_return=0.0,
            trades=[],
            orders=[],
            positions=[],
            equity_curve=[],
            summary=summary,
        )

    date_to_next: dict[pd.Timestamp, pd.Timestamp] = {}
    date_positions = {date: position for position, date in enumerate(dates)}
    for idx, date in enumerate(dates[:-1]):
        date_to_next[date] = dates[idx + 1]

    signal_book = _coerce_signal_book(picks_by_date, source=settings.strategy)
    pending_buys: dict[pd.Timestamp, list[OrderIntent]] = {}
    for signal_date in signal_book:
        signal_ts = pd.to_datetime(signal_date)
        buy_date = date_to_next.get(signal_ts)
        if buy_date is None:
            continue
        # Preserve signal priority (for example factor rank) while removing
        # duplicates. Sorting by symbol would silently discard ranking intent.
        unique: dict[str, Signal] = {}
        for signal in signal_book.signals_for(signal_ts):
            unique.setdefault(signal.symbol, signal)
        pending_buys.setdefault(buy_date, []).extend(
            OrderIntent(signal=signal, execution_date=_date_str(buy_date))
            for signal in unique.values()
        )

    cash = float(settings.initial_cash)
    positions: Dict[str, Position] = {}
    orders: List[dict] = []
    trades: List[dict] = []
    equity_rows: List[dict] = []

    for date in dates:
        date_label = _date_str(date)

        # Sells first, freeing cash before new buys.
        for code in sorted(list(positions)):
            position = positions[code]
            entry_ts = pd.to_datetime(position.entry_date)
            bars_held = _trading_bars_held(date_positions, entry_ts, date)
            if bars_held < settings.hold_days:
                continue
            if pd.to_datetime(position.entry_date) >= date:
                orders.append(
                    {
                        "date": date_label,
                        "code": code,
                        "side": "sell",
                        "status": "blocked",
                        "reason": "t_plus_1",
                        "signal_date": position.signal_date,
                        "price": "",
                        "shares": position.shares,
                        "cash_delta": 0.0,
                        "source": position.source,
                        "score": position.score,
                        "weight": position.weight,
                        "metadata": position.metadata,
                    }
                )
                continue
            row = _row_on_or_none(prepared.get(code), date)
            can_sell, reason = _can_sell(row)
            if not can_sell:
                orders.append(
                    {
                        "date": date_label,
                        "code": code,
                        "side": "sell",
                        "status": "blocked",
                        "reason": reason,
                        "signal_date": position.signal_date,
                        "price": "",
                        "shares": position.shares,
                        "cash_delta": 0.0,
                        "source": position.source,
                        "score": position.score,
                        "weight": position.weight,
                        "metadata": position.metadata,
                    }
                )
                continue

            price = _execution_price(row, settings.buy_mode, side="sell")
            gross = position.shares * price
            proceeds = gross * (1.0 - settings.fee_model.sell_cost_rate)
            cash += proceeds
            pnl = proceeds - position.entry_value
            trade = {
                "signal_date": position.signal_date,
                "entry_date": position.entry_date,
                "exit_date": date_label,
                "strategy": settings.strategy,
                "buy_mode": settings.buy_mode,
                "hold_days": settings.hold_days,
                "code": code,
                "shares": position.shares,
                "entry_price": position.entry_price,
                "exit_price": price,
                "entry_value": position.entry_value,
                "exit_value": proceeds,
                "return": pnl / position.entry_value if position.entry_value else float("nan"),
                "pnl": pnl,
                "source": position.source,
                "score": position.score,
                "weight": position.weight,
            }
            trades.append(trade)
            orders.append(
                {
                    "date": date_label,
                    "code": code,
                    "side": "sell",
                    "status": "filled",
                    "reason": "hold_days",
                    "signal_date": position.signal_date,
                    "price": price,
                    "shares": position.shares,
                    "cash_delta": proceeds,
                    "source": position.source,
                    "score": position.score,
                    "weight": position.weight,
                    "metadata": position.metadata,
                }
            )
            del positions[code]

        # Then buy signals from the prior trading day.
        for intent in pending_buys.get(date, []):
            signal = intent.signal
            code = signal.symbol
            signal_date = signal.date
            if code in positions:
                orders.append(
                    {
                        "date": date_label,
                        "code": code,
                        "side": "buy",
                        "status": "skipped",
                        "reason": "already_held",
                        "signal_date": signal_date,
                        "price": "",
                        "shares": 0,
                        "cash_delta": 0.0,
                        "source": signal.source,
                        "score": signal.score,
                        "weight": signal.weight,
                        "metadata": signal.metadata,
                    }
                )
                continue
            if len(positions) >= settings.max_positions:
                orders.append(
                    {
                        "date": date_label,
                        "code": code,
                        "side": "buy",
                        "status": "skipped",
                        "reason": "max_positions",
                        "signal_date": signal_date,
                        "price": "",
                        "shares": 0,
                        "cash_delta": 0.0,
                        "source": signal.source,
                        "score": signal.score,
                        "weight": signal.weight,
                        "metadata": signal.metadata,
                    }
                )
                continue
            row = _row_on_or_none(prepared.get(code), date)
            can_buy, reason = _can_buy(row)
            if not can_buy:
                orders.append(
                    {
                        "date": date_label,
                        "code": code,
                        "side": "buy",
                        "status": "blocked",
                        "reason": reason,
                        "signal_date": signal_date,
                        "price": "",
                        "shares": 0,
                        "cash_delta": 0.0,
                        "source": signal.source,
                        "score": signal.score,
                        "weight": signal.weight,
                        "metadata": signal.metadata,
                    }
                )
                continue

            price = _execution_price(row, settings.buy_mode, side="buy")
            equity_before_buy = _portfolio_value(cash, positions, prepared, date)
            target_cash = min(
                cash,
                equity_before_buy * settings.position_pct,
                equity_before_buy / settings.max_positions,
            )
            shares = _shares_for_cash(
                cash_budget=target_cash,
                price=price,
                fee_model=settings.fee_model,
                lot_size=settings.lot_size,
            )
            total_cost = shares * price * (1.0 + settings.fee_model.buy_cost_rate)
            if shares <= 0 or total_cost > cash:
                orders.append(
                    {
                        "date": date_label,
                        "code": code,
                        "side": "buy",
                        "status": "skipped",
                        "reason": "insufficient_cash",
                        "signal_date": signal_date,
                        "price": price,
                        "shares": 0,
                        "cash_delta": 0.0,
                        "source": signal.source,
                        "score": signal.score,
                        "weight": signal.weight,
                        "metadata": signal.metadata,
                    }
                )
                continue

            cash -= total_cost
            positions[code] = Position(
                symbol=code,
                shares=shares,
                entry_date=date_label,
                entry_price=price,
                entry_value=total_cost,
                signal_date=signal_date,
                hold_days=settings.hold_days,
                source=signal.source,
                score=signal.score,
                weight=signal.weight,
                metadata=signal.metadata,
            )
            orders.append(
                {
                    "date": date_label,
                    "code": code,
                    "side": "buy",
                    "status": "filled",
                    "reason": "signal",
                    "signal_date": signal_date,
                    "price": price,
                    "shares": shares,
                    "cash_delta": -total_cost,
                    "source": signal.source,
                    "score": signal.score,
                    "weight": signal.weight,
                    "metadata": signal.metadata,
                }
            )

        equity = _portfolio_value(cash, positions, prepared, date)
        equity_rows.append(
            {
                "date": date_label,
                "cash": cash,
                "market_value": equity - cash,
                "total_value": equity,
                "total_return": equity / settings.initial_cash - 1.0 if settings.initial_cash else float("nan"),
                "position_count": len(positions),
            }
        )

    final_equity = equity_rows[-1]["total_value"] if equity_rows else settings.initial_cash
    total_return = final_equity / settings.initial_cash - 1.0 if settings.initial_cash else float("nan")
    risk_metrics = calculate_risk_metrics(
        equity_rows=equity_rows,
        trades=trades,
        initial_cash=settings.initial_cash,
    )
    summary = {
        "run_date": dt.date.today().isoformat(),
        "initial_cash": settings.initial_cash,
        "final_cash": final_equity,
        "cash": cash,
        "market_value": final_equity - cash,
        "total_return": total_return,
        "strategy": settings.strategy,
        "buy_mode": settings.buy_mode,
        "hold_days": settings.hold_days,
        "max_positions": settings.max_positions,
        "position_pct": settings.position_pct,
        "lot_size": settings.lot_size,
        "trade_count": len(trades),
        "order_count": len(orders),
        "realized_trade_count": len(trades),
        "open_position_count": len(positions),
        "commission_rate": settings.fee_model.commission_rate,
        "stamp_tax_rate": settings.fee_model.stamp_tax_rate,
        "transfer_fee_rate": settings.fee_model.transfer_fee_rate,
        **risk_metrics,
    }
    return PortfolioBacktestResult(
        initial_cash=settings.initial_cash,
        final_cash=final_equity,
        total_return=total_return,
        trades=trades,
        orders=orders,
        positions=[
            {
                "code": position.code,
                "shares": position.shares,
                "entry_date": position.entry_date,
                "entry_price": position.entry_price,
                "entry_value": position.entry_value,
                "signal_date": position.signal_date,
                "market_value": _position_market_value(position, prepared, dates[-1]),
                "source": position.source,
                "score": position.score,
                "weight": position.weight,
            }
            for position in sorted(positions.values(), key=lambda item: item.code)
        ],
        equity_curve=equity_rows,
        summary=summary,
    )


def run_staggered_cohort_portfolio_from_prepared(
    *,
    prepared: Dict[str, pd.DataFrame],
    picks_by_date: Dict[pd.Timestamp, List[str]] | SignalBook,
    settings: PortfolioSettings,
    cohort_count: int,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    show_progress: bool = False,
) -> PortfolioBacktestResult:
    """Run a strict fixed-slot staggered cohort portfolio.

    Capital is divided into ``cohort_count`` independent sleeves. Exactly one
    scheduled sleeve may open on each trading day. A sleeve is reusable only
    after all of its matured positions are sold; blocked exits never create an
    extra sleeve or expand the portfolio's capital budget.
    """

    if cohort_count <= 0:
        raise ValueError("cohort_count must be positive")
    if settings.hold_days != cohort_count:
        raise ValueError("strict staggered portfolios require cohort_count == hold_days")
    if settings.buy_mode != BUY_MODE_NEXT_OPEN:
        raise ValueError("staggered factor portfolios require next_open execution")
    dates = _build_all_trade_dates(prepared, start_date=start_date, end_date=end_date)
    if not dates:
        return _empty_cohort_result(settings, cohort_count)

    date_position = {date: index for index, date in enumerate(dates)}
    date_to_next = {date: dates[index + 1] for index, date in enumerate(dates[:-1])}
    signal_book = _coerce_signal_book(picks_by_date, source=settings.strategy)
    pending_buys: dict[pd.Timestamp, tuple[list[OrderIntent], str]] = {}
    for signal_date in signal_book:
        signal_ts = pd.to_datetime(signal_date)
        buy_date = date_to_next.get(signal_ts)
        if buy_date is not None:
            unique: dict[str, Signal] = {}
            for signal in signal_book.signals_for(signal_ts):
                unique.setdefault(signal.symbol, signal)
            pending_buys[buy_date] = (
                [
                    OrderIntent(signal=signal, execution_date=_date_str(buy_date))
                    for signal in unique.values()
                ],
                _date_str(signal_ts),
            )

    sleeve_cash = float(settings.initial_cash) / cohort_count
    sleeves = [
        CohortSleeve(cohort_id=index + 1, cash=sleeve_cash, positions={})
        for index in range(cohort_count)
    ]
    orders: List[dict] = []
    trades: List[dict] = []
    equity_rows: List[dict] = []

    backtest_dates = tqdm(
        dates,
        desc="组合回测",
        unit="交易日",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    for day_index, date in enumerate(backtest_dates):
        date_label = _date_str(date)

        # Retry every matured exit daily, but never reuse a partially blocked sleeve.
        for sleeve in sleeves:
            for code in list(sleeve.positions):
                position = sleeve.positions[code]
                entry_index = date_position.get(pd.to_datetime(position.entry_date))
                if entry_index is None or day_index - entry_index < settings.hold_days:
                    continue
                row = _row_on_or_none(prepared.get(code), date)
                can_sell, reason = _can_sell(row)
                if not can_sell:
                    orders.append(
                        _cohort_order(
                            date=date_label,
                            cohort_id=sleeve.cohort_id,
                            code=code,
                            side="sell",
                            status="blocked",
                            reason=reason,
                            signal_date=position.signal_date,
                            shares=position.shares,
                            source=position.source,
                            score=position.score,
                            weight=position.weight,
                            metadata=position.metadata,
                        )
                    )
                    continue
                price = _execution_price(row, settings.buy_mode, side="sell")
                gross = position.shares * price
                proceeds = gross * (1.0 - settings.fee_model.sell_cost_rate)
                sleeve.cash += proceeds
                pnl = proceeds - position.entry_value
                trades.append(
                    {
                        "cohort_id": sleeve.cohort_id,
                        "signal_date": position.signal_date,
                        "entry_date": position.entry_date,
                        "exit_date": date_label,
                        "strategy": settings.strategy,
                        "buy_mode": settings.buy_mode,
                        "hold_days": settings.hold_days,
                        "code": code,
                        "shares": position.shares,
                        "entry_price": position.entry_price,
                        "exit_price": price,
                        "entry_value": position.entry_value,
                        "exit_value": proceeds,
                        "return": pnl / position.entry_value if position.entry_value else float("nan"),
                        "pnl": pnl,
                        "source": position.source,
                        "score": position.score,
                        "weight": position.weight,
                    }
                )
                orders.append(
                    _cohort_order(
                        date=date_label,
                        cohort_id=sleeve.cohort_id,
                        code=code,
                        side="sell",
                        status="filled",
                        reason="hold_days",
                        signal_date=position.signal_date,
                        price=price,
                        shares=position.shares,
                        cash_delta=proceeds,
                        source=position.source,
                        score=position.score,
                        weight=position.weight,
                        metadata=position.metadata,
                    )
                )
                del sleeve.positions[code]

        scheduled = sleeves[day_index % cohort_count]
        pending = pending_buys.get(date)
        if pending is not None:
            pending_intents, signal_date = pending
            pending_signals = [intent.signal for intent in pending_intents]
            signal_by_code = {signal.symbol: signal for signal in pending_signals}
            if scheduled.positions:
                orders.append(
                    _cohort_order(
                        date=date_label,
                        cohort_id=scheduled.cohort_id,
                        code="",
                        side="buy",
                        status="skipped",
                        reason="cohort_exit_blocked",
                        signal_date=signal_date,
                        signal=pending_signals[0] if pending_signals else None,
                    )
                )
            else:
                selected, target_cash, rejected = _cohort_buy_candidates(
                    [signal.symbol for signal in pending_signals],
                    prepared=prepared,
                    date=date,
                    sleeve_cash=scheduled.cash,
                    fee_model=settings.fee_model,
                    lot_size=settings.lot_size,
                )
                for code, reason in rejected:
                    signal = signal_by_code[code]
                    orders.append(
                        _cohort_order(
                            date=date_label,
                            cohort_id=scheduled.cohort_id,
                            code=code,
                            side="buy",
                            status="blocked" if reason in {"suspended", "limit_up", "missing_bar"} else "skipped",
                            reason=reason,
                            signal_date=signal_date,
                            signal=signal,
                        )
                    )
                for code in selected:
                    signal = signal_by_code[code]
                    row = _row_on_or_none(prepared.get(code), date)
                    price = _execution_price(row, settings.buy_mode, side="buy")
                    shares = _shares_for_cash(
                        cash_budget=min(target_cash, scheduled.cash),
                        price=price,
                        fee_model=settings.fee_model,
                        lot_size=settings.lot_size,
                    )
                    total_cost = shares * price * (1.0 + settings.fee_model.buy_cost_rate)
                    if shares <= 0 or total_cost > scheduled.cash:
                        orders.append(
                            _cohort_order(
                                date=date_label,
                                cohort_id=scheduled.cohort_id,
                                code=code,
                                side="buy",
                                status="skipped",
                                reason="insufficient_sleeve_cash",
                                signal_date=signal_date,
                                price=price,
                                signal=signal,
                            )
                        )
                        continue
                    scheduled.cash -= total_cost
                    scheduled.positions[code] = Position(
                        symbol=code,
                        shares=shares,
                        entry_date=date_label,
                        entry_price=price,
                        entry_value=total_cost,
                        signal_date=signal_date,
                        hold_days=settings.hold_days,
                        source=signal.source,
                        score=signal.score,
                        weight=signal.weight,
                        metadata=signal.metadata,
                    )
                    orders.append(
                        _cohort_order(
                            date=date_label,
                            cohort_id=scheduled.cohort_id,
                            code=code,
                            side="buy",
                            status="filled",
                            reason="signal",
                            signal_date=signal_date,
                            price=price,
                            shares=shares,
                            cash_delta=-total_cost,
                            signal=signal,
                        )
                    )

        cash = sum(sleeve.cash for sleeve in sleeves)
        market_value = sum(
            _position_market_value(position, prepared, date)
            for sleeve in sleeves
            for position in sleeve.positions.values()
        )
        position_count = sum(len(sleeve.positions) for sleeve in sleeves)
        active_cohort_count = sum(bool(sleeve.positions) for sleeve in sleeves)
        total_value = cash + market_value
        equity_rows.append(
            {
                "date": date_label,
                "cash": cash,
                "market_value": market_value,
                "total_value": total_value,
                "total_return": total_value / settings.initial_cash - 1.0,
                "position_count": position_count,
                "active_cohort_count": active_cohort_count,
            }
        )

    final_equity = equity_rows[-1]["total_value"]
    risk_metrics = calculate_risk_metrics(
        equity_rows=equity_rows,
        trades=trades,
        initial_cash=settings.initial_cash,
    )
    positions = [
        {
            "cohort_id": sleeve.cohort_id,
            "code": position.code,
            "shares": position.shares,
            "entry_date": position.entry_date,
            "entry_price": position.entry_price,
            "entry_value": position.entry_value,
            "signal_date": position.signal_date,
            "market_value": _position_market_value(position, prepared, dates[-1]),
            "source": position.source,
            "score": position.score,
            "weight": position.weight,
        }
        for sleeve in sleeves
        for position in sorted(sleeve.positions.values(), key=lambda item: item.code)
    ]
    summary = {
        "run_date": dt.date.today().isoformat(),
        "portfolio_mode": "strict_staggered_cohorts",
        "initial_cash": settings.initial_cash,
        "initial_cash_per_cohort": sleeve_cash,
        "final_cash": final_equity,
        "cash": equity_rows[-1]["cash"],
        "market_value": equity_rows[-1]["market_value"],
        "total_return": final_equity / settings.initial_cash - 1.0,
        "strategy": settings.strategy,
        "buy_mode": settings.buy_mode,
        "hold_days": settings.hold_days,
        "cohort_count": cohort_count,
        "lot_size": settings.lot_size,
        "trade_count": len(trades),
        "order_count": len(orders),
        "realized_trade_count": len(trades),
        "open_position_count": len(positions),
        "active_cohort_count": equity_rows[-1]["active_cohort_count"],
        "commission_rate": settings.fee_model.commission_rate,
        "stamp_tax_rate": settings.fee_model.stamp_tax_rate,
        "transfer_fee_rate": settings.fee_model.transfer_fee_rate,
        **risk_metrics,
    }
    return PortfolioBacktestResult(
        initial_cash=settings.initial_cash,
        final_cash=final_equity,
        total_return=summary["total_return"],
        trades=trades,
        orders=orders,
        positions=positions,
        equity_curve=equity_rows,
        summary=summary,
    )


def _cohort_buy_candidates(
    codes: Sequence[str],
    *,
    prepared: Dict[str, pd.DataFrame],
    date: pd.Timestamp,
    sleeve_cash: float,
    fee_model: FeeModel,
    lot_size: int,
) -> tuple[list[str], float, list[tuple[str, str]]]:
    """Choose the largest rank-preserving equal-weight set affordable by a sleeve."""

    viable: list[tuple[str, float]] = []
    rejected: list[tuple[str, str]] = []
    for code in codes:
        row = _row_on_or_none(prepared.get(code), date)
        can_buy, reason = _can_buy(row)
        if not can_buy:
            rejected.append((code, reason))
            continue
        price = _execution_price(row, BUY_MODE_NEXT_OPEN, side="buy")
        minimum_cost = lot_size * price * (1.0 + fee_model.buy_cost_rate)
        viable.append((code, minimum_cost))

    for count in range(len(viable), 0, -1):
        target_cash = sleeve_cash / count
        affordable = [code for code, minimum_cost in viable if minimum_cost <= target_cash]
        if len(affordable) >= count:
            selected = affordable[:count]
            rejected.extend(
                (code, "insufficient_sleeve_cash")
                for code, _ in viable
                if code not in selected
            )
            return selected, target_cash, rejected
    rejected.extend((code, "insufficient_sleeve_cash") for code, _ in viable)
    return [], 0.0, rejected


def _cohort_order(
    *,
    date: str,
    cohort_id: int,
    code: str,
    side: str,
    status: str,
    reason: str,
    signal_date: str,
    price: object = "",
    shares: int = 0,
    cash_delta: float = 0.0,
    signal: Signal | None = None,
    source: str = "",
    score: float | None = None,
    weight: float | None = None,
    metadata: dict | None = None,
) -> OrderResult:
    if signal is not None:
        source = signal.source
        score = signal.score
        weight = signal.weight
        metadata = signal.metadata
    return OrderResult(
        date=date,
        cohort_id=cohort_id,
        symbol=code or None,
        side=side,
        status=status,
        reason=reason,
        signal_date=signal_date,
        price=price,
        shares=shares,
        cash_delta=cash_delta,
        source=source,
        score=score,
        weight=weight,
        metadata=dict(metadata or {}),
    )


def _empty_cohort_result(
    settings: PortfolioSettings,
    cohort_count: int,
) -> PortfolioBacktestResult:
    summary = {
        "run_date": dt.date.today().isoformat(),
        "portfolio_mode": "strict_staggered_cohorts",
        "initial_cash": settings.initial_cash,
        "initial_cash_per_cohort": settings.initial_cash / cohort_count,
        "final_cash": settings.initial_cash,
        "cash": settings.initial_cash,
        "market_value": 0.0,
        "total_return": 0.0,
        "strategy": settings.strategy,
        "buy_mode": settings.buy_mode,
        "hold_days": settings.hold_days,
        "cohort_count": cohort_count,
        "lot_size": settings.lot_size,
        "trade_count": 0,
        "order_count": 0,
        "realized_trade_count": 0,
        "open_position_count": 0,
        "active_cohort_count": 0,
        "commission_rate": settings.fee_model.commission_rate,
        "stamp_tax_rate": settings.fee_model.stamp_tax_rate,
        "transfer_fee_rate": settings.fee_model.transfer_fee_rate,
        "max_drawdown": 0.0,
        "annualized_return_mean": None,
        "overall_annualized_return": None,
        "average_yearly_annualized_return": None,
        "year_count": 0,
        "annualized_volatility": None,
        "sharpe_ratio": None,
    }
    return PortfolioBacktestResult(
        initial_cash=settings.initial_cash,
        final_cash=settings.initial_cash,
        total_return=0.0,
        trades=[],
        orders=[],
        positions=[],
        equity_curve=[],
        summary=summary,
    )


def _write_trades_csv(path: Path, trades: List[dict]) -> None:
    fieldnames = [
        "cohort_id",
        "signal_date",
        "entry_date",
        "exit_date",
        "strategy",
        "buy_mode",
        "hold_days",
        "code",
        "shares",
        "entry_price",
        "exit_price",
        "entry_value",
        "exit_value",
        "return",
        "pnl",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for trade in trades:
            row = {key: trade.get(key, "") for key in fieldnames}
            row["return"] = format_percent(row["return"])
            writer.writerow(row)


def _write_equity_curve_csv(path: Path, rows: List[dict]) -> None:
    fieldnames = [
        "date",
        "cash",
        "market_value",
        "total_value",
        "total_return",
        "position_count",
        "active_cohort_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            formatted = {key: row.get(key, "") for key in fieldnames}
            formatted["total_return"] = format_percent(formatted["total_return"])
            writer.writerow(formatted)


def _write_yearly_returns_csv(path: Path, rows: List[dict]) -> None:
    fieldnames = [
        "year",
        "period_start_date",
        "period_end_date",
        "trading_days",
        "start_equity",
        "end_equity",
        "total_return",
        "annualized_return",
        "is_partial_year",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_orders_csv(path: Path, orders: List[dict]) -> None:
    fieldnames = [
        "date",
        "cohort_id",
        "code",
        "side",
        "status",
        "reason",
        "signal_date",
        "price",
        "shares",
        "cash_delta",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for order in orders:
            writer.writerow({key: order.get(key, "") for key in fieldnames})


def _write_positions_csv(path: Path, positions: List[dict]) -> None:
    fieldnames = [
        "cohort_id",
        "code",
        "shares",
        "entry_date",
        "entry_price",
        "entry_value",
        "signal_date",
        "market_value",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for position in positions:
            writer.writerow({key: position.get(key, "") for key in fieldnames})


def _write_equity_curve_html(path: Path, rows: List[dict], summary: dict) -> None:
    try:
        import plotly.graph_objects as go
    except ImportError:
        html = "<html><body><p>plotly is not installed. See equity_curve.csv.</p></body></html>"
        path.write_text(html, encoding="utf-8")
        return

    dates = [row["date"] for row in rows]
    equity_values = [row.get("total_value", row.get("cash")) for row in rows]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=equity_values,
            mode="lines",
            name="Portfolio value",
            hovertemplate="%{x}<br>value=%{y:,.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=(
            f"Portfolio Equity Curve | {summary['strategy']} | "
            f"{summary['buy_mode']} | hold {summary['hold_days']}d"
        ),
        xaxis_title="Date",
        yaxis_title="Cash",
        template="plotly_white",
        hovermode="x unified",
    )
    fig.write_html(str(path), include_plotlyjs="cdn")


def write_portfolio_backtest_outputs(
    result: PortfolioBacktestResult,
    output_dir: str | Path,
) -> WorkflowResult[PortfolioBacktestResult]:
    """Write the standard auditable portfolio-backtest artifact set."""

    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    trades_path = resolved_output_dir / "portfolio_trades.csv"
    orders_path = resolved_output_dir / "daily_trade_plan.csv"
    orders_json_path = resolved_output_dir / "daily_trade_plan.json"
    positions_path = resolved_output_dir / "open_positions.csv"
    summary_path = resolved_output_dir / "portfolio_summary.json"
    equity_curve_path = resolved_output_dir / "equity_curve.csv"
    equity_curve_html_path = resolved_output_dir / "equity_curve.html"
    yearly_returns_path = resolved_output_dir / "yearly_returns.csv"
    yearly_rows = yearly_return_rows(
        result.equity_curve,
        initial_cash=result.initial_cash,
    )
    _write_trades_csv(trades_path, result.trades)
    _write_orders_csv(orders_path, result.orders)
    _write_positions_csv(positions_path, result.positions)
    _write_equity_curve_csv(equity_curve_path, result.equity_curve)
    _write_yearly_returns_csv(yearly_returns_path, yearly_rows)
    _write_equity_curve_html(equity_curve_html_path, result.equity_curve, result.summary)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(result.summary.to_dict(), f, ensure_ascii=False, indent=2)
    with orders_json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {"orders": [order.to_legacy_dict() for order in result.orders]},
            f,
            ensure_ascii=False,
            indent=2,
        )
    return WorkflowResult.from_mapping(
        {
            "result": result,
            "trades_path": trades_path,
            "orders_path": orders_path,
            "orders_json_path": orders_json_path,
            "positions_path": positions_path,
            "summary_path": summary_path,
            "equity_curve_path": equity_curve_path,
            "equity_curve_html_path": equity_curve_html_path,
            "yearly_returns_path": yearly_returns_path,
        }
    )


def run_portfolio_backtest(
    *,
    initial_cash: float,
    strategy: str,
    buy_mode: str,
    hold_days: int,
    commission_wan: float,
    max_positions: int = 10,
    position_pct: float = 0.1,
    lot_size: int = 100,
    config_path: Optional[str] = None,
    data_dir: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> WorkflowResult[PortfolioBacktestResult]:
    cfg = load_config(config_path)
    global_cfg = cfg.get("global", {})
    resolved_data_dir = str(_resolve_cfg_path(data_dir or global_cfg.get("data_dir", "./data/raw")))
    resolved_output_dir = _resolve_cfg_path(output_dir or DEFAULT_OUTPUT_DIR)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    raw_data = load_raw_data(resolved_data_dir, end_date=None)
    raw_data = clean_market_data(raw_data)
    warmup = _calc_warmup(cfg, int(global_cfg.get("min_bars_buffer", 10)))
    n_turnover_days = int(global_cfg.get("n_turnover_days", 43))
    stock_pool_cfg = cfg.get("stock_pool", {})
    top_m = int(stock_pool_cfg.get("top_m", global_cfg.get("top_m", 5000)))
    excluded_boards = tuple(stock_pool_cfg.get("exclude_boards", ()))
    start_ts = pd.to_datetime(start_date) if start_date else None
    end_ts = pd.to_datetime(end_date) if end_date else None

    selectors = filter_selectors_by_strategy(build_enabled_selectors(cfg), (strategy,))
    if not selectors:
        raise ValueError(f"Strategy is not enabled or unknown: {strategy}")

    strategy_name, selector = selectors[0]
    preparer = MarketDataPreparer(
        start_date=start_ts,
        end_date=None,
        warmup_bars=warmup,
        n_turnover_days=n_turnover_days,
        selector=selector,
    )
    prepared = preparer.prepare(raw_data)
    top_pool = build_stock_pool_by_date(
        prepared,
        config=StockPoolConfig(
            top_m=top_m,
            min_price=float(stock_pool_cfg.get("min_price", 1.0)),
            min_turnover=float(stock_pool_cfg.get("min_turnover", 0.0)),
            exclude_boards=excluded_boards,
            require_tradeable=bool(stock_pool_cfg.get("require_tradeable", True)),
        ),
    )
    picks = SelectorPickPrecomputer(
        selector=selector,
        start_date=start_ts,
        end_date=end_ts,
    ).precompute(prepared, top_turnover_pool=top_pool)
    picks = filter_picks_by_date(picks, start_date=start_date, end_date=end_date)

    settings = PortfolioSettings(
        initial_cash=initial_cash,
        strategy=strategy_name,
        buy_mode=buy_mode,
        hold_days=hold_days,
        fee_model=FeeModel.from_commission_wan(commission_wan),
        max_positions=max_positions,
        position_pct=position_pct,
        lot_size=lot_size,
    )
    result = run_realistic_portfolio_from_prepared(
        prepared=prepared,
        picks_by_date=picks,
        settings=settings,
        start_date=start_date,
        end_date=end_date,
    )

    return write_portfolio_backtest_outputs(result, resolved_output_dir)
