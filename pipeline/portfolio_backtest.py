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

from .signal_returns import (
    BUY_MODE_NEXT_OPEN,
    BUY_MODE_SIGNAL_CLOSE,
    DEFAULT_HORIZONS,
    VALID_BUY_MODES,
    build_enabled_selectors,
    filter_picks_by_date,
    filter_selectors_by_strategy,
    format_percent,
)
from .pipeline_core import MarketDataPreparer, SelectorPickPrecomputer, TopTurnoverPoolBuilder
from .select_stock import _calc_warmup, _resolve_cfg_path, load_config, load_raw_data


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


@dataclass(frozen=True)
class PortfolioBacktestResult:
    initial_cash: float
    final_cash: float
    total_return: float
    trades: List[dict]
    summary: dict


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
) -> dict:
    max_drawdown = 0.0
    peak = None
    for row in equity_rows:
        cash = float(row["cash"])
        peak = cash if peak is None else max(peak, cash)
        if peak and peak > 0:
            drawdown = cash / peak - 1.0
            max_drawdown = max(max_drawdown, abs(min(drawdown, 0.0)))

    returns = pd.Series(
        [float(trade["basket_return"]) for trade in trades if math.isfinite(float(trade["basket_return"]))],
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

    return {
        "max_drawdown": max_drawdown,
        "annualized_return_mean": annualized_return_mean,
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
    picks_by_date: Dict[pd.Timestamp, List[str]],
    settings: PortfolioSettings,
) -> PortfolioBacktestResult:
    if settings.hold_days <= 0:
        raise ValueError("hold_days must be positive")
    if settings.buy_mode not in VALID_BUY_MODES:
        raise ValueError(f"buy_mode must be one of {sorted(VALID_BUY_MODES)}")

    cash = float(settings.initial_cash)
    trades: List[dict] = []

    for signal_date in sorted(picks_by_date):
        signal_ts = pd.to_datetime(signal_date)
        stock_returns: List[float] = []
        details: List[dict] = []

        for code in sorted(picks_by_date[signal_date]):
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
            }
        )

    total_return = cash / settings.initial_cash - 1.0 if settings.initial_cash else float("nan")
    equity_rows = build_equity_curve_rows(
        initial_cash=settings.initial_cash,
        trades=trades,
        start_date=None,
    )
    risk_metrics = calculate_risk_metrics(equity_rows=equity_rows, trades=trades)
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
        summary=summary,
    )


def _write_trades_csv(path: Path, trades: List[dict]) -> None:
    fieldnames = [
        "signal_date",
        "strategy",
        "buy_mode",
        "hold_days",
        "stock_count",
        "start_cash",
        "basket_return",
        "end_cash",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for trade in trades:
            row = {key: trade[key] for key in fieldnames}
            row["basket_return"] = format_percent(row["basket_return"])
            writer.writerow(row)


def _write_equity_curve_csv(path: Path, rows: List[dict]) -> None:
    fieldnames = ["date", "cash", "total_return"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            formatted = dict(row)
            formatted["total_return"] = format_percent(formatted["total_return"])
            writer.writerow(formatted)


def _write_equity_curve_html(path: Path, rows: List[dict], summary: dict) -> None:
    try:
        import plotly.graph_objects as go
    except ImportError:
        html = "<html><body><p>plotly is not installed. See equity_curve.csv.</p></body></html>"
        path.write_text(html, encoding="utf-8")
        return

    dates = [row["date"] for row in rows]
    cash_values = [row["cash"] for row in rows]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=cash_values,
            mode="lines",
            name="Portfolio value",
            hovertemplate="%{x}<br>cash=%{y:,.2f}<extra></extra>",
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


def run_portfolio_backtest(
    *,
    initial_cash: float,
    strategy: str,
    buy_mode: str,
    hold_days: int,
    commission_wan: float,
    config_path: Optional[str] = None,
    data_dir: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> dict:
    cfg = load_config(config_path)
    global_cfg = cfg.get("global", {})
    resolved_data_dir = str(_resolve_cfg_path(data_dir or global_cfg.get("data_dir", "./data/raw")))
    resolved_output_dir = _resolve_cfg_path(output_dir or DEFAULT_OUTPUT_DIR)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    raw_data = load_raw_data(resolved_data_dir, end_date=None)
    warmup = _calc_warmup(cfg, int(global_cfg.get("min_bars_buffer", 10)))
    n_turnover_days = int(global_cfg.get("n_turnover_days", 43))
    top_m = int(global_cfg.get("top_m", 5000))
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
    top_pool = TopTurnoverPoolBuilder(top_m=top_m).build(prepared)
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
    )
    result = run_portfolio_from_prepared(
        prepared=prepared,
        picks_by_date=picks,
        settings=settings,
    )

    trades_path = resolved_output_dir / "portfolio_trades.csv"
    summary_path = resolved_output_dir / "portfolio_summary.json"
    equity_curve_path = resolved_output_dir / "equity_curve.csv"
    equity_curve_html_path = resolved_output_dir / "equity_curve.html"
    equity_curve_rows = build_equity_curve_rows(
        initial_cash=initial_cash,
        trades=result.trades,
        start_date=start_date,
    )
    _write_trades_csv(trades_path, result.trades)
    _write_equity_curve_csv(equity_curve_path, equity_curve_rows)
    _write_equity_curve_html(equity_curve_html_path, equity_curve_rows, result.summary)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(result.summary, f, ensure_ascii=False, indent=2)

    return {
        "result": result,
        "trades_path": trades_path,
        "summary_path": summary_path,
        "equity_curve_path": equity_curve_path,
        "equity_curve_html_path": equity_curve_html_path,
    }
