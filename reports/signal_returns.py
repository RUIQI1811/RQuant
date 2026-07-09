from __future__ import annotations

import csv
import datetime as dt
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from pipeline.pipeline_core import MarketDataPreparer, SelectorPickPrecomputer, TopTurnoverPoolBuilder
from strategies.bdsr_macd_obv import BDSRMACDOBVSelector
from strategies.mbdsr import MBDSRSelector
from strategies.preselect import _calc_warmup, _resolve_cfg_path, _sorted_zx, load_config, load_raw_data
from strategies.selector import B1Selector, BrickChartSelector


DEFAULT_HORIZONS = (1, 5, 10)
DEFAULT_OUTPUT_DIR = Path("data") / "backtest"
BUY_MODE_SIGNAL_CLOSE = "signal_close"
BUY_MODE_NEXT_OPEN = "next_open"
VALID_BUY_MODES = {BUY_MODE_SIGNAL_CLOSE, BUY_MODE_NEXT_OPEN}


def format_percent(value: object) -> str:
    if value is None:
        return ""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(numeric):
        return ""
    return f"{numeric:.2%}"


def _as_timestamp(value: object) -> pd.Timestamp:
    return pd.to_datetime(value)


def _return_at_horizon(
    df: pd.DataFrame,
    base_pos: int,
    horizon: int,
    entry_price: float,
    *,
    exit_price_col: str,
) -> float:
    future_pos = base_pos + int(horizon)
    if future_pos >= len(df) or entry_price == 0:
        return float("nan")
    future_price = float(df.iloc[future_pos][exit_price_col])
    return future_price / entry_price - 1.0


def _resolve_entry(
    df: pd.DataFrame,
    signal_pos: int,
    buy_mode: str,
) -> tuple[int, str, float]:
    if buy_mode == BUY_MODE_SIGNAL_CLOSE:
        entry_pos = signal_pos
        entry_price = float(df.iloc[entry_pos]["close"])
    elif buy_mode == BUY_MODE_NEXT_OPEN:
        entry_pos = signal_pos + 1
        if entry_pos >= len(df):
            return entry_pos, "", float("nan")
        entry_price = float(df.iloc[entry_pos]["open"])
    else:
        raise ValueError(f"Unsupported buy_mode: {buy_mode}")

    entry_date = pd.to_datetime(df.iloc[entry_pos]["date"]).strftime("%Y-%m-%d")
    return entry_pos, entry_date, entry_price


def build_signal_return_rows(
    prepared: Dict[str, pd.DataFrame],
    picks_by_date: Dict[pd.Timestamp, List[str]],
    *,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    strategy: str,
    buy_mode: str = BUY_MODE_SIGNAL_CLOSE,
) -> List[dict]:
    if buy_mode not in VALID_BUY_MODES:
        raise ValueError(f"buy_mode must be one of {sorted(VALID_BUY_MODES)}")

    rows: List[dict] = []

    for pick_date in sorted(picks_by_date):
        pick_ts = _as_timestamp(pick_date)
        for code in sorted(picks_by_date[pick_date]):
            df = prepared.get(code)
            if df is None or df.empty:
                continue

            index = pd.DatetimeIndex(df.index)
            matches = index.get_indexer([pick_ts])
            pos = int(matches[0])
            if pos < 0:
                continue

            signal_close = float(df.iloc[pos]["close"])
            entry_pos, entry_date, entry_price = _resolve_entry(df, pos, buy_mode)
            exit_base_pos = pos if buy_mode == BUY_MODE_SIGNAL_CLOSE else entry_pos
            exit_price_col = "close" if buy_mode == BUY_MODE_SIGNAL_CLOSE else "open"
            row = {
                "date": pick_ts.strftime("%Y-%m-%d"),
                "code": code,
                "strategy": strategy,
                "buy_mode": buy_mode,
                "close": signal_close,
                "entry_date": entry_date,
                "entry_price": entry_price,
            }
            for horizon in horizons:
                row[f"return_{int(horizon)}d"] = _return_at_horizon(
                    df,
                    exit_base_pos,
                    int(horizon),
                    entry_price,
                    exit_price_col=exit_price_col,
                )
            rows.append(row)

    return rows


def filter_picks_by_date(
    picks_by_date: Dict[pd.Timestamp, List[str]],
    *,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[pd.Timestamp, List[str]]:
    start_ts = pd.to_datetime(start_date) if start_date else None
    end_ts = pd.to_datetime(end_date) if end_date else None
    filtered: Dict[pd.Timestamp, List[str]] = {}
    for pick_date, codes in picks_by_date.items():
        pick_ts = pd.to_datetime(pick_date)
        if start_ts is not None and pick_ts < start_ts:
            continue
        if end_ts is not None and pick_ts > end_ts:
            continue
        filtered[pick_ts] = codes
    return filtered


def summarize_signal_returns(
    rows: Iterable[dict],
    *,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
) -> dict:
    materialized = list(rows)
    summary = {}
    for horizon in horizons:
        key = f"return_{int(horizon)}d"
        values = [
            float(row[key])
            for row in materialized
            if key in row and math.isfinite(float(row[key]))
        ]
        series = pd.Series(values, dtype="float64")
        summary[key] = {
            "count": int(series.count()),
            "mean_return": float(series.mean()) if not series.empty else None,
            "median_return": float(series.median()) if not series.empty else None,
            "win_rate": float((series > 0).mean()) if not series.empty else None,
        }
    return summary


def summary_to_rows(summary: dict) -> List[dict]:
    return [
        {
            "horizon": horizon,
            "count": metrics["count"],
            "mean_return": metrics["mean_return"],
            "median_return": metrics["median_return"],
            "win_rate": metrics["win_rate"],
        }
        for horizon, metrics in summary.items()
    ]


def _make_b1_selector(cfg_b1: dict) -> B1Selector:
    zx_m1, zx_m2, zx_m3, zx_m4 = _sorted_zx(
        cfg_b1["zx_m1"],
        cfg_b1["zx_m2"],
        cfg_b1["zx_m3"],
        cfg_b1["zx_m4"],
    )
    return B1Selector(
        j_threshold=float(cfg_b1["j_threshold"]),
        j_q_threshold=float(cfg_b1["j_q_threshold"]),
        zx_m1=zx_m1,
        zx_m2=zx_m2,
        zx_m3=zx_m3,
        zx_m4=zx_m4,
    )


def _make_brick_selector(cfg_brick: dict) -> BrickChartSelector:
    return BrickChartSelector(
        daily_return_threshold=float(cfg_brick.get("daily_return_threshold", 0.05)),
        brick_growth_ratio=float(cfg_brick.get("brick_growth_ratio", 1.0)),
        min_prior_green_bars=int(cfg_brick.get("min_prior_green_bars", 2)),
        zxdq_ratio=cfg_brick.get("zxdq_ratio"),
        zxdq_span=int(cfg_brick.get("zxdq_span", 10)),
        require_zxdq_gt_zxdkx=bool(cfg_brick.get("require_zxdq_gt_zxdkx", True)),
        zxdkx_m1=int(cfg_brick.get("zxdkx_m1", 14)),
        zxdkx_m2=int(cfg_brick.get("zxdkx_m2", 28)),
        zxdkx_m3=int(cfg_brick.get("zxdkx_m3", 57)),
        zxdkx_m4=int(cfg_brick.get("zxdkx_m4", 114)),
        require_weekly_ma_bull=bool(cfg_brick.get("require_weekly_ma_bull", True)),
        wma_short=int(cfg_brick.get("wma_short", 20)),
        wma_mid=int(cfg_brick.get("wma_mid", 60)),
        wma_long=int(cfg_brick.get("wma_long", 120)),
        n=int(cfg_brick.get("n", 4)),
        m1=int(cfg_brick.get("m1", 4)),
        m2=int(cfg_brick.get("m2", 6)),
        m3=int(cfg_brick.get("m3", 6)),
        t=float(cfg_brick.get("t", 4.0)),
        shift1=float(cfg_brick.get("shift1", 90.0)),
        shift2=float(cfg_brick.get("shift2", 100.0)),
        sma_w1=int(cfg_brick.get("sma_w1", 1)),
        sma_w2=int(cfg_brick.get("sma_w2", 1)),
        sma_w3=int(cfg_brick.get("sma_w3", 1)),
    )


def _make_mbdsr_selector(cfg_mbdsr: dict) -> MBDSRSelector:
    return MBDSRSelector(
        use_next_confirm=bool(cfg_mbdsr.get("use_next_confirm", False)),
        extra_bars_buffer=int(cfg_mbdsr.get("extra_bars_buffer", 10)),
    )


def _make_bdsr_macd_obv_selector(cfg_strategy: dict) -> BDSRMACDOBVSelector:
    return BDSRMACDOBVSelector(
        bdsr_fast_window=int(cfg_strategy.get("bdsr_fast_window", 9)),
        bdsr_slow_window=int(cfg_strategy.get("bdsr_slow_window", 26)),
        macd_fast_period=int(cfg_strategy.get("macd_fast_period", 12)),
        macd_slow_period=int(cfg_strategy.get("macd_slow_period", 26)),
        macd_signal_period=int(cfg_strategy.get("macd_signal_period", 9)),
        obv_ma_window=int(cfg_strategy.get("obv_ma_window", 20)),
        obv_trend_lookback=int(cfg_strategy.get("obv_trend_lookback", 3)),
        extra_bars_buffer=int(cfg_strategy.get("extra_bars_buffer", 10)),
    )


def build_enabled_selectors(cfg: dict) -> List[Tuple[str, object]]:
    selectors: List[Tuple[str, object]] = []
    if cfg.get("b1", {}).get("enabled", True):
        selectors.append(("b1", _make_b1_selector(cfg["b1"])))
    if cfg.get("brick", {}).get("enabled", True):
        selectors.append(("brick", _make_brick_selector(cfg["brick"])))
    if cfg.get("bdsr_macd_obv", {}).get("enabled", False):
        selectors.append(
            (
                "bdsr_macd_obv",
                _make_bdsr_macd_obv_selector(cfg["bdsr_macd_obv"]),
            )
        )
    if cfg.get("mbdsr", {}).get("enabled", False):
        selector = _make_mbdsr_selector(cfg["mbdsr"])
        name = "mbdsr_confirm" if selector.use_next_confirm else "mbdsr"
        selectors.append((name, selector))
    return selectors


def filter_selectors_by_strategy(
    selectors: List[Tuple[str, object]],
    strategies: Optional[Sequence[str]] = None,
) -> List[Tuple[str, object]]:
    if not strategies:
        return selectors
    wanted = {strategy.strip().lower() for strategy in strategies if strategy.strip()}
    return [(name, selector) for name, selector in selectors if name.lower() in wanted]


def _write_csv(path: Path, rows: List[dict], horizons: Sequence[int]) -> None:
    fieldnames = ["date", "code", "strategy", "buy_mode", "close", "entry_date", "entry_price"] + [
        f"return_{int(h)}d" for h in horizons
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            formatted = dict(row)
            for horizon in horizons:
                key = f"return_{int(horizon)}d"
                formatted[key] = format_percent(formatted.get(key))
            writer.writerow(formatted)


def _write_summary_csv(path: Path, rows: List[dict]) -> None:
    fieldnames = ["horizon", "count", "mean_return", "median_return", "win_rate"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            formatted = dict(row)
            formatted["mean_return"] = format_percent(formatted.get("mean_return"))
            formatted["median_return"] = format_percent(formatted.get("median_return"))
            formatted["win_rate"] = format_percent(formatted.get("win_rate"))
            writer.writerow(formatted)


def prepare_strategy_data(
    *,
    raw_data: Dict[str, pd.DataFrame],
    selectors: Sequence[Tuple[str, object]],
    start_date: Optional[pd.Timestamp],
    warmup_bars: int,
    n_turnover_days: int,
    top_m: int,
    reuse_base_preparation: bool = False,
) -> Iterable[
    Tuple[
        str,
        object,
        Dict[str, pd.DataFrame],
        Dict[pd.Timestamp, List[str]],
    ]
]:
    if reuse_base_preparation:
        preparer = MarketDataPreparer(
            start_date=start_date,
            end_date=None,
            warmup_bars=warmup_bars,
            n_turnover_days=n_turnover_days,
            selector=None,
        )
        base_prepared = preparer.prepare_base_only(raw_data)
        top_pool = TopTurnoverPoolBuilder(top_m=top_m).build(base_prepared)
        for strategy, selector in selectors:
            prepared = preparer.apply_selector_features(base_prepared, selector)
            yield strategy, selector, prepared, top_pool
        return

    for strategy, selector in selectors:
        preparer = MarketDataPreparer(
            start_date=start_date,
            end_date=None,
            warmup_bars=warmup_bars,
            n_turnover_days=n_turnover_days,
            selector=selector,
        )
        prepared = preparer.prepare(raw_data)
        top_pool = TopTurnoverPoolBuilder(top_m=top_m).build(prepared)
        yield strategy, selector, prepared, top_pool


def run_signal_returns(
    *,
    config_path: Optional[str] = None,
    data_dir: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    output_dir: Optional[str] = None,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    strategies: Optional[Sequence[str]] = None,
    buy_mode: str = BUY_MODE_SIGNAL_CLOSE,
    reuse_base_preparation: bool = False,
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
    selectors = filter_selectors_by_strategy(build_enabled_selectors(cfg), strategies)
    all_rows: List[dict] = []

    prepared_strategies = prepare_strategy_data(
        raw_data=raw_data,
        selectors=selectors,
        start_date=start_ts,
        warmup_bars=warmup,
        n_turnover_days=n_turnover_days,
        top_m=top_m,
        reuse_base_preparation=reuse_base_preparation,
    )
    for strategy, selector, prepared, top_pool in prepared_strategies:
        picks = SelectorPickPrecomputer(
            selector=selector,
            start_date=start_ts,
            end_date=end_ts,
        ).precompute(prepared, top_turnover_pool=top_pool)
        picks = filter_picks_by_date(
            picks,
            start_date=start_date,
            end_date=end_date,
        )
        all_rows.extend(
            build_signal_return_rows(
                prepared,
                picks,
                horizons=horizons,
                strategy=strategy,
                buy_mode=buy_mode,
            )
        )

    all_rows.sort(key=lambda row: (row["date"], row["strategy"], row["code"]))
    metrics = summarize_signal_returns(all_rows, horizons=horizons)
    summary = {
        "run_date": dt.date.today().isoformat(),
        "start_date": start_date,
        "end_date": end_date,
        "horizons": [int(h) for h in horizons],
        "buy_mode": buy_mode,
        "reuse_base_preparation": bool(reuse_base_preparation),
        "total_signals": len(all_rows),
        "metrics": metrics,
    }

    csv_path = resolved_output_dir / "signal_returns.csv"
    summary_path = resolved_output_dir / "signal_summary.json"
    summary_csv_path = resolved_output_dir / "signal_summary.csv"
    _write_csv(csv_path, all_rows, horizons)
    _write_summary_csv(summary_csv_path, summary_to_rows(metrics))
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return {
        "rows": all_rows,
        "summary": summary,
        "csv_path": csv_path,
        "summary_path": summary_path,
        "summary_csv_path": summary_csv_path,
    }
