"""Command-line entrypoint for RQuant research workflows."""
from __future__ import annotations

import argparse
import datetime
import logging
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Callable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reports.research_report import DEFAULT_REPORT_OUTPUT_DIR, run_research_report
from domain.artifacts import WorkflowResult
from rquant.runtime import CommandResult

BUY_MODE_SIGNAL_CLOSE = "signal_close"
VALID_BUY_MODES = {BUY_MODE_SIGNAL_CLOSE, "next_open"}
DEFAULT_HORIZONS = (1, 5, 10)

run_signal_returns: Callable | None = None
run_portfolio_backtest: Callable | None = None
run_filter_rank_portfolio_backtest: Callable | None = None
run_rank_ensemble_portfolio_backtest: Callable | None = None
run_signal_portfolio_backtest: Callable | None = None

logger = logging.getLogger("cli")


def _add_log_file(log_dir: str, pick_date: str) -> None:
    """可选：追加文件日志到 data/logs/rquant_YYYY-MM-DD.log。"""
    p = Path(log_dir)
    p.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(p / f"rquant_{pick_date}.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logging.getLogger().addHandler(fh)


# =============================================================================
# preselect 子命令
# =============================================================================

def cmd_preselect(args: argparse.Namespace) -> None:
    from market.io import save_candidates
    from signals.candidates import CandidateRun
    from strategies.preselect import resolve_preselect_output_dir, run_preselect

    logger.info("===== 量化初选开始 =====")

    selection = run_preselect(
        config_path=args.config or None,
        data_dir=args.data or None,
        end_date=args.end_date or None,
        pick_date=args.date or None,
    )
    pick_ts, candidates = selection

    pick_date_str = pick_ts.strftime("%Y-%m-%d")
    run_date_str = datetime.date.today().isoformat()

    # 可选日志文件
    if args.log_dir:
        _add_log_file(args.log_dir, pick_date_str)

    run = CandidateRun(
        run_date=run_date_str,
        pick_date=pick_date_str,
        candidates=candidates,
        meta={
            "config": args.config,
            "data_dir": args.data,
            "total": len(candidates),
        },
    )

    resolved_output_dir = resolve_preselect_output_dir(
        config_path=args.config or None,
        output_dir=args.output or None,
    )

    paths = save_candidates(
        run,
        candidates_dir=resolved_output_dir,
    )

    logger.info("===== 初选完成 =====")
    logger.info("选股日期  : %s", pick_date_str)
    logger.info("候选数量  : %d 只", len(candidates))
    for key, path in paths.items():
        logger.info("%-8s → %s", key, path)

    # 终端摘要
    if candidates:
        print(f"\n{'代码':>8}  {'策略':>6}  {'收盘价':>8}  {'砖型增长':>10}")
        print("-" * 44)
        for c in candidates:
            bg = f"{c.brick_growth:.2f}x" if c.brick_growth is not None else "  —"
            print(f"{c.code:>8}  {c.strategy:>6}  {c.close:>8.2f}  {bg:>10}")
    else:
        print("\n(今日无候选股票)")
    return WorkflowResult.from_mapping({"result": selection, **paths})


# =============================================================================
# CLI 解析
# =============================================================================

def _parse_horizons(value: str) -> tuple[int, ...]:
    horizons = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not horizons or any(h <= 0 for h in horizons):
        raise argparse.ArgumentTypeError("horizons must be positive integers, e.g. 1,5,10")
    return horizons


def _parse_strategies(value: str) -> tuple[str, ...]:
    strategies = tuple(part.strip().lower() for part in value.split(",") if part.strip())
    if not strategies:
        raise argparse.ArgumentTypeError(
            "strategies must be strategy names, e.g. brick, mbdsr, or bdsr_macd_obv"
        )
    return strategies


def _parse_buy_mode(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in VALID_BUY_MODES:
        raise argparse.ArgumentTypeError(
            f"buy mode must be one of {', '.join(sorted(VALID_BUY_MODES))}"
        )
    return normalized


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _fraction(value: str) -> float:
    parsed = float(value)
    if not 0 < parsed <= 1:
        raise argparse.ArgumentTypeError("value must be in (0, 1]")
    return parsed


def cmd_signal_returns(args: argparse.Namespace) -> None:
    runner = _load_signal_returns_runner()
    result = runner(
        config_path=args.config or None,
        data_dir=args.data or None,
        start_date=args.start or None,
        end_date=args.end or None,
        output_dir=args.output or None,
        horizons=args.horizons,
        strategies=args.strategies,
        buy_mode=args.buy_mode,
        reuse_base_preparation=args.reuse_base_preparation,
    )

    summary = result["summary"]
    print("\nSignal return summary")
    print(f"signals: {summary['total_signals']}")
    for key, metrics in summary["metrics"].items():
        mean = metrics["mean_return"]
        median = metrics["median_return"]
        win_rate = metrics["win_rate"]
        if mean is None or median is None or win_rate is None:
            print(f"{key}: count=0")
        else:
            print(
                f"{key}: count={metrics['count']} "
                f"mean={mean:.4%} median={median:.4%} win_rate={win_rate:.2%}"
            )
    print(f"csv: {result['csv_path']}")
    print(f"summary: {result['summary_path']}")
    print(f"summary_csv: {result['summary_csv_path']}")
    return result


def _load_signal_returns_runner() -> Callable:
    global run_signal_returns
    if run_signal_returns is None:
        from reports.signal_returns import run_signal_returns as runner
        run_signal_returns = runner
    return run_signal_returns


def _load_portfolio_backtest_runner() -> Callable:
    global run_portfolio_backtest
    if run_portfolio_backtest is None:
        from backtest.portfolio import run_portfolio_backtest as runner
        run_portfolio_backtest = runner
    return run_portfolio_backtest


def _load_factor_portfolio_backtest_runner() -> Callable:
    global run_filter_rank_portfolio_backtest
    if run_filter_rank_portfolio_backtest is None:
        from backtest.factor_portfolio import run_filter_rank_portfolio_backtest as runner
        run_filter_rank_portfolio_backtest = runner
    return run_filter_rank_portfolio_backtest


def _load_rank_ensemble_portfolio_backtest_runner() -> Callable:
    global run_rank_ensemble_portfolio_backtest
    if run_rank_ensemble_portfolio_backtest is None:
        from backtest.factor_portfolio import run_rank_ensemble_portfolio_backtest as runner
        run_rank_ensemble_portfolio_backtest = runner
    return run_rank_ensemble_portfolio_backtest


def _load_signal_portfolio_backtest_runner() -> Callable:
    global run_signal_portfolio_backtest
    if run_signal_portfolio_backtest is None:
        from backtest.signal_portfolio import run_signal_portfolio_backtest as runner
        run_signal_portfolio_backtest = runner
    return run_signal_portfolio_backtest


def cmd_portfolio_backtest(args: argparse.Namespace) -> None:
    runner = _load_portfolio_backtest_runner()
    result = runner(
        config_path=args.config or None,
        data_dir=args.data or None,
        start_date=args.start or None,
        end_date=args.end or None,
        output_dir=args.output or None,
        initial_cash=args.initial_cash,
        strategy=args.strategy,
        buy_mode=args.buy_mode,
        hold_days=args.hold_days,
        commission_wan=args.commission_wan,
        max_positions=args.max_positions,
        position_pct=args.position_pct,
        lot_size=args.lot_size,
    )

    summary = result["result"].summary
    print("\nPortfolio backtest summary")
    print(f"date range: {args.start or ''} to {args.end or ''}")
    print(f"strategy: {summary['strategy']}")
    print(f"buy mode: {summary['buy_mode']}")
    print(f"hold days: {summary['hold_days']}")
    print(f"initial cash: {summary['initial_cash']:.2f}")
    print(f"final cash: {summary['final_cash']:.2f}")
    print(f"total return: {summary['total_return']:.2%}")
    print(f"max drawdown: {summary['max_drawdown']:.2%}")
    volatility = summary.get("annualized_volatility")
    sharpe = summary.get("sharpe_ratio")
    print("annualized volatility: n/a" if volatility is None else f"annualized volatility: {volatility:.2%}")
    print("sharpe ratio: n/a" if sharpe is None else f"sharpe ratio: {sharpe:.2f}")
    print(f"trade count: {summary['trade_count']}")
    print(f"trades: {result['trades_path']}")
    print(f"daily_trade_plan: {result['orders_path']}")
    print(f"daily_trade_plan_json: {result['orders_json_path']}")
    print(f"open_positions: {result['positions_path']}")
    print(f"summary: {result['summary_path']}")
    print(f"equity_curve: {result['equity_curve_path']}")
    print(f"equity_curve_html: {result['equity_curve_html_path']}")
    return result


def cmd_research_report(args: argparse.Namespace) -> None:
    result = run_research_report(
        signal_dir=args.signal_dir,
        portfolio_dir=args.portfolio_dir,
        candidates_path=args.candidates,
        review_path=args.review,
        output_dir=args.output,
        allow_inconsistent=args.allow_inconsistent,
    )

    summary = result["summary"]
    print("\nResearch report")
    print(f"pick date: {summary['candidates'].get('pick_date') or 'n/a'}")
    print(f"candidates: {summary['candidates'].get('count', 0)}")
    print(f"signals: {summary['signal_returns'].get('total_signals', 0)}")
    validation = summary["validation"]
    print(f"artifact validation: {validation['status']}")
    for warning in validation["warnings"]:
        print(f"WARNING: {warning}")
    portfolio = summary["portfolio"]
    total_return = portfolio.get("total_return")
    if total_return is None:
        print("portfolio return: n/a")
    else:
        print(f"portfolio return: {total_return:.2%}")
    print(f"json: {result['json_path']}")
    print(f"html: {result['html_path']}")
    return result


def cmd_factor_select(args: argparse.Namespace) -> None:
    """Run a two-stage factor selector and save unified factor signals."""

    import pandas as pd

    from factors.alpha101 import build_alpha101_panels
    from factors.filter_rank import (
        FilterRankConfig,
        build_filter_rank_frame,
        filter_then_rank,
        write_filter_rank_reports,
    )
    from strategies.preselect import load_raw_data

    if args.date and (args.start or args.end):
        raise ValueError("--date cannot be combined with --start or --end")
    raw_data = load_raw_data(args.data)
    metadata_path = Path(args.metadata) if args.metadata else None
    metadata = pd.read_csv(metadata_path) if metadata_path and metadata_path.exists() else None
    panels = build_alpha101_panels(raw_data, metadata=metadata)
    available_dates = panels.close.index
    if args.date:
        selection_dates = pd.DatetimeIndex([pd.Timestamp(args.date)])
    elif args.start or args.end:
        selection_dates = available_dates
        if args.start:
            selection_dates = selection_dates[selection_dates >= pd.Timestamp(args.start)]
        if args.end:
            selection_dates = selection_dates[selection_dates <= pd.Timestamp(args.end)]
    else:
        selection_dates = pd.DatetimeIndex([available_dates.max()])
    if selection_dates.empty:
        raise ValueError("no trading dates remain after applying the requested date range")

    config = FilterRankConfig(
        filter_factor=args.filter_factor,
        rank_factor=args.rank_factor,
        filter_top_quantile=args.filter_top_quantile,
        top_n=args.top_n,
        rank_start=args.rank_start,
        rank_end=args.rank_end,
        factor_lag_days=args.factor_lag_days,
        min_universe=args.min_universe,
        min_listing_days=args.min_listing_days,
        liquidity_lookback_days=args.liquidity_lookback_days,
        min_liquidity=args.min_liquidity,
        source=args.source,
    )
    factor_frame, filter_status = build_filter_rank_frame(
        panels,
        config=config,
        dates=selection_dates,
    )
    result = filter_then_rank(factor_frame, config=config, filter_status=filter_status)
    output = write_filter_rank_reports(result, args.output, config=config)

    print("\nFactor filter-rank selection")
    print(f"filter: top {config.filter_top_quantile:.0%} by {config.filter_factor}")
    print(
        f"rank: {config.rank_start}-{config.resolved_rank_end} "
        f"by {config.rank_factor}"
    )
    print(f"factor lag: {config.factor_lag_days} trading day(s)")
    print(f"signals: {len(result.signals)}")
    print(f"output: {output}")
    if not result.selections.empty:
        columns = ["date", "symbol", "filter_value", "rank_value", "rank_position", "weight"]
        preview = result.selections[columns].head(20)
        print(preview.to_string(index=False))
        if len(result.selections) > len(preview):
            print(
                f"... {len(result.selections) - len(preview)} more selections; "
                f"see {output / 'selections.csv'}"
            )
    return WorkflowResult.from_mapping(
        {"result": result, "signal_output_dir": output}
    )


def cmd_factor_backtest(args: argparse.Namespace) -> None:
    from factors.filter_rank import FilterRankConfig

    config = FilterRankConfig(
        filter_factor=args.filter_factor,
        rank_factor=args.rank_factor,
        filter_top_quantile=args.filter_top_quantile,
        top_n=args.top_n,
        rank_start=args.rank_start,
        rank_end=args.rank_end,
        factor_lag_days=args.factor_lag_days,
        min_universe=args.min_universe,
        min_listing_days=args.min_listing_days,
        liquidity_lookback_days=args.liquidity_lookback_days,
        min_liquidity=args.min_liquidity,
        source=args.source,
    )
    result = _load_factor_portfolio_backtest_runner()(
        data_dir=args.data,
        metadata_path=args.metadata,
        output_dir=args.output,
        start_date=args.start,
        end_date=args.end,
        selection_config=config,
        initial_cash=args.initial_cash,
        hold_days=args.hold_days,
        commission_wan=args.commission_wan,
        lot_size=args.lot_size,
        show_progress=not args.no_progress,
    )
    summary = result["result"].summary
    print("\nFactor portfolio backtest")
    print(f"date range: {summary['start_date']} to {summary['end_date']}")
    print(f"signals: {summary['signal_count']}")
    print(f"hold days: {summary['hold_days']}")
    print(f"total return: {summary['total_return']:.2%}")
    print(f"max drawdown: {summary['max_drawdown']:.2%}")
    sharpe = summary.get("sharpe_ratio")
    print("sharpe ratio: n/a" if sharpe is None else f"sharpe ratio: {sharpe:.2f}")
    print(f"realized trades: {summary['realized_trade_count']}")
    print(f"summary: {result['summary_path']}")
    print(f"trades: {result['trades_path']}")
    print(f"equity curve: {result['equity_curve_html_path']}")
    return result


def _rank_ensemble_config_from_args(args: argparse.Namespace):
    from factors.ensemble import RankEnsembleConfig

    return RankEnsembleConfig(
        factors=tuple(args.factors),
        weights=tuple(args.weights) if args.weights is not None else None,
        ascending_factors=tuple(args.ascending_factors),
        min_factor_coverage=args.min_factor_coverage,
        top_n=args.top_n,
        rank_start=args.rank_start,
        rank_end=args.rank_end,
        factor_lag_days=args.factor_lag_days,
        min_universe=args.min_universe,
        min_listing_days=args.min_listing_days,
        liquidity_lookback_days=args.liquidity_lookback_days,
        min_liquidity=args.min_liquidity,
        source=args.source,
    )


def cmd_factor_ensemble_select(args: argparse.Namespace) -> None:
    """Build auditable weighted-rank factor signals without running a portfolio."""

    import pandas as pd

    from factors.alpha101 import build_alpha101_panels
    from factors.ensemble import (
        build_alpha101_rank_ensemble_frame,
        rank_factor_ensemble,
        write_rank_ensemble_reports,
    )
    from strategies.preselect import load_raw_data

    if args.date and (args.start or args.end):
        raise ValueError("--date cannot be combined with --start or --end")
    config = _rank_ensemble_config_from_args(args)
    raw_data = load_raw_data(args.data)
    metadata_path = Path(args.metadata) if args.metadata else None
    metadata = pd.read_csv(metadata_path) if metadata_path and metadata_path.exists() else None
    panels = build_alpha101_panels(raw_data, metadata=metadata)
    available_dates = panels.close.index
    if args.date:
        selection_dates = pd.DatetimeIndex([pd.Timestamp(args.date)])
    elif args.start or args.end:
        selection_dates = available_dates
        if args.start:
            selection_dates = selection_dates[selection_dates >= pd.Timestamp(args.start)]
        if args.end:
            selection_dates = selection_dates[selection_dates <= pd.Timestamp(args.end)]
    else:
        selection_dates = pd.DatetimeIndex([available_dates.max()])
    if selection_dates.empty:
        raise ValueError("no trading dates remain after applying the requested date range")

    factor_frame, filter_status = build_alpha101_rank_ensemble_frame(
        panels,
        config=config,
        dates=selection_dates,
    )
    result = rank_factor_ensemble(
        factor_frame,
        config=config,
        filter_status=filter_status,
    )
    output = write_rank_ensemble_reports(result, args.output, config=config)

    print("\nFactor rank-ensemble selection")
    print(f"factors: {', '.join(config.factors)}")
    print(
        "weights: "
        + ", ".join(
            f"{factor}={weight:.4f}"
            for factor, weight in zip(config.factors, config.normalized_weights)
        )
    )
    print(
        "lower is better: "
        + (", ".join(config.ascending_factors) if config.ascending_factors else "none")
    )
    print(f"minimum factor coverage: {config.min_factor_coverage:.0%}")
    print(f"factor lag: {config.factor_lag_days} trading day(s)")
    print(f"signals: {len(result.signals)}")
    print(f"output: {output}")
    if not result.selections.empty:
        columns = [
            "date",
            "symbol",
            "ensemble_score",
            "factor_coverage",
            "rank_position",
            "weight",
        ]
        preview = result.selections[columns].head(20)
        print(preview.to_string(index=False))
        if len(result.selections) > len(preview):
            print(
                f"... {len(result.selections) - len(preview)} more selections; "
                f"see {output / 'selections.csv'}"
            )
    return WorkflowResult.from_mapping(
        {"result": result, "signal_output_dir": output}
    )


def cmd_factor_ensemble_backtest(args: argparse.Namespace) -> None:
    config = _rank_ensemble_config_from_args(args)
    result = _load_rank_ensemble_portfolio_backtest_runner()(
        data_dir=args.data,
        metadata_path=args.metadata,
        output_dir=args.output,
        start_date=args.start,
        end_date=args.end,
        selection_config=config,
        initial_cash=args.initial_cash,
        hold_days=args.hold_days,
        commission_wan=args.commission_wan,
        lot_size=args.lot_size,
        show_progress=not args.no_progress,
    )
    summary = result["result"].summary
    print("\nFactor rank-ensemble portfolio backtest")
    print(f"date range: {summary['start_date']} to {summary['end_date']}")
    print(f"factors: {', '.join(summary.get('factors', config.factors))}")
    print(f"signals: {summary['signal_count']}")
    print(f"hold days: {summary['hold_days']}")
    print(f"total return: {summary['total_return']:.2%}")
    print(f"max drawdown: {summary['max_drawdown']:.2%}")
    sharpe = summary.get("sharpe_ratio")
    print("sharpe ratio: n/a" if sharpe is None else f"sharpe ratio: {sharpe:.2f}")
    print(f"realized trades: {summary['realized_trade_count']}")
    print(f"summary: {result['summary_path']}")
    print(f"trades: {result['trades_path']}")
    print(f"equity curve: {result['equity_curve_html_path']}")
    return result


def cmd_train_model(args: argparse.Namespace) -> None:
    from training.train_walk_forward import run_from_args

    outputs = run_from_args(args)
    print("\nWalk-forward model training complete")
    for name, path in outputs.items():
        if name == "result":
            continue
        print(f"{name}: {path}")
    return outputs


def cmd_fit_multifactor(args: argparse.Namespace) -> None:
    from training.multifactor import run_from_args

    outputs = run_from_args(args)
    print("\nMulti-factor walk-forward fitting complete")
    for name, path in outputs.items():
        if name == "result":
            continue
        print(f"{name}: {path}")
    return outputs


def cmd_signal_backtest(args: argparse.Namespace) -> None:
    result = _load_signal_portfolio_backtest_runner()(
        signals_path=args.signals,
        data_dir=args.data,
        output_dir=args.output,
        source=args.source,
        start_date=args.start,
        end_date=args.end,
        initial_cash=args.initial_cash,
        hold_days=args.hold_days,
        commission_wan=args.commission_wan,
        stamp_tax_rate=args.stamp_tax_rate,
        transfer_fee_rate=args.transfer_fee_rate,
        max_positions=args.max_positions,
        lot_size=args.lot_size,
        show_progress=not args.no_progress,
    )
    summary = result["result"].summary
    print("\nUnified signal portfolio backtest")
    print(f"source: {summary['signal_source_filter']}")
    print(f"date range: {summary['start_date']} to {summary['end_date']}")
    print(f"signals: {summary['signal_count']}")
    print(f"hold days: {summary['hold_days']}")
    print(f"total return: {summary['total_return']:.2%}")
    print(f"max drawdown: {summary['max_drawdown']:.2%}")
    sharpe = summary.get("sharpe_ratio")
    print("sharpe ratio: n/a" if sharpe is None else f"sharpe ratio: {sharpe:.2f}")
    print(f"summary: {result['summary_path']}")
    print(f"trades: {result['trades_path']}")
    print(f"equity curve: {result['equity_curve_html_path']}")
    return result


def cmd_make_ml_dataset(args: argparse.Namespace) -> None:
    from training.build_dataset import run_from_args

    outputs = run_from_args(args)
    print("\nML dataset complete")
    for name, path in outputs.items():
        if name == "result":
            continue
        print(f"{name}: {path}")
    return outputs


def cmd_fetch_data(args: argparse.Namespace) -> None:
    from market.fetch_kline import run_from_args

    result = run_from_args(args)
    print("\nMarket data fetch complete")
    print(f"date range: {result['start']} to {result['end']}")
    print(f"symbols: {result['symbol_count']}")
    print(f"output: {result['output_dir']}")
    print(f"outcomes: {result['outcomes']}")
    print(f"manifest: {result['manifest_path']}")
    if not result["ok"]:
        print(f"failed symbols: {result['failed_codes']}")
        raise SystemExit(2)
    return WorkflowResult.from_mapping(
        {
            "result": result,
            "output_dir": Path(result["output_dir"]),
            "manifest_path": Path(result["manifest_path"]),
        }
    )


def cmd_fetch_context(args: argparse.Namespace) -> None:
    from market.fetch_context import run_from_args

    result = run_from_args(args)
    print("\nResearch context fetch complete")
    print(f"date range: {result['start']} to {result['end']}")
    print(f"requested dates: {result['requested_date_count']}")
    print(f"fetched dates: {result['fetched_date_count']}")
    print(f"reused dates: {result['reused_date_count']}")
    print(f"workers: {result['worker_count']}")
    print(f"output: {result['output_dir']}")
    print(f"manifest: {result['manifest_path']}")
    if not result["ok"]:
        print(f"failed dates: {result['failed_dates']}")
        raise SystemExit(2)
    return WorkflowResult.from_mapping(
        {
            "result": result,
            "output_dir": Path(result["output_dir"]),
            "manifest_path": Path(result["manifest_path"]),
        }
    )


def cmd_fetch_benchmark(args: argparse.Namespace) -> None:
    from market.fetch_benchmark import run_from_args

    result = run_from_args(args)
    print("\nBenchmark index fetch complete")
    print(f"index: {result['index_code']}")
    print(f"date range: {result['start']} to {result['end']}")
    print(f"rows: {result['row_count']}")
    print(f"reused: {result['reused']}")
    print(f"output: {result['output_file']}")
    print(f"manifest: {result['manifest_path']}")
    if not result["ok"]:
        print(f"error: {result['error']}")
        raise SystemExit(2)
    return WorkflowResult.from_mapping(
        {
            "result": result,
            "output_file": Path(result["output_file"]),
            "manifest_path": Path(result["manifest_path"]),
        }
    )


def cmd_build_style_factors(args: argparse.Namespace) -> None:
    from factors.style_returns import run_from_args

    result = run_from_args(args)
    print("\nMKT/SMB/HML construction complete")
    print(f"rows: {result['row_count']}")
    print(f"output: {result['output_file']}")
    print(f"manifest: {result['manifest_path']}")
    return WorkflowResult.from_mapping(
        {
            "result": result,
            "output_file": Path(result["output_file"]),
            "manifest_path": Path(result["manifest_path"]),
        }
    )


def cmd_factor_test(args: argparse.Namespace) -> None:
    from scripts.test_factor import run_from_args

    output = run_from_args(args)
    return WorkflowResult.from_mapping(
        {"result": {"factor": args.factor}, "output_dir": Path(output)}
    ) if output else WorkflowResult(result={"factor": args.factor})


def cmd_factor_batch(args: argparse.Namespace) -> None:
    from scripts.test_factor_batch import run_from_args

    exit_code = run_from_args(args)
    if exit_code:
        raise SystemExit(exit_code)


def cmd_factor_correlation(args: argparse.Namespace) -> None:
    from scripts.factor_correlation import run_from_args

    exit_code = run_from_args(args)
    if exit_code:
        raise SystemExit(exit_code)
    return WorkflowResult.from_mapping(
        {"result": {"factors": list(args.factors)}, "output_dir": Path(args.output)}
    )


def cmd_factor_run_all(args: argparse.Namespace) -> WorkflowResult:
    from reports.factor_research_pipeline import run_from_args

    outputs = run_from_args(args)
    print("\nFactor research run-all complete")
    for name, path in outputs.items():
        if name == "result":
            continue
        print(f"{name}: {path}")
    return outputs


def cmd_doctor(args: argparse.Namespace) -> None:
    from reports.system_doctor import run_system_doctor

    report = run_system_doctor(
        data_dir=args.data,
        output_path=args.output,
        deep=args.deep,
        max_market_data_age_days=args.max_data_age_days,
    )
    print("\nRQuant system doctor")
    print(f"status: {report['status']}")
    print(f"python: {report['runtime']['python_executable']}")
    print(f"dependencies: {report['dependencies']['status']}")
    print(f"configs: {report['configs']['status']}")
    print(f"secrets: {report['secrets']['status']} (values never displayed)")
    print(f"workflow artifacts: {report['workflow_artifacts']['status']}")
    market = report["market_data"]
    print(
        "market data: "
        f"{market['status']} ({market['inspected_file_count']}/{market['file_count']} files inspected)"
    )
    print(
        f"issues: {report['summary']['error_count']} errors, "
        f"{report['summary']['warning_count']} warnings"
    )
    for section_name in (
        "dependencies",
        "configs",
        "secrets",
        "workflow_artifacts",
        "market_data",
    ):
        section = report[section_name]
        for message in section["errors"]:
            print(f"ERROR [{section_name}] {message}")
        for message in section["warnings"]:
            print(f"WARNING [{section_name}] {message}")
    if report.get("output_path"):
        print(f"report: {report['output_path']}")
    if not report["ok"]:
        raise SystemExit(1)
    values = {"result": report}
    if report.get("output_path"):
        values["report_path"] = Path(report["output_path"])
    return WorkflowResult.from_mapping(values)


def _add_rank_ensemble_selection_arguments(
    parser: argparse.ArgumentParser,
    *,
    top_n: int,
) -> None:
    parser.add_argument(
        "--factors",
        nargs="+",
        required=True,
        help="Explicit Alpha101 components, e.g. alpha_040 alpha_069 alpha_077",
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        type=_positive_float,
        default=None,
        help="Optional positive component weights in the same order as --factors",
    )
    parser.add_argument(
        "--ascending-factors",
        nargs="*",
        default=[],
        help="Subset whose lower raw values are better; all others use higher-is-better",
    )
    parser.add_argument(
        "--min-factor-coverage",
        type=_fraction,
        default=1.0,
        help="Minimum available normalized factor weight per stock, default 1.0",
    )
    parser.add_argument("--top-n", type=_positive_int, default=top_n)
    parser.add_argument(
        "--rank-start",
        type=_positive_int,
        default=1,
        help="First ensemble rank to select, 1-based and inclusive",
    )
    parser.add_argument(
        "--rank-end",
        type=_positive_int,
        default=None,
        help="Last ensemble rank to select, inclusive; overrides --top-n",
    )
    parser.add_argument("--factor-lag-days", type=_non_negative_int, default=1)
    parser.add_argument("--min-universe", type=_positive_int, default=20)
    parser.add_argument("--min-listing-days", type=_non_negative_int, default=60)
    parser.add_argument("--liquidity-lookback-days", type=_positive_int, default=20)
    parser.add_argument("--min-liquidity", type=_non_negative_float, default=0.0)
    parser.add_argument("--source", default="factor_rank_ensemble")


def build_parser(*, prog: str = "scripts.quant_cli") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="RQuant 量化研究 CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("preselect", help="运行量化初选")
    p.add_argument("--config", default=None, help="rules_preselect.yaml 路径")
    p.add_argument("--data",   default=None, help="CSV 数据目录（覆盖配置文件）")
    p.add_argument("--date",   default=None, help="选股基准日期 YYYY-MM-DD（默认最新）")
    p.add_argument("--end-date", dest="end_date", default=None,
                   help="数据截断日期（回测用）")
    p.add_argument("--output", default=None, help="候选输出目录（默认 data/candidates/）")
    p.add_argument("--log-dir", dest="log_dir", default=None,
                   help="流水日志目录（默认 data/logs/）")

    p = sub.add_parser("signal-returns", help="Evaluate future returns after selector signals")
    p.add_argument("--config", default=None, help="rules_preselect.yaml path")
    p.add_argument("--data", default=None, help="CSV data directory")
    p.add_argument("--start", default=None, help="Start signal date YYYY-MM-DD")
    p.add_argument("--end", default=None, help="End signal date YYYY-MM-DD")
    p.add_argument("--output", default=None, help="Output directory, default data/backtest")
    p.add_argument(
        "--horizons",
        type=_parse_horizons,
        default=DEFAULT_HORIZONS,
        help="Comma-separated holding horizons, default 1,5,10",
    )
    p.add_argument(
        "--strategies",
        type=_parse_strategies,
        default=None,
        help="Comma-separated strategies, e.g. brick, mbdsr, or bdsr_macd_obv",
    )
    p.add_argument(
        "--buy-mode",
        type=_parse_buy_mode,
        default=BUY_MODE_SIGNAL_CLOSE,
        help="Entry price mode: signal_close or next_open",
    )
    p.add_argument(
        "--reuse-base-preparation",
        action="store_true",
        help="Opt in to sharing base market-data preparation across strategies",
    )

    p = sub.add_parser("portfolio-backtest", help="Run portfolio-level strategy backtest")
    p.add_argument("--config", default=None, help="rules_preselect.yaml path")
    p.add_argument("--data", default=None, help="CSV data directory")
    p.add_argument("--start", default=None, help="Start signal date YYYY-MM-DD")
    p.add_argument("--end", default=None, help="End signal date YYYY-MM-DD")
    p.add_argument("--output", default=None, help="Output directory, default data/portfolio_backtest")
    p.add_argument(
        "--strategy",
        default="brick",
        help="Strategy name, e.g. brick, mbdsr, or bdsr_macd_obv",
    )
    p.add_argument(
        "--buy-mode",
        type=_parse_buy_mode,
        default=BUY_MODE_SIGNAL_CLOSE,
        help="Entry price mode: signal_close or next_open",
    )
    p.add_argument(
        "--hold-days",
        type=_positive_int,
        default=1,
        help="Holding period in trading bars, default 1",
    )
    p.add_argument(
        "--initial-cash",
        type=_positive_float,
        default=100000000.0,
        help="Initial portfolio cash, default 100000000",
    )
    p.add_argument(
        "--commission-wan",
        type=_non_negative_float,
        default=0.8,
        help="Commission in wan units, e.g. 0.8 means 0.008%%",
    )
    p.add_argument(
        "--max-positions",
        type=_positive_int,
        default=10,
        help="Maximum concurrent positions, default 10",
    )
    p.add_argument(
        "--position-pct",
        type=_positive_float,
        default=0.1,
        help="Target fraction of equity for each new position, default 0.1",
    )
    p.add_argument(
        "--lot-size",
        type=_positive_int,
        default=100,
        help="Minimum trade lot size, default 100 shares",
    )

    p = sub.add_parser("research-report", help="Build a JSON and HTML research report from outputs")
    p.add_argument("--signal-dir", required=True, help="Directory containing signal_summary.json")
    p.add_argument("--portfolio-dir", required=True, help="Directory containing portfolio_summary.json")
    p.add_argument(
        "--candidates",
        default="data/candidates/candidates_latest.json",
        help="Candidates JSON path",
    )
    p.add_argument(
        "--review",
        default=None,
        help="Optional Gemini suggestion.json path",
    )
    p.add_argument(
        "--output",
        default=str(DEFAULT_REPORT_OUTPUT_DIR),
        help="Report output directory, default data/reports",
    )
    p.add_argument(
        "--allow-inconsistent",
        action="store_true",
        help="Write a diagnostic report even when artifact consistency checks fail",
    )

    p = sub.add_parser(
        "factor-select",
        help="Filter the factor universe with one Alpha101 factor, then rank with another",
    )
    p.add_argument("--data", default="data/raw", help="Raw per-symbol OHLCV CSV directory")
    p.add_argument("--metadata", default="config/stocklist.csv", help="Optional classification CSV")
    p.add_argument("--date", default=None, help="One selection date; defaults to latest trading date")
    p.add_argument("--start", default=None, help="First selection date for a historical signal range")
    p.add_argument("--end", default=None, help="Last selection date for a historical signal range")
    p.add_argument("--filter-factor", default="alpha_077")
    p.add_argument("--rank-factor", default="alpha_040")
    p.add_argument(
        "--filter-top-quantile",
        type=_fraction,
        default=0.5,
        help="Fraction retained by the filter factor, default 0.5",
    )
    p.add_argument("--top-n", type=_positive_int, default=10)
    p.add_argument(
        "--rank-start",
        type=_positive_int,
        default=1,
        help="First post-filter rank to select, 1-based and inclusive",
    )
    p.add_argument(
        "--rank-end",
        type=_positive_int,
        default=None,
        help="Last post-filter rank to select, inclusive; overrides --top-n",
    )
    p.add_argument("--factor-lag-days", type=_non_negative_int, default=1)
    p.add_argument("--min-universe", type=_positive_int, default=20)
    p.add_argument("--min-listing-days", type=_non_negative_int, default=60)
    p.add_argument("--liquidity-lookback-days", type=_positive_int, default=20)
    p.add_argument("--min-liquidity", type=_non_negative_float, default=0.0)
    p.add_argument("--source", default="alpha077_filter_alpha040_rank")
    p.add_argument(
        "--output",
        default="data/factor_signals/alpha077_alpha040",
        help="Output directory for signals and audit files",
    )

    p = sub.add_parser(
        "factor-backtest",
        help="Generate filter/rank factor signals and run the realistic portfolio backtest",
    )
    p.add_argument("--data", default="data/raw", help="Raw per-symbol OHLCV CSV directory")
    p.add_argument("--metadata", default="config/stocklist.csv", help="Optional classification CSV")
    p.add_argument("--start", default=None, help="First signal/backtest date")
    p.add_argument("--end", default=None, help="Last signal/backtest date")
    p.add_argument("--filter-factor", default="alpha_077")
    p.add_argument("--rank-factor", default="alpha_040")
    p.add_argument(
        "--filter-top-quantile",
        type=_fraction,
        default=0.8,
        help="Fraction retained by the filter factor, default 0.8",
    )
    p.add_argument("--top-n", type=_positive_int, default=500, help="Top stocks, default 500")
    p.add_argument(
        "--rank-start",
        type=_positive_int,
        default=1,
        help="First alpha rank after filtering, 1-based and inclusive",
    )
    p.add_argument(
        "--rank-end",
        type=_positive_int,
        default=None,
        help="Last alpha rank after filtering, inclusive; overrides --top-n",
    )
    p.add_argument("--factor-lag-days", type=_non_negative_int, default=1)
    p.add_argument("--min-universe", type=_positive_int, default=20)
    p.add_argument("--min-listing-days", type=_non_negative_int, default=60)
    p.add_argument("--liquidity-lookback-days", type=_positive_int, default=20)
    p.add_argument("--min-liquidity", type=_non_negative_float, default=0.0)
    p.add_argument("--source", default="alpha077_filter_alpha040_rank")
    p.add_argument(
        "--hold-days",
        type=_positive_int,
        default=20,
        help="Holding days and fixed capital-sleeve count, default 20",
    )
    p.add_argument(
        "--initial-cash",
        type=_positive_float,
        default=10000000.0,
        help="Initial portfolio cash, default 10000000",
    )
    p.add_argument("--commission-wan", type=_non_negative_float, default=0.8)
    p.add_argument("--lot-size", type=_positive_int, default=100)
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the portfolio backtest progress bar",
    )
    p.add_argument(
        "--output",
        default="data/portfolio_backtest_alpha077_alpha040",
        help="Backtest output directory",
    )

    p = sub.add_parser(
        "factor-ensemble-select",
        help="Combine explicit Alpha101 factors by weighted daily percentile rank",
    )
    p.add_argument("--data", default="data/raw", help="Raw per-symbol OHLCV CSV directory")
    p.add_argument("--metadata", default="config/stocklist.csv", help="Optional classification CSV")
    p.add_argument("--date", default=None, help="One selection date; defaults to latest trading date")
    p.add_argument("--start", default=None, help="First selection date for a historical signal range")
    p.add_argument("--end", default=None, help="Last selection date for a historical signal range")
    _add_rank_ensemble_selection_arguments(p, top_n=10)
    p.add_argument(
        "--output",
        default="data/factor_signals/ensemble",
        help="Output directory for signals and component audit files",
    )

    p = sub.add_parser(
        "factor-ensemble-backtest",
        help="Generate weighted-rank factor signals and run the realistic portfolio backtest",
    )
    p.add_argument("--data", default="data/raw", help="Raw per-symbol OHLCV CSV directory")
    p.add_argument("--metadata", default="config/stocklist.csv", help="Optional classification CSV")
    p.add_argument("--start", default=None, help="First signal/backtest date")
    p.add_argument("--end", default=None, help="Last signal/backtest date")
    _add_rank_ensemble_selection_arguments(p, top_n=500)
    p.add_argument(
        "--hold-days",
        type=_positive_int,
        default=20,
        help="Holding days and fixed capital-sleeve count, default 20",
    )
    p.add_argument(
        "--initial-cash",
        type=_positive_float,
        default=10000000.0,
        help="Initial portfolio cash, default 10000000",
    )
    p.add_argument("--commission-wan", type=_non_negative_float, default=0.8)
    p.add_argument("--lot-size", type=_positive_int, default=100)
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the portfolio backtest progress bar",
    )
    p.add_argument(
        "--output",
        default="data/portfolio_backtest_factor_ensemble",
        help="Backtest output directory",
    )

    from training.train_walk_forward import add_arguments as add_train_model_arguments

    p = sub.add_parser(
        "train-model",
        help="Train resumable walk-forward out-of-sample model scores",
    )
    add_train_model_arguments(p)

    from training.multifactor import add_arguments as add_multifactor_arguments

    p = sub.add_parser(
        "fit-multifactor",
        help="Build ranked factor features and compare walk-forward ML fitters",
    )
    add_multifactor_arguments(p)

    p = sub.add_parser(
        "signal-backtest",
        help="Run the constrained portfolio from one unified signals.csv source",
    )
    p.add_argument("--signals", required=True, help="Unified signals.csv path")
    p.add_argument("--data", default="data/raw", help="Raw per-symbol OHLCV CSV directory")
    p.add_argument("--source", default=None, help="Required when the file has multiple sources")
    p.add_argument("--start", default=None, help="First signal/backtest date")
    p.add_argument("--end", default=None, help="Last backtest date; defaults to latest market date")
    p.add_argument("--hold-days", type=_positive_int, default=20)
    p.add_argument("--initial-cash", type=_positive_float, default=10000000.0)
    p.add_argument("--commission-wan", type=_non_negative_float, default=0.8)
    p.add_argument("--stamp-tax-rate", type=_non_negative_float, default=0.0005)
    p.add_argument("--transfer-fee-rate", type=_non_negative_float, default=0.00001)
    p.add_argument("--max-positions", type=_positive_int, default=10)
    p.add_argument("--lot-size", type=_positive_int, default=100)
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--output", default="data/portfolio_backtest_signals")

    from training.build_dataset import add_arguments as add_ml_dataset_arguments

    p = sub.add_parser(
        "make-ml-dataset",
        help="Build one-day-lagged factor features and forward-return labels",
    )
    add_ml_dataset_arguments(p)

    p = sub.add_parser("fetch-data", help="Fetch and update local qfq daily bars")
    p.add_argument("--config", default="config/fetch_kline.yaml")
    p.add_argument("--start", default=None, help="YYYYMMDD, YYYY-MM-DD, or today")
    p.add_argument("--end", default=None, help="YYYYMMDD, YYYY-MM-DD, or today")
    p.add_argument("--out", default=None, help="Override output directory")
    p.add_argument("--workers", type=_positive_int, default=None)
    p.add_argument(
        "--max-requests-per-minute",
        type=_non_negative_int,
        default=None,
        help="Evenly throttle Tushare calls; 0 disables, default YAML value is 180",
    )
    p.add_argument("--max-symbols", type=_positive_int, default=None)
    p.add_argument("--log", default=None, help="Override log file path")
    p.add_argument(
        "--manifest",
        default=None,
        help="Checkpoint JSON path; defaults to <out>/_fetch_manifest.json",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed symbols from a signature-matching manifest",
    )

    p = sub.add_parser(
        "fetch-context",
        help="Fetch point-in-time Tushare daily_basic market-cap context",
    )
    p.add_argument("--start", required=True, help="First date, e.g. 20180101")
    p.add_argument("--end", default=None, help="Last date; defaults to today")
    p.add_argument("--out", default="data/context/daily_basic")
    p.add_argument("--manifest", default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-requests-per-minute", type=_non_negative_int, default=180)
    p.add_argument("--workers", type=_positive_int, default=8)
    p.add_argument(
        "--max-dates",
        type=_positive_int,
        default=None,
        help="Smoke-test only: fetch the first N open dates",
    )

    p = sub.add_parser(
        "build-style-factors",
        help="Build lagged-characteristic daily MKT/SMB/HML for GTJA030",
    )
    p.add_argument("--data", default="data/raw")
    p.add_argument("--context", default="data/context/daily_basic")
    p.add_argument("--out", default="data/context/style_factors.csv")
    p.add_argument("--manifest", default=None)
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--size-quantile", type=float, default=0.5)
    p.add_argument("--value-low-quantile", type=float, default=0.3)
    p.add_argument("--value-high-quantile", type=float, default=0.7)
    p.add_argument("--min-stocks-per-portfolio", type=_positive_int, default=5)
    p.add_argument("--max-symbols", type=_positive_int, default=None)

    p = sub.add_parser(
        "fetch-benchmark",
        help="Fetch a Tushare index_daily benchmark for market-related GTJA factors",
    )
    p.add_argument("--start", required=True, help="First date, e.g. 20180101")
    p.add_argument("--end", default=None, help="Last date; defaults to today")
    p.add_argument("--index-code", default="000300.SH", help="Tushare index code")
    p.add_argument("--out", default="data/context/benchmark_000300.csv")
    p.add_argument("--manifest", default=None)
    p.add_argument("--resume", action="store_true")

    from scripts.test_factor import add_arguments as add_factor_test_arguments

    p = sub.add_parser("factor-test", help="Run one long-only FactorTester report")
    add_factor_test_arguments(p)

    from scripts.test_factor_batch import add_arguments as add_factor_batch_arguments

    p = sub.add_parser(
        "factor-batch",
        help="Run resumable lifecycle-aware Alpha101 or GTJA191 factor batches",
    )
    add_factor_batch_arguments(p)

    from scripts.factor_correlation import add_arguments as add_factor_correlation_arguments

    p = sub.add_parser(
        "factor-correlation",
        help="Build factor correlation matrices and an auditable |Spearman| deduplication list",
    )
    add_factor_correlation_arguments(p)

    from reports.factor_research_pipeline import add_arguments as add_factor_run_all_arguments

    p = sub.add_parser(
        "factor-run-all",
        help="Run factor evaluation, correlation deduplication, and 3y-to-1y long-only ML",
    )
    add_factor_run_all_arguments(p)

    p = sub.add_parser(
        "doctor",
        help="Check runtime, dependencies, configs, secrets, and local market data",
    )
    p.add_argument("--data", default="data/raw", help="Raw market-data directory")
    p.add_argument("--output", default=None, help="Optional atomic JSON report path")
    p.add_argument(
        "--deep",
        action="store_true",
        help="Inspect every market CSV instead of the first 25 files",
    )
    p.add_argument(
        "--max-data-age-days",
        type=_non_negative_int,
        default=7,
        help="Warn when the latest local bar is older than this many calendar days",
    )

    return parser


_COMMAND_HANDLERS: dict[str, Callable[[argparse.Namespace], object]] = {
    "preselect": cmd_preselect,
    "signal-returns": cmd_signal_returns,
    "portfolio-backtest": cmd_portfolio_backtest,
    "research-report": cmd_research_report,
    "factor-select": cmd_factor_select,
    "factor-backtest": cmd_factor_backtest,
    "factor-ensemble-select": cmd_factor_ensemble_select,
    "factor-ensemble-backtest": cmd_factor_ensemble_backtest,
    "train-model": cmd_train_model,
    "fit-multifactor": cmd_fit_multifactor,
    "signal-backtest": cmd_signal_backtest,
    "make-ml-dataset": cmd_make_ml_dataset,
    "fetch-data": cmd_fetch_data,
    "fetch-context": cmd_fetch_context,
    "fetch-benchmark": cmd_fetch_benchmark,
    "build-style-factors": cmd_build_style_factors,
    "factor-test": cmd_factor_test,
    "factor-batch": cmd_factor_batch,
    "factor-correlation": cmd_factor_correlation,
    "factor-run-all": cmd_factor_run_all,
    "doctor": cmd_doctor,
}


def dispatch(args: argparse.Namespace) -> CommandResult:
    """Dispatch one parsed command and normalize its framework result."""

    handler = _COMMAND_HANDLERS.get(args.command)
    if handler is None:
        raise ValueError(f"unknown command: {args.command}")
    returned = handler(args)
    if isinstance(returned, CommandResult):
        return returned

    if isinstance(returned, WorkflowResult):
        outputs = {
            name: str(reference.path)
            for name, reference in returned.artifacts.items()
        }
        return CommandResult(
            status=returned.status,
            outputs=outputs,
            summary={
                "command": args.command,
                "handler": handler.__name__,
                "domain_result_type": type(returned.result).__name__,
                "artifact_count": len(returned.artifacts),
            },
            workflow=returned,
        )

    if isinstance(returned, Mapping):
        outputs = {
            name: str(value)
            for name, value in returned.items()
            if isinstance(value, Path)
        }
        return CommandResult(
            outputs=outputs,
            summary={"command": args.command, "handler": handler.__name__},
        )

    outputs: dict[str, str] = {}
    for name in ("output", "out", "manifest", "log", "log_dir"):
        value = getattr(args, name, None)
        if value:
            outputs[name] = str(value)
    return CommandResult(
        outputs=outputs,
        summary={"command": args.command, "handler": handler.__name__},
    )


def execute(
    argv: Sequence[str] | None = None,
    *,
    prog: str = "scripts.quant_cli",
) -> CommandResult:
    parser = build_parser(prog=prog)
    args = parser.parse_args(list(argv) if argv is not None else None)
    return dispatch(args)


def main(argv: Sequence[str] | None = None) -> int:
    """Compatibility callable; governed execution is provided by ``rquant``."""

    result = execute(argv)
    return result.exit_code

def test():
    """简单测试函数，验证 CLI 逻辑（不依赖外部数据）。"""
    class Args:
        command = "preselect"
        config = None
        data = None
        date = None
        end_date = None
        output = "./data/candidates"
        log_dir = "./data/logs"

    args = Args()
    cmd_preselect(args)


if __name__ == "__main__":
    from rquant.cli import legacy_main

    raise SystemExit(legacy_main(backend=sys.modules[__name__]))
