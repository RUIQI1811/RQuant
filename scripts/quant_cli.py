"""Command-line entrypoint for RQuant research workflows."""
from __future__ import annotations

import argparse
import datetime
import logging
import sys
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reports.research_report import DEFAULT_REPORT_OUTPUT_DIR, run_research_report

BUY_MODE_SIGNAL_CLOSE = "signal_close"
VALID_BUY_MODES = {BUY_MODE_SIGNAL_CLOSE, "next_open"}
DEFAULT_HORIZONS = (1, 5, 10)

run_signal_returns: Callable | None = None
run_portfolio_backtest: Callable | None = None
run_filter_rank_portfolio_backtest: Callable | None = None

# ── 日志配置 ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
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

    pick_ts, candidates = run_preselect(
        config_path=args.config or None,
        data_dir=args.data or None,
        end_date=args.end_date or None,
        pick_date=args.date or None,
    )

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


def cmd_research_report(args: argparse.Namespace) -> None:
    result = run_research_report(
        signal_dir=args.signal_dir,
        portfolio_dir=args.portfolio_dir,
        candidates_path=args.candidates,
        review_path=args.review,
        output_dir=args.output,
    )

    summary = result["summary"]
    print("\nResearch report")
    print(f"pick date: {summary['candidates'].get('pick_date') or 'n/a'}")
    print(f"candidates: {summary['candidates'].get('count', 0)}")
    print(f"signals: {summary['signal_returns'].get('total_signals', 0)}")
    portfolio = summary["portfolio"]
    total_return = portfolio.get("total_return")
    if total_return is None:
        print("portfolio return: n/a")
    else:
        print(f"portfolio return: {total_return:.2%}")
    print(f"json: {result['json_path']}")
    print(f"html: {result['html_path']}")


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scripts.quant_cli",
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

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "preselect":
        cmd_preselect(args)
    elif args.command == "signal-returns":
        cmd_signal_returns(args)
    elif args.command == "portfolio-backtest":
        cmd_portfolio_backtest(args)
    elif args.command == "research-report":
        cmd_research_report(args)
    elif args.command == "factor-select":
        cmd_factor_select(args)
    elif args.command == "factor-backtest":
        cmd_factor_backtest(args)
    else:
        parser.print_help()
        sys.exit(1)

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
    main()
    # test()
