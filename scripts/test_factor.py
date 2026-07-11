from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reports.factor_tester import (  # noqa: E402
    FactorTester,
    FactorTesterConfig,
    build_long_factor_frame_from_raw,
)
from factors.alpha101 import ALPHA101_NAMES  # noqa: E402
from factors.gtja191 import GTJA191_NAMES  # noqa: E402
from factors.brick import (  # noqa: E402
    LISTED_BRICK_FACTORS,
    is_brick_factor,
)
from factors.custom import (  # noqa: E402
    CUSTOM_FACTOR_NAMES,
    is_custom_factor,
    normalize_custom_factor_name,
)
from strategies.preselect import load_config, load_raw_data  # noqa: E402


def _parse_windows(values: list[str]) -> tuple[int, ...]:
    windows = tuple(int(value) for value in values)
    if not windows or any(window <= 0 for window in windows):
        raise argparse.ArgumentTypeError("windows must be positive integers")
    return windows


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--factor",
        help=(
            "Factor name, e.g. custom_001, brick, "
            "momentum_20d, or alpha_001"
        ),
    )
    parser.add_argument("--list-factors", action="store_true", help="List built-in named factors")
    parser.add_argument(
        "--factor-file",
        default=None,
        help="Optional long-format factor CSV. If omitted, a supported built-in factor is computed from raw OHLCV.",
    )
    parser.add_argument(
        "--strategy-config",
        default="config/rules_preselect.yaml",
        help="Strategy config used by BrickChart-derived factors",
    )
    parser.add_argument("--data", default="data/raw", help="Raw OHLCV data directory")
    parser.add_argument(
        "--metadata",
        default="config/stocklist.csv",
        help="Optional symbol metadata CSV with industry/sector/subindustry columns",
    )
    parser.add_argument("--benchmark-file", default=None, help="Optional benchmark OHLC CSV for GTJA factors")
    parser.add_argument("--style-factor-file", default=None, help="Optional date,mkt,smb,hml CSV for GTJA factors")
    parser.add_argument("--output", default="factor_report", help="Report output root")
    parser.add_argument("--windows", nargs="+", default=["1", "5", "10", "20"], help="Forward return windows")
    parser.add_argument("--groups", type=int, default=10, help="Number of quantile groups")
    parser.add_argument(
        "--top-counts",
        nargs="+",
        type=int,
        default=[1, 5, 10, 20, 50, 100],
        help="Long-only TopN buckets to report, default: 1 5 10 20 50 100",
    )
    parser.add_argument("--date-col", default="date")
    parser.add_argument("--symbol-col", default="symbol")
    parser.add_argument("--factor-col", default="factor_value")
    parser.add_argument("--close-col", default="close")
    parser.add_argument("--industry-col", default="industry")
    parser.add_argument("--market-cap-col", default="market_cap")
    parser.add_argument("--universe-col", default=None)
    parser.add_argument("--tradeable-col", default="is_tradeable")
    parser.add_argument("--suspended-col", default="is_suspended")
    parser.add_argument("--limit-up-col", default="is_limit_up")
    parser.add_argument("--limit-down-col", default="is_limit_down")
    parser.add_argument("--st-col", default="is_st")
    parser.add_argument("--min-listing-days", type=int, default=60)
    parser.add_argument("--liquidity-lookback-days", type=int, default=20)
    parser.add_argument("--min-liquidity", type=float, default=0.0)
    parser.add_argument("--commission-rate", type=float, default=0.0003)
    parser.add_argument("--slippage-rate", type=float, default=0.0005)
    parser.add_argument("--stamp-tax-rate", type=float, default=0.0005)
    parser.add_argument("--oos-start-date", default=None)
    parser.add_argument("--oos-fraction", type=float, default=0.3)
    parser.add_argument("--winsorize", action="store_true", help="Winsorize factor values by date")
    parser.add_argument("--zscore", action="store_true", help="Z-score standardize factor values by date")
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run FactorTester for one factor.")
    return add_arguments(parser)


def load_factor_data(args: argparse.Namespace) -> pd.DataFrame:
    """Load a long-format factor file or adapt current raw OHLCV CSVs."""
    if args.factor_file:
        return pd.read_csv(args.factor_file)
    raw_data = load_raw_data(args.data)
    metadata = None
    if args.metadata:
        metadata_path = Path(args.metadata)
        if metadata_path.exists():
            metadata = pd.read_csv(metadata_path)
    factor_config = None
    if is_brick_factor(args.factor):
        factor_config = load_config(args.strategy_config).get("brick", {})
    benchmark_data = pd.read_csv(args.benchmark_file) if args.benchmark_file else None
    style_factor_data = pd.read_csv(args.style_factor_file) if args.style_factor_file else None
    return build_long_factor_frame_from_raw(
        raw_data,
        factor_name=args.factor,
        factor_config=factor_config,
        metadata=metadata,
        benchmark_data=benchmark_data,
        style_factor_data=style_factor_data,
        date_col=args.date_col,
        symbol_col=args.symbol_col,
        close_col=args.close_col,
    )


def run_from_args(args: argparse.Namespace) -> Path | None:
    if args.list_factors:
        print(
            "\n".join(
                (*CUSTOM_FACTOR_NAMES, *LISTED_BRICK_FACTORS, *ALPHA101_NAMES, *GTJA191_NAMES)
            )
        )
        return None
    if not args.factor:
        raise ValueError("--factor is required unless --list-factors is used")
    if is_custom_factor(args.factor):
        args.factor = normalize_custom_factor_name(args.factor)
    windows = _parse_windows(args.windows)
    if args.groups not in (5, 10):
        raise ValueError("--groups currently supports 5 or 10")
    if not args.top_counts or any(value <= 0 for value in args.top_counts):
        raise ValueError("--top-counts must contain positive integers")

    data = load_factor_data(args)
    config = FactorTesterConfig(
        date_col=args.date_col,
        symbol_col=args.symbol_col,
        factor_col=args.factor_col,
        close_col=args.close_col,
        universe_col=args.universe_col,
        industry_col=args.industry_col,
        market_cap_col=args.market_cap_col,
        tradeable_col=args.tradeable_col,
        suspended_col=args.suspended_col,
        limit_up_col=args.limit_up_col,
        limit_down_col=args.limit_down_col,
        st_col=args.st_col,
        groups=args.groups,
        top_n_counts=tuple(args.top_counts),
        forward_return_windows=windows,
        winsorize=args.winsorize,
        zscore=args.zscore,
        min_listing_days=args.min_listing_days,
        liquidity_lookback_days=args.liquidity_lookback_days,
        min_liquidity=args.min_liquidity,
        commission_rate=args.commission_rate,
        slippage_rate=args.slippage_rate,
        stamp_tax_rate=args.stamp_tax_rate,
        oos_start_date=args.oos_start_date,
        oos_fraction=args.oos_fraction,
    )
    tester = FactorTester(data, factor_name=args.factor, config=config)
    report_dir = tester.write_reports(args.output)
    print(f"factor report: {report_dir}")
    return report_dir


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        run_from_args(args)
    except (argparse.ArgumentTypeError, KeyError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
