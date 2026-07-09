from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reports.alpha101_batch import (  # noqa: E402
    Alpha101BatchConfig,
    Alpha101BatchRunner,
    directory_signature,
    files_signature,
    parse_factor_selection,
)
from reports.factor_tester import FactorTester  # noqa: E402
from factors.alpha101 import ALPHA101_NAMES, Alpha101, build_alpha101_panels  # noqa: E402
from factors.catalog import FactorCatalog, load_factor_catalog  # noqa: E402
from strategies.preselect import load_raw_data  # noqa: E402


def _positive_windows(values: list[str]) -> tuple[int, ...]:
    windows = tuple(int(value) for value in values)
    if not windows or any(window <= 0 for window in windows):
        raise argparse.ArgumentTypeError("windows must be positive integers")
    return windows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run resumable, memory-bounded batch tests for Alpha101 factors."
    )
    parser.add_argument("--data", default="data/raw", help="Raw per-symbol OHLCV CSV directory")
    parser.add_argument("--metadata", default="pipeline/stocklist.csv", help="Optional classification CSV")
    parser.add_argument("--output", default="factor_report/alpha101_batch", help="Batch output directory")
    parser.add_argument(
        "--factor-config",
        default="config/factors.yaml",
        help="YAML lifecycle config for active/watch/disabled factors",
    )
    parser.add_argument(
        "--ignore-factor-config",
        action="store_true",
        help="Temporarily run the requested factors regardless of configured status",
    )
    parser.add_argument(
        "--factors",
        nargs="+",
        default=["all"],
        help="Names/numbers/ranges, e.g. all, 1-20, alpha_001 alpha_101",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="Factors to exclude, using the same syntax as --factors",
    )
    parser.add_argument("--windows", nargs="+", default=["1", "5", "10", "20"])
    parser.add_argument("--groups", type=int, choices=(5, 10), default=10)
    parser.add_argument(
        "--top-counts",
        nargs="+",
        type=int,
        default=[1, 5, 10, 20, 50, 100],
        help="Long-only TopN buckets to report, default: 1 5 10 20 50 100",
    )
    parser.add_argument("--start-date", default=None, help="First evaluation date (history is retained for warmup)")
    parser.add_argument("--end-date", default=None, help="Last evaluation date")
    parser.add_argument("--winsorize", action="store_true")
    parser.add_argument("--zscore", action="store_true")
    parser.add_argument("--min-periods", type=int, default=3)
    parser.add_argument("--min-listing-days", type=int, default=60)
    parser.add_argument("--liquidity-lookback-days", type=int, default=20)
    parser.add_argument("--min-liquidity", type=float, default=0.0)
    parser.add_argument("--commission-rate", type=float, default=0.0003)
    parser.add_argument("--slippage-rate", type=float, default=0.0005)
    parser.add_argument("--stamp-tax-rate", type=float, default=0.0005)
    parser.add_argument("--oos-start-date", default=None)
    parser.add_argument("--oos-fraction", type=float, default=0.3)
    parser.add_argument("--force", action="store_true", help="Recompute even when a matching report exists")
    parser.add_argument("--fail-fast", action="store_true", help="Stop at the first factor failure")
    parser.add_argument(
        "--max-symbols",
        type=int,
        default=None,
        help="Smoke-test only: retain the first N sorted symbols",
    )
    parser.add_argument("--list-factors", action="store_true")
    parser.add_argument(
        "--list-factor-status",
        action="store_true",
        help="List every Alpha101 factor with its configured lifecycle status",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.list_factors:
        print("\n".join(ALPHA101_NAMES))
        return 0

    try:
        catalog = FactorCatalog() if args.ignore_factor_config else load_factor_catalog(args.factor_config)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        parser.error(str(exc))
    if args.list_factor_status:
        print("factor,status")
        for factor, status in catalog.status_map().items():
            print(f"{factor},{status}")
        return 0

    try:
        requested_factors = parse_factor_selection(args.factors, args.exclude)
        factors = catalog.select(requested_factors)
        windows = _positive_windows(args.windows)
        top_counts = _positive_windows([str(value) for value in args.top_counts])
    except (KeyError, ValueError, argparse.ArgumentTypeError) as exc:
        parser.error(str(exc))
    if not factors:
        parser.error(
            "factor selection is empty after exclusions and lifecycle filtering; "
            "use --ignore-factor-config for a one-off override"
        )
    if args.max_symbols is not None and args.max_symbols <= 0:
        parser.error("--max-symbols must be positive")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    selected_statuses = catalog.status_map(factors)
    active_count = sum(status == "active" for status in selected_statuses.values())
    watch_count = sum(status == "watch" for status in selected_statuses.values())
    filtered_count = len(requested_factors) - len(factors)
    logging.info(
        "factor lifecycle selection: active=%d watch=%d disabled/filtered=%d",
        active_count,
        watch_count,
        filtered_count,
    )
    if args.max_symbols:
        selected_symbols = sorted(path.stem for path in Path(args.data).glob("*.csv"))[: args.max_symbols]
        raw_data = load_raw_data(args.data, symbols=selected_symbols)
        logging.warning("smoke-test universe limited to %d symbols", len(raw_data))
    else:
        raw_data = load_raw_data(args.data)

    metadata_path = Path(args.metadata) if args.metadata else None
    metadata = pd.read_csv(metadata_path) if metadata_path and metadata_path.exists() else None
    panels = build_alpha101_panels(raw_data, metadata=metadata)

    config = Alpha101BatchConfig(
        windows=windows,
        groups=args.groups,
        top_n_counts=top_counts,
        start_date=args.start_date,
        end_date=args.end_date,
        winsorize=args.winsorize,
        zscore=args.zscore,
        min_periods=args.min_periods,
        min_listing_days=args.min_listing_days,
        liquidity_lookback_days=args.liquidity_lookback_days,
        min_liquidity=args.min_liquidity,
        commission_rate=args.commission_rate,
        slippage_rate=args.slippage_rate,
        stamp_tax_rate=args.stamp_tax_rate,
        oos_start_date=args.oos_start_date,
        oos_fraction=args.oos_fraction,
        force=args.force,
        fail_fast=args.fail_fast,
    )
    implementation_signature = files_signature(
        [
            Path(__file__),
            Path(sys.modules[Alpha101.__module__].__file__),
            Path(sys.modules[FactorTester.__module__].__file__),
            ROOT / "pipeline" / "alpha101_batch.py",
        ]
    )
    data_sig = directory_signature(args.data, metadata_path)
    if args.max_symbols:
        data_sig = f"{data_sig}:max-symbols={args.max_symbols}"

    result = Alpha101BatchRunner(
        panels,
        factors=factors,
        output_dir=args.output,
        config=config,
        data_signature=data_sig,
        implementation_signature=implementation_signature,
        factor_statuses=selected_statuses,
    ).run()
    print(f"batch status: {result.output_dir / 'batch_status.csv'}")
    print(f"leaderboard: {result.output_dir / 'leaderboard.csv'}")
    if result.failed_factors:
        print(f"failed factors: {', '.join(result.failed_factors)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
