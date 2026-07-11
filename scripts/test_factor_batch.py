from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reports.alpha101_batch import (  # noqa: E402
    Alpha101BatchConfig,
    Alpha101BatchRunner,
    directory_signature,
    files_signature,
    parse_factor_selection as parse_alpha101_selection,
)
from reports.factor_tester import FactorTester  # noqa: E402
from reports.gtja191_batch import (  # noqa: E402
    GTJA191BatchConfig,
    GTJA191BatchRunner,
    filter_gtja_selection_from_start,
    parse_gtja_selection,
)
import factors.operators as factor_operators  # noqa: E402
from factors.alpha101 import ALPHA101_NAMES, Alpha101, build_alpha101_panels  # noqa: E402
from factors.catalog import FactorCatalog, load_factor_catalog  # noqa: E402
from factors.gtja191 import (  # noqa: E402
    GTJA191,
    GTJA191_NAMES,
    build_gtja191_panels,
    normalize_gtja_name,
)
from strategies.preselect import load_raw_data  # noqa: E402


ALPHA101_OUTPUT = "factor_report/alpha101_batch"
ALPHA101_FACTOR_CONFIG = "config/factors.yaml"
GTJA191_OUTPUT = "factor_report/gtja191_batch"
GTJA191_FACTOR_CONFIG = "config/gtja191_factors.yaml"


def _positive_windows(values: list[str] | list[int]) -> tuple[int, ...]:
    windows = tuple(int(value) for value in values)
    if not windows or any(window <= 0 for window in windows):
        raise argparse.ArgumentTypeError("windows must be positive integers")
    return windows


def _optional_csv(path: str | None) -> pd.DataFrame | None:
    return pd.read_csv(path) if path else None


def _gtja_factor_statuses(path: str, factors: tuple[str, ...]) -> dict[str, str]:
    config_path = Path(path)
    if not config_path.exists():
        return {name: "active" for name in factors}
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    default = str(payload.get("default_status", "active")).strip().lower()
    entries = payload.get("factors", {}) or {}
    statuses = {}
    for name in factors:
        entry = entries.get(name, default)
        if isinstance(entry, dict):
            entry = entry.get("status", default)
        statuses[name] = str(entry).strip().lower()
    return statuses


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--family",
        choices=("alpha101", "gtja191"),
        default="alpha101",
        help="Factor family to evaluate; default: alpha101",
    )
    parser.add_argument("--data", default="data/raw", help="Raw per-symbol OHLCV CSV directory")
    parser.add_argument("--metadata", default="config/stocklist.csv", help="Optional classification CSV")
    parser.add_argument(
        "--benchmark-file",
        default=None,
        help="GTJA191 only: optional benchmark factor CSV with date,mkt style fields",
    )
    parser.add_argument(
        "--style-factor-file",
        default=None,
        help="GTJA191 only: optional style-factor CSV with date,mkt,smb,hml fields",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=f"Batch output directory; defaults to {ALPHA101_OUTPUT} or {GTJA191_OUTPUT}",
    )
    parser.add_argument(
        "--factor-config",
        default=None,
        help=f"YAML lifecycle config; defaults to {ALPHA101_FACTOR_CONFIG} or {GTJA191_FACTOR_CONFIG}",
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
        help="Names/numbers/ranges, e.g. all, 1-20, alpha_001 alpha_101, gtja_001",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="Factors to exclude, using the same syntax as --factors",
    )
    parser.add_argument(
        "--start-factor",
        default=None,
        help="GTJA191 only: start from this factor in registry order, e.g. 37 or gtja_037",
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
    parser.add_argument("--start-date", default=None, help="Alpha101 only: first evaluation date")
    parser.add_argument("--end-date", default=None, help="Alpha101 only: last evaluation date")
    parser.add_argument("--winsorize", action="store_true", help="Alpha101 only")
    parser.add_argument("--zscore", action="store_true", help="Alpha101 only")
    parser.add_argument("--min-periods", type=int, default=3)
    parser.add_argument("--min-listing-days", type=int, default=60)
    parser.add_argument("--liquidity-lookback-days", type=int, default=20, help="Alpha101 only")
    parser.add_argument("--min-liquidity", type=float, default=0.0, help="Alpha101 only")
    parser.add_argument("--commission-rate", type=float, default=0.0003, help="Alpha101 only")
    parser.add_argument("--slippage-rate", type=float, default=0.0005, help="Alpha101 only")
    parser.add_argument("--stamp-tax-rate", type=float, default=0.0005, help="Alpha101 only")
    parser.add_argument("--oos-start-date", default=None, help="Alpha101 only")
    parser.add_argument("--oos-fraction", type=float, default=0.3, help="Alpha101 only")
    parser.add_argument("--force", action="store_true", help="Recompute even when a matching report exists")
    parser.add_argument("--fail-fast", action="store_true", help="Stop at the first factor failure")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="GTJA191 only: disable the factor progress bar",
    )
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
        help="List every factor with its configured lifecycle status",
    )
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run resumable, memory-bounded batch tests for factor families.",
        epilog="GTJA191 reports are written to OUTPUT/leaderboard.csv.",
    )
    return add_arguments(parser)


def _run_alpha101(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser | None = None,
) -> int:
    if args.list_factors:
        print("\n".join(ALPHA101_NAMES))
        return 0

    factor_config = args.factor_config or ALPHA101_FACTOR_CONFIG
    output = args.output or ALPHA101_OUTPUT
    try:
        catalog = FactorCatalog() if args.ignore_factor_config else load_factor_catalog(factor_config)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        _argument_error(parser, str(exc))
    if args.list_factor_status:
        print("factor,status")
        for factor, status in catalog.status_map().items():
            print(f"{factor},{status}")
        return 0

    try:
        requested_factors = parse_alpha101_selection(args.factors, args.exclude)
        factors = catalog.select(requested_factors)
        windows = _positive_windows(args.windows)
        top_counts = _positive_windows(args.top_counts)
    except (KeyError, ValueError, argparse.ArgumentTypeError) as exc:
        _argument_error(parser, str(exc))
    if args.start_factor:
        _argument_error(parser, "--start-factor is only supported with --family gtja191")
    if not factors:
        _argument_error(
            parser,
            "factor selection is empty after exclusions and lifecycle filtering; "
            "use --ignore-factor-config for a one-off override"
        )
    if args.max_symbols is not None and args.max_symbols <= 0:
        _argument_error(parser, "--max-symbols must be positive")

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
            Path(factor_operators.__file__),
            Path(sys.modules[FactorTester.__module__].__file__),
            ROOT / "reports" / "alpha101_batch.py",
        ]
    )
    data_sig = directory_signature(args.data, metadata_path)
    if args.max_symbols:
        data_sig = f"{data_sig}:max-symbols={args.max_symbols}"

    result = Alpha101BatchRunner(
        panels,
        factors=factors,
        output_dir=output,
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


def _run_gtja191(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser | None = None,
) -> int:
    if args.list_factors:
        print("\n".join(GTJA191_NAMES))
        return 0

    factor_config = args.factor_config or GTJA191_FACTOR_CONFIG
    output = args.output or GTJA191_OUTPUT
    try:
        factors = parse_gtja_selection(args.factors, args.exclude)
        windows = _positive_windows(args.windows)
        top_counts = _positive_windows(args.top_counts)
    except (KeyError, ValueError, argparse.ArgumentTypeError) as exc:
        _argument_error(parser, str(exc))
    statuses = _gtja_factor_statuses(factor_config, factors)
    if args.list_factor_status:
        print("factor,status")
        for factor, status in statuses.items():
            print(f"{factor},{status}")
        return 0
    if not args.ignore_factor_config:
        factors = tuple(name for name in factors if statuses[name] in ("active", "watch"))
    factors = filter_gtja_selection_from_start(factors, args.start_factor)
    if not factors:
        _argument_error(
            parser,
            "factor selection is empty after exclusions, lifecycle filtering, and start-factor",
        )
    if args.max_symbols is not None and args.max_symbols <= 0:
        _argument_error(parser, "--max-symbols must be positive")

    if args.max_symbols:
        selected_symbols = sorted(path.stem for path in Path(args.data).glob("*.csv"))[: args.max_symbols]
        raw_data = load_raw_data(args.data, symbols=selected_symbols)
    else:
        raw_data = load_raw_data(args.data)
    metadata = pd.read_csv(args.metadata) if args.metadata and Path(args.metadata).exists() else None
    panels = build_gtja191_panels(
        raw_data,
        metadata=metadata,
        benchmark_data=_optional_csv(args.benchmark_file),
        style_factor_data=_optional_csv(args.style_factor_file),
    )
    config = GTJA191BatchConfig(
        windows=windows,
        groups=args.groups,
        top_n_counts=top_counts,
        min_periods=args.min_periods,
        min_listing_days=args.min_listing_days,
        force=args.force,
        fail_fast=args.fail_fast,
        show_progress=not args.no_progress,
    )
    implementation_signature = files_signature(
        [
            Path(__file__),
            Path(sys.modules[GTJA191.__module__].__file__),
            Path(factor_operators.__file__),
            Path(sys.modules[FactorTester.__module__].__file__),
            ROOT / "reports" / "alpha101_batch.py",
            ROOT / "reports" / "gtja191_batch.py",
        ]
    )
    result = GTJA191BatchRunner(
        panels,
        factors=factors,
        output_dir=output,
        config=config,
        implementation_signature=implementation_signature,
        factor_statuses={normalize_gtja_name(name): status for name, status in statuses.items()},
    ).run()
    counts = result.status["status"].value_counts().to_dict()
    print(f"GTJA191 batch report: {result.output_dir}")
    print(f"batch status: {result.output_dir / 'batch_status.csv'}")
    print(f"leaderboard: {result.output_dir / 'leaderboard.csv'}")
    print(f"status counts: {counts}")
    return 0


def run_from_args(
    args: argparse.Namespace,
    *,
    parser: argparse.ArgumentParser | None = None,
) -> int:
    if args.family == "gtja191":
        return _run_gtja191(args, parser)
    return _run_alpha101(args, parser)


def _argument_error(parser: argparse.ArgumentParser | None, message: str) -> None:
    if parser is not None:
        parser.error(message)
    raise ValueError(message)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return run_from_args(args, parser=parser)


if __name__ == "__main__":
    raise SystemExit(main())
