from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from factors.alpha101 import build_alpha101_panels  # noqa: E402
from factors.catalog import FactorCatalog, load_factor_catalog  # noqa: E402
from factors.correlation import (  # noqa: E402
    FactorCorrelationConfig,
    calculate_factor_correlations,
    write_factor_correlation_reports,
)
from reports.alpha101_batch import parse_factor_selection  # noqa: E402
from strategies.preselect import load_raw_data  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build mean daily cross-sectional Spearman and Pearson correlation "
            "matrices for Alpha101 factors."
        )
    )
    parser.add_argument("--data", default="data/raw", help="Raw per-symbol OHLCV CSV directory")
    parser.add_argument("--metadata", default="config/stocklist.csv", help="Optional classification CSV")
    parser.add_argument(
        "--output",
        default="factor_report/factor_correlation",
        help="Correlation report directory",
    )
    parser.add_argument(
        "--factor-config",
        default="config/factors.yaml",
        help="YAML lifecycle config for active/watch/disabled factors",
    )
    parser.add_argument(
        "--ignore-factor-config",
        action="store_true",
        help="Run requested factors regardless of configured lifecycle status",
    )
    parser.add_argument(
        "--factors",
        nargs="+",
        default=["all"],
        help="Names/numbers/ranges, e.g. all, 1-20, alpha_003 alpha_040",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="Factors to exclude, using the same syntax as --factors",
    )
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--factor-lag-days", type=int, default=1)
    parser.add_argument("--min-observations", type=int, default=20)
    parser.add_argument("--min-dates", type=int, default=20)
    parser.add_argument("--high-correlation-threshold", type=float, default=0.8)
    parser.add_argument(
        "--max-symbols",
        type=int,
        default=None,
        help="Smoke-test only: load the first N sorted symbols",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.max_symbols is not None and args.max_symbols <= 0:
        parser.error("--max-symbols must be positive")

    try:
        catalog = FactorCatalog() if args.ignore_factor_config else load_factor_catalog(args.factor_config)
        requested = parse_factor_selection(args.factors, args.exclude)
        factors = requested if args.ignore_factor_config else catalog.select(requested)
        config = FactorCorrelationConfig(
            start_date=args.start_date,
            end_date=args.end_date,
            factor_lag_days=args.factor_lag_days,
            min_observations=args.min_observations,
            min_dates=args.min_dates,
            high_correlation_threshold=args.high_correlation_threshold,
        )
    except (FileNotFoundError, KeyError, ValueError) as exc:
        parser.error(str(exc))
    if len(factors) < 2:
        parser.error(
            "at least two factors must remain after exclusions and lifecycle filtering; "
            "use --ignore-factor-config for a one-off override"
        )

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.info("selected %d factors", len(factors))
    if args.max_symbols:
        symbols = sorted(path.stem for path in Path(args.data).glob("*.csv"))[: args.max_symbols]
        raw_data = load_raw_data(args.data, symbols=symbols)
        logging.warning("smoke-test universe limited to %d symbols", len(raw_data))
    else:
        raw_data = load_raw_data(args.data)

    metadata_path = Path(args.metadata) if args.metadata else None
    metadata = pd.read_csv(metadata_path) if metadata_path and metadata_path.exists() else None
    panels = build_alpha101_panels(raw_data, metadata=metadata)
    result = calculate_factor_correlations(
        panels,
        factors,
        config=config,
        factor_statuses=catalog.status_map(factors),
    )
    output_dir = write_factor_correlation_reports(result, args.output, config=config)
    print(f"spearman matrix: {output_dir / 'spearman_matrix.csv'}")
    print(f"pearson matrix: {output_dir / 'pearson_matrix.csv'}")
    print(f"pair ranking: {output_dir / 'correlation_pairs.csv'}")
    print(f"heatmap: {output_dir / 'spearman_heatmap.html'}")
    if not result.pairs.empty:
        strongest = result.pairs.iloc[0]
        print(
            "strongest pair: "
            f"{strongest['factor_a']} / {strongest['factor_b']} "
            f"(Spearman={strongest['spearman']:.4f})"
        )
    if result.failed_factors:
        print(f"failed factors: {', '.join(result.failed_factors)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
