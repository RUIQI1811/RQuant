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

from factors.alpha101 import build_alpha101_panels, normalize_alpha_name  # noqa: E402
from factors.catalog import FactorCatalog, load_factor_catalog  # noqa: E402
from factors.directions import load_gtja_factor_directions  # noqa: E402
from factors.correlation import (  # noqa: E402
    FactorCorrelationConfig,
    calculate_external_factor_correlations,
    calculate_factor_correlations,
    calculate_gtja_factor_correlations,
    write_factor_correlation_reports,
)
from factors.external import (  # noqa: E402
    load_external_factor_file,
    normalize_external_factor_name,
)
from reports.alpha101_batch import parse_factor_selection  # noqa: E402
from reports.gtja191_batch import parse_gtja_selection  # noqa: E402
from factors.gtja191 import (  # noqa: E402
    GTJA191_NAMES,
    build_gtja191_panels,
    normalize_gtja_name,
)
from strategies.preselect import load_raw_data  # noqa: E402


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--family",
        choices=("alpha101", "gtja191", "external"),
        default="alpha101",
        help="Factor family to correlate; --factor-file also selects external",
    )
    parser.add_argument("--data", default="data/raw", help="Raw per-symbol OHLCV CSV directory")
    parser.add_argument("--metadata", default="config/stocklist.csv", help="Optional classification CSV")
    parser.add_argument("--benchmark-file", default=None, help="GTJA191 benchmark daily CSV")
    parser.add_argument("--style-factor-file", default=None, help="GTJA191 MKT/SMB/HML daily CSV")
    parser.add_argument(
        "--factor-file",
        default=None,
        help="Optional external wide/long factor CSV; bypasses the Alpha101 calculator",
    )
    parser.add_argument(
        "--factor-layout",
        choices=("auto", "wide", "long"),
        default="auto",
    )
    parser.add_argument("--date-col", default="date")
    parser.add_argument("--symbol-col", default="symbol")
    parser.add_argument("--factor-name-col", default="factor")
    parser.add_argument("--factor-value-col", default="factor_value")
    parser.add_argument(
        "--output",
        default="factor_report/factor_correlation",
        help="Correlation report directory",
    )
    parser.add_argument(
        "--factor-config",
        default=None,
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
        "--priority-file",
        default=None,
        help="Optional CSV used to retain the highest-quality factor in each correlation cluster",
    )
    parser.add_argument("--priority-factor-col", default="factor")
    parser.add_argument(
        "--priority-score-col",
        default="tradable_top_quantile_sharpe",
        help="Higher score wins inside each |Spearman| threshold cluster",
    )
    parser.add_argument(
        "--priority-window",
        type=int,
        default=None,
        help="Optional leaderboard window used for both quality and eligibility",
    )
    parser.add_argument(
        "--eligibility-col",
        default=None,
        help=(
            "Optional boolean priority-file column; only eligible deduplicated "
            "representatives are written to ml_candidate_factors.csv"
        ),
    )
    parser.add_argument(
        "--max-symbols",
        type=int,
        default=None,
        help="Smoke-test only: load the first N sorted symbols",
    )
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build mean daily cross-sectional Spearman and Pearson correlation "
            "matrices for Alpha101 or external factors."
        )
    )
    return add_arguments(parser)


def run_from_args(
    args: argparse.Namespace,
    *,
    parser: argparse.ArgumentParser | None = None,
) -> int:
    if args.max_symbols is not None and args.max_symbols <= 0:
        _argument_error(parser, "--max-symbols must be positive")

    try:
        config = FactorCorrelationConfig(
            start_date=args.start_date,
            end_date=args.end_date,
            factor_lag_days=args.factor_lag_days,
            min_observations=args.min_observations,
            min_dates=args.min_dates,
            high_correlation_threshold=args.high_correlation_threshold,
        )
    except (FileNotFoundError, KeyError, ValueError) as exc:
        _argument_error(parser, str(exc))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    family = "external" if args.factor_file else args.family
    if family == "external" and not args.factor_file:
        _argument_error(parser, "--family external requires --factor-file")
    catalog: FactorCatalog | None = None
    gtja_statuses: dict[str, str] | None = None
    gtja_directions: dict[str, int] | None = None
    external = None
    if family == "external":
        requested_external = None
        if not any(str(value).strip().lower() == "all" for value in args.factors):
            requested_external = tuple(args.factors)
        try:
            external = load_external_factor_file(
                args.factor_file,
                factors=requested_external,
                date_col=args.date_col,
                symbol_col=args.symbol_col,
                layout=args.factor_layout,
                factor_name_col=args.factor_name_col,
                factor_value_col=args.factor_value_col,
            )
        except (FileNotFoundError, ValueError) as exc:
            _argument_error(parser, str(exc))
        excluded = {normalize_external_factor_name(value) for value in args.exclude}
        factors = tuple(name for name in external.factors if name not in excluded)
    elif family == "gtja191":
        factor_config = args.factor_config or "config/gtja191_factors.yaml"
        try:
            requested = parse_gtja_selection(args.factors, args.exclude)
            gtja_statuses = _load_gtja_statuses(factor_config)
            gtja_directions = load_gtja_factor_directions(
                factor_config,
                GTJA191_NAMES,
            )
            factors = (
                requested
                if args.ignore_factor_config
                else tuple(
                    name
                    for status in ("active", "watch")
                    for name in requested
                    if gtja_statuses.get(name, "disabled") == status
                )
            )
        except (FileNotFoundError, KeyError, ValueError) as exc:
            _argument_error(parser, str(exc))
    else:
        factor_config = args.factor_config or "config/factors.yaml"
        try:
            catalog = (
                FactorCatalog()
                if args.ignore_factor_config
                else load_factor_catalog(factor_config)
            )
            requested = parse_factor_selection(args.factors, args.exclude)
            factors = requested if args.ignore_factor_config else catalog.select(requested)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            _argument_error(parser, str(exc))
    if len(factors) < 2:
        _argument_error(parser, "at least two factors must remain after filtering")
    logging.info("selected %d factors", len(factors))
    priority_scores = None
    eligible_factors = None
    eligibility_settings = None
    if args.priority_file:
        priority = pd.read_csv(args.priority_file)
        missing = {
            args.priority_factor_col,
            args.priority_score_col,
        }.difference(priority.columns)
        if missing:
            _argument_error(
                parser,
                "priority file missing columns: " + ", ".join(sorted(missing))
            )
        if family == "external":
            normalizer = normalize_external_factor_name
        elif family == "gtja191":
            normalizer = normalize_gtja_name
        else:
            normalizer = normalize_alpha_name
        priority[args.priority_factor_col] = priority[args.priority_factor_col].map(normalizer)
        if args.priority_window is not None:
            if "window" not in priority.columns:
                _argument_error(parser, "priority file missing columns: window")
            priority = priority.loc[
                pd.to_numeric(priority["window"], errors="coerce").eq(
                    int(args.priority_window)
                )
            ]
            if priority.empty:
                _argument_error(
                    parser,
                    f"priority file has no rows for window={args.priority_window}",
                )
        priority[args.priority_score_col] = pd.to_numeric(
            priority[args.priority_score_col], errors="coerce"
        )
        priority_scores = (
            priority.dropna(subset=[args.priority_score_col])
            .groupby(args.priority_factor_col)[args.priority_score_col]
            .max()
            .to_dict()
        )
        if args.eligibility_col:
            if args.eligibility_col not in priority.columns:
                _argument_error(
                    parser,
                    f"priority file missing columns: {args.eligibility_col}",
                )
            try:
                eligibility = priority[args.eligibility_col].map(_as_bool)
            except ValueError as exc:
                _argument_error(parser, str(exc))
            eligible_factors = tuple(
                sorted(
                    priority.loc[eligibility, args.priority_factor_col]
                    .dropna()
                    .astype(str)
                    .unique()
                )
            )
            eligibility_settings = {
                "source_file": str(args.priority_file),
                "column": args.eligibility_col,
                "window": args.priority_window,
            }
    elif args.priority_window is not None or args.eligibility_col:
        _argument_error(
            parser,
            "--priority-window/--eligibility-col require --priority-file",
        )
    if external is not None:
        external_frame = external.frame
        if args.max_symbols:
            symbols = sorted(external_frame["symbol"].unique())[: args.max_symbols]
            external_frame = external_frame.loc[external_frame["symbol"].isin(symbols)]
            logging.warning("smoke-test universe limited to %d symbols", len(symbols))
        result = calculate_external_factor_correlations(
            external_frame,
            factors,
            config=config,
            factor_statuses={factor: "active" for factor in factors},
            priority_scores=priority_scores,
        )
    else:
        if args.max_symbols:
            symbols = sorted(path.stem for path in Path(args.data).glob("*.csv"))[
                : args.max_symbols
            ]
            raw_data = load_raw_data(args.data, symbols=symbols)
            logging.warning("smoke-test universe limited to %d symbols", len(raw_data))
        else:
            raw_data = load_raw_data(args.data)
        metadata_path = Path(args.metadata) if args.metadata else None
        metadata = (
            pd.read_csv(metadata_path)
            if metadata_path and metadata_path.exists()
            else None
        )
        if family == "gtja191":
            benchmark = pd.read_csv(args.benchmark_file) if args.benchmark_file else None
            style_factors = (
                pd.read_csv(args.style_factor_file) if args.style_factor_file else None
            )
            panels = build_gtja191_panels(
                raw_data,
                metadata=metadata,
                benchmark_data=benchmark,
                style_factor_data=style_factors,
            )
            result = calculate_gtja_factor_correlations(
                panels,
                factors,
                config=config,
                factor_statuses=gtja_statuses,
                factor_directions=gtja_directions,
                priority_scores=priority_scores,
            )
        else:
            panels = build_alpha101_panels(raw_data, metadata=metadata)
            result = calculate_factor_correlations(
                panels,
                factors,
                config=config,
                factor_statuses=catalog.status_map(factors) if catalog else None,
                priority_scores=priority_scores,
            )
    output_dir = write_factor_correlation_reports(
        result,
        args.output,
        config=config,
        eligible_factors=eligible_factors,
        eligibility_settings=eligibility_settings,
    )
    print(f"spearman matrix: {output_dir / 'spearman_matrix.csv'}")
    print(f"pearson matrix: {output_dir / 'pearson_matrix.csv'}")
    print(f"pair ranking: {output_dir / 'correlation_pairs.csv'}")
    print(f"deduplicated factors: {output_dir / 'deduplicated_factors.csv'}")
    if eligible_factors is not None:
        print(f"ML candidate factors: {output_dir / 'ml_candidate_factors.csv'}")
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


def _argument_error(parser: argparse.ArgumentParser | None, message: str) -> None:
    if parser is not None:
        parser.error(message)
    raise ValueError(message)


def _as_bool(value: object) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n", ""}:
        return False
    raise ValueError(f"invalid boolean eligibility value: {value!r}")


def _load_gtja_statuses(path: str | Path) -> dict[str, str]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"factor config not found: {config_path}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("factor config root must be a mapping")
    default = str(payload.get("default_status", "active")).strip().lower()
    if default not in {"active", "watch", "disabled"}:
        raise ValueError(f"invalid default_status: {default!r}")
    entries = payload.get("factors", {}) or {}
    if not isinstance(entries, dict):
        raise ValueError("factor config 'factors' must be a mapping")
    statuses: dict[str, str] = {}
    for name in GTJA191_NAMES:
        entry = entries.get(name, default)
        status = entry.get("status", default) if isinstance(entry, dict) else entry
        normalized = str(status).strip().lower()
        if normalized not in {"active", "watch", "disabled"}:
            raise ValueError(f"invalid status for {name}: {status!r}")
        statuses[name] = normalized
    return statuses


def main() -> int:
    parser = build_parser()
    return run_from_args(parser.parse_args(), parser=parser)


if __name__ == "__main__":
    raise SystemExit(main())
