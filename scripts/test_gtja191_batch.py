from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from factors.gtja191 import build_gtja191_panels, normalize_gtja_name  # noqa: E402
from reports.gtja191_batch import (  # noqa: E402
    GTJA191BatchConfig,
    GTJA191BatchRunner,
    filter_gtja_selection_from_start,
    parse_gtja_selection,
)
from strategies.preselect import load_raw_data  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate GTJA191 factors sequentially with resumable reports.")
    parser.add_argument("--data", default="data/raw")
    parser.add_argument("--metadata", default="pipeline/stocklist.csv")
    parser.add_argument("--benchmark-file", default=None)
    parser.add_argument("--style-factor-file", default=None)
    parser.add_argument("--output", default="factor_report/gtja191_batch")
    parser.add_argument("--factors", nargs="+", default=["all"])
    parser.add_argument("--exclude", nargs="*", default=[])
    parser.add_argument(
        "--start-factor",
        default=None,
        help="Start from this GTJA factor in registry order, e.g. 37 or gtja_037",
    )
    parser.add_argument("--factor-config", default="config/gtja191_factors.yaml")
    parser.add_argument("--ignore-factor-config", action="store_true")
    parser.add_argument("--windows", nargs="+", type=int, default=[1, 5, 10, 20])
    parser.add_argument("--groups", type=int, choices=(5, 10), default=10)
    parser.add_argument(
        "--top-counts",
        nargs="+",
        type=int,
        default=[1, 5, 10, 20, 50, 100],
        help="Long-only TopN buckets to report, default: 1 5 10 20 50 100",
    )
    parser.add_argument("--min-periods", type=int, default=3)
    parser.add_argument("--min-listing-days", type=int, default=60)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--no-progress", action="store_true", help="Disable the factor progress bar")
    parser.epilog = "The combined performance table is written to OUTPUT/leaderboard.csv."
    return parser


def _optional_csv(path: str | None) -> pd.DataFrame | None:
    return pd.read_csv(path) if path else None


def _factor_statuses(path: str, factors: tuple[str, ...]) -> dict[str, str]:
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
            entry = entry.get("status", entry.get("decision", default))
        statuses[name] = str(entry).strip().lower()
    return statuses


def main() -> None:
    args = build_parser().parse_args()
    if not args.top_counts or any(value <= 0 for value in args.top_counts):
        raise SystemExit("--top-counts must contain positive integers")
    factors = parse_gtja_selection(args.factors, args.exclude)
    statuses = _factor_statuses(args.factor_config, factors)
    if not args.ignore_factor_config:
        factors = tuple(name for name in factors if statuses[name] in ("active", "watch"))
    factors = filter_gtja_selection_from_start(factors, args.start_factor)
    if not factors:
        raise SystemExit(
            "factor selection is empty after exclusions, lifecycle filtering, and start-factor"
        )

    raw_data = load_raw_data(args.data)
    metadata = pd.read_csv(args.metadata) if args.metadata and Path(args.metadata).exists() else None
    panels = build_gtja191_panels(
        raw_data,
        metadata=metadata,
        benchmark_data=_optional_csv(args.benchmark_file),
        style_factor_data=_optional_csv(args.style_factor_file),
    )
    config = GTJA191BatchConfig(
        windows=tuple(args.windows),
        groups=args.groups,
        top_n_counts=tuple(args.top_counts),
        min_periods=args.min_periods,
        min_listing_days=args.min_listing_days,
        force=args.force,
        fail_fast=args.fail_fast,
        show_progress=not args.no_progress,
    )
    result = GTJA191BatchRunner(
        panels,
        factors=factors,
        output_dir=args.output,
        config=config,
        factor_statuses={normalize_gtja_name(name): status for name, status in statuses.items()},
    ).run()
    counts = result.status["status"].value_counts().to_dict()
    print(f"GTJA191 batch report: {result.output_dir}")
    print(f"batch status: {result.output_dir / 'batch_status.csv'}")
    print(f"leaderboard: {result.output_dir / 'leaderboard.csv'}")
    print(f"status counts: {counts}")


if __name__ == "__main__":
    main()
