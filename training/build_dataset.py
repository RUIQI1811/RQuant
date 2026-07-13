"""Build lagged factor features and forward-return labels from local market data."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from domain.artifacts import WorkflowResult
from domain.research import MLDatasetResult
from domain.values import DateRange

import factors.alpha101 as alpha101_module
import factors.custom as custom_module
import factors.gtja191 as gtja191_module
import labels.make_forward_return as labels_module
from factors.alpha101 import Alpha101, build_alpha101_panels, normalize_alpha_name
from factors.custom import CustomFactors, normalize_custom_factor_name
from factors.gtja191 import GTJA191, build_gtja191_panels, normalize_gtja_name
from labels.make_forward_return import make_forward_returns, make_next_open_returns
from strategies.preselect import load_raw_data


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MLDatasetConfig:
    factors: tuple[str, ...]
    target_windows: tuple[int, ...] = (20,)
    factor_lag_days: int = 1
    label_mode: str = "next_open"
    feature_transform: str = "raw"
    target_transform: str = "raw"
    start_date: str | None = None
    end_date: str | None = None

    def __post_init__(self) -> None:
        factors = tuple(_normalize_factor_name(value) for value in self.factors)
        if not factors:
            raise ValueError("factors must not be empty")
        if len(set(factors)) != len(factors):
            raise ValueError("factors must be unique")
        object.__setattr__(self, "factors", factors)
        windows = tuple(int(value) for value in self.target_windows)
        if not windows or any(value <= 0 for value in windows):
            raise ValueError("target_windows must contain positive integers")
        if len(set(windows)) != len(windows):
            raise ValueError("target_windows must be unique")
        object.__setattr__(self, "target_windows", windows)
        if self.factor_lag_days != 1:
            raise ValueError("ML factor features must preserve factor_lag_days=1")
        label_mode = str(self.label_mode).strip().lower()
        if label_mode not in {"next_open", "close_to_close"}:
            raise ValueError("label_mode must be next_open or close_to_close")
        object.__setattr__(self, "label_mode", label_mode)
        valid_transforms = {"raw", "rank", "zscore"}
        feature_transform = str(self.feature_transform).strip().lower()
        target_transform = str(self.target_transform).strip().lower()
        if feature_transform not in valid_transforms:
            raise ValueError("feature_transform must be raw, rank, or zscore")
        if target_transform not in valid_transforms:
            raise ValueError("target_transform must be raw, rank, or zscore")
        object.__setattr__(self, "feature_transform", feature_transform)
        object.__setattr__(self, "target_transform", target_transform)
        if self.start_date and self.end_date:
            if pd.Timestamp(self.start_date) > pd.Timestamp(self.end_date):
                raise ValueError("start_date must not be after end_date")


def build_ml_dataset(
    *,
    data_dir: str | Path,
    output_dir: str | Path,
    config: MLDatasetConfig,
    metadata_path: str | Path | None = "config/stocklist.csv",
    benchmark_file: str | Path | None = None,
    style_factor_file: str | Path | None = None,
) -> WorkflowResult[MLDatasetResult]:
    """Calculate one-day-lagged factor features plus point-in-time labels."""

    started = time.perf_counter()
    logger.info("Loading raw market data from %s", Path(data_dir))
    raw_data = load_raw_data(str(data_dir), end_date=None)
    if not raw_data:
        raise ValueError("no raw market data found")
    logger.info("Loaded raw market data for %d symbols", len(raw_data))
    metadata_file = Path(metadata_path) if metadata_path else None
    metadata = (
        pd.read_csv(metadata_file)
        if metadata_file is not None and metadata_file.exists()
        else None
    )
    benchmark = _optional_csv(benchmark_file)
    style_factors = _optional_csv(style_factor_file)
    base_panels = build_alpha101_panels(raw_data, metadata=metadata)
    alpha_calculator = Alpha101(base_panels)
    custom_calculator = CustomFactors(base_panels)
    gtja_calculator = None

    dates = base_panels.close.index
    symbols = base_panels.close.columns
    logger.info(
        "Prepared base panels: %d trading dates, %d symbols",
        len(dates),
        len(symbols),
    )
    index = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
    features = pd.DataFrame(index=index)
    for position, factor in enumerate(config.factors, start=1):
        factor_started = time.perf_counter()
        logger.info("[%d/%d] Calculating factor %s", position, len(config.factors), factor)
        if factor.startswith("alpha_"):
            values = alpha_calculator.calculate(factor)
        elif factor.startswith("custom_"):
            values = custom_calculator.calculate(factor)
        elif factor.startswith("gtja_"):
            if gtja_calculator is None:
                gtja_panels = build_gtja191_panels(
                    raw_data,
                    metadata=metadata,
                    benchmark_data=benchmark,
                    style_factor_data=style_factors,
                )
                gtja_calculator = GTJA191(gtja_panels)
            values = gtja_calculator.calculate(factor)
        else:
            raise ValueError(f"unsupported factor family: {factor}")
        lagged = (
            values.reindex(index=dates, columns=symbols)
            .shift(config.factor_lag_days)
        )
        features[factor] = _to_long(lagged, factor).reindex(index)
        logger.info(
            "[%d/%d] Factor %s complete in %.2fs",
            position,
            len(config.factors),
            factor,
            time.perf_counter() - factor_started,
        )

    feature_frame = features.reset_index()
    feature_frame["symbol"] = feature_frame["symbol"].astype(str).str.zfill(6)
    if config.feature_transform != "raw":
        feature_frame = _cross_sectional_transform(
            feature_frame,
            columns=config.factors,
            transform=config.feature_transform,
        )
    if config.label_mode == "next_open":
        price_frame = _to_long(base_panels.open, "open").dropna().reset_index()
        price_frame["symbol"] = price_frame["symbol"].astype(str).str.zfill(6)
        label_frame = make_next_open_returns(price_frame, windows=config.target_windows)
        label_columns = [f"next_open_return_{window}d" for window in config.target_windows]
    else:
        price_frame = _to_long(base_panels.close, "close").dropna().reset_index()
        price_frame["symbol"] = price_frame["symbol"].astype(str).str.zfill(6)
        label_frame = make_forward_returns(price_frame, windows=config.target_windows)
        label_columns = [f"forward_return_{window}d" for window in config.target_windows]

    transformed_label_columns: list[str] = []
    if config.target_transform != "raw":
        transformed = _cross_sectional_transform(
            label_frame[["date", "symbol", *label_columns]],
            columns=label_columns,
            transform=config.target_transform,
        )
        suffix = f"cs_{config.target_transform}"
        for column in label_columns:
            transformed_column = f"{column}_{suffix}"
            label_frame[transformed_column] = transformed[column]
            transformed_label_columns.append(transformed_column)

    feature_frame = _filter_dates(feature_frame, config)
    label_frame = _filter_dates(label_frame, config)
    feature_frame["date"] = pd.to_datetime(feature_frame["date"]).dt.strftime("%Y-%m-%d")
    label_frame["date"] = pd.to_datetime(label_frame["date"]).dt.strftime("%Y-%m-%d")
    feature_frame = feature_frame.sort_values(["date", "symbol"], kind="mergesort")
    label_frame = label_frame.sort_values(["date", "symbol"], kind="mergesort")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    features_path = destination / "features.csv"
    labels_path = destination / "labels.csv"
    manifest_path = destination / "manifest.json"
    _atomic_write_csv(features_path, feature_frame)
    _atomic_write_csv(labels_path, label_frame)
    factor_missing = {
        factor: int(feature_frame[factor].isna().sum())
        for factor in config.factors
    }
    all_label_columns = [*label_columns, *transformed_label_columns]
    label_missing = {
        column: int(label_frame[column].isna().sum())
        for column in all_label_columns
    }
    manifest = {
        "config": asdict(config),
        "inputs": {
            "data_dir": str(Path(data_dir).resolve()),
            "metadata": str(metadata_file.resolve()) if metadata_file and metadata_file.exists() else None,
            "benchmark_file": str(Path(benchmark_file).resolve()) if benchmark_file else None,
            "style_factor_file": str(Path(style_factor_file).resolve()) if style_factor_file else None,
        },
        "data_signature": _data_signature(
            Path(data_dir),
            metadata_file,
            Path(benchmark_file) if benchmark_file else None,
            Path(style_factor_file) if style_factor_file else None,
        ),
        "implementation_sha256": {
            path.name: _file_sha256(path)
            for path in _implementation_files()
        },
        "feature_rows": len(feature_frame),
        "label_rows": len(label_frame),
        "factor_missing_rows": factor_missing,
        "label_missing_rows": label_missing,
        "target_columns": {
            "raw": label_columns,
            "fitted": transformed_label_columns or label_columns,
        },
        "date_range": {
            "start": feature_frame["date"].min() if not feature_frame.empty else None,
            "end": feature_frame["date"].max() if not feature_frame.empty else None,
        },
        "outputs": {"features": features_path.name, "labels": labels_path.name},
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_write_json(manifest_path, manifest)
    logger.info(
        "ML dataset complete in %.2fs: feature_rows=%d, label_rows=%d, output=%s",
        time.perf_counter() - started,
        len(feature_frame),
        len(label_frame),
        destination.resolve(),
    )
    result = MLDatasetResult(
        factors=tuple(config.factors),
        target_columns=tuple(all_label_columns),
        feature_rows=len(feature_frame),
        label_rows=len(label_frame),
        date_range=DateRange(
            feature_frame["date"].min() if not feature_frame.empty else None,
            feature_frame["date"].max() if not feature_frame.empty else None,
        ),
        factor_missing_rows=factor_missing,
        label_missing_rows=label_missing,
    )
    return WorkflowResult.from_mapping(
        {
            "result": result,
            "features_path": features_path,
            "labels_path": labels_path,
            "manifest_path": manifest_path,
        }
    )


def _normalize_factor_name(value: object) -> str:
    name = str(value).strip().lower().replace("-", "_")
    if name.startswith("alpha") or name.isdigit():
        return normalize_alpha_name(name)
    if name.startswith("custom"):
        return normalize_custom_factor_name(name)
    if name.startswith("gtja"):
        return normalize_gtja_name(name)
    raise ValueError(f"unsupported factor name: {value}")


def _filter_dates(frame: pd.DataFrame, config: MLDatasetConfig) -> pd.DataFrame:
    dates = pd.to_datetime(frame["date"])
    selected = pd.Series(True, index=frame.index)
    if config.start_date:
        selected &= dates >= pd.Timestamp(config.start_date)
    if config.end_date:
        selected &= dates <= pd.Timestamp(config.end_date)
    return frame.loc[selected].copy()


def _cross_sectional_transform(
    frame: pd.DataFrame,
    *,
    columns: tuple[str, ...] | list[str],
    transform: str,
) -> pd.DataFrame:
    """Transform values within each date without using any future date."""

    result = frame.copy()
    for column in columns:
        numeric = pd.to_numeric(result[column], errors="coerce")
        if transform == "rank":
            result[column] = numeric.groupby(result["date"], sort=False).rank(
                method="average",
                pct=True,
            )
        elif transform == "zscore":
            grouped = numeric.groupby(result["date"], sort=False)
            mean = grouped.transform("mean")
            scale = grouped.transform("std")
            result[column] = ((numeric - mean) / scale).where(scale.gt(1e-12), 0.0)
        else:
            raise ValueError(f"unsupported cross-sectional transform: {transform}")
    return result


def _optional_csv(path: str | Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"optional input file not found: {resolved}")
    return pd.read_csv(resolved)


def _to_long(panel: pd.DataFrame, name: str) -> pd.Series:
    return panel.rename_axis(index="date", columns="symbol").stack(future_stack=True).rename(name)


def _data_signature(data_dir: Path, *extra_files: Path | None) -> str:
    digest = hashlib.sha256()
    files = sorted(data_dir.glob("*.csv"))
    if not files:
        raise ValueError(f"no CSV files found in {data_dir}")
    for path in [*files, *(value for value in extra_files if value is not None and value.exists())]:
        stat = path.stat()
        digest.update(str(path.resolve()).encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return digest.hexdigest()


def _implementation_files() -> tuple[Path, ...]:
    return (
        Path(__file__).resolve(),
        Path(alpha101_module.__file__).resolve(),
        Path(custom_module.__file__).resolve(),
        Path(gtja191_module.__file__).resolve(),
        Path(labels_module.__file__).resolve(),
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    temp = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temp, index=False)
    os.replace(temp, path)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    os.replace(temp, path)


def _json_default(value: object) -> object:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--data", default="data/raw")
    parser.add_argument("--metadata", default="config/stocklist.csv")
    parser.add_argument("--benchmark-file", default=None)
    parser.add_argument("--style-factor-file", default=None)
    parser.add_argument("--factors", nargs="+", required=True)
    parser.add_argument("--target-windows", nargs="+", type=int, default=[20])
    parser.add_argument("--factor-lag-days", type=int, choices=(1,), default=1)
    parser.add_argument(
        "--label-mode",
        choices=("next_open", "close_to_close"),
        default="next_open",
    )
    parser.add_argument(
        "--feature-transform",
        choices=("raw", "rank", "zscore"),
        default="raw",
        help="Per-date transform after the mandatory one-day factor lag",
    )
    parser.add_argument(
        "--target-transform",
        choices=("raw", "rank", "zscore"),
        default="raw",
        help="Also emit a per-date transformed target for cross-sectional fitting",
    )
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--output", required=True)
    return parser


def config_from_args(args: argparse.Namespace) -> MLDatasetConfig:
    return MLDatasetConfig(
        factors=tuple(args.factors),
        target_windows=tuple(args.target_windows),
        factor_lag_days=args.factor_lag_days,
        label_mode=args.label_mode,
        feature_transform=args.feature_transform,
        target_transform=args.target_transform,
        start_date=args.start,
        end_date=args.end,
    )


def run_from_args(args: argparse.Namespace) -> WorkflowResult[MLDatasetResult]:
    return build_ml_dataset(
        data_dir=args.data,
        output_dir=args.output,
        config=config_from_args(args),
        metadata_path=args.metadata,
        benchmark_file=args.benchmark_file,
        style_factor_file=args.style_factor_file,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build lagged ML features and forward labels")
    return add_arguments(parser)


def main() -> None:
    outputs = run_from_args(build_parser().parse_args())
    print("ML dataset complete")
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
