"""End-to-end multi-factor fitting with comparable walk-forward models."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from backtest.signal_portfolio import run_signal_portfolio_backtest
from domain.artifacts import WorkflowResult
from domain.research import MultifactorComparisonResult
from models.elasticnet import DEFAULT_ELASTICNET_ALPHA, DEFAULT_ELASTICNET_L1_RATIO

from training.build_dataset import MLDatasetConfig, build_ml_dataset
from training.train_walk_forward import (
    DEFAULT_MLP_EPOCHS,
    MODEL_NAMES,
    WalkForwardTrainingConfig,
    run_walk_forward_training,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MultifactorFitConfig:
    factors: tuple[str, ...]
    models: tuple[str, ...] = ("ridge", "lightgbm", "doubleensemble", "mlp")
    target_window: int = 20
    label_mode: str = "next_open"
    feature_transform: str = "rank"
    target_transform: str = "rank"
    train_size: int = 504
    test_size: int = 21
    window_mode: str = "trading_days"
    train_years: int = 3
    test_years: int = 1
    purge_days: int | None = None
    signal_top_n: int = 10
    ridge_alpha: float = 1.0
    elasticnet_alpha: float = DEFAULT_ELASTICNET_ALPHA
    elasticnet_l1_ratio: float = DEFAULT_ELASTICNET_L1_RATIO
    lightgbm_estimators: int = 200
    lightgbm_n_jobs: int = 1
    qlib_valid_ratio: float = 0.2
    doubleensemble_num_models: int = 6
    mlp_hidden_sizes: tuple[int, ...] = (64, 32)
    mlp_epochs: int = DEFAULT_MLP_EPOCHS
    mlp_batch_size: int = 256
    mlp_learning_rate: float = 1e-3
    mlp_weight_decay: float = 0.0
    mlp_dropout: float = 0.0
    random_state: int = 42
    device: str = "auto"
    start_date: str | None = None
    end_date: str | None = None
    run_backtests: bool = False
    backtest_initial_cash: float = 10000000.0
    backtest_commission_wan: float = 0.8
    backtest_stamp_tax_rate: float = 0.0005
    backtest_transfer_fee_rate: float = 0.00001
    backtest_lot_size: int = 100

    def __post_init__(self) -> None:
        factors = tuple(str(value).strip() for value in self.factors if str(value).strip())
        if not factors:
            raise ValueError("factors must not be empty")
        object.__setattr__(self, "factors", factors)
        models = tuple(str(value).strip().lower() for value in self.models)
        if not models or len(set(models)) != len(models):
            raise ValueError("models must be non-empty and unique")
        invalid = sorted(set(models).difference(MODEL_NAMES))
        if invalid:
            raise ValueError(f"unsupported models: {invalid}")
        object.__setattr__(self, "models", models)
        if self.target_window <= 0:
            raise ValueError("target_window must be positive")
        label_mode = str(self.label_mode).strip().lower()
        if label_mode not in {"next_open", "close_to_close"}:
            raise ValueError("label_mode must be next_open or close_to_close")
        object.__setattr__(self, "label_mode", label_mode)
        for field_name in ("feature_transform", "target_transform"):
            value = str(getattr(self, field_name)).strip().lower()
            if value not in {"raw", "rank", "zscore"}:
                raise ValueError(f"{field_name} must be raw, rank, or zscore")
            object.__setattr__(self, field_name, value)
        if self.train_size <= 0 or self.test_size <= 0 or self.signal_top_n <= 0:
            raise ValueError("train_size, test_size, and signal_top_n must be positive")
        window_mode = str(self.window_mode).strip().lower().replace("-", "_")
        if window_mode not in {"trading_days", "calendar_years"}:
            raise ValueError("window_mode must be trading_days or calendar_years")
        object.__setattr__(self, "window_mode", window_mode)
        if self.train_years <= 0 or self.test_years <= 0:
            raise ValueError("train_years and test_years must be positive")
        if not 0 < self.qlib_valid_ratio < 1:
            raise ValueError("qlib_valid_ratio must be in (0, 1)")
        if self.doubleensemble_num_models <= 0:
            raise ValueError("doubleensemble_num_models must be positive")
        if self.start_date and self.end_date:
            if pd.Timestamp(self.start_date) > pd.Timestamp(self.end_date):
                raise ValueError("start_date must not be after end_date")
        if self.backtest_initial_cash <= 0 or self.backtest_lot_size <= 0:
            raise ValueError("backtest_initial_cash and backtest_lot_size must be positive")
        if (
            self.backtest_commission_wan < 0
            or self.backtest_stamp_tax_rate < 0
            or self.backtest_transfer_fee_rate < 0
        ):
            raise ValueError("backtest transaction-cost rates must be non-negative")


def run_multifactor_fit(
    *,
    data_dir: str | Path,
    output_dir: str | Path,
    config: MultifactorFitConfig,
    metadata_path: str | Path | None = "config/stocklist.csv",
    benchmark_file: str | Path | None = None,
    style_factor_file: str | Path | None = None,
    factor_file: str | Path | None = None,
    factor_layout: str = "auto",
    factor_date_col: str = "date",
    factor_symbol_col: str = "symbol",
    factor_name_col: str = "factor",
    factor_value_col: str = "factor_value",
    factor_selection_file: str | Path | None = None,
    context_file: str | Path | None = None,
    context_date_col: str = "date",
    context_symbol_col: str = "symbol",
    force: bool = False,
) -> WorkflowResult[MultifactorComparisonResult]:
    """Build the factor matrix, train each model, and rank OOS diagnostics."""

    started = time.perf_counter()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Starting multi-factor fit: factors=%d, models=%s, target_window=%dd, output=%s",
        len(config.factors),
        ",".join(config.models),
        config.target_window,
        destination.resolve(),
    )
    dataset_config = MLDatasetConfig(
        factors=config.factors,
        target_windows=(config.target_window,),
        factor_lag_days=1,
        label_mode=config.label_mode,
        feature_transform=config.feature_transform,
        target_transform=config.target_transform,
        start_date=config.start_date,
        end_date=config.end_date,
    )
    dataset_outputs = build_ml_dataset(
        data_dir=data_dir,
        output_dir=destination / "dataset",
        config=dataset_config,
        metadata_path=metadata_path,
        benchmark_file=benchmark_file,
        style_factor_file=style_factor_file,
        factor_file=factor_file,
        factor_layout=factor_layout,
        factor_date_col=factor_date_col,
        factor_symbol_col=factor_symbol_col,
        factor_name_col=factor_name_col,
        factor_value_col=factor_value_col,
        context_file=context_file,
        context_date_col=context_date_col,
        context_symbol_col=context_symbol_col,
    )
    logger.info(
        "Shared dataset ready: features=%s, labels=%s",
        dataset_outputs["features_path"],
        dataset_outputs["labels_path"],
    )
    target_col = _target_column(config)
    leaderboard_rows: list[dict[str, object]] = []
    model_outputs: dict[str, dict[str, Path]] = {}
    model_backtests: dict[str, dict[str, dict[str, Path]]] = {}
    for position, model_name in enumerate(config.models, start=1):
        model_started = time.perf_counter()
        logger.info(
            "[%d/%d] Starting model %s",
            position,
            len(config.models),
            model_name,
        )
        outputs = run_walk_forward_training(
            features_path=dataset_outputs["features_path"],
            labels_path=dataset_outputs["labels_path"],
            output_dir=destination / "models" / model_name,
            config=WalkForwardTrainingConfig(
                feature_cols=dataset_config.factors,
                target_col=target_col,
                model=model_name,
                train_size=config.train_size,
                test_size=config.test_size,
                window_mode=config.window_mode,
                train_years=config.train_years,
                test_years=config.test_years,
                purge_days=config.purge_days,
                signal_top_n=config.signal_top_n,
                ridge_alpha=config.ridge_alpha,
                elasticnet_alpha=config.elasticnet_alpha,
                elasticnet_l1_ratio=config.elasticnet_l1_ratio,
                lightgbm_estimators=config.lightgbm_estimators,
                lightgbm_n_jobs=config.lightgbm_n_jobs,
                qlib_valid_ratio=config.qlib_valid_ratio,
                doubleensemble_num_models=config.doubleensemble_num_models,
                mlp_hidden_sizes=config.mlp_hidden_sizes,
                mlp_epochs=config.mlp_epochs,
                mlp_batch_size=config.mlp_batch_size,
                mlp_learning_rate=config.mlp_learning_rate,
                mlp_weight_decay=config.mlp_weight_decay,
                mlp_dropout=config.mlp_dropout,
                random_state=config.random_state,
                device=config.device,
            ),
            force=force,
        )
        model_outputs[model_name] = outputs
        summary = json.loads(outputs["summary_path"].read_text(encoding="utf-8"))
        metrics = summary["out_of_sample_metrics"]
        backtest_metrics, backtest_outputs = _run_model_backtests(
            model_name=model_name,
            signals_path=outputs["signals_path"],
            data_dir=data_dir,
            destination=destination,
            config=config,
        )
        model_backtests[model_name] = backtest_outputs
        leaderboard_rows.append(
            {
                "model": model_name,
                "rank_ic_mean": metrics.get("rank_ic_mean"),
                "rank_ic_date_count": metrics.get("rank_ic_date_count"),
                "pearson": metrics.get("pearson"),
                "mse": metrics.get("mse"),
                "mae": metrics.get("mae"),
                "window_count": summary.get("window_count"),
                "prediction_count": summary.get("prediction_count"),
                "signal_count": summary.get("signal_count"),
                "signals_path": str(outputs["signals_path"].resolve()),
                "summary_path": str(outputs["summary_path"].resolve()),
                **backtest_metrics,
            }
        )
        logger.info(
            "[%d/%d] Model %s complete in %.2fs: rank_ic=%s, predictions=%s, signals=%s",
            position,
            len(config.models),
            model_name,
            time.perf_counter() - model_started,
            metrics.get("rank_ic_mean"),
            summary.get("prediction_count"),
            summary.get("signal_count"),
        )

    leaderboard = pd.DataFrame(leaderboard_rows).sort_values(
        ["rank_ic_mean", "model"],
        ascending=[False, True],
        na_position="last",
        kind="mergesort",
    )
    leaderboard_path = destination / "leaderboard.csv"
    profitable_models_path = destination / "profitable_models.csv"
    manifest_path = destination / "manifest.json"
    _atomic_write_csv(leaderboard_path, leaderboard)
    if config.run_backtests:
        profitable = leaderboard.loc[
            leaderboard["profitable_before_cost"].fillna(False).astype(bool)
            | leaderboard["profitable_after_cost"].fillna(False).astype(bool)
        ].sort_values(
            ["net_sharpe", "model"],
            ascending=[False, True],
            na_position="last",
            kind="mergesort",
        )
        _atomic_write_csv(profitable_models_path, profitable)
    _atomic_write_json(
        manifest_path,
        {
            "config": asdict(config),
            "normalized_factors": list(dataset_config.factors),
            "target_col": target_col,
            "dataset_manifest": str(dataset_outputs["manifest_path"].resolve()),
            "external_factor_file": (
                str(Path(factor_file).resolve()) if factor_file else None
            ),
            "factor_selection_file": (
                str(Path(factor_selection_file).resolve())
                if factor_selection_file
                else None
            ),
            "research_context_file": (
                str(Path(context_file).resolve()) if context_file else None
            ),
            "models": {
                name: {
                    "signals": str(outputs["signals_path"].resolve()),
                    "summary": str(outputs["summary_path"].resolve()),
                    "manifest": str(outputs["manifest_path"].resolve()),
                    "backtests": {
                        scenario: {
                            key: str(path.resolve())
                            for key, path in scenario_outputs.items()
                        }
                        for scenario, scenario_outputs in model_backtests.get(
                            name, {}
                        ).items()
                    },
                }
                for name, outputs in model_outputs.items()
            },
            "leaderboard": leaderboard_path.name,
            "profitable_models": (
                profitable_models_path.name if config.run_backtests else None
            ),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    result = MultifactorComparisonResult(
        models=tuple(config.models),
        factors=tuple(dataset_config.factors),
        target_column=target_col,
        leaderboard=tuple(leaderboard.to_dict("records")),
    )
    logger.info(
        "Multi-factor fit complete in %.2fs: leaderboard=%s, manifest=%s",
        time.perf_counter() - started,
        leaderboard_path.resolve(),
        manifest_path.resolve(),
    )
    output_mapping: dict[str, object] = {
        "result": result,
        "dataset_dir": destination / "dataset",
        "models_dir": destination / "models",
        "leaderboard_path": leaderboard_path,
        "manifest_path": manifest_path,
    }
    if config.run_backtests:
        output_mapping["profitable_models_path"] = profitable_models_path
    return WorkflowResult.from_mapping(output_mapping)


def _target_column(config: MultifactorFitConfig) -> str:
    prefix = "next_open_return" if config.label_mode == "next_open" else "forward_return"
    raw = f"{prefix}_{config.target_window}d"
    return raw if config.target_transform == "raw" else f"{raw}_cs_{config.target_transform}"


def _run_model_backtests(
    *,
    model_name: str,
    signals_path: str | Path,
    data_dir: str | Path,
    destination: Path,
    config: MultifactorFitConfig,
) -> tuple[dict[str, object], dict[str, dict[str, Path]]]:
    metrics: dict[str, object] = {
        "backtest_status": "not_run",
        "gross_total_return": None,
        "gross_annualized_return": None,
        "gross_sharpe": None,
        "net_total_return": None,
        "net_annualized_return": None,
        "net_sharpe": None,
        "net_max_drawdown": None,
        "net_realized_trade_count": None,
        "profitable_before_cost": None,
        "profitable_after_cost": None,
    }
    if not config.run_backtests:
        return metrics, {}

    artifacts: dict[str, dict[str, Path]] = {}
    for scenario, costs in (
        ("gross", (0.0, 0.0, 0.0)),
        (
            "net",
            (
                config.backtest_commission_wan,
                config.backtest_stamp_tax_rate,
                config.backtest_transfer_fee_rate,
            ),
        ),
    ):
        commission_wan, stamp_tax_rate, transfer_fee_rate = costs
        logger.info("Running %s long-only backtest for model %s", scenario, model_name)
        backtest = run_signal_portfolio_backtest(
            signals_path=signals_path,
            data_dir=data_dir,
            output_dir=destination / "backtests" / model_name / scenario,
            source=f"model_{model_name}",
            start_date=None,
            end_date=config.end_date,
            initial_cash=config.backtest_initial_cash,
            hold_days=config.target_window,
            commission_wan=commission_wan,
            stamp_tax_rate=stamp_tax_rate,
            transfer_fee_rate=transfer_fee_rate,
            max_positions=config.signal_top_n,
            lot_size=config.backtest_lot_size,
            show_progress=False,
        )
        artifacts[scenario] = {
            key: Path(value)
            for key, value in backtest.items()
            if key != "result" and isinstance(value, (str, Path))
        }
        summary = backtest["result"].summary
        metrics[f"{scenario}_total_return"] = summary.get("total_return")
        metrics[f"{scenario}_annualized_return"] = summary.get(
            "annualized_return_mean"
        )
        metrics[f"{scenario}_sharpe"] = summary.get("sharpe_ratio")
        if scenario == "net":
            metrics["net_max_drawdown"] = summary.get("max_drawdown")
            metrics["net_realized_trade_count"] = summary.get(
                "realized_trade_count"
            )

    metrics["backtest_status"] = "success"
    metrics["profitable_before_cost"] = _is_positive(metrics["gross_total_return"])
    metrics["profitable_after_cost"] = _is_positive(metrics["net_total_return"])
    return metrics, artifacts


def _is_positive(value: object) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return bool(pd.notna(numeric) and numeric > 0)


def load_lifecycle_factors(
    path: str | Path,
    *,
    statuses: tuple[str, ...] = ("active",),
) -> tuple[str, ...]:
    """Return configured factors whose lifecycle status is explicitly selected."""

    selected_statuses = {str(value).strip().lower() for value in statuses}
    if not selected_statuses:
        raise ValueError("lifecycle statuses must not be empty")
    invalid = selected_statuses.difference({"active", "watch", "disabled"})
    if invalid:
        raise ValueError(f"unsupported lifecycle statuses: {sorted(invalid)}")

    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    default = str(payload.get("default_status", "active")).strip().lower()
    entries = payload.get("factors", {}) or {}
    if not isinstance(entries, dict):
        raise ValueError(f"lifecycle config factors must be a mapping: {config_path}")

    factors: list[str] = []
    for name, entry in entries.items():
        status = entry.get("status", default) if isinstance(entry, dict) else entry
        if str(status).strip().lower() in selected_statuses:
            factors.append(str(name).strip())
    return tuple(factors)


def load_factor_selection_file(
    path: str | Path,
    *,
    factor_col: str = "factor",
) -> tuple[str, ...]:
    """Load an ordered factor list such as correlation deduplication output."""

    selection_path = Path(path)
    if not selection_path.exists():
        raise FileNotFoundError(f"factor selection file not found: {selection_path}")
    frame = pd.read_csv(selection_path, dtype={factor_col: str})
    if factor_col not in frame.columns:
        raise ValueError(
            f"factor selection file missing column {factor_col!r}: {selection_path}"
        )
    factors = tuple(
        dict.fromkeys(
            str(value).strip()
            for value in frame[factor_col].dropna()
            if str(value).strip()
        )
    )
    if not factors:
        raise ValueError(f"factor selection file is empty: {selection_path}")
    return factors


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    temp = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temp, index=False)
    os.replace(temp, path)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--data", default="data/raw")
    parser.add_argument("--metadata", default="config/stocklist.csv")
    parser.add_argument("--benchmark-file", default=None)
    parser.add_argument("--style-factor-file", default=None)
    parser.add_argument(
        "--factor-file",
        default=None,
        help="Optional external wide/long factor CSV; columns may be mixed with built-ins",
    )
    parser.add_argument(
        "--factor-layout",
        choices=("auto", "wide", "long"),
        default="auto",
    )
    parser.add_argument("--factor-date-col", default="date")
    parser.add_argument("--factor-symbol-col", default="symbol")
    parser.add_argument("--factor-name-col", default="factor")
    parser.add_argument("--factor-value-col", default="factor_value")
    parser.add_argument(
        "--context-file",
        default=None,
        help="Optional point-in-time date,symbol market-cap/classification/regime CSV",
    )
    parser.add_argument("--context-date-col", default="date")
    parser.add_argument("--context-symbol-col", default="symbol")
    parser.add_argument(
        "--factors",
        nargs="+",
        default=[],
        help="Explicit factor names; can be combined with --factor-config",
    )
    parser.add_argument(
        "--factor-config",
        default=None,
        help="Lifecycle YAML whose selected factors are added to the ML feature set",
    )
    parser.add_argument(
        "--factor-selection-file",
        default=None,
        help="CSV factor list, e.g. factor-correlation/deduplicated_factors.csv",
    )
    parser.add_argument("--factor-selection-col", default="factor")
    parser.add_argument(
        "--lifecycle-statuses",
        nargs="+",
        choices=("active", "watch", "disabled"),
        default=["active"],
        help="Statuses to import from --factor-config; default: active",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_NAMES,
        default=["ridge", "lightgbm", "doubleensemble", "mlp"],
    )
    parser.add_argument("--target-window", type=int, default=20)
    parser.add_argument("--label-mode", choices=("next_open", "close_to_close"), default="next_open")
    parser.add_argument("--feature-transform", choices=("raw", "rank", "zscore"), default="rank")
    parser.add_argument("--target-transform", choices=("raw", "rank", "zscore"), default="rank")
    parser.add_argument("--train-size", type=int, default=504)
    parser.add_argument("--test-size", type=int, default=21)
    parser.add_argument(
        "--window-mode",
        choices=("trading-days", "calendar-years"),
        default="trading-days",
    )
    parser.add_argument("--train-years", type=int, default=3)
    parser.add_argument("--test-years", type=int, default=1)
    parser.add_argument("--purge-days", type=int, default=None)
    parser.add_argument("--signal-top-n", type=int, default=10)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument(
        "--elasticnet-alpha",
        type=float,
        default=DEFAULT_ELASTICNET_ALPHA,
    )
    parser.add_argument(
        "--elasticnet-l1-ratio",
        type=float,
        default=DEFAULT_ELASTICNET_L1_RATIO,
    )
    parser.add_argument("--lightgbm-estimators", type=int, default=200)
    parser.add_argument("--lightgbm-n-jobs", type=int, default=1)
    parser.add_argument("--qlib-valid-ratio", type=float, default=0.2)
    parser.add_argument("--doubleensemble-num-models", type=int, default=6)
    parser.add_argument("--mlp-hidden-sizes", nargs="+", type=int, default=[64, 32])
    parser.add_argument("--mlp-epochs", type=int, default=DEFAULT_MLP_EPOCHS)
    parser.add_argument("--mlp-batch-size", type=int, default=256)
    parser.add_argument("--mlp-learning-rate", type=float, default=1e-3)
    parser.add_argument("--mlp-weight-decay", type=float, default=0.0)
    parser.add_argument("--mlp-dropout", type=float, default=0.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument(
        "--run-backtests",
        action="store_true",
        help="Run gross and cost-aware long-only portfolio backtests for every model",
    )
    parser.add_argument("--backtest-initial-cash", type=float, default=10000000.0)
    parser.add_argument("--backtest-commission-wan", type=float, default=0.8)
    parser.add_argument("--backtest-stamp-tax-rate", type=float, default=0.0005)
    parser.add_argument("--backtest-transfer-fee-rate", type=float, default=0.00001)
    parser.add_argument("--backtest-lot-size", type=int, default=100)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output", default="data/ml/multifactor")
    return parser


def config_from_args(args: argparse.Namespace) -> MultifactorFitConfig:
    factors = list(args.factors)
    if args.factor_selection_file:
        factors.extend(
            load_factor_selection_file(
                args.factor_selection_file,
                factor_col=args.factor_selection_col,
            )
        )
    if args.factor_config:
        factors.extend(
            load_lifecycle_factors(
                args.factor_config,
                statuses=tuple(args.lifecycle_statuses),
            )
        )
    return MultifactorFitConfig(
        factors=tuple(dict.fromkeys(factors)),
        models=tuple(args.models),
        target_window=args.target_window,
        label_mode=args.label_mode,
        feature_transform=args.feature_transform,
        target_transform=args.target_transform,
        train_size=args.train_size,
        test_size=args.test_size,
        window_mode=args.window_mode,
        train_years=args.train_years,
        test_years=args.test_years,
        purge_days=args.purge_days,
        signal_top_n=args.signal_top_n,
        ridge_alpha=args.ridge_alpha,
        elasticnet_alpha=args.elasticnet_alpha,
        elasticnet_l1_ratio=args.elasticnet_l1_ratio,
        lightgbm_estimators=args.lightgbm_estimators,
        lightgbm_n_jobs=args.lightgbm_n_jobs,
        qlib_valid_ratio=args.qlib_valid_ratio,
        doubleensemble_num_models=args.doubleensemble_num_models,
        mlp_hidden_sizes=tuple(args.mlp_hidden_sizes),
        mlp_epochs=args.mlp_epochs,
        mlp_batch_size=args.mlp_batch_size,
        mlp_learning_rate=args.mlp_learning_rate,
        mlp_weight_decay=args.mlp_weight_decay,
        mlp_dropout=args.mlp_dropout,
        random_state=args.random_state,
        device=args.device,
        start_date=args.start,
        end_date=args.end,
        run_backtests=args.run_backtests,
        backtest_initial_cash=args.backtest_initial_cash,
        backtest_commission_wan=args.backtest_commission_wan,
        backtest_stamp_tax_rate=args.backtest_stamp_tax_rate,
        backtest_transfer_fee_rate=args.backtest_transfer_fee_rate,
        backtest_lot_size=args.backtest_lot_size,
    )


def run_from_args(args: argparse.Namespace) -> WorkflowResult[MultifactorComparisonResult]:
    return run_multifactor_fit(
        data_dir=args.data,
        output_dir=args.output,
        config=config_from_args(args),
        metadata_path=args.metadata,
        benchmark_file=args.benchmark_file,
        style_factor_file=args.style_factor_file,
        factor_file=args.factor_file,
        factor_layout=args.factor_layout,
        factor_date_col=args.factor_date_col,
        factor_symbol_col=args.factor_symbol_col,
        factor_name_col=args.factor_name_col,
        factor_value_col=args.factor_value_col,
        factor_selection_file=args.factor_selection_file,
        context_file=args.context_file,
        context_date_col=args.context_date_col,
        context_symbol_col=args.context_symbol_col,
        force=args.force,
    )
