"""Resumable walk-forward training for out-of-sample stock scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import pickle
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from domain.artifacts import WorkflowResult
from domain.research import ModelFitResult

from models.elasticnet import ElasticNetModel
from models.lightgbm_model import LightGBMModel
from models.linear_ridge import RidgeModel
from models.mlp_torch import TorchMLPModel
from training.predict_score import scores_to_signals
from training.validation import build_walk_forward_windows, validate_feature_label_frame
import training.predict_score as predict_score_module
import training.validation as validation_module
import models.elasticnet as elasticnet_module
import models.lightgbm_model as lightgbm_module
import models.linear_ridge as ridge_module
import models.mlp_torch as mlp_module


logger = logging.getLogger(__name__)


MODEL_NAMES = ("ridge", "elasticnet", "lightgbm", "mlp")


@dataclass(frozen=True)
class WalkForwardTrainingConfig:
    feature_cols: tuple[str, ...]
    target_col: str
    model: str = "ridge"
    train_size: int = 504
    test_size: int = 21
    purge_days: int | None = None
    signal_top_n: int = 10
    ridge_alpha: float = 1.0
    elasticnet_alpha: float = 0.1
    elasticnet_l1_ratio: float = 0.5
    lightgbm_estimators: int = 200
    lightgbm_n_jobs: int = 1
    mlp_hidden_sizes: tuple[int, ...] = (64, 32)
    mlp_epochs: int = 100
    mlp_batch_size: int = 256
    mlp_learning_rate: float = 1e-3
    mlp_weight_decay: float = 0.0
    mlp_dropout: float = 0.0
    random_state: int = 42
    device: str = "auto"
    date_col: str = "date"
    symbol_col: str = "symbol"

    def __post_init__(self) -> None:
        features = tuple(str(value).strip() for value in self.feature_cols)
        if not features or any(not value for value in features):
            raise ValueError("feature_cols must contain explicit non-blank columns")
        if len(set(features)) != len(features):
            raise ValueError("feature_cols must be unique")
        reserved = {self.date_col, self.symbol_col, self.target_col}
        overlap = set(features).intersection(reserved)
        if overlap:
            raise ValueError(
                "feature_cols must not include date, symbol, or target columns: "
                + ", ".join(sorted(overlap))
            )
        object.__setattr__(self, "feature_cols", features)

        model = str(self.model).strip().lower()
        if model not in MODEL_NAMES:
            raise ValueError(f"model must be one of {', '.join(MODEL_NAMES)}")
        object.__setattr__(self, "model", model)
        if self.train_size <= 0 or self.test_size <= 0:
            raise ValueError("train_size and test_size must be positive")
        inferred = infer_target_horizon(self.target_col)
        purge_days = self.purge_days
        if purge_days is None:
            if inferred is None:
                raise ValueError(
                    "cannot infer purge_days from target_col; pass --purge-days explicitly"
                )
            purge_days = inferred
        if purge_days < 0:
            raise ValueError("purge_days must be non-negative")
        if inferred is not None and purge_days < inferred:
            raise ValueError(
                f"purge_days {purge_days} is shorter than inferred target horizon {inferred}"
            )
        object.__setattr__(self, "purge_days", int(purge_days))
        if self.signal_top_n <= 0:
            raise ValueError("signal_top_n must be positive")
        if self.ridge_alpha < 0:
            raise ValueError("ridge_alpha must be non-negative")
        if self.elasticnet_alpha < 0:
            raise ValueError("elasticnet_alpha must be non-negative")
        if not 0 <= self.elasticnet_l1_ratio <= 1:
            raise ValueError("elasticnet_l1_ratio must be in [0, 1]")
        if self.lightgbm_estimators <= 0:
            raise ValueError("lightgbm_estimators must be positive")
        if self.lightgbm_n_jobs == 0:
            raise ValueError("lightgbm_n_jobs must be -1 or a non-zero worker count")
        hidden = tuple(int(value) for value in self.mlp_hidden_sizes)
        if not hidden or any(value <= 0 for value in hidden):
            raise ValueError("mlp_hidden_sizes must contain positive integers")
        object.__setattr__(self, "mlp_hidden_sizes", hidden)
        if self.mlp_epochs <= 0 or self.mlp_batch_size <= 0 or self.mlp_learning_rate <= 0:
            raise ValueError("MLP epochs, batch size, and learning rate must be positive")
        if self.mlp_weight_decay < 0:
            raise ValueError("mlp_weight_decay must be non-negative")
        if not 0 <= self.mlp_dropout < 1:
            raise ValueError("mlp_dropout must be in [0, 1)")


def infer_target_horizon(target_col: str) -> int | None:
    name = str(target_col).strip().lower()
    transform_suffix = r"(?:_cs_(?:rank|zscore))?"
    next_open_match = re.fullmatch(
        rf"next_open_return_(\d+)d{transform_suffix}",
        name,
    )
    if next_open_match:
        return int(next_open_match.group(1)) + 1
    close_match = re.fullmatch(rf"forward_return_(\d+)d{transform_suffix}", name)
    return int(close_match.group(1)) if close_match else None


def run_walk_forward_training(
    *,
    features_path: str | Path,
    labels_path: str | Path,
    output_dir: str | Path,
    config: WalkForwardTrainingConfig,
    force: bool = False,
) -> WorkflowResult[ModelFitResult]:
    """Train one model per window and persist only out-of-sample predictions."""

    features_file = Path(features_path)
    labels_file = Path(labels_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    logger.info(
        "Preparing walk-forward training: model=%s, features=%d, target=%s",
        config.model,
        len(config.feature_cols),
        config.target_col,
    )
    frame, input_audit = _load_training_frame(features_file, labels_file, config)
    windows = build_walk_forward_windows(
        frame[config.date_col],
        train_size=config.train_size,
        test_size=config.test_size,
        purge_size=int(config.purge_days),
    )
    if not windows:
        unique_dates = frame[config.date_col].nunique()
        required_dates = config.train_size + int(config.purge_days) + config.test_size
        raise ValueError(
            f"not enough trading dates for one walk-forward window: "
            f"available={unique_dates}, required={required_dates}"
        )

    run_signature = _run_signature(features_file, labels_file, config)
    prediction_frames: list[pd.DataFrame] = []
    window_rows: list[dict[str, object]] = []
    window_metric_rows: list[dict[str, object]] = []
    windows_root = destination / "windows"
    windows_root.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Model %s has %d walk-forward windows (%d training rows, %d trading dates)",
        config.model,
        len(windows),
        input_audit["training_rows"],
        input_audit["trading_dates"],
    )
    for position, window in enumerate(windows, start=1):
        window_id = position - 1
        window_started = time.perf_counter()
        logger.info(
            "[%s %d/%d] train=%s..%s, test=%s..%s",
            config.model,
            position,
            len(windows),
            window.train_start.strftime("%Y-%m-%d"),
            window.train_end.strftime("%Y-%m-%d"),
            window.test_start.strftime("%Y-%m-%d"),
            window.test_end.strftime("%Y-%m-%d"),
        )
        window_dir = windows_root / f"window_{window_id:03d}"
        window_dir.mkdir(parents=True, exist_ok=True)
        predictions_path = window_dir / "predictions.csv"
        metrics_path = window_dir / "metrics.json"
        manifest_path = window_dir / "manifest.json"
        window_signature = _window_signature(run_signature, window_id, window)
        reused = False
        if not force and _window_is_reusable(
            manifest_path,
            predictions_path,
            metrics_path,
            window_signature,
        ):
            predictions = pd.read_csv(predictions_path, dtype={config.symbol_col: str})
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            model_artifact = json.loads(manifest_path.read_text(encoding="utf-8")).get(
                "model_artifact"
            )
            reused = True
        else:
            train = frame.loc[
                frame[config.date_col].between(window.train_start, window.train_end)
            ].copy()
            test = frame.loc[
                frame[config.date_col].between(window.test_start, window.test_end)
            ].copy()
            if train.empty or test.empty:
                raise ValueError(f"window {window_id} has empty train or test rows")
            model = _build_model(config)
            model.fit(train[list(config.feature_cols)], train[config.target_col])
            scores = model.predict(test[list(config.feature_cols)])
            predictions = test[
                [config.date_col, config.symbol_col, config.target_col]
            ].copy()
            predictions["score"] = pd.to_numeric(scores, errors="coerce").to_numpy()
            predictions["window_id"] = window_id
            predictions["model"] = config.model
            predictions["train_start"] = window.train_start.strftime("%Y-%m-%d")
            predictions["train_end"] = window.train_end.strftime("%Y-%m-%d")
            predictions["test_start"] = window.test_start.strftime("%Y-%m-%d")
            predictions["test_end"] = window.test_end.strftime("%Y-%m-%d")
            predictions["purge_days"] = int(config.purge_days)
            predictions[config.date_col] = pd.to_datetime(
                predictions[config.date_col]
            ).dt.strftime("%Y-%m-%d")
            predictions[config.symbol_col] = (
                predictions[config.symbol_col].astype(str).str.zfill(6)
            )
            if not np.isfinite(predictions["score"]).all():
                raise ValueError(f"window {window_id} produced non-finite predictions")
            metrics = _prediction_metrics(
                predictions,
                target_col=config.target_col,
                date_col=config.date_col,
            )
            model_artifact = _save_model(model, window_dir, config.model)
            _atomic_write_csv(predictions_path, predictions)
            _atomic_write_json(metrics_path, metrics)
            _atomic_write_json(
                manifest_path,
                {
                    "window_signature": window_signature,
                    "model_artifact": model_artifact,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                },
            )

        prediction_frames.append(predictions)
        window_row = {
            "window_id": window_id,
            "train_start": window.train_start.strftime("%Y-%m-%d"),
            "train_end": window.train_end.strftime("%Y-%m-%d"),
            "purge_start": window.purge_start.strftime("%Y-%m-%d")
            if window.purge_start is not None
            else None,
            "purge_end": window.purge_end.strftime("%Y-%m-%d")
            if window.purge_end is not None
            else None,
            "test_start": window.test_start.strftime("%Y-%m-%d"),
            "test_end": window.test_end.strftime("%Y-%m-%d"),
            "prediction_count": len(predictions),
            "reused": reused,
            "model_artifact": model_artifact,
        }
        window_rows.append(window_row)
        window_metric_rows.append({"window_id": window_id, **metrics})
        logger.info(
            "[%s %d/%d] %s in %.2fs: predictions=%d, rank_ic=%s",
            config.model,
            position,
            len(windows),
            "reused" if reused else "trained",
            time.perf_counter() - window_started,
            len(predictions),
            metrics.get("rank_ic_mean"),
        )

    predictions = pd.concat(prediction_frames, ignore_index=True).sort_values(
        [config.date_col, config.symbol_col],
        kind="mergesort",
    )
    if predictions.duplicated([config.date_col, config.symbol_col]).any():
        raise ValueError("walk-forward windows produced overlapping date/symbol predictions")
    signals = scores_to_signals(
        predictions,
        source=config.model,
        date_col=config.date_col,
        symbol_col=config.symbol_col,
        score_col="score",
        top_n=config.signal_top_n,
    )
    overall_metrics = _prediction_metrics(
        predictions,
        target_col=config.target_col,
        date_col=config.date_col,
    )
    summary = {
        "model": config.model,
        "feature_cols": list(config.feature_cols),
        "target_col": config.target_col,
        "train_size": config.train_size,
        "test_size": config.test_size,
        "purge_days": int(config.purge_days),
        "window_count": len(windows),
        "reused_window_count": sum(bool(row["reused"]) for row in window_rows),
        "prediction_count": len(predictions),
        "signal_count": len(signals),
        "signal_top_n": config.signal_top_n,
        "input_audit": input_audit,
        "out_of_sample_metrics": overall_metrics,
    }

    predictions_path = destination / "predictions.csv"
    signals_path = destination / "signals.csv"
    signals_json_path = destination / "signals.json"
    windows_path = destination / "windows.csv"
    metrics_path = destination / "metrics.csv"
    summary_path = destination / "summary.json"
    manifest_path = destination / "manifest.json"
    signal_csv = signals.copy()
    signal_csv["metadata"] = signal_csv["metadata"].map(
        lambda value: json.dumps(value, ensure_ascii=False, sort_keys=True)
    )
    _atomic_write_csv(predictions_path, predictions)
    _atomic_write_csv(signals_path, signal_csv)
    _atomic_write_json(signals_json_path, {"signals": signals.to_dict("records")})
    _atomic_write_csv(windows_path, pd.DataFrame(window_rows))
    _atomic_write_csv(metrics_path, pd.DataFrame(window_metric_rows))
    _atomic_write_json(summary_path, summary)
    _atomic_write_json(
        manifest_path,
        {
            "run_signature": run_signature,
            "config": asdict(config),
            "inputs": {
                "features": str(features_file.resolve()),
                "labels": str(labels_file.resolve()),
            },
            "outputs": {
                "predictions": predictions_path.name,
                "signals": signals_path.name,
                "signals_json": signals_json_path.name,
                "windows": windows_path.name,
                "metrics": metrics_path.name,
                "summary": summary_path.name,
                "window_root": windows_root.name,
            },
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    result = ModelFitResult(
        model=config.model,
        feature_columns=tuple(config.feature_cols),
        target_column=config.target_col,
        window_count=len(windows),
        prediction_count=len(predictions),
        signal_count=len(signals),
        out_of_sample_metrics=overall_metrics,
    )
    logger.info(
        "Walk-forward model %s complete in %.2fs: windows=%d, reused=%d, predictions=%d, signals=%d, output=%s",
        config.model,
        time.perf_counter() - started,
        len(windows),
        summary["reused_window_count"],
        len(predictions),
        len(signals),
        destination.resolve(),
    )
    return WorkflowResult.from_mapping(
        {
            "result": result,
            "predictions_path": predictions_path,
            "signals_path": signals_path,
            "signals_json_path": signals_json_path,
            "windows_path": windows_path,
            "metrics_path": metrics_path,
            "summary_path": summary_path,
            "manifest_path": manifest_path,
            "windows_root": windows_root,
        }
    )


def _load_training_frame(
    features_path: Path,
    labels_path: Path,
    config: WalkForwardTrainingConfig,
) -> tuple[pd.DataFrame, dict[str, int]]:
    if not features_path.exists():
        raise FileNotFoundError(f"features file not found: {features_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"labels file not found: {labels_path}")
    features = pd.read_csv(features_path, dtype={config.symbol_col: str})
    labels = pd.read_csv(labels_path, dtype={config.symbol_col: str})
    feature_required = {config.date_col, config.symbol_col, *config.feature_cols}
    label_required = {config.date_col, config.symbol_col, config.target_col}
    missing_features = sorted(feature_required.difference(features.columns))
    missing_labels = sorted(label_required.difference(labels.columns))
    if missing_features:
        raise ValueError(f"missing feature columns: {missing_features}")
    if missing_labels:
        raise ValueError(f"missing label columns: {missing_labels}")
    features = features[[config.date_col, config.symbol_col, *config.feature_cols]].copy()
    labels = labels[[config.date_col, config.symbol_col, config.target_col]].copy()
    for source_name, source in (("features", features), ("labels", labels)):
        source[config.date_col] = pd.to_datetime(source[config.date_col])
        source[config.symbol_col] = source[config.symbol_col].astype(str).str.zfill(6)
        if source.duplicated([config.date_col, config.symbol_col]).any():
            raise ValueError(f"duplicate date/symbol rows in {source_name}")

    merged = features.merge(
        labels,
        on=[config.date_col, config.symbol_col],
        how="inner",
        validate="one_to_one",
    )
    matched_rows = len(merged)
    numeric_columns = [*config.feature_cols, config.target_col]
    for column in numeric_columns:
        merged[column] = pd.to_numeric(merged[column], errors="coerce")
    missing_mask = merged[numeric_columns].isna().any(axis=1)
    dropped_missing = int(missing_mask.sum())
    merged = merged.loc[~missing_mask].copy()
    if merged.empty:
        raise ValueError("no complete feature/label rows remain after alignment")
    frame = validate_feature_label_frame(
        merged,
        feature_cols=config.feature_cols,
        target_col=config.target_col,
        date_col=config.date_col,
        symbol_col=config.symbol_col,
    )
    audit = {
        "feature_rows": len(features),
        "label_rows": len(labels),
        "matched_rows_before_missing_filter": matched_rows,
        "unmatched_feature_rows": len(features) - matched_rows,
        "unmatched_label_rows": len(labels) - matched_rows,
        "dropped_missing_rows": dropped_missing,
        "training_rows": len(frame),
        "trading_dates": int(frame[config.date_col].nunique()),
    }
    return frame, audit


def _build_model(config: WalkForwardTrainingConfig):
    if config.model == "ridge":
        return RidgeModel(alpha=config.ridge_alpha)
    if config.model == "elasticnet":
        return ElasticNetModel(
            alpha=config.elasticnet_alpha,
            l1_ratio=config.elasticnet_l1_ratio,
        )
    if config.model == "lightgbm":
        return LightGBMModel(
            n_estimators=config.lightgbm_estimators,
            n_jobs=config.lightgbm_n_jobs,
            random_state=config.random_state,
        )
    if config.model == "mlp":
        return TorchMLPModel(
            hidden_sizes=config.mlp_hidden_sizes,
            learning_rate=config.mlp_learning_rate,
            epochs=config.mlp_epochs,
            batch_size=config.mlp_batch_size,
            weight_decay=config.mlp_weight_decay,
            dropout=config.mlp_dropout,
            random_state=config.random_state,
            device=config.device,
        )
    raise ValueError(f"unsupported model: {config.model}")


def _prediction_metrics(
    predictions: pd.DataFrame,
    *,
    target_col: str,
    date_col: str,
) -> dict[str, float | int | None]:
    actual = pd.to_numeric(predictions[target_col], errors="coerce")
    score = pd.to_numeric(predictions["score"], errors="coerce")
    residual = score - actual
    pearson = (
        actual.corr(score, method="pearson")
        if len(predictions) >= 2 and actual.nunique() > 1 and score.nunique() > 1
        else np.nan
    )
    daily_rank_ic = []
    for _, daily in predictions.groupby(date_col, sort=True):
        if (
            len(daily) < 2
            or daily[target_col].nunique() <= 1
            or daily["score"].nunique() <= 1
        ):
            continue
        value = daily[target_col].corr(daily["score"], method="spearman")
        if pd.notna(value):
            daily_rank_ic.append(float(value))
    return {
        "count": len(predictions),
        "mse": float((residual**2).mean()),
        "mae": float(residual.abs().mean()),
        "pearson": float(pearson) if pd.notna(pearson) else None,
        "rank_ic_mean": float(np.mean(daily_rank_ic)) if daily_rank_ic else None,
        "rank_ic_date_count": len(daily_rank_ic),
    }


def _save_model(model: object, window_dir: Path, model_name: str) -> str:
    if model_name == "mlp":
        path = window_dir / "model.pt"
        model.save(path)
        return path.name
    path = window_dir / "model.pkl"
    temp = path.with_name(f".{path.name}.tmp")
    with temp.open("wb") as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temp, path)
    return path.name


def _run_signature(
    features_path: Path,
    labels_path: Path,
    config: WalkForwardTrainingConfig,
) -> str:
    payload = {
        "features_sha256": _file_sha256(features_path),
        "labels_sha256": _file_sha256(labels_path),
        "implementation_sha256": {
            str(path.name): _file_sha256(path)
            for path in _implementation_files()
        },
        "config": asdict(config),
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _window_signature(run_signature: str, window_id: int, window: object) -> str:
    payload = {
        "run_signature": run_signature,
        "window_id": window_id,
        "window": {
            key: value.strftime("%Y-%m-%d") if value is not None else None
            for key, value in asdict(window).items()
        },
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _window_is_reusable(
    manifest_path: Path,
    predictions_path: Path,
    metrics_path: Path,
    expected_signature: str,
) -> bool:
    if not (manifest_path.exists() and predictions_path.exists() and metrics_path.exists()):
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return manifest.get("window_signature") == expected_signature


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _implementation_files() -> tuple[Path, ...]:
    modules = (
        validation_module,
        predict_score_module,
        ridge_module,
        elasticnet_module,
        lightgbm_module,
        mlp_module,
    )
    paths = [Path(__file__).resolve()]
    paths.extend(Path(module.__file__).resolve() for module in modules)
    return tuple(paths)


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
    parser.add_argument("--features", required=True, help="Long-format feature CSV path")
    parser.add_argument("--labels", required=True, help="Long-format label CSV path")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--feature-cols", nargs="+", required=True)
    parser.add_argument("--target-col", required=True)
    parser.add_argument("--model", default="ridge", choices=MODEL_NAMES)
    parser.add_argument("--train-size", type=int, default=504, help="Training trading dates")
    parser.add_argument("--test-size", type=int, default=21, help="Test trading dates per window")
    parser.add_argument(
        "--purge-days",
        type=int,
        default=None,
        help="Gap in trading dates; inferred from forward_return_Nd when omitted",
    )
    parser.add_argument("--signal-top-n", type=int, default=10)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--elasticnet-alpha", type=float, default=0.1)
    parser.add_argument("--elasticnet-l1-ratio", type=float, default=0.5)
    parser.add_argument("--lightgbm-estimators", type=int, default=200)
    parser.add_argument(
        "--lightgbm-n-jobs",
        type=int,
        default=1,
        help="LightGBM worker count; default 1 for deterministic local runs",
    )
    parser.add_argument("--mlp-hidden-sizes", nargs="+", type=int, default=[64, 32])
    parser.add_argument("--mlp-epochs", type=int, default=100)
    parser.add_argument("--mlp-batch-size", type=int, default=256)
    parser.add_argument("--mlp-learning-rate", type=float, default=1e-3)
    parser.add_argument("--mlp-weight-decay", type=float, default=0.0)
    parser.add_argument("--mlp-dropout", type=float, default=0.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--device", default="auto", help="MLP device: auto, cpu, mps, cuda")
    parser.add_argument("--force", action="store_true", help="Recompute completed windows")
    return parser


def config_from_args(args: argparse.Namespace) -> WalkForwardTrainingConfig:
    return WalkForwardTrainingConfig(
        feature_cols=tuple(args.feature_cols),
        target_col=args.target_col,
        model=args.model,
        train_size=args.train_size,
        test_size=args.test_size,
        purge_days=args.purge_days,
        signal_top_n=args.signal_top_n,
        ridge_alpha=args.ridge_alpha,
        elasticnet_alpha=args.elasticnet_alpha,
        elasticnet_l1_ratio=args.elasticnet_l1_ratio,
        lightgbm_estimators=args.lightgbm_estimators,
        lightgbm_n_jobs=args.lightgbm_n_jobs,
        mlp_hidden_sizes=tuple(args.mlp_hidden_sizes),
        mlp_epochs=args.mlp_epochs,
        mlp_batch_size=args.mlp_batch_size,
        mlp_learning_rate=args.mlp_learning_rate,
        mlp_weight_decay=args.mlp_weight_decay,
        mlp_dropout=args.mlp_dropout,
        random_state=args.random_state,
        device=args.device,
    )


def run_from_args(args: argparse.Namespace) -> WorkflowResult[ModelFitResult]:
    return run_walk_forward_training(
        features_path=args.features,
        labels_path=args.labels,
        output_dir=args.output,
        config=config_from_args(args),
        force=args.force,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train resumable walk-forward stock scores")
    return add_arguments(parser)


def main() -> None:
    outputs = run_from_args(build_parser().parse_args())
    print("Walk-forward training complete")
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
