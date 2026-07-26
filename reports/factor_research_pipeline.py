"""Configuration-driven end-to-end factor research orchestration.

The workflow deliberately composes the existing batch, correlation, ML, signal,
and portfolio boundaries.  It does not implement factor formulas or a second
backtest engine.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from domain.artifacts import WorkflowResult
from domain.research import FactorResearchPipelineResult


LOGGER = logging.getLogger(__name__)
SCHEMA_VERSION = 1
FACTOR_FAMILIES = ("alpha101", "gtja191", "external")

_ROOT_KEYS = {
    "version",
    "factor_library",
    "evaluation",
    "correlation",
    "machine_learning",
    "execution",
}
_LIBRARY_KEYS = {
    "family",
    "data",
    "metadata",
    "factor_file",
    "factor_layout",
    "factor_config",
    "factors",
    "exclude",
    "ignore_factor_config",
    "date_col",
    "symbol_col",
    "factor_name_col",
    "factor_value_col",
    "context_file",
    "context_date_col",
    "context_symbol_col",
    "benchmark_file",
    "style_factor_file",
    "industry_col",
    "market_cap_col",
}
_EVALUATION_KEYS = {
    "windows",
    "groups",
    "top_counts",
    "start_date",
    "end_date",
    "winsorize",
    "zscore",
    "min_periods",
    "min_listing_days",
    "liquidity_lookback_days",
    "min_liquidity",
    "commission_rate",
    "slippage_rate",
    "stamp_tax_rate",
    "market_cap_groups",
    "market_regime_col",
    "market_regime_lookback_days",
    "market_regime_min_periods",
    "bull_return_threshold",
    "bear_return_threshold",
    "oos_start_date",
    "oos_fraction",
    "fail_fast",
    "max_symbols",
}
_CORRELATION_KEYS = {
    "factor_lag_days",
    "min_observations",
    "min_dates",
    "high_correlation_threshold",
    "priority_factor_col",
    "priority_score_col",
    "priority_window",
    "eligibility_col",
}
_ML_KEYS = {
    "enabled",
    "models",
    "target_window",
    "label_mode",
    "feature_transform",
    "target_transform",
    "window_mode",
    "train_years",
    "test_years",
    "purge_days",
    "signal_top_n",
    "ridge_alpha",
    "elasticnet_alpha",
    "elasticnet_l1_ratio",
    "lightgbm_estimators",
    "lightgbm_n_jobs",
    "qlib_valid_ratio",
    "doubleensemble_num_models",
    "mlp_hidden_sizes",
    "mlp_epochs",
    "mlp_batch_size",
    "mlp_learning_rate",
    "mlp_weight_decay",
    "mlp_dropout",
    "random_state",
    "device",
    "start",
    "end",
    "run_backtests",
    "backtest_initial_cash",
    "backtest_commission_wan",
    "backtest_stamp_tax_rate",
    "backtest_transfer_fee_rate",
    "backtest_lot_size",
}
_EXECUTION_KEYS = {"output", "force", "require_classification"}


@dataclass(frozen=True)
class FactorResearchRunConfig:
    """Validated configuration for the one-command factor research pipeline."""

    source_path: Path
    factor_library: dict[str, Any]
    evaluation: dict[str, Any]
    correlation: dict[str, Any]
    machine_learning: dict[str, Any]
    output_dir: Path
    force: bool = False
    require_classification: bool = True

    @property
    def family(self) -> str:
        return str(self.factor_library["family"])


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--config",
        default="config/factor_research_run_all.yaml",
        help="YAML describing the selected factor library and all research stages",
    )
    parser.add_argument("--family", choices=FACTOR_FAMILIES, default=None)
    parser.add_argument("--factor-file", default=None, help="Override an external factor CSV")
    parser.add_argument("--factor-config", default=None, help="Override lifecycle/category YAML")
    parser.add_argument("--factors", nargs="+", default=None, help="Override selected factor names")
    parser.add_argument("--output", default=None, help="Override the workflow output directory")
    parser.add_argument("--force", action="store_true", help="Recompute resumable stages")
    parser.add_argument("--skip-ml", action="store_true", help="Run research and deduplication only")
    return parser


def load_factor_research_config(
    path: str | Path,
    *,
    family: str | None = None,
    factor_file: str | None = None,
    factor_config: str | None = None,
    factors: Sequence[str] | None = None,
    output: str | None = None,
    force: bool = False,
    skip_ml: bool = False,
) -> FactorResearchRunConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"factor research run-all config not found: {config_path}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ValueError("factor research run-all config must be a mapping")
    _reject_unknown(payload, _ROOT_KEYS, "root")
    version = int(payload.get("version", SCHEMA_VERSION))
    if version != SCHEMA_VERSION:
        raise ValueError(f"unsupported factor research config version: {version}")

    library = _mapping_section(payload, "factor_library", _LIBRARY_KEYS, required=True)
    evaluation = _mapping_section(payload, "evaluation", _EVALUATION_KEYS)
    correlation = _mapping_section(payload, "correlation", _CORRELATION_KEYS)
    machine_learning = _mapping_section(payload, "machine_learning", _ML_KEYS)
    execution = _mapping_section(payload, "execution", _EXECUTION_KEYS)

    if factor_file is not None:
        library["factor_file"] = factor_file
        library["family"] = family or "external"
    if family is not None:
        library["family"] = family
    if factor_config is not None:
        library["factor_config"] = factor_config
    if factors is not None:
        library["factors"] = list(factors)

    selected_family = str(library.get("family", "")).strip().lower()
    if selected_family not in FACTOR_FAMILIES:
        raise ValueError(f"unsupported factor family: {selected_family or '<missing>'}")
    library["family"] = selected_family
    if selected_family == "external" and not library.get("factor_file"):
        raise ValueError("external factor research requires factor_library.factor_file")
    if selected_family != "external" and library.get("factor_file"):
        raise ValueError("factor_file can only be used with family=external")

    machine_learning.setdefault("enabled", True)
    if skip_ml:
        machine_learning["enabled"] = False
    _validate_long_only_ml_contract(machine_learning)
    target_window = int(machine_learning.get("target_window", 20))
    priority_window = int(correlation.get("priority_window", target_window))
    if priority_window != target_window:
        raise ValueError(
            "correlation.priority_window must equal machine_learning.target_window "
            "so ML selection uses the same holding horizon"
        )
    evaluation_windows = tuple(int(value) for value in evaluation.get("windows", (1, 5, 10, 20)))
    if target_window not in evaluation_windows:
        raise ValueError(
            "machine_learning.target_window must be included in evaluation.windows"
        )

    resolved_output = Path(output or execution.get("output", "factor_report/factor_run_all"))
    return FactorResearchRunConfig(
        source_path=config_path,
        factor_library=library,
        evaluation=evaluation,
        correlation=correlation,
        machine_learning=machine_learning,
        output_dir=resolved_output,
        force=bool(force or execution.get("force", False)),
        require_classification=bool(execution.get("require_classification", True)),
    )


def run_factor_research_pipeline(
    config: FactorResearchRunConfig,
    *,
    batch_runner: Callable[[argparse.Namespace], int] | None = None,
    correlation_runner: Callable[[argparse.Namespace], int] | None = None,
    ml_runner: Callable[[argparse.Namespace], WorkflowResult[Any]] | None = None,
) -> WorkflowResult[FactorResearchPipelineResult]:
    """Run batch evaluation, correlation deduplication, then long-only ML."""

    from scripts.factor_correlation import build_parser as build_correlation_parser
    from scripts.test_factor_batch import build_parser as build_batch_parser
    from training.multifactor import add_arguments as add_multifactor_arguments

    destination = config.output_dir
    batch_dir = destination / "batch"
    correlation_dir = destination / "correlation"
    ml_dir = destination / "ml"
    manifest_path = destination / "manifest.json"
    summary_path = destination / "summary.json"
    destination.mkdir(parents=True, exist_ok=True)

    batch_runner = batch_runner or _default_batch_runner
    correlation_runner = correlation_runner or _default_correlation_runner
    ml_runner = ml_runner or _default_ml_runner
    manifest = _initial_manifest(config, batch_dir, correlation_dir, ml_dir)
    _atomic_write_json(manifest_path, manifest)

    warnings: list[str] = []
    active_stage = "preflight"
    try:
        LOGGER.info("[1/3] Validating selected factor library and classifications")
        _validate_external_classification(config, destination)
        manifest["stages"]["preflight"] = {"status": "complete"}
        _atomic_write_json(manifest_path, manifest)

        active_stage = "batch"
        LOGGER.info("[1/3] Running full factor evaluation: %s", config.family)
        batch_args = _batch_args(config, build_batch_parser(), batch_dir)
        batch_code = int(batch_runner(batch_args) or 0)
        if batch_code != 0:
            raise RuntimeError(f"factor batch returned non-zero exit code: {batch_code}")
        batch_status = batch_dir / "batch_status.csv"
        leaderboard_path = batch_dir / "leaderboard.csv"
        _validate_batch_outputs(batch_status, leaderboard_path)
        evaluated_factors = _factor_names(leaderboard_path)
        if len(evaluated_factors) < 2:
            raise ValueError("factor correlation requires at least two successfully evaluated factors")
        manifest["stages"]["batch"] = {
            "status": "complete",
            "factor_count": len(evaluated_factors),
            "leaderboard": str(leaderboard_path.resolve()),
        }
        _atomic_write_json(manifest_path, manifest)

        active_stage = "correlation"
        LOGGER.info("[2/3] Calculating pairwise correlations and |rho| deduplication")
        correlation_args = _correlation_args(
            config,
            build_correlation_parser(),
            correlation_dir,
            leaderboard_path,
            evaluated_factors,
        )
        correlation_code = int(correlation_runner(correlation_args) or 0)
        if correlation_code != 0:
            raise RuntimeError(
                f"factor correlation returned non-zero exit code: {correlation_code}"
            )
        deduplicated_path = correlation_dir / "deduplicated_factors.csv"
        candidates_path = correlation_dir / "ml_candidate_factors.csv"
        deduplicated_factors = _factor_names(deduplicated_path)
        ml_candidates = _factor_names(candidates_path) if candidates_path.exists() else ()
        manifest["stages"]["correlation"] = {
            "status": "complete",
            "deduplicated_factor_count": len(deduplicated_factors),
            "ml_candidate_factor_count": len(ml_candidates),
            "threshold": float(correlation_args.high_correlation_threshold),
        }
        _atomic_write_json(manifest_path, manifest)

        ml_status = "disabled"
        models: tuple[str, ...] = ()
        if bool(config.machine_learning.get("enabled", True)) and ml_candidates:
            active_stage = "machine_learning"
            LOGGER.info(
                "[3/3] Running 3-calendar-year -> 1-calendar-year ML and long-only backtests"
            )
            ml_parser = argparse.ArgumentParser(add_help=False)
            add_multifactor_arguments(ml_parser)
            ml_args = _ml_args(config, ml_parser, ml_dir, candidates_path)
            ml_outputs = ml_runner(ml_args)
            if not (ml_dir / "leaderboard.csv").exists():
                raise FileNotFoundError("ML stage did not write leaderboard.csv")
            if not (ml_dir / "profitable_models.csv").exists():
                raise FileNotFoundError("ML stage did not write profitable_models.csv")
            models = tuple(str(value) for value in ml_args.models)
            ml_status = "complete"
            manifest["stages"]["machine_learning"] = {
                "status": "complete",
                "models": list(models),
                "leaderboard": str((ml_dir / "leaderboard.csv").resolve()),
                "gross_and_net_long_only_backtests": True,
            }
            if isinstance(ml_outputs, WorkflowResult):
                warnings.extend(ml_outputs.warnings)
        elif not bool(config.machine_learning.get("enabled", True)):
            manifest["stages"]["machine_learning"] = {
                "status": "skipped",
                "reason": "disabled by configuration or --skip-ml",
            }
        else:
            ml_status = "skipped_no_profitable_candidates"
            message = (
                "ML skipped because no deduplicated factor was profitable after costs "
                "at the configured priority horizon"
            )
            warnings.append(message)
            LOGGER.warning(message)
            manifest["stages"]["machine_learning"] = {
                "status": "skipped",
                "reason": "no after-cost profitable deduplicated factors",
            }

        result = FactorResearchPipelineResult(
            factor_family=config.family,
            evaluated_factors=evaluated_factors,
            deduplicated_factors=deduplicated_factors,
            ml_candidate_factors=ml_candidates,
            models=models,
            ml_status=ml_status,
        )
        summary = _build_summary(config, result, batch_dir, correlation_dir, ml_dir)
        _atomic_write_json(summary_path, summary)
        manifest["status"] = "complete"
        manifest["finished_at"] = _utc_now()
        manifest["summary"] = summary
        manifest["warnings"] = warnings
        _atomic_write_json(manifest_path, manifest)

        LOGGER.info("Factor run-all complete: %s", manifest_path.resolve())
        workflow = WorkflowResult.from_mapping(
            {
                "result": result,
                "output_dir": destination,
                "manifest_path": manifest_path,
                "summary_path": summary_path,
                "batch_leaderboard_path": leaderboard_path,
                "long_only_profitability_path": batch_dir / "long_only_profitability.csv",
                "profitable_long_only_path": batch_dir / "profitable_long_only.csv",
                "correlation_pairs_path": correlation_dir / "correlation_pairs.csv",
                "deduplicated_factors_path": deduplicated_path,
                "ml_candidate_factors_path": candidates_path,
            }
        )
        if ml_status == "complete":
            workflow["ml_leaderboard_path"] = ml_dir / "leaderboard.csv"
            workflow["profitable_models_path"] = ml_dir / "profitable_models.csv"
        workflow.warnings.extend(warnings)
        return workflow
    except BaseException as exc:
        manifest["stages"][active_stage] = {
            "status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
        manifest["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        manifest["finished_at"] = _utc_now()
        manifest["error"] = {"type": type(exc).__name__, "message": str(exc)}
        _atomic_write_json(manifest_path, manifest)
        raise


def run_from_args(args: argparse.Namespace) -> WorkflowResult[FactorResearchPipelineResult]:
    config = load_factor_research_config(
        args.config,
        family=args.family,
        factor_file=args.factor_file,
        factor_config=args.factor_config,
        factors=args.factors,
        output=args.output,
        force=args.force,
        skip_ml=args.skip_ml,
    )
    return run_factor_research_pipeline(config)


def _default_batch_runner(args: argparse.Namespace) -> int:
    from scripts.test_factor_batch import run_from_args

    return run_from_args(args)


def _default_correlation_runner(args: argparse.Namespace) -> int:
    from scripts.factor_correlation import run_from_args

    return run_from_args(args)


def _default_ml_runner(args: argparse.Namespace) -> WorkflowResult[Any]:
    from training.multifactor import run_from_args

    return run_from_args(args)


def _batch_args(
    config: FactorResearchRunConfig,
    parser: argparse.ArgumentParser,
    output_dir: Path,
) -> argparse.Namespace:
    args = parser.parse_args([])
    _assign(args, config.factor_library, "factor_library")
    _assign(args, config.evaluation, "evaluation")
    args.output = str(output_dir)
    args.force = config.force
    args.list_factors = False
    args.list_factor_status = False
    args.no_progress = True
    return args


def _correlation_args(
    config: FactorResearchRunConfig,
    parser: argparse.ArgumentParser,
    output_dir: Path,
    leaderboard_path: Path,
    evaluated_factors: Sequence[str],
) -> argparse.Namespace:
    args = parser.parse_args([])
    shared = {
        key: value
        for key, value in config.factor_library.items()
        if key in vars(args)
    }
    _assign(args, shared, "factor_library")
    _assign(args, config.correlation, "correlation")
    args.factors = list(evaluated_factors)
    args.exclude = []
    args.start_date = config.evaluation.get("start_date")
    args.end_date = config.evaluation.get("end_date")
    args.priority_file = str(leaderboard_path)
    args.priority_factor_col = config.correlation.get("priority_factor_col", "factor")
    args.priority_score_col = config.correlation.get(
        "priority_score_col", "preferred_net_sharpe"
    )
    args.priority_window = int(
        config.correlation.get(
            "priority_window", config.machine_learning.get("target_window", 20)
        )
    )
    args.eligibility_col = config.correlation.get(
        "eligibility_col", "preferred_profitable_after_cost"
    )
    args.output = str(output_dir)
    return args


def _ml_args(
    config: FactorResearchRunConfig,
    parser: argparse.ArgumentParser,
    output_dir: Path,
    candidates_path: Path,
) -> argparse.Namespace:
    args = parser.parse_args([])
    _assign(
        args,
        {key: value for key, value in config.machine_learning.items() if key != "enabled"},
        "machine_learning",
    )
    library = config.factor_library
    args.data = library.get("data", args.data)
    args.metadata = library.get("metadata", args.metadata)
    args.benchmark_file = library.get("benchmark_file")
    args.style_factor_file = library.get("style_factor_file")
    args.factor_file = library.get("factor_file")
    args.factor_layout = library.get("factor_layout", args.factor_layout)
    args.factor_date_col = library.get("date_col", args.factor_date_col)
    args.factor_symbol_col = library.get("symbol_col", args.factor_symbol_col)
    args.factor_name_col = library.get("factor_name_col", args.factor_name_col)
    args.factor_value_col = library.get("factor_value_col", args.factor_value_col)
    args.context_file = library.get("context_file")
    args.context_date_col = library.get("context_date_col", args.context_date_col)
    args.context_symbol_col = library.get("context_symbol_col", args.context_symbol_col)
    args.factors = []
    args.factor_config = None
    args.factor_selection_file = str(candidates_path)
    args.factor_selection_col = "factor"
    args.window_mode = "calendar-years"
    args.train_years = 3
    args.test_years = 1
    args.run_backtests = True
    if "start" not in config.machine_learning:
        args.start = config.evaluation.get("start_date")
    if "end" not in config.machine_learning:
        args.end = config.evaluation.get("end_date")
    args.force = config.force
    args.output = str(output_dir)
    return args


def _validate_long_only_ml_contract(values: Mapping[str, Any]) -> None:
    if not bool(values.get("enabled", True)):
        return
    required = {
        "label_mode": "next_open",
        "window_mode": "calendar-years",
        "train_years": 3,
        "test_years": 1,
        "run_backtests": True,
    }
    for name, expected in required.items():
        actual = values.get(name, expected)
        if actual != expected:
            raise ValueError(
                f"machine_learning.{name} must be {expected!r} in factor-run-all; "
                "use fit-multifactor directly for a different experiment"
            )


def _validate_external_classification(
    config: FactorResearchRunConfig,
    destination: Path,
) -> None:
    if config.family != "external" or not config.require_classification:
        return
    from factors.external import load_external_factor_file, normalize_external_factor_name

    library = config.factor_library
    requested = None
    raw_requested = list(library.get("factors", ["all"]))
    if not any(str(value).strip().lower() == "all" for value in raw_requested):
        requested = tuple(str(value) for value in raw_requested)
    external = load_external_factor_file(
        library["factor_file"],
        factors=requested,
        date_col=library.get("date_col", "date"),
        symbol_col=library.get("symbol_col", "symbol"),
        layout=library.get("factor_layout", "auto"),
        factor_name_col=library.get("factor_name_col", "factor"),
        factor_value_col=library.get("factor_value_col", "factor_value"),
    )
    excluded = {
        normalize_external_factor_name(value) for value in library.get("exclude", [])
    }
    factors = tuple(name for name in external.factors if name not in excluded)
    config_path_value = library.get("factor_config")
    payload: dict[str, Any] = {}
    if config_path_value and Path(config_path_value).exists():
        loaded = yaml.safe_load(Path(config_path_value).read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, Mapping):
            raise ValueError("external factor lifecycle/category config must be a mapping")
        payload = dict(loaded)
    categories = payload.get("categories", {}) or {}
    entries = payload.get("factors", {}) or {}
    missing: list[str] = []
    for factor in factors:
        entry = entries.get(factor, {}) if isinstance(entries, Mapping) else {}
        category = categories.get(factor) if isinstance(categories, Mapping) else None
        if category is None and isinstance(entry, Mapping):
            category = entry.get("category")
        if not str(category or "").strip() or str(category).strip().lower() == "unclassified":
            missing.append(factor)
    if not missing:
        return

    template_path = destination / "factor_classification_template.yaml"
    template = {
        "default_status": str(payload.get("default_status", "active")),
        "factors": {
            factor: (
                entries.get(factor, "active")
                if isinstance(entries, Mapping)
                else "active"
            )
            for factor in factors
        },
        "categories": {
            factor: _configured_category(factor, categories, entries)
            for factor in factors
        },
    }
    _atomic_write_yaml(template_path, template)
    raise ValueError(
        f"{len(missing)} external factors are missing research categories; "
        f"complete {template_path} and set factor_library.factor_config"
    )


def _configured_category(
    factor: str,
    categories: object,
    entries: object,
) -> str:
    category = categories.get(factor) if isinstance(categories, Mapping) else None
    entry = entries.get(factor) if isinstance(entries, Mapping) else None
    if category is None and isinstance(entry, Mapping):
        category = entry.get("category")
    return str(category).strip() if category else "unclassified"


def _validate_batch_outputs(status_path: Path, leaderboard_path: Path) -> None:
    if not status_path.exists() or not leaderboard_path.exists():
        raise FileNotFoundError("factor batch did not write batch_status.csv and leaderboard.csv")
    status = pd.read_csv(status_path)
    if "status" not in status.columns:
        raise ValueError("batch_status.csv is missing status column")
    failed = status.loc[~status["status"].astype(str).isin(("success", "skipped"))]
    if not failed.empty:
        names = failed.get("factor", pd.Series(dtype=str)).astype(str).tolist()
        raise RuntimeError("factor batch is partial; failed factors: " + ", ".join(names))


def _factor_names(path: Path) -> tuple[str, ...]:
    if not path.exists():
        raise FileNotFoundError(f"required factor artifact not found: {path}")
    frame = pd.read_csv(path, dtype={"factor": str})
    if "factor" not in frame.columns:
        raise ValueError(f"factor artifact is missing factor column: {path}")
    return tuple(
        dict.fromkeys(
            str(value).strip()
            for value in frame["factor"].dropna()
            if str(value).strip()
        )
    )


def _build_summary(
    config: FactorResearchRunConfig,
    result: FactorResearchPipelineResult,
    batch_dir: Path,
    correlation_dir: Path,
    ml_dir: Path,
) -> dict[str, Any]:
    profitable_path = batch_dir / "profitable_long_only.csv"
    profitable_rows = len(pd.read_csv(profitable_path)) if profitable_path.exists() else 0
    profitable_models_path = ml_dir / "profitable_models.csv"
    profitable_models = (
        len(pd.read_csv(profitable_models_path)) if profitable_models_path.exists() else 0
    )
    return {
        "factor_family": result.factor_family,
        "evaluated_factor_count": len(result.evaluated_factors),
        "profitable_long_only_rows": profitable_rows,
        "deduplicated_factor_count": len(result.deduplicated_factors),
        "ml_candidate_factor_count": len(result.ml_candidate_factors),
        "ml_status": result.ml_status,
        "models": list(result.models),
        "profitable_model_count": profitable_models,
        "research_contract": {
            "factor_lag_days": 1,
            "portfolio_side": "long_only",
            "top_side": "buy_high_factor_values",
            "bottom_side": "buy_low_factor_values",
            "short_positions": False,
            "correlation_primary": "mean_daily_cross_sectional_spearman",
            "correlation_threshold": float(
                config.correlation.get("high_correlation_threshold", 0.8)
            ),
            "ml_window": "3_calendar_years_train_1_calendar_year_test",
            "ml_backtests": "gross_and_after_cost",
        },
        "outputs": {
            "batch": str(batch_dir.resolve()),
            "correlation": str(correlation_dir.resolve()),
            "machine_learning": str(ml_dir.resolve()),
        },
        "generated_at": _utc_now(),
    }


def _initial_manifest(
    config: FactorResearchRunConfig,
    batch_dir: Path,
    correlation_dir: Path,
    ml_dir: Path,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "started_at": _utc_now(),
        "finished_at": None,
        "config": str(config.source_path.resolve()),
        "config_sha256": _sha256(config.source_path),
        "factor_family": config.family,
        "long_only_only": True,
        "force": config.force,
        "settings": {
            "factor_library": config.factor_library,
            "evaluation": config.evaluation,
            "correlation": config.correlation,
            "machine_learning": config.machine_learning,
        },
        "outputs": {
            "batch": str(batch_dir.resolve()),
            "correlation": str(correlation_dir.resolve()),
            "machine_learning": str(ml_dir.resolve()),
        },
        "stages": {
            "preflight": {"status": "pending"},
            "batch": {"status": "pending"},
            "correlation": {"status": "pending"},
            "machine_learning": {"status": "pending"},
        },
        "warnings": [],
        "error": None,
    }


def _mapping_section(
    payload: Mapping[str, Any],
    name: str,
    allowed: set[str],
    *,
    required: bool = False,
) -> dict[str, Any]:
    raw = payload.get(name)
    if raw is None:
        if required:
            raise ValueError(f"missing required config section: {name}")
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"config section {name} must be a mapping")
    _reject_unknown(raw, allowed, name)
    return dict(raw)


def _reject_unknown(values: Mapping[str, Any], allowed: set[str], section: str) -> None:
    unknown = sorted(set(values).difference(allowed))
    if unknown:
        raise ValueError(f"unknown {section} config keys: {', '.join(unknown)}")


def _assign(args: argparse.Namespace, values: Mapping[str, Any], section: str) -> None:
    missing = sorted(set(values).difference(vars(args)))
    if missing:
        raise ValueError(f"unsupported {section} runner keys: {', '.join(missing)}")
    for name, value in values.items():
        setattr(args, name, value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        yaml.safe_dump(dict(payload), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)
