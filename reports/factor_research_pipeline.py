"""Thin CLI orchestration for the complete factor-research workflow.

``factor-run-all`` deliberately owns no factor, correlation, model, signal, or
backtest implementation.  It launches the governed public commands exactly as
a user would run them and only returns lightweight parent-run metadata.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from rquant.runtime import CommandResult, generate_run_id


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
    # Accepted for compatibility, but never forwarded by factor-run-all.
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
_EXECUTION_KEYS = {
    "output",  # Deprecated compatibility root.
    "batch_output",
    "correlation_output",
    "ml_output",
    "force",
    "require_classification",
}

_BATCH_LIBRARY_KEYS = (
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
)
_CORRELATION_LIBRARY_KEYS = (
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
    "benchmark_file",
    "style_factor_file",
)
_ML_LIBRARY_KEYS = (
    "data",
    "metadata",
    "benchmark_file",
    "style_factor_file",
    "factor_file",
    "factor_layout",
    "context_file",
    "context_date_col",
    "context_symbol_col",
)
_EVALUATION_CLI_KEYS = (
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
)
_CORRELATION_CLI_KEYS = (
    "factor_lag_days",
    "min_observations",
    "min_dates",
    "high_correlation_threshold",
    "priority_factor_col",
    "priority_score_col",
    "priority_window",
)
_ML_CLI_KEYS = (
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
)


@dataclass(frozen=True)
class FactorResearchRunConfig:
    """Validated settings used only to construct public CLI commands."""

    source_path: Path
    factor_library: dict[str, Any]
    evaluation: dict[str, Any]
    correlation: dict[str, Any]
    machine_learning: dict[str, Any]
    batch_output: Path
    correlation_output: Path
    ml_output: Path
    force: bool = False
    require_classification: bool = True
    warnings: tuple[str, ...] = ()

    @property
    def family(self) -> str:
        return str(self.factor_library["family"])

    @property
    def library_key(self) -> str:
        return _library_key(self.factor_library)


@dataclass(frozen=True)
class StageCommand:
    """One governed child command and its expected stable output location."""

    stage: str
    run_id: str
    argv: tuple[str, ...]
    output_dir: Path
    run_manifest: Path

    def summary(self, exit_code: int) -> dict[str, object]:
        return {
            "stage": self.stage,
            "command": list(self.argv),
            "run_id": self.run_id,
            "exit_code": int(exit_code),
            "output_dir": str(self.output_dir.resolve()),
            "run_manifest": str(self.run_manifest.resolve()),
        }


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--config",
        default="config/factor_research_run_all.yaml",
        help="YAML used to build the three normal factor-research CLI commands",
    )
    parser.add_argument("--family", choices=FACTOR_FAMILIES, default=None)
    parser.add_argument("--factor-file", default=None, help="Override an external factor CSV")
    parser.add_argument("--factor-config", default=None, help="Override lifecycle/category YAML")
    parser.add_argument("--factors", nargs="+", default=None, help="Override selected factor names")
    parser.add_argument(
        "--batch-output",
        default=None,
        help="Override the native factor-batch output directory",
    )
    parser.add_argument(
        "--correlation-output",
        default=None,
        help="Override the native factor-correlation output directory",
    )
    parser.add_argument(
        "--ml-output",
        default=None,
        help="Override the native fit-multifactor output directory",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Deprecated: compatibility root containing batch/correlation/ml subdirectories",
    )
    parser.add_argument("--force", action="store_true", help="Recompute resumable stages")
    parser.add_argument("--skip-ml", action="store_true", help="Run batch and correlation only")
    return parser


def load_factor_research_config(
    path: str | Path,
    *,
    family: str | None = None,
    factor_file: str | None = None,
    factor_config: str | None = None,
    factors: Sequence[str] | None = None,
    batch_output: str | None = None,
    correlation_output: str | None = None,
    ml_output: str | None = None,
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
            "so correlation quality uses the same holding horizon"
        )
    evaluation_windows = tuple(int(value) for value in evaluation.get("windows", (1, 5, 10, 20)))
    if target_window not in evaluation_windows:
        raise ValueError("machine_learning.target_window must be included in evaluation.windows")

    warnings: list[str] = []
    if correlation.get("eligibility_col") not in (None, "", False):
        warnings.append(
            "correlation.eligibility_col is deprecated and ignored; all deduplicated "
            "factors are passed to fit-multifactor"
        )

    legacy_root_value = output or execution.get("output")
    legacy_root = Path(legacy_root_value) if legacy_root_value else None
    if legacy_root is not None:
        warnings.append(
            "execution.output/--output is deprecated; use batch_output, "
            "correlation_output, and ml_output"
        )

    defaults = _default_stage_outputs(library)
    resolved_batch = Path(
        batch_output
        or execution.get("batch_output")
        or (legacy_root / "batch" if legacy_root else defaults["batch"])
    )
    resolved_correlation = Path(
        correlation_output
        or execution.get("correlation_output")
        or (legacy_root / "correlation" if legacy_root else defaults["correlation"])
    )
    resolved_ml = Path(
        ml_output
        or execution.get("ml_output")
        or (legacy_root / "ml" if legacy_root else defaults["ml"])
    )

    return FactorResearchRunConfig(
        source_path=config_path,
        factor_library=library,
        evaluation=evaluation,
        correlation=correlation,
        machine_learning=machine_learning,
        batch_output=resolved_batch,
        correlation_output=resolved_correlation,
        ml_output=resolved_ml,
        force=bool(force or execution.get("force", False)),
        require_classification=bool(execution.get("require_classification", True)),
        warnings=tuple(warnings),
    )


def build_stage_commands(
    config: FactorResearchRunConfig,
    *,
    python_executable: str | None = None,
    project_root: str | Path | None = None,
    runs_dir: str | Path = "data/runs",
    run_id_factory: Callable[[str], str] | None = None,
) -> tuple[StageCommand, ...]:
    """Build the exact public CLI commands executed by ``factor-run-all``."""

    executable = python_executable or sys.executable
    make_run_id = run_id_factory or _generate_child_run_id
    run_root = Path(runs_dir)
    if not run_root.is_absolute() and project_root is not None:
        run_root = Path(project_root) / run_root
    stage_specs: list[tuple[str, list[str], Path]] = [
        ("factor-batch", _batch_cli_args(config), config.batch_output),
        (
            "factor-correlation",
            _correlation_cli_args(config),
            config.correlation_output,
        ),
    ]
    if bool(config.machine_learning.get("enabled", True)):
        stage_specs.append(("fit-multifactor", _ml_cli_args(config), config.ml_output))

    commands: list[StageCommand] = []
    for stage, stage_args, output_dir in stage_specs:
        run_id = make_run_id(stage)
        global_args: list[str] = []
        if project_root is not None:
            global_args.extend(("--project-root", str(project_root)))
        global_args.extend(("--runs-dir", str(runs_dir), "--run-id", run_id))
        argv = (executable, "-m", "rquant", *global_args, stage, *stage_args)
        commands.append(
            StageCommand(
                stage=stage,
                run_id=run_id,
                argv=tuple(argv),
                output_dir=output_dir,
                run_manifest=run_root / run_id / "run.json",
            )
        )
    return tuple(commands)


def run_factor_research_pipeline(
    config: FactorResearchRunConfig,
    *,
    command_runner: Callable[[Sequence[str]], int] | None = None,
    python_executable: str | None = None,
    project_root: str | Path | None = None,
    runs_dir: str | Path = "data/runs",
    run_id_factory: Callable[[str], str] | None = None,
) -> CommandResult:
    """Run each governed research command once and fail fast on non-zero exit."""

    runner = command_runner or _run_child_command
    commands = build_stage_commands(
        config,
        python_executable=python_executable,
        project_root=project_root,
        runs_dir=runs_dir,
        run_id_factory=run_id_factory,
    )
    for warning in config.warnings:
        LOGGER.warning(warning)

    completed: list[dict[str, object]] = []
    outputs: dict[str, str] = {}
    for position, command in enumerate(commands, start=1):
        LOGGER.info(
            "[%d/%d] Launching governed command: %s",
            position,
            len(commands),
            command.stage,
        )
        exit_code = int(runner(command.argv) or 0)
        completed.append(command.summary(exit_code))
        outputs[f"{_stage_key(command.stage)}_output"] = str(command.output_dir)
        outputs[f"{_stage_key(command.stage)}_run_manifest"] = str(command.run_manifest)
        if exit_code != 0:
            LOGGER.error("Child command %s failed with exit code %d", command.stage, exit_code)
            return CommandResult(
                status="failed",
                exit_code=exit_code,
                outputs=outputs,
                summary={
                    "factor_family": config.family,
                    "orchestration": "governed_public_cli_commands",
                    "failed_stage": command.stage,
                    "stages": completed,
                    "warnings": list(config.warnings),
                },
            )

    return CommandResult(
        outputs=outputs,
        summary={
            "factor_family": config.family,
            "orchestration": "governed_public_cli_commands",
            "failed_stage": None,
            "stages": completed,
            "warnings": list(config.warnings),
        },
    )


def run_from_args(args: argparse.Namespace) -> CommandResult:
    config = load_factor_research_config(
        args.config,
        family=args.family,
        factor_file=args.factor_file,
        factor_config=args.factor_config,
        factors=args.factors,
        batch_output=args.batch_output,
        correlation_output=args.correlation_output,
        ml_output=args.ml_output,
        output=args.output,
        force=args.force,
        skip_ml=args.skip_ml,
    )
    return run_factor_research_pipeline(
        config,
        project_root=os.environ.get("RQUANT_PROJECT_ROOT"),
        runs_dir=os.environ.get("RQUANT_RUNS_DIR", "data/runs"),
    )


def _batch_cli_args(config: FactorResearchRunConfig) -> list[str]:
    argv: list[str] = []
    _append_values(argv, config.factor_library, _BATCH_LIBRARY_KEYS)
    _append_values(argv, config.evaluation, _EVALUATION_CLI_KEYS)
    if config.family == "external" and config.require_classification:
        argv.append("--require-classification")
    if config.force:
        argv.append("--force")
    argv.extend(("--output", str(config.batch_output)))
    return argv


def _correlation_cli_args(config: FactorResearchRunConfig) -> list[str]:
    argv: list[str] = []
    _append_values(argv, config.factor_library, _CORRELATION_LIBRARY_KEYS)
    correlation = {
        key: value
        for key, value in config.correlation.items()
        if key != "eligibility_col"
    }
    correlation.setdefault("priority_score_col", "preferred_gross_sharpe")
    correlation.setdefault(
        "priority_window", int(config.machine_learning.get("target_window", 20))
    )
    _append_values(argv, correlation, _CORRELATION_CLI_KEYS)
    _append_option(argv, "start_date", config.evaluation.get("start_date"))
    _append_option(argv, "end_date", config.evaluation.get("end_date"))
    _append_option(argv, "max_symbols", config.evaluation.get("max_symbols"))
    argv.extend(("--priority-file", str(config.batch_output / "leaderboard.csv")))
    argv.extend(("--output", str(config.correlation_output)))
    return argv


def _ml_cli_args(config: FactorResearchRunConfig) -> list[str]:
    argv: list[str] = []
    _append_values(argv, config.factor_library, _ML_LIBRARY_KEYS)
    library = config.factor_library
    _append_option(argv, "factor_date_col", library.get("date_col"))
    _append_option(argv, "factor_symbol_col", library.get("symbol_col"))
    _append_option(argv, "factor_name_col", library.get("factor_name_col"))
    _append_option(argv, "factor_value_col", library.get("factor_value_col"))
    ml_values = {
        key: value
        for key, value in config.machine_learning.items()
        if key != "enabled"
    }
    _append_values(argv, ml_values, _ML_CLI_KEYS)
    if "start" not in ml_values:
        _append_option(argv, "start", config.evaluation.get("start_date"))
    if "end" not in ml_values:
        _append_option(argv, "end", config.evaluation.get("end_date"))
    argv.extend(
        (
            "--factor-selection-file",
            str(config.correlation_output / "deduplicated_factors.csv"),
            "--factor-selection-col",
            "factor",
        )
    )
    if config.force:
        argv.append("--force")
    argv.extend(("--output", str(config.ml_output)))
    return argv


def _append_values(
    argv: list[str],
    values: Mapping[str, Any],
    names: Sequence[str],
) -> None:
    for name in names:
        if name in values:
            _append_option(argv, name, values[name])


def _append_option(argv: list[str], name: str, value: object) -> None:
    if value is None or value is False:
        return
    flag = "--" + name.replace("_", "-")
    if value is True:
        argv.append(flag)
        return
    if isinstance(value, (list, tuple)):
        if not value:
            return
        argv.append(flag)
        argv.extend(str(item) for item in value)
        return
    argv.extend((flag, str(value)))


def _default_stage_outputs(library: Mapping[str, Any]) -> dict[str, Path]:
    key = _library_key(library)
    return {
        "batch": Path("factor_report") / f"{key}_batch",
        "correlation": Path("factor_report") / f"{key}_correlation",
        "ml": Path("data") / "ml" / f"{key}_multifactor",
    }


def _library_key(library: Mapping[str, Any]) -> str:
    family = str(library.get("family", "")).strip().lower()
    if family in ("alpha101", "gtja191"):
        return family
    stem = Path(str(library.get("factor_file", "external"))).stem.lower()
    normalized = re.sub(r"[^a-z0-9._-]+", "_", stem).strip("._-")
    return normalized or "external"


def _generate_child_run_id(stage: str) -> str:
    return f"{stage}-{generate_run_id()}"


def _run_child_command(argv: Sequence[str]) -> int:
    completed = subprocess.run(list(argv), check=False)
    return int(completed.returncode)


def _stage_key(stage: str) -> str:
    return {
        "factor-batch": "batch",
        "factor-correlation": "correlation",
        "fit-multifactor": "ml",
    }[stage]


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
