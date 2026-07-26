"""Resumable batch research runner for the repository's Alpha101 factors."""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import re
import shutil
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from reports.factor_tester import FactorTester, FactorTesterConfig, forward_return_col
from factors.alpha101 import (
    ALPHA101_NAMES,
    Alpha101,
    Alpha101Panels,
    normalize_alpha_name,
)
from factors.catalog import FACTOR_STATUSES


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Alpha101BatchConfig:
    """Settings which affect factor-test results and safe resume behavior."""

    windows: tuple[int, ...] = (1, 5, 10, 20)
    groups: int = 10
    top_n_counts: tuple[int, ...] = (1, 5, 10, 20, 50, 100)
    start_date: str | None = None
    end_date: str | None = None
    winsorize: bool = False
    zscore: bool = False
    min_periods: int = 3
    min_listing_days: int = 60
    liquidity_lookback_days: int = 20
    min_liquidity: float = 0.0
    commission_rate: float = 0.0003
    slippage_rate: float = 0.0005
    stamp_tax_rate: float = 0.0005
    industry_col: str | None = "industry"
    market_cap_col: str | None = "market_cap"
    market_cap_groups: int = 3
    market_regime_col: str | None = "market_regime"
    market_regime_lookback_days: int = 60
    market_regime_min_periods: int = 20
    bull_return_threshold: float = 0.10
    bear_return_threshold: float = -0.10
    oos_start_date: str | None = None
    oos_fraction: float = 0.3
    force: bool = False
    fail_fast: bool = False
    show_progress: bool = False

    def __post_init__(self) -> None:
        if not self.windows or any(int(window) <= 0 for window in self.windows):
            raise ValueError("windows must contain positive integers")
        if self.groups not in (5, 10):
            raise ValueError("groups must be 5 or 10")
        if not self.top_n_counts or any(int(value) <= 0 for value in self.top_n_counts):
            raise ValueError("top_n_counts must contain positive integers")
        if self.min_periods < 2:
            raise ValueError("min_periods must be at least 2")
        if self.min_listing_days < 0:
            raise ValueError("min_listing_days must be non-negative")
        if self.liquidity_lookback_days <= 0:
            raise ValueError("liquidity_lookback_days must be positive")
        if self.min_liquidity < 0:
            raise ValueError("min_liquidity must be non-negative")
        if any(rate < 0 for rate in (self.commission_rate, self.slippage_rate, self.stamp_tax_rate)):
            raise ValueError("trading cost rates must be non-negative")
        if self.market_cap_groups < 2:
            raise ValueError("market_cap_groups must be at least 2")
        if self.market_regime_lookback_days <= 0:
            raise ValueError("market_regime_lookback_days must be positive")
        if not 1 <= self.market_regime_min_periods <= self.market_regime_lookback_days:
            raise ValueError(
                "market_regime_min_periods must be between 1 and "
                "market_regime_lookback_days"
            )
        if self.bear_return_threshold >= self.bull_return_threshold:
            raise ValueError("bear_return_threshold must be below bull_return_threshold")
        if not 0.0 < self.oos_fraction < 1.0:
            raise ValueError("oos_fraction must be between 0 and 1")
        if self.start_date and self.end_date:
            if pd.Timestamp(self.start_date) > pd.Timestamp(self.end_date):
                raise ValueError("start_date must not be after end_date")

    def result_settings(self) -> dict[str, object]:
        """Return only settings that change generated factor reports."""
        return {
            "windows": [int(window) for window in self.windows],
            "groups": self.groups,
            "top_n_counts": [int(value) for value in self.top_n_counts],
            "start_date": self.start_date,
            "end_date": self.end_date,
            "winsorize": self.winsorize,
            "zscore": self.zscore,
            "min_periods": self.min_periods,
            "factor_lag_days": 1,
            "min_listing_days": self.min_listing_days,
            "liquidity_lookback_days": self.liquidity_lookback_days,
            "min_liquidity": self.min_liquidity,
            "commission_rate": self.commission_rate,
            "slippage_rate": self.slippage_rate,
            "stamp_tax_rate": self.stamp_tax_rate,
            "industry_col": self.industry_col,
            "market_cap_col": self.market_cap_col,
            "market_cap_groups": self.market_cap_groups,
            "market_regime_col": self.market_regime_col,
            "market_regime_lookback_days": self.market_regime_lookback_days,
            "market_regime_min_periods": self.market_regime_min_periods,
            "bull_return_threshold": self.bull_return_threshold,
            "bear_return_threshold": self.bear_return_threshold,
            "oos_start_date": self.oos_start_date,
            "oos_fraction": self.oos_fraction,
        }


@dataclass(frozen=True)
class Alpha101BatchResult:
    output_dir: Path
    status: pd.DataFrame
    leaderboard: pd.DataFrame

    @property
    def failed_factors(self) -> tuple[str, ...]:
        if self.status.empty:
            return ()
        return tuple(self.status.loc[self.status["status"].eq("failed"), "factor"].astype(str))


def parse_factor_selection(
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Parse factor names, comma lists, and inclusive ranges into registry order."""

    selected = set(_expand_factor_tokens(include or ("all",)))
    selected.difference_update(_expand_factor_tokens(exclude or ()))
    return tuple(name for name in ALPHA101_NAMES if name in selected)


def _expand_factor_tokens(tokens: Sequence[str]) -> set[str]:
    names: set[str] = set()
    for token_group in tokens:
        for raw_token in str(token_group).split(","):
            token = raw_token.strip().lower()
            if not token:
                continue
            if token == "all":
                names.update(ALPHA101_NAMES)
                continue
            match = re.fullmatch(r"(?:alpha_?)?(\d+)\s*-\s*(?:alpha_?)?(\d+)", token)
            if match:
                start, end = (int(value) for value in match.groups())
                if start > end:
                    raise ValueError(f"invalid descending factor range: {raw_token}")
                names.update(normalize_alpha_name(number) for number in range(start, end + 1))
                continue
            names.add(normalize_alpha_name(token))
    return names


def directory_signature(data_dir: str | Path, metadata_path: str | Path | None = None) -> str:
    """Hash file names, sizes, and mtimes so resume notices changed input data."""

    root = Path(data_dir).resolve()
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.csv")):
        stat = path.stat()
        digest.update(
            f"{path.name}\0{stat.st_size}\0{stat.st_mtime_ns}\0"
            f"{getattr(stat, 'st_flags', 0)}\n".encode()
        )
    if metadata_path:
        path = Path(metadata_path).resolve()
        if path.exists():
            stat = path.stat()
            digest.update(
                f"metadata\0{path}\0{stat.st_size}\0{stat.st_mtime_ns}\0"
                f"{getattr(stat, 'st_flags', 0)}\n".encode()
            )
    return digest.hexdigest()


def files_signature(paths: Iterable[str | Path]) -> str:
    """Hash implementation files which can alter factor results."""

    digest = hashlib.sha256()
    for path_like in sorted((Path(path).resolve() for path in paths), key=str):
        digest.update(str(path_like).encode())
        digest.update(path_like.read_bytes())
    return digest.hexdigest()


def build_run_fingerprint(
    config: Alpha101BatchConfig,
    *,
    data_signature: str,
    implementation_signature: str,
) -> str:
    payload = {
        "settings": config.result_settings(),
        "data_signature": data_signature,
        "implementation_signature": implementation_signature,
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def build_forward_return_frame(
    panels: Alpha101Panels,
    windows: Sequence[int],
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    liquidity_lookback_days: int = 20,
) -> pd.DataFrame:
    """Build the reusable long-form universe and forward returns once."""

    close = panels.close.sort_index()
    evaluation_dates = close.index
    if start_date:
        evaluation_dates = evaluation_dates[evaluation_dates >= pd.Timestamp(start_date)]
    if end_date:
        evaluation_dates = evaluation_dates[evaluation_dates <= pd.Timestamp(end_date)]
    if evaluation_dates.empty:
        raise ValueError("no trading dates remain after applying start_date/end_date")

    close_evaluation = close.loc[evaluation_dates]
    close_long = (
        close_evaluation.rename_axis(index="date", columns="symbol")
        .stack(future_stack=True)
        .dropna()
    )
    base = pd.DataFrame(index=close_long.index)
    base["close"] = close_long
    volume_long = (
        panels.volume.reindex(index=evaluation_dates, columns=close.columns)
        .rename_axis(index="date", columns="symbol")
        .stack(future_stack=True)
    )
    base["volume"] = volume_long.reindex(base.index)
    turnover_panel = (
        panels.turnover_value.reindex(index=close.index, columns=close.columns)
        if panels.turnover_value is not None
        else close * panels.volume.reindex(index=close.index, columns=close.columns)
    )
    turnover_long = (
        turnover_panel.loc[evaluation_dates]
        .rename_axis(index="date", columns="symbol")
        .stack(future_stack=True)
    )
    base["turnover_value"] = turnover_long.reindex(base.index)
    avg_turnover = turnover_panel.rolling(
        int(liquidity_lookback_days),
        min_periods=1,
    ).mean().shift(1).loc[evaluation_dates]
    avg_turnover_long = avg_turnover.rename_axis(index="date", columns="symbol").stack(
        future_stack=True
    )
    base["avg_turnover_lagged"] = avg_turnover_long.reindex(base.index)
    listing_age = close.notna().cumsum().loc[evaluation_dates]
    listing_age_long = listing_age.rename_axis(index="date", columns="symbol").stack(
        future_stack=True
    )
    base["listing_age_days"] = listing_age_long.reindex(base.index)
    daily_returns = close.pct_change(fill_method=None).loc[evaluation_dates]
    daily_return_long = daily_returns.rename_axis(index="date", columns="symbol").stack(
        future_stack=True
    )
    base["daily_return"] = daily_return_long.reindex(base.index)
    if panels.industry is not None:
        industry_long = (
            panels.industry.reindex(index=evaluation_dates, columns=close.columns)
            .rename_axis(index="date", columns="symbol")
            .stack(future_stack=True)
        )
        base["industry"] = industry_long.reindex(base.index)
    if panels.cap is not None:
        cap_long = (
            panels.cap.reindex(index=evaluation_dates, columns=close.columns)
            .rename_axis(index="date", columns="symbol")
            .stack(future_stack=True)
        )
        base["market_cap"] = cap_long.reindex(base.index)
    if panels.is_st is not None:
        is_st_long = (
            panels.is_st.reindex(index=evaluation_dates, columns=close.columns)
            .rename_axis(index="date", columns="symbol")
            .stack(future_stack=True)
        )
        base["is_st"] = is_st_long.reindex(base.index)
    market_regime_panel = getattr(panels, "market_regime", None)
    if market_regime_panel is not None:
        regime_long = (
            market_regime_panel.reindex(index=evaluation_dates, columns=close.columns)
            .rename_axis(index="date", columns="symbol")
            .stack(future_stack=True)
        )
        base["market_regime"] = regime_long.reindex(base.index)
    for window in windows:
        returns = close.shift(-int(window)).div(close).sub(1.0).loc[evaluation_dates]
        return_long = returns.rename_axis(index="date", columns="symbol").stack(future_stack=True)
        base[forward_return_col("forward_return", int(window))] = return_long.reindex(base.index)
    return base.reset_index()


class Alpha101BatchRunner:
    """Calculate, test, checkpoint, and summarize Alpha101 factors sequentially."""

    def __init__(
        self,
        panels: Alpha101Panels,
        *,
        factors: Sequence[str],
        output_dir: str | Path,
        config: Alpha101BatchConfig | None = None,
        data_signature: str = "unspecified-data",
        implementation_signature: str = "unspecified-implementation",
        factor_statuses: Mapping[str, str] | None = None,
        factor_categories: Mapping[str, str] | None = None,
    ) -> None:
        normalized = tuple(normalize_alpha_name(name) for name in factors)
        if not normalized:
            raise ValueError("at least one Alpha101 factor must be selected")
        self.panels = panels
        self.factors = tuple(dict.fromkeys(normalized))
        self.output_dir = Path(output_dir)
        self.config = config or Alpha101BatchConfig()
        self.data_signature = data_signature
        self.implementation_signature = implementation_signature
        supplied_statuses = factor_statuses or {}
        self.factor_statuses = {
            name: str(supplied_statuses.get(name, "active")).strip().lower()
            for name in self.factors
        }
        supplied_categories = factor_categories or {}
        self.factor_categories = {
            name: str(supplied_categories.get(name, "unclassified")).strip()
            or "unclassified"
            for name in self.factors
        }
        invalid_statuses = set(self.factor_statuses.values()).difference(FACTOR_STATUSES)
        if invalid_statuses:
            raise ValueError(f"unknown factor statuses: {', '.join(sorted(invalid_statuses))}")
        self.fingerprint = build_run_fingerprint(
            self.config,
            data_signature=data_signature,
            implementation_signature=implementation_signature,
        )
        self.calculator = Alpha101(panels)

    def run(self) -> Alpha101BatchResult:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
        (self.output_dir / ".tmp").mkdir(exist_ok=True)
        base_returns = build_forward_return_frame(
            self.panels,
            self.config.windows,
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            liquidity_lookback_days=self.config.liquidity_lookback_days,
        )
        manifest = self._manifest(status="running", base_rows=len(base_returns))
        _atomic_write_json(self.output_dir / "batch_manifest.json", manifest)
        self._leaderboard_cache = build_leaderboard(
            self.output_dir,
            self.factors,
            fingerprint=self.fingerprint,
            factor_statuses=self.factor_statuses,
            factor_categories=self.factor_categories,
        )

        status_rows: list[dict[str, object]] = []
        progress = tqdm(
            enumerate(self.factors, start=1),
            total=len(self.factors),
            desc="因子批处理",
            unit="因子",
            dynamic_ncols=True,
            disable=not self.config.show_progress,
        )
        for position, factor_name in progress:
            progress.set_postfix_str(factor_name)
            logger.info("[%d/%d] %s", position, len(self.factors), factor_name)
            started = time.perf_counter()
            final_dir = self.output_dir / factor_name
            existing_metadata = _read_json(final_dir / "run_metadata.json")
            if (
                not self.config.force
                and (final_dir / "summary.csv").exists()
                and existing_metadata.get("fingerprint") == self.fingerprint
            ):
                row_count = existing_metadata.get("row_count", len(base_returns))
                status_rows.append(
                    self._status_row(
                        factor_name,
                        "skipped",
                        time.perf_counter() - started,
                        row_count=row_count,
                        message="matching completed report already exists",
                    )
                )
                self._checkpoint(status_rows, factor_name=factor_name)
                continue

            factor_frame: pd.DataFrame | None = None
            factor_values: pd.DataFrame | None = None
            try:
                factor_values = self.calculator.calculate(factor_name)
                factor_frame = self._factor_frame(base_returns, factor_values)
                tester = FactorTester(
                    factor_frame,
                    factor_name=factor_name,
                    config=FactorTesterConfig(
                        groups=self.config.groups,
                        top_n_counts=tuple(int(value) for value in self.config.top_n_counts),
                        forward_return_windows=tuple(int(value) for value in self.config.windows),
                        winsorize=self.config.winsorize,
                        zscore=self.config.zscore,
                        min_periods=self.config.min_periods,
                        min_listing_days=self.config.min_listing_days,
                        liquidity_lookback_days=self.config.liquidity_lookback_days,
                        min_liquidity=self.config.min_liquidity,
                        commission_rate=self.config.commission_rate,
                        slippage_rate=self.config.slippage_rate,
                        stamp_tax_rate=self.config.stamp_tax_rate,
                        industry_col=self.config.industry_col,
                        market_cap_col=self.config.market_cap_col,
                        market_cap_groups=self.config.market_cap_groups,
                        market_regime_col=self.config.market_regime_col,
                        market_regime_lookback_days=(
                            self.config.market_regime_lookback_days
                        ),
                        market_regime_min_periods=(
                            self.config.market_regime_min_periods
                        ),
                        bull_return_threshold=self.config.bull_return_threshold,
                        bear_return_threshold=self.config.bear_return_threshold,
                        oos_start_date=self.config.oos_start_date,
                        oos_fraction=self.config.oos_fraction,
                    ),
                )
                temp_root = self.output_dir / ".tmp"
                temp_factor_dir = temp_root / factor_name
                if temp_factor_dir.exists():
                    shutil.rmtree(temp_factor_dir)
                tester.write_reports(temp_root)
                metadata = {
                    "factor": factor_name,
                    "fingerprint": self.fingerprint,
                    "row_count": len(factor_frame),
                    "completed_at": _utc_now(),
                    "settings": self.config.result_settings(),
                }
                _atomic_write_json(temp_factor_dir / "run_metadata.json", metadata)
                if final_dir.exists():
                    shutil.rmtree(final_dir)
                os.replace(temp_factor_dir, final_dir)
                failure_log = self.output_dir / "logs" / f"{factor_name}.log"
                if failure_log.exists():
                    failure_log.unlink()
                status_rows.append(
                    self._status_row(
                        factor_name,
                        "success",
                        time.perf_counter() - started,
                        row_count=len(factor_frame),
                    )
                )
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                (self.output_dir / "logs" / f"{factor_name}.log").write_text(
                    traceback.format_exc(), encoding="utf-8"
                )
                status_rows.append(
                    self._status_row(
                        factor_name,
                        "failed",
                        time.perf_counter() - started,
                        row_count=len(factor_frame) if factor_frame is not None else 0,
                        message=message,
                    )
                )
                logger.error("%s failed: %s", factor_name, message)
                self._checkpoint(status_rows, factor_name=factor_name)
                if self.config.fail_fast:
                    raise
            else:
                self._checkpoint(status_rows, factor_name=factor_name)
            finally:
                del factor_frame
                del factor_values
                gc.collect()

        status = pd.DataFrame(status_rows)
        leaderboard = self._leaderboard_cache.copy()
        manifest = self._manifest(status="completed", base_rows=len(base_returns))
        manifest["completed_at"] = _utc_now()
        manifest["success_count"] = int(status["status"].isin(["success", "skipped"]).sum())
        manifest["failed_count"] = int(status["status"].eq("failed").sum())
        write_long_only_profitability_reports(self.output_dir, leaderboard)
        _atomic_write_json(self.output_dir / "batch_manifest.json", manifest)
        return Alpha101BatchResult(self.output_dir, status, leaderboard)

    def _factor_frame(self, base_returns: pd.DataFrame, factor_values: pd.DataFrame) -> pd.DataFrame:
        values = (
            factor_values.reindex(index=self.panels.close.index, columns=self.panels.close.columns)
            .rename_axis(index="date", columns="symbol")
            .stack(future_stack=True)
        )
        base_index = pd.MultiIndex.from_frame(base_returns[["date", "symbol"]])
        frame = base_returns.copy()
        frame["factor_value"] = values.reindex(base_index).to_numpy()
        return frame

    def _status_row(
        self,
        factor: str,
        status: str,
        duration: float,
        *,
        row_count: object,
        message: str = "",
    ) -> dict[str, object]:
        return {
            "factor": factor,
            "status": status,
            "duration_seconds": round(float(duration), 3),
            "row_count": row_count,
            "message": message,
            "report_dir": str(self.output_dir / factor),
            "factor_status": self.factor_statuses[factor],
            "finished_at": _utc_now(),
        }

    def _checkpoint(
        self,
        status_rows: list[dict[str, object]],
        *,
        factor_name: str,
    ) -> None:
        status = pd.DataFrame(status_rows)
        _atomic_write_csv(self.output_dir / "batch_status.csv", status)
        latest = build_leaderboard(
            self.output_dir,
            (factor_name,),
            fingerprint=self.fingerprint,
            factor_statuses=self.factor_statuses,
        )
        if not latest.empty:
            cached = self._leaderboard_cache
            if not cached.empty and "factor" in cached.columns:
                cached = cached.loc[cached["factor"] != factor_name]
            combined = pd.DataFrame.from_records(
                [*cached.to_dict("records"), *latest.to_dict("records")],
                columns=latest.columns,
            )
            status_order = pd.Categorical(
                combined["factor_status"],
                categories=list(FACTOR_STATUSES),
                ordered=True,
            )
            self._leaderboard_cache = (
                combined.assign(_status_order=status_order)
                .sort_values(
                    ["window", "_status_order", "abs_rank_icir", "factor"],
                    ascending=[True, True, False, True],
                    na_position="last",
                )
                .drop(columns="_status_order")
                .reset_index(drop=True)
            )
        _atomic_write_csv(self.output_dir / "leaderboard.csv", self._leaderboard_cache)

    def _manifest(self, *, status: str, base_rows: int) -> dict[str, object]:
        return {
            "status": status,
            "fingerprint": self.fingerprint,
            "selected_factors": list(self.factors),
            "factor_statuses": self.factor_statuses,
            "settings": self.config.result_settings(),
            "data_signature": self.data_signature,
            "implementation_signature": self.implementation_signature,
            "base_rows": base_rows,
            "updated_at": _utc_now(),
        }


def build_leaderboard(
    output_dir: str | Path,
    factors: Sequence[str],
    *,
    fingerprint: str | None = None,
    factor_statuses: Mapping[str, str] | None = None,
    factor_categories: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Combine successful factor reports into one row per factor and horizon."""

    rows: list[dict[str, object]] = []
    for factor in factors:
        factor_dir = Path(output_dir) / factor
        ic_path = factor_dir / "ic_summary.csv"
        if not ic_path.exists():
            continue
        if fingerprint is not None:
            metadata = _read_json(factor_dir / "run_metadata.json")
            if metadata.get("fingerprint") != fingerprint:
                continue
        ic = pd.read_csv(ic_path)
        group = _read_csv(factor_dir / "group_summary.csv")
        top_n = _read_csv(factor_dir / "top_n_summary.csv")
        tradable_top_n = _read_csv(factor_dir / "tradable_top_n.csv")
        tradable_top_quantile = _read_csv(factor_dir / "tradable_top_quantile.csv")
        horizon_effectiveness = _read_csv(factor_dir / "horizon_effectiveness.csv")
        neutralized = _read_csv(factor_dir / "neutralized_ic_summary.csv")
        sample = _read_csv(factor_dir / "sample_performance.csv")
        tradable = _read_csv(factor_dir / "tradable_long_short.csv")
        coverage = _read_csv(factor_dir / "coverage.csv")
        turnover = _read_csv(factor_dir / "turnover.csv")
        avg_coverage = (
            pd.to_numeric(coverage["coverage"], errors="coerce").mean()
            if "coverage" in coverage.columns
            else np.nan
        )
        avg_turnover = (
            pd.to_numeric(turnover["turnover"], errors="coerce").mean()
            if "turnover" in turnover.columns
            else np.nan
        )
        group_by_window = group.set_index("window") if "window" in group.columns else pd.DataFrame()
        neutralized_by_window = (
            neutralized.set_index("window") if "window" in neutralized.columns else pd.DataFrame()
        )
        top_n_by_window: dict[int, dict[str, object]] = {}
        if {"window", "top_n"}.issubset(top_n.columns):
            for _, top_row in top_n.iterrows():
                window_key = int(top_row["window"])
                top_count = int(top_row["top_n"])
                top_n_by_window.setdefault(window_key, {}).update(
                    {
                        f"top_{top_count}_mean_return": top_row.get("mean_forward_return"),
                        f"top_{top_count}_annualized_return": top_row.get("annualized_return"),
                        f"top_{top_count}_sharpe": top_row.get("sharpe"),
                        f"top_{top_count}_selected_count": top_row.get("average_selected_count"),
                    }
                )
        tradable_top_n_by_window: dict[int, dict[str, object]] = {}
        if {"window", "top_n", "date"}.issubset(tradable_top_n.columns):
            latest_top_n = (
                tradable_top_n.sort_values("date").groupby(["window", "top_n"]).tail(1)
            )
            for _, top_row in latest_top_n.iterrows():
                window_key = int(top_row["window"])
                top_count = int(top_row["top_n"])
                tradable_top_n_by_window.setdefault(window_key, {}).update(
                    {
                        f"tradable_top_{top_count}_annualized_return": top_row.get(
                            "annualized_return"
                        ),
                        f"tradable_top_{top_count}_max_drawdown": top_row.get(
                            "max_drawdown"
                        ),
                        f"tradable_top_{top_count}_sharpe": top_row.get("sharpe"),
                        f"tradable_top_{top_count}_cum_nav": top_row.get(
                            "tradable_top_n_cum_nav"
                        ),
                        f"tradable_top_{top_count}_selected_count": top_row.get(
                            "selected_count"
                        ),
                    }
                )
        oos_by_window = (
            sample.loc[sample.get("sample", pd.Series(dtype=str)).eq("out_of_sample")].set_index(
                "window"
            )
            if "window" in sample.columns
            else pd.DataFrame()
        )
        tradable_by_window = (
            tradable.sort_values("date").groupby("window").tail(1).set_index("window")
            if {"window", "date"}.issubset(tradable.columns)
            else pd.DataFrame()
        )
        tradable_quantile_by_window = (
            tradable_top_quantile.sort_values("date").groupby("window").tail(1).set_index(
                "window"
            )
            if {"window", "date"}.issubset(tradable_top_quantile.columns)
            else pd.DataFrame()
        )
        effectiveness_by_window = (
            horizon_effectiveness.set_index("window")
            if "window" in horizon_effectiveness.columns
            else pd.DataFrame()
        )
        for _, ic_row in ic.iterrows():
            window = int(ic_row["window"])
            group_row = (
                group_by_window.loc[window]
                if not group_by_window.empty and window in group_by_window.index
                else pd.Series(dtype=object)
            )
            neutralized_row = (
                neutralized_by_window.loc[window]
                if not neutralized_by_window.empty and window in neutralized_by_window.index
                else pd.Series(dtype=object)
            )
            oos_row = (
                oos_by_window.loc[window]
                if not oos_by_window.empty and window in oos_by_window.index
                else pd.Series(dtype=object)
            )
            tradable_row = (
                tradable_by_window.loc[window]
                if not tradable_by_window.empty and window in tradable_by_window.index
                else pd.Series(dtype=object)
            )
            tradable_quantile_row = (
                tradable_quantile_by_window.loc[window]
                if not tradable_quantile_by_window.empty
                and window in tradable_quantile_by_window.index
                else pd.Series(dtype=object)
            )
            effectiveness_row = (
                effectiveness_by_window.loc[window]
                if not effectiveness_by_window.empty
                and window in effectiveness_by_window.index
                else pd.Series(dtype=object)
            )
            preferred_long_side = effectiveness_row.get("preferred_long_side")
            if preferred_long_side == "high_factor":
                preferred_prefix = "high"
            elif preferred_long_side == "low_factor":
                preferred_prefix = "low"
            else:
                preferred_prefix = None

            def preferred_metric(metric: str) -> object:
                if preferred_prefix is None:
                    return np.nan
                return effectiveness_row.get(f"{preferred_prefix}_{metric}")

            rank_icir = pd.to_numeric(pd.Series([ic_row.get("rank_icir")]), errors="coerce").iloc[0]
            rank_ic_mean = pd.to_numeric(pd.Series([ic_row.get("rank_ic_mean")]), errors="coerce").iloc[0]
            if pd.isna(rank_ic_mean):
                direction = "unknown"
            else:
                direction = "positive" if rank_ic_mean >= 0 else "negative"
            row = {
                "factor": factor,
                "factor_status": (factor_statuses or {}).get(factor, "active"),
                "factor_category": (factor_categories or {}).get(
                    factor, "unclassified"
                ),
                "window": window,
                "direction": direction,
                "rank_ic_mean": rank_ic_mean,
                "rank_icir": rank_icir,
                "abs_rank_icir": abs(rank_icir) if pd.notna(rank_icir) else np.nan,
                "rank_ic_win_rate": ic_row.get("rank_ic_win_rate"),
                "ic_mean": ic_row.get("ic_mean"),
                "icir": ic_row.get("icir"),
                "ic_win_rate": ic_row.get("ic_win_rate"),
                "top_bottom_return": group_row.get("top_bottom_return"),
                "monotonic": group_row.get("monotonic"),
                "neutralized_rank_ic_mean": neutralized_row.get(
                    "neutralized_rank_ic_mean"
                ),
                "neutralized_rank_icir": neutralized_row.get("neutralized_rank_icir"),
                "oos_rank_ic_mean": oos_row.get("rank_ic_mean"),
                "oos_tradable_period_return": _first_available(
                    oos_row.get("tradable_top_quantile_period_return"),
                    oos_row.get("tradable_period_return"),
                ),
                "tradable_top_quantile": tradable_quantile_row.get("top_quantile"),
                "tradable_top_quantile_annualized_return": tradable_quantile_row.get(
                    "annualized_return"
                ),
                "tradable_top_quantile_max_drawdown": tradable_quantile_row.get(
                    "max_drawdown"
                ),
                "tradable_top_quantile_sharpe": tradable_quantile_row.get("sharpe"),
                "tradable_top_quantile_cum_nav": tradable_quantile_row.get(
                    "tradable_top_quantile_cum_nav"
                ),
                "tradable_top_quantile_selected_count": tradable_quantile_row.get(
                    "selected_count"
                ),
                "preferred_long_side": preferred_long_side,
                "preferred_gross_annualized_return": preferred_metric(
                    "gross_annualized_return"
                ),
                "preferred_gross_sharpe": preferred_metric("gross_sharpe"),
                "preferred_net_annualized_return": preferred_metric(
                    "net_annualized_return"
                ),
                "preferred_net_sharpe": preferred_metric("net_sharpe"),
                "preferred_profitable_before_cost": preferred_metric(
                    "profitable_before_cost"
                ),
                "preferred_profitable_after_cost": preferred_metric(
                    "profitable_after_cost"
                ),
                "high_gross_annualized_return": effectiveness_row.get(
                    "high_gross_annualized_return"
                ),
                "high_gross_sharpe": effectiveness_row.get("high_gross_sharpe"),
                "high_net_annualized_return": effectiveness_row.get(
                    "high_net_annualized_return"
                ),
                "high_net_sharpe": effectiveness_row.get("high_net_sharpe"),
                "low_gross_annualized_return": effectiveness_row.get(
                    "low_gross_annualized_return"
                ),
                "low_gross_sharpe": effectiveness_row.get("low_gross_sharpe"),
                "low_net_annualized_return": effectiveness_row.get(
                    "low_net_annualized_return"
                ),
                "low_net_sharpe": effectiveness_row.get("low_net_sharpe"),
                "high_profitable_before_cost": effectiveness_row.get(
                    "high_profitable_before_cost"
                ),
                "high_profitable_after_cost": effectiveness_row.get(
                    "high_profitable_after_cost"
                ),
                "low_profitable_before_cost": effectiveness_row.get(
                    "low_profitable_before_cost"
                ),
                "low_profitable_after_cost": effectiveness_row.get(
                    "low_profitable_after_cost"
                ),
                "tradable_annualized_return": _first_available(
                    tradable_quantile_row.get("annualized_return"),
                    tradable_row.get("annualized_return"),
                ),
                "tradable_max_drawdown": _first_available(
                    tradable_quantile_row.get("max_drawdown"),
                    tradable_row.get("max_drawdown"),
                ),
                "tradable_sharpe": _first_available(
                    tradable_quantile_row.get("sharpe"),
                    tradable_row.get("sharpe"),
                ),
                "avg_coverage": avg_coverage,
                "avg_turnover": avg_turnover,
                "observation_count": ic_row.get("count"),
            }
            row.update(top_n_by_window.get(window, {}))
            row.update(tradable_top_n_by_window.get(window, {}))
            rows.append(row)
    columns = [
        "factor",
        "factor_status",
        "factor_category",
        "window",
        "direction",
        "rank_ic_mean",
        "rank_icir",
        "abs_rank_icir",
        "rank_ic_win_rate",
        "ic_mean",
        "icir",
        "ic_win_rate",
        "top_1_mean_return",
        "top_5_mean_return",
        "top_10_mean_return",
        "top_20_mean_return",
        "top_50_mean_return",
        "top_100_mean_return",
        "top_1_annualized_return",
        "top_5_annualized_return",
        "top_10_annualized_return",
        "top_20_annualized_return",
        "top_50_annualized_return",
        "top_100_annualized_return",
        "top_1_sharpe",
        "top_5_sharpe",
        "top_10_sharpe",
        "top_20_sharpe",
        "top_50_sharpe",
        "top_100_sharpe",
        "top_1_selected_count",
        "top_5_selected_count",
        "top_10_selected_count",
        "top_20_selected_count",
        "top_50_selected_count",
        "top_100_selected_count",
        "tradable_top_1_annualized_return",
        "tradable_top_5_annualized_return",
        "tradable_top_10_annualized_return",
        "tradable_top_20_annualized_return",
        "tradable_top_50_annualized_return",
        "tradable_top_100_annualized_return",
        "tradable_top_1_max_drawdown",
        "tradable_top_5_max_drawdown",
        "tradable_top_10_max_drawdown",
        "tradable_top_20_max_drawdown",
        "tradable_top_50_max_drawdown",
        "tradable_top_100_max_drawdown",
        "tradable_top_1_sharpe",
        "tradable_top_5_sharpe",
        "tradable_top_10_sharpe",
        "tradable_top_20_sharpe",
        "tradable_top_50_sharpe",
        "tradable_top_100_sharpe",
        "tradable_top_1_cum_nav",
        "tradable_top_5_cum_nav",
        "tradable_top_10_cum_nav",
        "tradable_top_20_cum_nav",
        "tradable_top_50_cum_nav",
        "tradable_top_100_cum_nav",
        "tradable_top_1_selected_count",
        "tradable_top_5_selected_count",
        "tradable_top_10_selected_count",
        "tradable_top_20_selected_count",
        "tradable_top_50_selected_count",
        "tradable_top_100_selected_count",
        "tradable_top_quantile",
        "tradable_top_quantile_annualized_return",
        "tradable_top_quantile_max_drawdown",
        "tradable_top_quantile_sharpe",
        "tradable_top_quantile_cum_nav",
        "tradable_top_quantile_selected_count",
        "preferred_long_side",
        "preferred_gross_annualized_return",
        "preferred_gross_sharpe",
        "preferred_net_annualized_return",
        "preferred_net_sharpe",
        "preferred_profitable_before_cost",
        "preferred_profitable_after_cost",
        "high_gross_annualized_return",
        "high_gross_sharpe",
        "high_net_annualized_return",
        "high_net_sharpe",
        "low_gross_annualized_return",
        "low_gross_sharpe",
        "low_net_annualized_return",
        "low_net_sharpe",
        "high_profitable_before_cost",
        "high_profitable_after_cost",
        "low_profitable_before_cost",
        "low_profitable_after_cost",
        "top_bottom_return",
        "monotonic",
        "neutralized_rank_ic_mean",
        "neutralized_rank_icir",
        "oos_rank_ic_mean",
        "oos_tradable_period_return",
        "tradable_annualized_return",
        "tradable_max_drawdown",
        "tradable_sharpe",
        "avg_coverage",
        "avg_turnover",
        "observation_count",
    ]
    leaderboard = pd.DataFrame(rows, columns=columns)
    if leaderboard.empty:
        return leaderboard
    status_order = pd.Categorical(
        leaderboard["factor_status"], categories=list(FACTOR_STATUSES), ordered=True
    )
    return (
        leaderboard.assign(_status_order=status_order)
        .sort_values(
            ["window", "_status_order", "abs_rank_icir", "factor"],
            ascending=[True, True, False, True],
            na_position="last",
        )
        .drop(columns="_status_order")
        .reset_index(drop=True)
    )


def build_long_only_profitability(leaderboard: pd.DataFrame) -> pd.DataFrame:
    """Normalize high/low long-only results into one auditable row per side."""

    columns = [
        "factor",
        "factor_status",
        "factor_category",
        "window",
        "ic_direction",
        "rank_ic_mean",
        "side",
        "preferred_by_ic",
        "gross_annualized_return",
        "gross_sharpe",
        "net_annualized_return",
        "net_sharpe",
        "profitable_before_cost",
        "profitable_after_cost",
    ]
    rows: list[dict[str, object]] = []

    def as_bool(value: object) -> bool:
        return bool(value) if pd.notna(value) else False

    for _, row in leaderboard.iterrows():
        for prefix, side in (("high", "high_factor"), ("low", "low_factor")):
            rows.append(
                {
                    "factor": row.get("factor"),
                    "factor_status": row.get("factor_status"),
                    "factor_category": row.get("factor_category"),
                    "window": row.get("window"),
                    "ic_direction": row.get("direction"),
                    "rank_ic_mean": row.get("rank_ic_mean"),
                    "side": side,
                    "preferred_by_ic": row.get("preferred_long_side") == side,
                    "gross_annualized_return": row.get(
                        f"{prefix}_gross_annualized_return"
                    ),
                    "gross_sharpe": row.get(f"{prefix}_gross_sharpe"),
                    "net_annualized_return": row.get(
                        f"{prefix}_net_annualized_return"
                    ),
                    "net_sharpe": row.get(f"{prefix}_net_sharpe"),
                    "profitable_before_cost": as_bool(
                        row.get(f"{prefix}_profitable_before_cost", False)
                    ),
                    "profitable_after_cost": as_bool(
                        row.get(f"{prefix}_profitable_after_cost", False)
                    ),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def write_long_only_profitability_reports(
    output_dir: str | Path,
    leaderboard: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write complete and profitable-only long-only factor selections."""

    profitability = build_long_only_profitability(leaderboard)
    profitable = (
        profitability.loc[
            profitability["profitable_before_cost"]
            | profitability["profitable_after_cost"]
        ].reset_index(drop=True)
        if not profitability.empty
        else profitability
    )
    destination = Path(output_dir)
    _atomic_write_csv(destination / "long_only_profitability.csv", profitability)
    _atomic_write_csv(destination / "profitable_long_only.csv", profitable)
    return profitability, profitable


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _first_available(*values: object) -> object:
    for value in values:
        if pd.notna(value):
            return value
    return np.nan


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temp, index=False)
    os.replace(temp, path)


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
