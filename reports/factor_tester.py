from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from domain.factors import FactorEvaluationResult


LOGGER = logging.getLogger(__name__)

DEFAULT_FORWARD_WINDOWS = (1, 5, 10, 20)
DEFAULT_QUANTILES = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
DEFAULT_TOP_N_COUNTS = (1, 5, 10, 20, 50, 100)
TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class FactorTesterConfig:
    """Column mapping and test settings for long-format factor data."""

    date_col: str = "date"
    symbol_col: str = "symbol"
    factor_col: str = "factor_value"
    close_col: str = "close"
    daily_return_col: str = "daily_return"
    volume_col: str = "volume"
    liquidity_col: str = "turnover_value"
    liquidity_metric_col: str = "avg_turnover_lagged"
    universe_col: Optional[str] = None
    forward_return_prefix: str = "forward_return"
    forward_return_col: Optional[str] = None
    industry_col: Optional[str] = "industry"
    market_cap_col: Optional[str] = "market_cap"
    tradeable_col: Optional[str] = "is_tradeable"
    suspended_col: Optional[str] = "is_suspended"
    limit_up_col: Optional[str] = "is_limit_up"
    limit_down_col: Optional[str] = "is_limit_down"
    st_col: Optional[str] = "is_st"
    listing_age_col: str = "listing_age_days"
    groups: int = 10
    top_n_counts: tuple[int, ...] = DEFAULT_TOP_N_COUNTS
    forward_return_windows: tuple[int, ...] = DEFAULT_FORWARD_WINDOWS
    factor_lag_days: int = 1
    winsorize: bool = False
    winsorize_limits: tuple[float, float] = (0.01, 0.99)
    zscore: bool = False
    min_periods: int = 3
    exclude_st: bool = True
    min_listing_days: int = 60
    liquidity_lookback_days: int = 20
    min_liquidity: float = 0.0
    commission_rate: float = 0.0003
    slippage_rate: float = 0.0005
    stamp_tax_rate: float = 0.0005
    oos_start_date: Optional[str] = None
    oos_fraction: float = 0.3
    market_cap_groups: int = 3
    market_regime_col: Optional[str] = "market_regime"
    market_regime_lookback_days: int = 60
    market_regime_min_periods: int = 20
    bull_return_threshold: float = 0.10
    bear_return_threshold: float = -0.10

    def __post_init__(self) -> None:
        if self.factor_lag_days < 1:
            raise ValueError("factor_lag_days must be at least 1")
        if not self.top_n_counts or any(int(value) <= 0 for value in self.top_n_counts):
            raise ValueError("top_n_counts must contain positive integers")
        if self.min_listing_days < 0:
            raise ValueError("min_listing_days must be non-negative")
        if self.liquidity_lookback_days <= 0:
            raise ValueError("liquidity_lookback_days must be positive")
        if self.min_liquidity < 0:
            raise ValueError("min_liquidity must be non-negative")
        if any(rate < 0 for rate in (self.commission_rate, self.slippage_rate, self.stamp_tax_rate)):
            raise ValueError("trading cost rates must be non-negative")
        if not 0.0 < self.oos_fraction < 1.0:
            raise ValueError("oos_fraction must be between 0 and 1")
        if self.market_cap_groups < 2:
            raise ValueError("market_cap_groups must be at least 2")
        if self.market_regime_lookback_days <= 0:
            raise ValueError("market_regime_lookback_days must be positive")
        if not 1 <= self.market_regime_min_periods <= self.market_regime_lookback_days:
            raise ValueError(
                "market_regime_min_periods must be between 1 and market_regime_lookback_days"
            )
        if self.bear_return_threshold >= self.bull_return_threshold:
            raise ValueError("bear_return_threshold must be below bull_return_threshold")


@dataclass
class _TradableCohort:
    """One long-short sleeve with fixed per-symbol weights and delayed exits."""

    long_weights: dict[str, float]
    short_weights: dict[str, float]
    remaining_days: int


@dataclass
class _LongOnlyCohort:
    """One long-only sleeve with fixed per-symbol weights and delayed exits."""

    long_weights: dict[str, float]
    remaining_days: int


def _safe_std(series: pd.Series) -> float:
    value = float(series.std(ddof=1))
    return value if math.isfinite(value) else float("nan")


def _icir(mean: Optional[float], std: Optional[float]) -> Optional[float]:
    if mean is None or std is None or not math.isfinite(std) or std == 0:
        return None
    return float(mean) / float(std)


def _max_drawdown(nav: pd.Series) -> float:
    """Return max drawdown for a cumulative NAV series."""
    if nav.empty:
        return 0.0
    nav_with_start = pd.concat(
        [pd.Series([1.0], dtype="float64"), nav.reset_index(drop=True)],
        ignore_index=True,
    )
    peak = nav_with_start.cummax()
    drawdown = nav_with_start / peak - 1.0
    return abs(float(drawdown.min()))


def _annualized_return(
    returns: pd.Series,
    *,
    return_horizon_days: int = 1,
) -> Optional[float]:
    """Convert fixed-horizon returns to daily returns, then annualize."""
    clean = returns.dropna()
    if clean.empty:
        return None
    if return_horizon_days <= 0:
        return None
    gross_returns = 1.0 + clean
    if bool((gross_returns <= 0).any()):
        return None
    daily_returns = gross_returns.pow(1.0 / return_horizon_days) - 1.0
    cumulative = float((1.0 + daily_returns).prod())
    years = len(daily_returns) / TRADING_DAYS_PER_YEAR
    if years <= 0 or cumulative <= 0:
        return None
    return cumulative ** (1.0 / years) - 1.0


def _sharpe(
    returns: pd.Series,
    *,
    return_horizon_days: int = 1,
) -> Optional[float]:
    """Return zero-rate Sharpe using sqrt(252 / return_horizon_days)."""
    clean = returns.dropna()
    if len(clean) < 2:
        return None
    if return_horizon_days <= 0:
        return None
    std = float(clean.std(ddof=1))
    if not math.isfinite(std) or std == 0:
        return None
    periods_per_year = TRADING_DAYS_PER_YEAR / return_horizon_days
    return float(clean.mean()) / std * math.sqrt(periods_per_year)


def _monotonic(values: Sequence[float]) -> bool:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if len(clean) < 2:
        return False
    return all(a <= b for a, b in zip(clean, clean[1:])) or all(
        a >= b for a, b in zip(clean, clean[1:])
    )


def _rank_autocorr(prev: set[str], current: set[str]) -> float:
    if not prev and not current:
        return float("nan")
    union = prev | current
    if not union:
        return float("nan")
    return len(prev & current) / len(union)


def forward_return_col(prefix: str, window: int) -> str:
    """Build the default forward return column name for a window."""
    return f"{prefix}_{int(window)}d"


class FactorTester:
    """Run coverage, distribution, IC, grouping, long-short, turnover, and exposure tests."""

    def __init__(
        self,
        data: pd.DataFrame,
        *,
        factor_name: str,
        config: Optional[FactorTesterConfig] = None,
    ) -> None:
        self.raw_data = data.copy()
        self.factor_name = factor_name
        self.config = config or FactorTesterConfig()
        self.data: Optional[pd.DataFrame] = None
        self.valuation_data: Optional[pd.DataFrame] = None
        self._valuation_by_date_cache: Optional[dict[pd.Timestamp, pd.DataFrame]] = None
        self.filter_availability: dict[str, bool] = {}

    def prepare_data(self) -> pd.DataFrame:
        """Normalize point-in-time fields, lag the factor, and create research returns."""
        cfg = self.config
        required = [cfg.date_col, cfg.symbol_col, cfg.factor_col]
        missing = [col for col in required if col not in self.raw_data.columns]
        if missing:
            raise ValueError(f"missing required columns: {', '.join(missing)}")

        df = self.raw_data.copy()
        df[cfg.date_col] = pd.to_datetime(df[cfg.date_col])
        df[cfg.symbol_col] = df[cfg.symbol_col].astype(str).str.zfill(6)
        df[cfg.factor_col] = pd.to_numeric(df[cfg.factor_col], errors="coerce")
        if cfg.close_col in df.columns:
            df[cfg.close_col] = pd.to_numeric(df[cfg.close_col], errors="coerce")
        if cfg.volume_col in df.columns:
            df[cfg.volume_col] = pd.to_numeric(df[cfg.volume_col], errors="coerce")
        if cfg.market_cap_col and cfg.market_cap_col in df.columns:
            df[cfg.market_cap_col] = pd.to_numeric(df[cfg.market_cap_col], errors="coerce")
        df = df.sort_values([cfg.symbol_col, cfg.date_col]).reset_index(drop=True)

        if cfg.daily_return_col in df.columns:
            df[cfg.daily_return_col] = pd.to_numeric(
                df[cfg.daily_return_col],
                errors="coerce",
            )
        elif cfg.close_col in df.columns:
            df[cfg.daily_return_col] = df.groupby(cfg.symbol_col)[cfg.close_col].pct_change(
                fill_method=None
            )

        st_available = bool(cfg.st_col and cfg.st_col in df.columns)
        suspended_available = bool(cfg.suspended_col and cfg.suspended_col in df.columns)
        tradeable_available = bool(cfg.tradeable_col and cfg.tradeable_col in df.columns)
        limit_up_available = bool(cfg.limit_up_col and cfg.limit_up_col in df.columns)
        limit_down_available = bool(cfg.limit_down_col and cfg.limit_down_col in df.columns)
        liquidity_available = cfg.liquidity_metric_col in df.columns or cfg.liquidity_col in df.columns or (
            cfg.close_col in df.columns and cfg.volume_col in df.columns
        )

        df["_is_st"] = (
            self._as_bool(df[cfg.st_col])
            if st_available and cfg.st_col
            else False
        )
        if suspended_available and cfg.suspended_col:
            df["_is_suspended"] = self._as_bool(df[cfg.suspended_col])
        elif cfg.volume_col in df.columns:
            df["_is_suspended"] = df[cfg.volume_col].fillna(0.0) <= 0
            suspended_available = True
        else:
            df["_is_suspended"] = False

        if tradeable_available and cfg.tradeable_col:
            df["_is_tradeable"] = self._as_bool(df[cfg.tradeable_col])
        else:
            df["_is_tradeable"] = ~df["_is_suspended"]
            tradeable_available = suspended_available

        limit_rates = pd.Series(0.10, index=df.index, dtype="float64")
        symbols = df[cfg.symbol_col]
        limit_rates.loc[symbols.str.startswith(("300", "301", "688"))] = 0.20
        limit_rates.loc[symbols.str.startswith(("4", "8"))] = 0.30
        limit_rates.loc[df["_is_st"]] = 0.05
        if limit_up_available and cfg.limit_up_col:
            df["_is_limit_up"] = self._as_bool(df[cfg.limit_up_col])
        elif cfg.daily_return_col in df.columns:
            df["_is_limit_up"] = df[cfg.daily_return_col] >= limit_rates - 0.001
            limit_up_available = True
        else:
            df["_is_limit_up"] = False
        if limit_down_available and cfg.limit_down_col:
            df["_is_limit_down"] = self._as_bool(df[cfg.limit_down_col])
        elif cfg.daily_return_col in df.columns:
            df["_is_limit_down"] = df[cfg.daily_return_col] <= -limit_rates + 0.001
            limit_down_available = True
        else:
            df["_is_limit_down"] = False

        if cfg.liquidity_metric_col in df.columns:
            df["_liquidity"] = pd.to_numeric(
                df[cfg.liquidity_metric_col], errors="coerce"
            )
            liquidity = df["_liquidity"]
        elif cfg.liquidity_col in df.columns:
            liquidity = pd.to_numeric(df[cfg.liquidity_col], errors="coerce")
        elif cfg.close_col in df.columns and cfg.volume_col in df.columns:
            liquidity = df[cfg.close_col] * df[cfg.volume_col]
            df[cfg.liquidity_col] = liquidity
        else:
            liquidity = pd.Series(np.nan, index=df.index, dtype="float64")
        if cfg.liquidity_metric_col not in df.columns:
            df["_liquidity"] = liquidity.groupby(df[cfg.symbol_col]).transform(
                lambda values: values.rolling(
                    cfg.liquidity_lookback_days,
                    min_periods=1,
                ).mean().shift(1)
            )

        if cfg.listing_age_col in df.columns:
            df["_listing_age_days"] = pd.to_numeric(
                df[cfg.listing_age_col], errors="coerce"
            )
        else:
            observed = (
                df[cfg.close_col].notna()
                if cfg.close_col in df.columns
                else pd.Series(True, index=df.index)
            )
            df["_listing_age_days"] = observed.groupby(df[cfg.symbol_col]).cumsum()

        df["_exclude_untradeable"] = ~df["_is_tradeable"] | df["_is_suspended"]
        df["_exclude_st"] = bool(cfg.exclude_st) & df["_is_st"]
        df["_exclude_new_stock"] = df["_listing_age_days"] < cfg.min_listing_days
        df["_exclude_low_liquidity"] = (
            (df["_liquidity"] < cfg.min_liquidity) | df["_liquidity"].isna()
            if cfg.min_liquidity > 0 and liquidity_available
            else False
        )
        df["_eligible"] = ~df[
            [
                "_exclude_untradeable",
                "_exclude_st",
                "_exclude_new_stock",
                "_exclude_low_liquidity",
            ]
        ].any(axis=1)

        df = self._attach_market_regime(df)

        self.filter_availability = {
            "tradeable": tradeable_available,
            "suspended": suspended_available,
            "limit_up": limit_up_available,
            "limit_down": limit_down_available,
            "st": st_available,
            "listing_age": True,
            "liquidity": liquidity_available,
            "industry": bool(cfg.industry_col and cfg.industry_col in df.columns),
            "market_cap": bool(cfg.market_cap_col and cfg.market_cap_col in df.columns),
            "market_regime": bool(df["_market_regime"].notna().any()),
        }

        df["factor_raw"] = df[cfg.factor_col]
        if cfg.market_cap_col and cfg.market_cap_col in df.columns:
            df["_market_cap_lagged"] = df.groupby(cfg.symbol_col)[cfg.market_cap_col].shift(
                cfg.factor_lag_days
            )
        df["factor_lagged"] = df.groupby(cfg.symbol_col)[cfg.factor_col].shift(
            cfg.factor_lag_days
        )
        df["factor_processed"] = df["factor_lagged"].where(df["_eligible"])

        if cfg.winsorize:
            df["factor_processed"] = self._winsorize_by_date(
                df["factor_processed"],
                df[cfg.date_col],
                lower_q=cfg.winsorize_limits[0],
                upper_q=cfg.winsorize_limits[1],
            )
        if cfg.zscore:
            df["factor_processed"] = self._zscore_by_date(df["factor_processed"], df[cfg.date_col])

        df = self._ensure_forward_returns(df)
        state_cols = [
            cfg.date_col,
            cfg.symbol_col,
            cfg.daily_return_col,
            "_is_tradeable",
            "_is_suspended",
            "_is_limit_up",
            "_is_limit_down",
            "_is_st",
            "_listing_age_days",
            "_liquidity",
            "_eligible",
        ]
        self.valuation_data = (
            df[state_cols].copy()
            if cfg.daily_return_col in df.columns
            else None
        )
        self.data = df
        return df

    def run_all(self) -> FactorEvaluationResult:
        """Run all factor tests and return report tables keyed by report name."""
        started_at = time.monotonic()
        LOGGER.info("factor %s: preparing evaluation data", self.factor_name)
        df = self._prepared()
        LOGGER.info(
            "factor %s: prepared %d rows in %.1fs",
            self.factor_name,
            len(df),
            time.monotonic() - started_at,
        )
        coverage, coverage_summary = self.coverage_test()
        distribution, distribution_summary = self.distribution_test()
        ic, ic_summary = self.ic_test()
        market_cap_ic, market_cap_ic_summary = self.market_cap_ic_test()
        stage_started_at = time.monotonic()
        LOGGER.info("factor %s: computing industry IC", self.factor_name)
        industry_ic, industry_ic_summary = self.industry_ic_test()
        LOGGER.info(
            "factor %s: industry IC completed in %.1fs",
            self.factor_name,
            time.monotonic() - stage_started_at,
        )
        market_regime_ic, market_regime_ic_summary = self.market_regime_ic_test(ic)
        annual_ic = self.annual_ic_test(ic)
        neutralized_ic, neutralized_ic_summary = self.neutralized_ic_test()
        group_return, group_summary = self.group_return_test()
        top_n_return, top_n_summary = self.top_n_return_test()
        tradable_top_n = self.tradable_top_n_test()
        tradable_top_quantile = self.tradable_top_quantile_test()
        tradable_bottom_n = self.tradable_bottom_n_test()
        tradable_bottom_quantile = self.tradable_bottom_quantile_test()
        stat_long_short = self.long_short_test()
        tradable_long_short = self.tradable_long_short_test()
        turnover = self.turnover_test()
        exposure = self.exposure_test()
        universe_filter, filter_status = self.universe_filter_test()
        oos_start = self._resolve_oos_start()
        ic = self._add_sample_split(ic, oos_start)
        neutralized_ic = self._add_sample_split(neutralized_ic, oos_start)
        group_return = self._add_sample_split(group_return, oos_start)
        top_n_return = self._add_sample_split(top_n_return, oos_start)
        tradable_top_n = self._add_sample_split(tradable_top_n, oos_start)
        tradable_top_quantile = self._add_sample_split(tradable_top_quantile, oos_start)
        tradable_bottom_n = self._add_sample_split(tradable_bottom_n, oos_start)
        tradable_bottom_quantile = self._add_sample_split(tradable_bottom_quantile, oos_start)
        stat_long_short = self._add_sample_split(stat_long_short, oos_start)
        tradable_long_short = self._add_sample_split(tradable_long_short, oos_start)
        annual_performance = self.annual_performance_test(
            stat_long_short,
            tradable_long_short,
        )
        annual_long_only = self.annual_long_only_performance_test(
            tradable_top_n=tradable_top_n,
            tradable_top_quantile=tradable_top_quantile,
            tradable_bottom_n=tradable_bottom_n,
            tradable_bottom_quantile=tradable_bottom_quantile,
        )
        horizon_effectiveness = self.horizon_effectiveness_test(
            ic_summary=ic_summary,
            tradable_top_quantile=tradable_top_quantile,
            tradable_bottom_quantile=tradable_bottom_quantile,
        )
        sample_performance = self.sample_performance_test(
            ic=ic,
            neutralized_ic=neutralized_ic,
            group_return=group_return,
            tradable_top_quantile=tradable_top_quantile,
            stat_long_short=stat_long_short,
            tradable_long_short=tradable_long_short,
            oos_start=oos_start,
        )
        LOGGER.info(
            "factor %s: all evaluation stages completed in %.1fs",
            self.factor_name,
            time.monotonic() - started_at,
        )

        summary = self._build_summary(
            coverage_summary=coverage_summary,
            distribution_summary=distribution_summary,
            ic_summary=ic_summary,
            group_summary=group_summary,
            top_n_summary=top_n_summary,
            tradable_top_n=tradable_top_n,
            tradable_top_quantile=tradable_top_quantile,
            stat_long_short=stat_long_short,
            tradable_long_short=tradable_long_short,
            exposure=exposure,
            neutralized_ic_summary=neutralized_ic_summary,
            oos_start=oos_start,
            row_count=len(df),
        )
        rank_ic = ic[[self.config.date_col, "window", "rank_ic", "sample"]]
        return FactorEvaluationResult(self.factor_name, {
            "summary": summary,
            "coverage": coverage,
            "distribution": distribution,
            "ic": ic[[self.config.date_col, "window", "ic", "sample"]],
            "rank_ic": rank_ic,
            "ic_summary": ic_summary,
            "market_cap_ic": market_cap_ic,
            "market_cap_ic_summary": market_cap_ic_summary,
            "industry_ic": industry_ic,
            "industry_ic_summary": industry_ic_summary,
            "market_regime_ic": market_regime_ic,
            "market_regime_ic_summary": market_regime_ic_summary,
            "annual_ic": annual_ic,
            "neutralized_ic": neutralized_ic,
            "neutralized_ic_summary": neutralized_ic_summary,
            "group_return": group_return,
            "group_summary": group_summary,
            "top_n_return": top_n_return,
            "top_n_summary": top_n_summary,
            "tradable_top_n": tradable_top_n,
            "tradable_top_quantile": tradable_top_quantile,
            "tradable_bottom_n": tradable_bottom_n,
            "tradable_bottom_quantile": tradable_bottom_quantile,
            "long_short": stat_long_short,
            "stat_long_short": stat_long_short,
            "tradable_long_short": tradable_long_short,
            "turnover": turnover,
            "exposure": exposure,
            "universe_filter": universe_filter,
            "filter_status": filter_status,
            "annual_performance": annual_performance,
            "annual_long_only": annual_long_only,
            "horizon_effectiveness": horizon_effectiveness,
            "sample_performance": sample_performance,
        })

    def write_reports(self, output_root: str | Path = "factor_report") -> Path:
        """Write CSV reports under factor_report/{factor_name}/ and return the directory."""
        reports = self.run_all()
        out_dir = Path(output_root) / self.factor_name
        out_dir.mkdir(parents=True, exist_ok=True)

        file_map = {
            "summary": "summary.csv",
            "coverage": "coverage.csv",
            "distribution": "distribution.csv",
            "ic": "ic.csv",
            "rank_ic": "rank_ic.csv",
            "ic_summary": "ic_summary.csv",
            "market_cap_ic": "market_cap_ic.csv",
            "market_cap_ic_summary": "market_cap_ic_summary.csv",
            "industry_ic": "industry_ic.csv",
            "industry_ic_summary": "industry_ic_summary.csv",
            "market_regime_ic": "market_regime_ic.csv",
            "market_regime_ic_summary": "market_regime_ic_summary.csv",
            "annual_ic": "annual_ic.csv",
            "neutralized_ic": "neutralized_ic.csv",
            "neutralized_ic_summary": "neutralized_ic_summary.csv",
            "group_return": "group_return.csv",
            "group_summary": "group_summary.csv",
            "top_n_return": "top_n_return.csv",
            "top_n_summary": "top_n_summary.csv",
            "tradable_top_n": "tradable_top_n.csv",
            "tradable_top_quantile": "tradable_top_quantile.csv",
            "tradable_bottom_n": "tradable_bottom_n.csv",
            "tradable_bottom_quantile": "tradable_bottom_quantile.csv",
            "long_short": "long_short.csv",
            "stat_long_short": "stat_long_short.csv",
            "tradable_long_short": "tradable_long_short.csv",
            "turnover": "turnover.csv",
            "exposure": "exposure.csv",
            "universe_filter": "universe_filter.csv",
            "filter_status": "filter_status.csv",
            "annual_performance": "annual_performance.csv",
            "annual_long_only": "annual_long_only.csv",
            "horizon_effectiveness": "horizon_effectiveness.csv",
            "sample_performance": "sample_performance.csv",
        }
        for key, filename in file_map.items():
            reports[key].to_csv(out_dir / filename, index=False)
        return out_dir

    def coverage_test(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Compute daily non-null factor count and coverage against the stock universe."""
        df = self._prepared()
        cfg = self.config
        grouped = df.groupby(cfg.date_col)
        if cfg.universe_col and cfg.universe_col in df.columns:
            universe_count = grouped[cfg.universe_col].sum().astype(int)
        else:
            universe_count = (
                df.loc[df["_eligible"]]
                .groupby(cfg.date_col)[cfg.symbol_col]
                .nunique()
                .reindex(grouped.size().index, fill_value=0)
                .astype(int)
            )
        non_null_count = grouped["factor_processed"].apply(lambda s: s.notna().sum()).astype(int)
        out = pd.DataFrame(
            {
                cfg.date_col: universe_count.index,
                "universe_count": universe_count.values,
                "non_null_count": non_null_count.reindex(universe_count.index).fillna(0).astype(int).values,
            }
        )
        out["coverage"] = np.where(
            out["universe_count"] > 0,
            out["non_null_count"] / out["universe_count"],
            np.nan,
        )
        summary = pd.DataFrame(
            [
                {"metric": "avg_coverage", "value": out["coverage"].mean()},
                {"metric": "min_coverage", "value": out["coverage"].min()},
            ]
        )
        return out, summary

    def distribution_test(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Compute daily factor distribution and overall outlier statistics."""
        df = self._prepared()
        cfg = self.config
        rows = []
        for date, daily in df.groupby(cfg.date_col):
            values = daily["factor_processed"].dropna()
            row: dict[str, Any] = {
                cfg.date_col: date,
                "count": int(values.count()),
                "mean": values.mean() if not values.empty else np.nan,
                "std": _safe_std(values) if len(values) > 1 else np.nan,
                "min": values.min() if not values.empty else np.nan,
                "max": values.max() if not values.empty else np.nan,
            }
            for q in DEFAULT_QUANTILES:
                row[f"q{int(q * 100):02d}"] = values.quantile(q) if not values.empty else np.nan
            rows.append(row)
        daily_distribution = pd.DataFrame(rows)

        raw = df["factor_raw"].dropna()
        if len(raw) > 1:
            z = (raw - raw.mean()) / raw.std(ddof=1)
            outlier_count = int((z.abs() > 5).sum())
        else:
            outlier_count = 0
        summary = pd.DataFrame(
            [
                {"metric": "raw_non_null_count", "value": int(raw.count())},
                {"metric": "outlier_abs_z_gt_5_count", "value": outlier_count},
                {
                    "metric": "outlier_abs_z_gt_5_rate",
                    "value": outlier_count / int(raw.count()) if raw.count() else np.nan,
                },
                {"metric": "winsorize_enabled", "value": int(bool(cfg.winsorize))},
                {"metric": "zscore_enabled", "value": int(bool(cfg.zscore))},
            ]
        )
        return daily_distribution, summary

    def ic_test(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Compute daily Pearson IC and Spearman Rank IC for each forward window."""
        df = self._prepared()
        cfg = self.config
        rows = []
        for window in cfg.forward_return_windows:
            ret_col = self._return_col(window)
            for date, daily in df.groupby(cfg.date_col):
                pair = daily[["factor_processed", ret_col]].dropna()
                if (
                    len(pair) < cfg.min_periods
                    or pair["factor_processed"].nunique() < 2
                    or pair[ret_col].nunique() < 2
                ):
                    ic = np.nan
                    rank_ic = np.nan
                else:
                    ic = pair["factor_processed"].corr(pair[ret_col], method="pearson")
                    rank_ic = pair["factor_processed"].corr(pair[ret_col], method="spearman")
                rows.append({cfg.date_col: date, "window": int(window), "ic": ic, "rank_ic": rank_ic})
        ic_df = pd.DataFrame(rows)

        summary_rows = []
        for window, part in ic_df.groupby("window"):
            ic_series = part["ic"].dropna()
            rank_series = part["rank_ic"].dropna()
            ic_mean = float(ic_series.mean()) if not ic_series.empty else None
            ic_std = float(ic_series.std(ddof=1)) if len(ic_series) > 1 else None
            rank_mean = float(rank_series.mean()) if not rank_series.empty else None
            rank_std = float(rank_series.std(ddof=1)) if len(rank_series) > 1 else None
            summary_rows.append(
                {
                    "window": int(window),
                    "ic_mean": ic_mean,
                    "ic_std": ic_std,
                    "icir": _icir(ic_mean, ic_std),
                    "ic_win_rate": float((ic_series > 0).mean()) if not ic_series.empty else None,
                    "rank_ic_mean": rank_mean,
                    "rank_ic_std": rank_std,
                    "rank_icir": _icir(rank_mean, rank_std),
                    "rank_ic_win_rate": float((rank_series > 0).mean()) if not rank_series.empty else None,
                    "count": int(ic_series.count()),
                }
            )
        return ic_df, pd.DataFrame(summary_rows)

    def market_cap_ic_test(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Compute IC independently inside point-in-time market-cap buckets."""
        df = self._prepared()
        cfg = self.config
        columns = [
            cfg.date_col,
            "window",
            "market_cap_bucket",
            "market_cap_bucket_label",
            "ic",
            "rank_ic",
            "count",
        ]
        if "_market_cap_lagged" not in df.columns:
            return pd.DataFrame(columns=columns), pd.DataFrame()

        rows: list[dict[str, Any]] = []
        for window in cfg.forward_return_windows:
            ret_col = self._return_col(window)
            for date, daily in df.groupby(cfg.date_col):
                valid = daily[
                    ["factor_processed", ret_col, "_market_cap_lagged"]
                ].dropna().copy()
                if len(valid) < cfg.market_cap_groups * cfg.min_periods:
                    continue
                valid["market_cap_bucket"] = self._assign_groups(
                    valid["_market_cap_lagged"],
                    cfg.market_cap_groups,
                )
                for bucket, part in valid.dropna(subset=["market_cap_bucket"]).groupby(
                    "market_cap_bucket"
                ):
                    ic, rank_ic = self._cross_sectional_ic(part, ret_col)
                    rows.append(
                        {
                            cfg.date_col: date,
                            "window": int(window),
                            "market_cap_bucket": int(bucket),
                            "market_cap_bucket_label": self._market_cap_bucket_label(
                                int(bucket)
                            ),
                            "ic": ic,
                            "rank_ic": rank_ic,
                            "count": int(len(part)),
                        }
                    )
        result = pd.DataFrame(rows, columns=columns)
        return result, self._summarize_segmented_ic(
            result,
            ["window", "market_cap_bucket", "market_cap_bucket_label"],
        )

    def industry_ic_test(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Compute IC independently inside each point-in-time industry."""
        df = self._prepared()
        cfg = self.config
        columns = [cfg.date_col, "window", "industry", "ic", "rank_ic", "count"]
        if not cfg.industry_col or cfg.industry_col not in df.columns:
            return pd.DataFrame(columns=columns), pd.DataFrame()

        group_keys = (
            df[[cfg.date_col, cfg.industry_col]]
            .dropna()
            .drop_duplicates()
            .sort_values([cfg.date_col, cfg.industry_col])
        )
        frames: list[pd.DataFrame] = []
        for window in cfg.forward_return_windows:
            ret_col = self._return_col(window)
            grouped = self._grouped_ic(
                df,
                group_cols=(cfg.date_col, cfg.industry_col),
                return_col=ret_col,
            )
            grouped = group_keys.merge(
                grouped,
                on=[cfg.date_col, cfg.industry_col],
                how="left",
                validate="one_to_one",
            )
            grouped["count"] = grouped["count"].fillna(0).astype("int64")
            grouped = grouped.rename(columns={cfg.industry_col: "industry"})
            grouped.insert(1, "window", int(window))
            frames.append(grouped)
        result = (
            pd.concat(frames, ignore_index=True)[columns]
            if frames
            else pd.DataFrame(columns=columns)
        )
        return result, self._summarize_segmented_ic(
            result,
            ["window", "industry"],
        )

    def _grouped_ic(
        self,
        frame: pd.DataFrame,
        *,
        group_cols: Sequence[str],
        return_col: str,
    ) -> pd.DataFrame:
        """Vectorized Pearson and Spearman correlations for many small groups."""

        value_cols = ["factor_processed", return_col]
        valid = frame[[*group_cols, *value_cols]].replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        output_cols = [*group_cols, "ic", "rank_ic", "count"]
        if valid.empty:
            return pd.DataFrame(columns=output_cols)

        keys = pd.MultiIndex.from_frame(valid[list(group_cols)], names=group_cols)
        codes, unique_keys = pd.factorize(keys, sort=False)
        group_count = len(unique_keys)
        x = valid["factor_processed"].to_numpy(dtype="float64", copy=False)
        y = valid[return_col].to_numpy(dtype="float64", copy=False)

        def correlations(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            count = np.bincount(codes, minlength=group_count).astype("int64")
            count_float = count.astype("float64")
            left_sum = np.bincount(codes, weights=left, minlength=group_count)
            right_sum = np.bincount(codes, weights=right, minlength=group_count)
            left_square_sum = np.bincount(
                codes, weights=left * left, minlength=group_count
            )
            right_square_sum = np.bincount(
                codes, weights=right * right, minlength=group_count
            )
            cross_sum = np.bincount(
                codes, weights=left * right, minlength=group_count
            )
            numerator = count_float * cross_sum - left_sum * right_sum
            left_variance = np.maximum(
                count_float * left_square_sum - left_sum * left_sum,
                0.0,
            )
            right_variance = np.maximum(
                count_float * right_square_sum - right_sum * right_sum,
                0.0,
            )
            denominator = np.sqrt(left_variance * right_variance)
            result = np.full(group_count, np.nan, dtype="float64")
            eligible = (count >= self.config.min_periods) & (denominator > 0.0)
            result[eligible] = numerator[eligible] / denominator[eligible]
            result[eligible] = np.clip(result[eligible], -1.0, 1.0)
            return result, count

        pearson, counts = correlations(x, y)
        ranked_x = (
            pd.Series(x, copy=False).groupby(codes, sort=False).rank(method="average")
        ).to_numpy(dtype="float64", copy=False)
        ranked_y = (
            pd.Series(y, copy=False).groupby(codes, sort=False).rank(method="average")
        ).to_numpy(dtype="float64", copy=False)
        spearman, _ = correlations(ranked_x, ranked_y)

        result = unique_keys.to_frame(index=False)
        result.columns = list(group_cols)
        result["ic"] = pearson
        result["rank_ic"] = spearman
        result["count"] = counts
        return result[output_cols]

    def market_regime_ic_test(
        self,
        ic: Optional[pd.DataFrame] = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Attach bull/bear/sideways labels to daily IC and summarize by regime."""
        df = self._prepared()
        cfg = self.config
        columns = [
            cfg.date_col,
            "window",
            "market_regime",
            "market_trailing_return",
            "ic",
            "rank_ic",
        ]
        if not df["_market_regime"].notna().any():
            return pd.DataFrame(columns=columns), pd.DataFrame()
        inconsistent = df.groupby(cfg.date_col)["_market_regime"].nunique(dropna=True)
        if bool((inconsistent > 1).any()):
            raise ValueError("market regime must have at most one value per trading date")
        daily_regime = (
            df[[cfg.date_col, "_market_regime", "_market_trailing_return"]]
            .drop_duplicates(cfg.date_col, keep="last")
            .rename(
                columns={
                    "_market_regime": "market_regime",
                    "_market_trailing_return": "market_trailing_return",
                }
            )
        )
        ic_frame = ic.copy() if ic is not None else self.ic_test()[0]
        result = ic_frame.merge(daily_regime, on=cfg.date_col, how="left")
        result = result[columns].dropna(subset=["ic", "rank_ic"], how="all")
        return result, self._summarize_segmented_ic(
            result.dropna(subset=["market_regime"]),
            ["window", "market_regime"],
        )

    def annual_ic_test(self, ic: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Summarize IC direction and stability for each calendar year."""
        cfg = self.config
        frame = ic.copy() if ic is not None else self.ic_test()[0]
        if frame.empty:
            return pd.DataFrame()
        frame["year"] = pd.to_datetime(frame[cfg.date_col]).dt.year
        return self._summarize_segmented_ic(frame, ["window", "year"])

    def _cross_sectional_ic(
        self,
        pair: pd.DataFrame,
        return_col: str,
    ) -> tuple[float, float]:
        cfg = self.config
        if (
            len(pair) < cfg.min_periods
            or pair["factor_processed"].nunique() < 2
            or pair[return_col].nunique() < 2
        ):
            return np.nan, np.nan
        return (
            pair["factor_processed"].corr(pair[return_col], method="pearson"),
            pair["factor_processed"].corr(pair[return_col], method="spearman"),
        )

    @staticmethod
    def _summarize_segmented_ic(
        frame: pd.DataFrame,
        group_cols: Sequence[str],
    ) -> pd.DataFrame:
        columns = [
            *group_cols,
            "ic_mean",
            "ic_std",
            "icir",
            "ic_win_rate",
            "rank_ic_mean",
            "rank_ic_std",
            "rank_icir",
            "rank_ic_win_rate",
            "observation_count",
        ]
        if frame.empty:
            return pd.DataFrame(columns=columns)
        rows: list[dict[str, Any]] = []
        for keys, part in frame.groupby(list(group_cols), dropna=False):
            key_values = keys if isinstance(keys, tuple) else (keys,)
            ic_values = pd.to_numeric(part["ic"], errors="coerce").dropna()
            rank_values = pd.to_numeric(part["rank_ic"], errors="coerce").dropna()
            ic_mean = float(ic_values.mean()) if not ic_values.empty else np.nan
            ic_std = float(ic_values.std(ddof=1)) if len(ic_values) > 1 else np.nan
            rank_mean = float(rank_values.mean()) if not rank_values.empty else np.nan
            rank_std = float(rank_values.std(ddof=1)) if len(rank_values) > 1 else np.nan
            row = dict(zip(group_cols, key_values))
            row.update(
                {
                    "ic_mean": ic_mean,
                    "ic_std": ic_std,
                    "icir": _icir(ic_mean, ic_std),
                    "ic_win_rate": float((ic_values > 0).mean())
                    if not ic_values.empty
                    else np.nan,
                    "rank_ic_mean": rank_mean,
                    "rank_ic_std": rank_std,
                    "rank_icir": _icir(rank_mean, rank_std),
                    "rank_ic_win_rate": float((rank_values > 0).mean())
                    if not rank_values.empty
                    else np.nan,
                    "observation_count": int(rank_values.count()),
                }
            )
            rows.append(row)
        return pd.DataFrame(rows, columns=columns)

    def _market_cap_bucket_label(self, bucket: int) -> str:
        groups = self.config.market_cap_groups
        if groups == 3:
            return {1: "small", 2: "mid", 3: "large"}[bucket]
        return f"cap_q{bucket}_of_{groups}"

    def group_return_test(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Group stocks by daily factor quantile and compute average forward returns."""
        df = self._prepared()
        cfg = self.config
        rows = []
        for window in cfg.forward_return_windows:
            ret_col = self._return_col(window)
            for date, daily in df.groupby(cfg.date_col):
                valid = daily[["factor_processed", ret_col]].dropna().copy()
                if len(valid) < cfg.groups:
                    continue
                valid["group"] = self._assign_groups(valid["factor_processed"], cfg.groups)
                for group_id, part in valid.dropna(subset=["group"]).groupby("group"):
                    rows.append(
                        {
                            cfg.date_col: date,
                            "window": int(window),
                            "group": int(group_id),
                            "mean_forward_return": part[ret_col].mean(),
                            "count": int(part[ret_col].count()),
                        }
                    )
        group_df = pd.DataFrame(rows)
        if group_df.empty:
            return group_df, pd.DataFrame(columns=["window", "top_bottom_return", "monotonic"])

        summary_rows = []
        for window, part in group_df.groupby("window"):
            avg_by_group = part.groupby("group")["mean_forward_return"].mean().sort_index()
            top = avg_by_group.iloc[-1] if not avg_by_group.empty else np.nan
            bottom = avg_by_group.iloc[0] if not avg_by_group.empty else np.nan
            summary_rows.append(
                {
                    "window": int(window),
                    "top_group": int(avg_by_group.index[-1]),
                    "bottom_group": int(avg_by_group.index[0]),
                    "top_return": top,
                    "bottom_return": bottom,
                    "top_bottom_return": top - bottom,
                    "monotonic": bool(_monotonic(avg_by_group.tolist())),
                }
            )
        return group_df, pd.DataFrame(summary_rows)

    def top_n_return_test(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Compute long-only forward returns for fixed TopN factor buckets."""
        df = self._prepared()
        cfg = self.config
        rows = []
        top_counts = tuple(dict.fromkeys(int(value) for value in cfg.top_n_counts))
        for window in cfg.forward_return_windows:
            ret_col = self._return_col(window)
            for date, daily in df.groupby(cfg.date_col):
                valid = (
                    daily[[cfg.symbol_col, "factor_processed", ret_col]]
                    .dropna()
                    .drop_duplicates(cfg.symbol_col, keep="last")
                    .copy()
                )
                if valid.empty:
                    continue
                valid = valid.sort_values(
                    ["factor_processed", cfg.symbol_col],
                    ascending=[False, True],
                    kind="mergesort",
                )
                for top_n in top_counts:
                    selected = valid.head(top_n)
                    if selected.empty:
                        continue
                    rows.append(
                        {
                            cfg.date_col: date,
                            "window": int(window),
                            "top_n": int(top_n),
                            "mean_forward_return": selected[ret_col].mean(),
                            "selected_count": int(selected[ret_col].count()),
                        }
                    )
        top_n_df = pd.DataFrame(rows)
        if top_n_df.empty:
            return top_n_df, pd.DataFrame(
                columns=[
                    "window",
                    "top_n",
                    "mean_forward_return",
                    "annualized_return",
                    "sharpe",
                    "observation_count",
                    "average_selected_count",
                ]
            )

        summary_rows = []
        for (window, top_n), part in top_n_df.groupby(["window", "top_n"]):
            returns = pd.to_numeric(part["mean_forward_return"], errors="coerce").dropna()
            summary_rows.append(
                {
                    "window": int(window),
                    "top_n": int(top_n),
                    "mean_forward_return": float(returns.mean()) if not returns.empty else np.nan,
                    "annualized_return": _annualized_return(
                        returns,
                        return_horizon_days=int(window),
                    ),
                    "sharpe": _sharpe(
                        returns,
                        return_horizon_days=int(window),
                    ),
                    "observation_count": int(returns.count()),
                    "average_selected_count": float(part["selected_count"].mean()),
                }
            )
        return top_n_df, pd.DataFrame(summary_rows)

    def tradable_top_n_test(self) -> pd.DataFrame:
        """Build daily long-only tradable NAVs for fixed TopN factor buckets."""
        cfg = self.config
        return self._tradable_long_only_test(
            bucket_name="top_n",
            bucket_values=tuple(dict.fromkeys(int(value) for value in cfg.top_n_counts)),
            nav_col="tradable_top_n_cum_nav",
            selector=self._select_top_n_symbols,
        )

    def tradable_top_quantile_test(self) -> pd.DataFrame:
        """Build a daily long-only tradable NAV for the highest factor quantile."""
        cfg = self.config
        return self._tradable_long_only_test(
            bucket_name="top_quantile",
            bucket_values=(1.0 / int(cfg.groups),),
            nav_col="tradable_top_quantile_cum_nav",
            selector=self._select_top_quantile_symbols,
        )

    def tradable_bottom_n_test(self) -> pd.DataFrame:
        """Build daily long-only tradable NAVs for fixed lowest-factor buckets."""
        cfg = self.config
        return self._tradable_long_only_test(
            bucket_name="bottom_n",
            bucket_values=tuple(dict.fromkeys(int(value) for value in cfg.top_n_counts)),
            nav_col="tradable_bottom_n_cum_nav",
            selector=self._select_bottom_n_symbols,
        )

    def tradable_bottom_quantile_test(self) -> pd.DataFrame:
        """Build a daily long-only tradable NAV for the lowest factor quantile."""
        cfg = self.config
        return self._tradable_long_only_test(
            bucket_name="bottom_quantile",
            bucket_values=(1.0 / int(cfg.groups),),
            nav_col="tradable_bottom_quantile_cum_nav",
            selector=self._select_bottom_quantile_symbols,
        )

    def _tradable_long_only_test(
        self,
        *,
        bucket_name: str,
        bucket_values: Sequence[int | float],
        nav_col: str,
        selector: Any,
    ) -> pd.DataFrame:
        df = self._prepared()
        cfg = self.config
        if self.valuation_data is None:
            raise ValueError(
                "long-only tradable NAV requires either a close column "
                f"('{cfg.close_col}') or daily return column ('{cfg.daily_return_col}')"
            )

        selection_columns = [
            cfg.date_col,
            cfg.symbol_col,
            "factor_processed",
            "_eligible",
            "_is_tradeable",
            "_is_suspended",
            "_is_limit_up",
            "_is_limit_down",
        ]
        state_by_date = {
            pd.Timestamp(date): daily.drop_duplicates(
                cfg.symbol_col, keep="last"
            ).set_index(cfg.symbol_col)
            for date, daily in df[selection_columns].groupby(cfg.date_col, sort=True)
        }
        valuation_by_date = self._valuation_states_by_date()
        valuation_dates = sorted(valuation_by_date)
        candidate_cache: dict[tuple[pd.Timestamp, int | float], tuple[str, ...]] = {}
        rows: list[dict[str, Any]] = []
        for window in cfg.forward_return_windows:
            holding_days = int(window)
            for bucket_value in bucket_values:
                active_cohorts: list[_LongOnlyCohort] = []
                nav = 1.0
                started = False
                for date_index in range(len(valuation_dates) - 1):
                    signal_date = valuation_dates[date_index]
                    return_date = valuation_dates[date_index + 1]
                    daily_selection = state_by_date.get(signal_date)
                    entry_turnover = 0.0
                    entry_cost = 0.0
                    blocked_entries = 0
                    selected_count = 0
                    candidate_count = 0
                    if daily_selection is not None and len(active_cohorts) < holding_days:
                        candidate_key = (signal_date, bucket_value)
                        candidate_symbols = candidate_cache.get(candidate_key)
                        if candidate_symbols is None:
                            valid = (
                                daily_selection[["factor_processed"]]
                                .dropna()
                                .reset_index()
                            )
                            candidate_symbols = tuple(
                                str(symbol) for symbol in selector(valid, bucket_value)
                            )
                            candidate_cache[candidate_key] = candidate_symbols
                        (
                            cohort,
                            blocked_entries,
                            selected_count,
                            candidate_count,
                        ) = self._build_long_only_cohort(
                            daily_selection,
                            candidate_symbols=candidate_symbols,
                            holding_days=holding_days,
                        )
                        if cohort is not None:
                            active_cohorts.append(cohort)
                            started = True
                            long_notional = sum(cohort.long_weights.values()) / holding_days
                            entry_turnover = long_notional
                            entry_cost = long_notional * (
                                cfg.commission_rate + cfg.slippage_rate
                            )

                    if not started:
                        continue

                    realized_state = valuation_by_date[return_date]
                    realized_returns = realized_state[cfg.daily_return_col]
                    gross_return = sum(
                        self._weighted_holding_return(realized_returns, cohort.long_weights)
                        for cohort in active_cohorts
                    ) / holding_days

                    exit_long_notional = 0.0
                    blocked_exits = 0
                    surviving_cohorts: list[_LongOnlyCohort] = []
                    for cohort in active_cohorts:
                        cohort.remaining_days -= 1
                        if cohort.remaining_days <= 0:
                            for symbol in tuple(cohort.long_weights):
                                if self._can_exit(realized_state, symbol, side="long"):
                                    exit_long_notional += (
                                        cohort.long_weights.pop(symbol) / holding_days
                                    )
                                else:
                                    blocked_exits += 1
                        if cohort.long_weights:
                            surviving_cohorts.append(cohort)
                    active_cohorts = surviving_cohorts

                    exit_turnover = exit_long_notional
                    exit_cost = exit_long_notional * (
                        cfg.commission_rate + cfg.slippage_rate + cfg.stamp_tax_rate
                    )
                    transaction_cost = entry_cost + exit_cost
                    net_return = gross_return - transaction_cost
                    nav *= 1.0 + net_return
                    row: dict[str, Any] = {
                        cfg.date_col: return_date,
                        "window": holding_days,
                        bucket_name: bucket_value,
                        "gross_return": gross_return,
                        "transaction_cost": transaction_cost,
                        "net_return": net_return,
                        "entry_turnover": entry_turnover,
                        "exit_turnover": exit_turnover,
                        "blocked_entries": blocked_entries,
                        "blocked_exits": blocked_exits,
                        "active_cohorts": len(active_cohorts),
                        "selected_count": selected_count,
                        "candidate_count": candidate_count,
                        nav_col: nav,
                    }
                    rows.append(row)
        out = pd.DataFrame(rows)
        if out.empty:
            return out
        metrics = []
        for keys, part in out.groupby(["window", bucket_name]):
            window, bucket_value = keys
            gross_returns = pd.to_numeric(part["gross_return"], errors="coerce").dropna()
            net_returns = pd.to_numeric(part["net_return"], errors="coerce").dropna()
            nav_series = part[nav_col].dropna()
            metrics.append(
                {
                    "window": int(window),
                    bucket_name: bucket_value,
                    "gross_period_return": (1.0 + gross_returns).prod() - 1.0,
                    "gross_annualized_return": _annualized_return(gross_returns),
                    "gross_sharpe": _sharpe(gross_returns),
                    "net_period_return": (1.0 + net_returns).prod() - 1.0,
                    "net_annualized_return": _annualized_return(net_returns),
                    "net_sharpe": _sharpe(net_returns),
                    "annualized_return": _annualized_return(net_returns),
                    "max_drawdown": _max_drawdown(nav_series),
                    "sharpe": _sharpe(net_returns),
                }
            )
        return out.merge(pd.DataFrame(metrics), on=["window", bucket_name], how="left")

    def _build_long_only_cohort(
        self,
        state: pd.DataFrame,
        *,
        candidate_symbols: Sequence[str],
        holding_days: int,
    ) -> tuple[Optional[_LongOnlyCohort], int, int, int]:
        if not candidate_symbols:
            return None, 0, 0, 0
        long_symbols = tuple(
            symbol for symbol in candidate_symbols if self._can_enter(state, symbol, side="long")
        )
        blocked_entries = len(candidate_symbols) - len(long_symbols)
        if not long_symbols:
            return None, blocked_entries, 0, len(candidate_symbols)
        return (
            _LongOnlyCohort(
                long_weights={symbol: 1.0 / len(long_symbols) for symbol in long_symbols},
                remaining_days=holding_days,
            ),
            blocked_entries,
            len(long_symbols),
            len(candidate_symbols),
        )

    def _valuation_states_by_date(self) -> dict[pd.Timestamp, pd.DataFrame]:
        if self._valuation_by_date_cache is not None:
            return self._valuation_by_date_cache
        if self.valuation_data is None:
            return {}
        cfg = self.config
        self._valuation_by_date_cache = {
            pd.Timestamp(date): daily.drop_duplicates(
                cfg.symbol_col, keep="last"
            ).set_index(cfg.symbol_col)
            for date, daily in self.valuation_data.groupby(cfg.date_col, sort=True)
        }
        return self._valuation_by_date_cache

    def _select_top_n_symbols(self, valid: pd.DataFrame, top_n: int | float) -> tuple[str, ...]:
        cfg = self.config
        selected = valid.sort_values(
            ["factor_processed", cfg.symbol_col],
            ascending=[False, True],
            kind="mergesort",
        ).head(int(top_n))
        return tuple(selected[cfg.symbol_col].astype(str))

    def _select_top_quantile_symbols(
        self,
        valid: pd.DataFrame,
        _: int | float,
    ) -> tuple[str, ...]:
        cfg = self.config
        if len(valid) < cfg.groups:
            return ()
        grouped = valid.copy()
        grouped["group"] = self._assign_groups(grouped["factor_processed"], cfg.groups)
        selected = grouped.loc[grouped["group"] == cfg.groups]
        return tuple(selected[cfg.symbol_col].astype(str))

    def _select_bottom_n_symbols(
        self,
        valid: pd.DataFrame,
        bottom_n: int | float,
    ) -> tuple[str, ...]:
        cfg = self.config
        selected = valid.sort_values(
            ["factor_processed", cfg.symbol_col],
            ascending=[True, True],
            kind="mergesort",
        ).head(int(bottom_n))
        return tuple(selected[cfg.symbol_col].astype(str))

    def _select_bottom_quantile_symbols(
        self,
        valid: pd.DataFrame,
        _: int | float,
    ) -> tuple[str, ...]:
        cfg = self.config
        if len(valid) < cfg.groups:
            return ()
        grouped = valid.copy()
        grouped["group"] = self._assign_groups(grouped["factor_processed"], cfg.groups)
        selected = grouped.loc[grouped["group"] == 1]
        return tuple(selected[cfg.symbol_col].astype(str))

    def long_short_test(self) -> pd.DataFrame:
        """Build explicitly statistical NAVs from forward-return observations."""
        df = self._prepared()
        cfg = self.config
        rows = []
        for window in cfg.forward_return_windows:
            ret_col = self._return_col(window)
            stat_nav = 1.0
            for date, daily in df.groupby(cfg.date_col, sort=True):
                valid = daily[["factor_processed", ret_col]].dropna().copy()
                if len(valid) < cfg.groups:
                    continue
                valid["group"] = self._assign_groups(valid["factor_processed"], cfg.groups)
                bottom = valid.loc[valid["group"] == 1, ret_col].mean()
                top = valid.loc[valid["group"] == cfg.groups, ret_col].mean()
                stat_return = float(top - bottom)
                stat_nav *= 1.0 + stat_return
                rows.append(
                    {
                        cfg.date_col: date,
                        "window": int(window),
                        "long_forward_return": top,
                        "short_forward_return": bottom,
                        "stat_return": stat_return,
                        "stat_cum_nav": stat_nav,
                    }
                )
        out = pd.DataFrame(rows)
        if out.empty:
            return out
        metrics = []
        for window, part in out.groupby("window"):
            returns = part["stat_return"].dropna()
            metrics.append(
                {
                    "window": int(window),
                    "annualized_return": _annualized_return(
                        returns,
                        return_horizon_days=int(window),
                    ),
                    "max_drawdown": _max_drawdown(part["stat_cum_nav"].dropna()),
                    "sharpe": _sharpe(
                        returns,
                        return_horizon_days=int(window),
                    ),
                }
            )
        return out.merge(pd.DataFrame(metrics), on="window", how="left")

    def tradable_long_short_test(self) -> pd.DataFrame:
        """Build a daily long-short NAV from staggered fixed-horizon cohorts."""
        df = self._prepared()
        cfg = self.config
        if self.valuation_data is None:
            raise ValueError(
                "long-short NAV requires either a close column "
                f"('{cfg.close_col}') or daily return column ('{cfg.daily_return_col}')"
            )

        state_by_date = {
            pd.Timestamp(date): daily
            for date, daily in df.groupby(cfg.date_col, sort=True)
        }
        valuation_by_date = self._valuation_states_by_date()
        valuation_dates = sorted(valuation_by_date)
        rows = []
        for window in cfg.forward_return_windows:
            active_cohorts: list[_TradableCohort] = []
            nav = 1.0
            started = False
            for date_index in range(len(valuation_dates) - 1):
                signal_date = valuation_dates[date_index]
                return_date = valuation_dates[date_index + 1]
                daily_selection = state_by_date.get(signal_date)
                entry_turnover = 0.0
                entry_cost = 0.0
                blocked_long_entries = 0
                blocked_short_entries = 0
                if daily_selection is not None:
                    cohort, blocked_long_entries, blocked_short_entries = self._build_tradable_cohort(
                        daily_selection,
                        holding_days=int(window),
                    )
                    if cohort is not None:
                        active_cohorts.append(cohort)
                        started = True
                        long_notional = sum(cohort.long_weights.values()) / int(window)
                        short_notional = sum(cohort.short_weights.values()) / int(window)
                        entry_turnover = long_notional + short_notional
                        entry_cost = (
                            long_notional * (cfg.commission_rate + cfg.slippage_rate)
                            + short_notional
                            * (cfg.commission_rate + cfg.slippage_rate + cfg.stamp_tax_rate)
                        )

                if not started:
                    continue

                realized_state = valuation_by_date[return_date]
                realized_returns = realized_state[cfg.daily_return_col]
                long_return = sum(
                    self._weighted_holding_return(realized_returns, cohort.long_weights)
                    for cohort in active_cohorts
                ) / int(window)
                short_return = sum(
                    self._weighted_holding_return(realized_returns, cohort.short_weights)
                    for cohort in active_cohorts
                ) / int(window)
                gross_return = long_return - short_return

                exit_long_notional = 0.0
                exit_short_notional = 0.0
                blocked_long_exits = 0
                blocked_short_exits = 0
                surviving_cohorts: list[_TradableCohort] = []
                for cohort in active_cohorts:
                    cohort.remaining_days -= 1
                    if cohort.remaining_days <= 0:
                        for symbol in tuple(cohort.long_weights):
                            if self._can_exit(realized_state, symbol, side="long"):
                                exit_long_notional += cohort.long_weights.pop(symbol) / int(window)
                            else:
                                blocked_long_exits += 1
                        for symbol in tuple(cohort.short_weights):
                            if self._can_exit(realized_state, symbol, side="short"):
                                exit_short_notional += cohort.short_weights.pop(symbol) / int(window)
                            else:
                                blocked_short_exits += 1
                    if cohort.long_weights or cohort.short_weights:
                        surviving_cohorts.append(cohort)
                active_cohorts = surviving_cohorts

                exit_turnover = exit_long_notional + exit_short_notional
                exit_cost = (
                    exit_long_notional
                    * (cfg.commission_rate + cfg.slippage_rate + cfg.stamp_tax_rate)
                    + exit_short_notional * (cfg.commission_rate + cfg.slippage_rate)
                )
                transaction_cost = entry_cost + exit_cost
                net_return = gross_return - transaction_cost
                nav *= 1.0 + net_return
                rows.append(
                    {
                        cfg.date_col: return_date,
                        "window": int(window),
                        "long_return": long_return,
                        "short_return": short_return,
                        "gross_return": gross_return,
                        "transaction_cost": transaction_cost,
                        "net_return": net_return,
                        "entry_turnover": entry_turnover,
                        "exit_turnover": exit_turnover,
                        "blocked_long_entries": blocked_long_entries,
                        "blocked_short_entries": blocked_short_entries,
                        "blocked_long_exits": blocked_long_exits,
                        "blocked_short_exits": blocked_short_exits,
                        "active_cohorts": len(active_cohorts),
                        "tradable_cum_nav": nav,
                    }
                )
        out = pd.DataFrame(rows)
        if out.empty:
            return out
        metrics = []
        for window, part in out.groupby("window"):
            returns = part["net_return"].dropna()
            nav_series = part["tradable_cum_nav"].dropna()
            metrics.append(
                {
                    "window": int(window),
                    "annualized_return": _annualized_return(returns),
                    "max_drawdown": _max_drawdown(nav_series),
                    "sharpe": _sharpe(returns),
                }
            )
        return out.merge(pd.DataFrame(metrics), on="window", how="left")

    def _build_tradable_cohort(
        self,
        daily: pd.DataFrame,
        *,
        holding_days: int,
    ) -> tuple[Optional[_TradableCohort], int, int]:
        cfg = self.config
        valid = (
            daily[[cfg.symbol_col, "factor_processed"]]
            .dropna()
            .drop_duplicates(cfg.symbol_col, keep="last")
            .copy()
        )
        if len(valid) < cfg.groups:
            return None, 0, 0
        valid["group"] = self._assign_groups(valid["factor_processed"], cfg.groups)
        long_candidates = valid.loc[valid["group"] == cfg.groups, cfg.symbol_col].astype(str)
        short_candidates = valid.loc[valid["group"] == 1, cfg.symbol_col].astype(str)
        state = daily.drop_duplicates(cfg.symbol_col, keep="last").set_index(cfg.symbol_col)
        long_symbols = tuple(
            symbol for symbol in long_candidates if self._can_enter(state, symbol, side="long")
        )
        short_symbols = tuple(
            symbol for symbol in short_candidates if self._can_enter(state, symbol, side="short")
        )
        blocked_long = len(long_candidates) - len(long_symbols)
        blocked_short = len(short_candidates) - len(short_symbols)
        if not long_symbols or not short_symbols:
            return None, blocked_long, blocked_short
        return (
            _TradableCohort(
                long_weights={symbol: 1.0 / len(long_symbols) for symbol in long_symbols},
                short_weights={symbol: 1.0 / len(short_symbols) for symbol in short_symbols},
                remaining_days=holding_days,
            ),
            blocked_long,
            blocked_short,
        )

    @staticmethod
    def _weighted_holding_return(
        returns_by_symbol: pd.Series,
        weights: dict[str, float],
    ) -> float:
        if not weights:
            return 0.0
        values = pd.to_numeric(
            returns_by_symbol.reindex(list(weights)),
            errors="coerce",
        ).replace([np.inf, -np.inf], np.nan)
        weight_series = pd.Series(weights, dtype="float64")
        return float((values.fillna(0.0) * weight_series).sum())

    @staticmethod
    def _can_enter(state: pd.DataFrame, symbol: str, *, side: str) -> bool:
        if symbol not in state.index:
            return False
        row = state.loc[symbol]
        if not bool(row.get("_eligible", False)):
            return False
        if not bool(row.get("_is_tradeable", False)) or bool(row.get("_is_suspended", False)):
            return False
        if side == "long":
            return not bool(row.get("_is_limit_up", False))
        return not bool(row.get("_is_limit_down", False))

    @staticmethod
    def _can_exit(state: pd.DataFrame, symbol: str, *, side: str) -> bool:
        if symbol not in state.index:
            return False
        row = state.loc[symbol]
        if not bool(row.get("_is_tradeable", False)) or bool(row.get("_is_suspended", False)):
            return False
        if side == "long":
            return not bool(row.get("_is_limit_down", False))
        return not bool(row.get("_is_limit_up", False))

    def neutralized_ic_test(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Compute IC after point-in-time industry and log-cap neutralization."""
        df = self._prepared()
        cfg = self.config
        has_industry = bool(cfg.industry_col and cfg.industry_col in df.columns)
        has_market_cap = "_market_cap_lagged" in df.columns
        rows = []
        for window in cfg.forward_return_windows:
            ret_col = self._return_col(window)
            for date, daily in df.groupby(cfg.date_col, sort=True):
                columns = ["factor_processed", ret_col]
                if has_industry and cfg.industry_col:
                    columns.append(cfg.industry_col)
                if has_market_cap:
                    columns.append("_market_cap_lagged")
                valid = daily[columns].dropna(subset=["factor_processed", ret_col]).copy()
                residual = valid["factor_processed"].astype("float64")
                controls: list[str] = []
                if has_industry and cfg.industry_col:
                    industry_valid = valid[cfg.industry_col].notna()
                    residual.loc[industry_valid] = residual.loc[industry_valid] - residual.loc[
                        industry_valid
                    ].groupby(valid.loc[industry_valid, cfg.industry_col]).transform("mean")
                    controls.append("industry")
                if has_market_cap:
                    cap = pd.to_numeric(valid["_market_cap_lagged"], errors="coerce")
                    cap_valid = cap > 0
                    if int(cap_valid.sum()) >= cfg.min_periods:
                        x = np.column_stack(
                            [np.ones(int(cap_valid.sum())), np.log(cap.loc[cap_valid].to_numpy())]
                        )
                        y = residual.loc[cap_valid].to_numpy()
                        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
                        residual.loc[cap_valid] = y - x @ beta
                        residual.loc[~cap_valid] = np.nan
                        controls.append("log_market_cap")
                pair = pd.DataFrame(
                    {"neutralized_factor": residual, "forward_return": valid[ret_col]}
                ).dropna()
                if not controls:
                    status = "skipped"
                    neutral_ic = np.nan
                    neutral_rank_ic = np.nan
                elif (
                    len(pair) < cfg.min_periods
                    or pair["neutralized_factor"].nunique() < 2
                    or pair["forward_return"].nunique() < 2
                ):
                    status = "insufficient_data"
                    neutral_ic = np.nan
                    neutral_rank_ic = np.nan
                else:
                    status = "ok"
                    neutral_ic = pair["neutralized_factor"].corr(pair["forward_return"])
                    neutral_rank_ic = pair["neutralized_factor"].corr(
                        pair["forward_return"], method="spearman"
                    )
                rows.append(
                    {
                        cfg.date_col: date,
                        "window": int(window),
                        "neutralized_ic": neutral_ic,
                        "neutralized_rank_ic": neutral_rank_ic,
                        "controls": "+".join(controls) if controls else "none",
                        "status": status,
                    }
                )
        out = pd.DataFrame(rows)
        summary_rows = []
        if not out.empty:
            for window, part in out.groupby("window"):
                ic_values = part["neutralized_ic"].dropna()
                rank_values = part["neutralized_rank_ic"].dropna()
                summary_rows.append(
                    {
                        "window": int(window),
                        "neutralized_ic_mean": ic_values.mean() if not ic_values.empty else None,
                        "neutralized_icir": _icir(
                            float(ic_values.mean()) if not ic_values.empty else None,
                            float(ic_values.std(ddof=1)) if len(ic_values) > 1 else None,
                        ),
                        "neutralized_rank_ic_mean": rank_values.mean()
                        if not rank_values.empty
                        else None,
                        "neutralized_rank_icir": _icir(
                            float(rank_values.mean()) if not rank_values.empty else None,
                            float(rank_values.std(ddof=1)) if len(rank_values) > 1 else None,
                        ),
                        "count": int(rank_values.count()),
                    }
                )
        return out, pd.DataFrame(summary_rows)

    def universe_filter_test(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Report point-in-time eligibility filters and unavailable inputs."""
        df = self._prepared()
        cfg = self.config
        rows = []
        for date, daily in df.groupby(cfg.date_col, sort=True):
            rows.append(
                {
                    cfg.date_col: date,
                    "total_count": int(daily[cfg.symbol_col].nunique()),
                    "eligible_count": int(daily.loc[daily["_eligible"], cfg.symbol_col].nunique()),
                    "excluded_untradeable": int(daily["_exclude_untradeable"].sum()),
                    "excluded_st": int(daily["_exclude_st"].sum()),
                    "excluded_new_stock": int(daily["_exclude_new_stock"].sum()),
                    "excluded_low_liquidity": int(daily["_exclude_low_liquidity"].sum()),
                }
            )
        enabled = {
            "tradeable": True,
            "suspended": True,
            "limit_up": True,
            "limit_down": True,
            "st": cfg.exclude_st,
            "listing_age": cfg.min_listing_days > 0,
            "liquidity": cfg.min_liquidity > 0,
            "industry": True,
            "market_cap": True,
            "market_regime": True,
        }
        status = pd.DataFrame(
            [
                {
                    "check": name,
                    "enabled": bool(enabled.get(name, False)),
                    "available": bool(available),
                    "status": "active"
                    if enabled.get(name, False) and available
                    else ("missing_input" if enabled.get(name, False) else "disabled"),
                }
                for name, available in self.filter_availability.items()
            ]
        )
        return pd.DataFrame(rows), status

    def annual_performance_test(
        self,
        stat_long_short: pd.DataFrame,
        tradable_long_short: pd.DataFrame,
    ) -> pd.DataFrame:
        """Summarize statistical and tradable performance by calendar year."""
        cfg = self.config
        rows: list[dict[str, Any]] = []
        definitions = (
            ("stat", stat_long_short, "stat_return", "stat_cum_nav"),
            ("tradable", tradable_long_short, "net_return", "tradable_cum_nav"),
        )
        for nav_type, frame, return_col, _ in definitions:
            if frame.empty:
                continue
            work = frame.copy()
            work["year"] = pd.to_datetime(work[cfg.date_col]).dt.year
            for (window, year), part in work.groupby(["window", "year"]):
                returns = pd.to_numeric(part[return_col], errors="coerce").dropna()
                if returns.empty:
                    continue
                local_nav = (1.0 + returns).cumprod()
                horizon = int(window) if nav_type == "stat" else 1
                rows.append(
                    {
                        "nav_type": nav_type,
                        "window": int(window),
                        "year": int(year),
                        "observation_count": int(returns.count()),
                        "period_return": float((1.0 + returns).prod() - 1.0),
                        "annualized_return": _annualized_return(
                            returns,
                            return_horizon_days=horizon,
                        ),
                        "max_drawdown": _max_drawdown(local_nav),
                        "sharpe": _sharpe(returns, return_horizon_days=horizon),
                    }
                )
        return pd.DataFrame(rows)

    def annual_long_only_performance_test(
        self,
        *,
        tradable_top_n: pd.DataFrame,
        tradable_top_quantile: pd.DataFrame,
        tradable_bottom_n: pd.DataFrame,
        tradable_bottom_quantile: pd.DataFrame,
    ) -> pd.DataFrame:
        """Report annual gross and after-cost results for both long-only directions."""
        cfg = self.config
        rows: list[dict[str, Any]] = []
        definitions = (
            ("top_n", "high_factor", tradable_top_n, "top_n"),
            ("top_quantile", "high_factor", tradable_top_quantile, "top_quantile"),
            ("bottom_n", "low_factor", tradable_bottom_n, "bottom_n"),
            (
                "bottom_quantile",
                "low_factor",
                tradable_bottom_quantile,
                "bottom_quantile",
            ),
        )
        for selection, side, frame, bucket_col in definitions:
            if frame.empty:
                continue
            work = frame.copy()
            work["year"] = pd.to_datetime(work[cfg.date_col]).dt.year
            for (window, bucket, year), part in work.groupby(
                ["window", bucket_col, "year"]
            ):
                gross = pd.to_numeric(part["gross_return"], errors="coerce").dropna()
                net = pd.to_numeric(part["net_return"], errors="coerce").dropna()
                if gross.empty and net.empty:
                    continue
                rows.append(
                    {
                        "selection": selection,
                        "side": side,
                        "window": int(window),
                        "bucket": bucket,
                        "year": int(year),
                        "observation_count": int(max(gross.count(), net.count())),
                        "gross_period_return": (1.0 + gross).prod() - 1.0
                        if not gross.empty
                        else np.nan,
                        "gross_annualized_return": _annualized_return(gross),
                        "gross_sharpe": _sharpe(gross),
                        "net_period_return": (1.0 + net).prod() - 1.0
                        if not net.empty
                        else np.nan,
                        "net_annualized_return": _annualized_return(net),
                        "net_sharpe": _sharpe(net),
                        "transaction_cost": float(
                            pd.to_numeric(
                                part["transaction_cost"], errors="coerce"
                            ).fillna(0.0).sum()
                        ),
                    }
                )
        return pd.DataFrame(rows)

    def horizon_effectiveness_test(
        self,
        *,
        ic_summary: pd.DataFrame,
        tradable_top_quantile: pd.DataFrame,
        tradable_bottom_quantile: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compare high/low long-only profitability for every research horizon."""
        rows: list[dict[str, Any]] = []
        for window in self.config.forward_return_windows:
            ic_part = ic_summary.loc[ic_summary["window"].eq(int(window))]
            rank_ic_mean = (
                float(ic_part.iloc[0]["rank_ic_mean"])
                if not ic_part.empty and pd.notna(ic_part.iloc[0]["rank_ic_mean"])
                else np.nan
            )
            if pd.isna(rank_ic_mean) or abs(rank_ic_mean) < 1e-12:
                direction = "flat"
                preferred = "undetermined"
            elif rank_ic_mean > 0:
                direction = "positive"
                preferred = "high_factor"
            else:
                direction = "negative"
                preferred = "low_factor"

            def metrics(frame: pd.DataFrame) -> dict[str, float]:
                if frame.empty or "window" not in frame.columns:
                    part = pd.DataFrame()
                else:
                    part = frame.loc[frame["window"].eq(int(window))]
                if part.empty:
                    return {
                        "gross_annualized_return": np.nan,
                        "gross_sharpe": np.nan,
                        "net_annualized_return": np.nan,
                        "net_sharpe": np.nan,
                    }
                first = part.iloc[0]
                return {
                    key: float(first[key]) if pd.notna(first.get(key)) else np.nan
                    for key in (
                        "gross_annualized_return",
                        "gross_sharpe",
                        "net_annualized_return",
                        "net_sharpe",
                    )
                }

            high = metrics(tradable_top_quantile)
            low = metrics(tradable_bottom_quantile)
            rows.append(
                {
                    "window": int(window),
                    "rank_ic_mean": rank_ic_mean,
                    "ic_direction": direction,
                    "preferred_long_side": preferred,
                    **{f"high_{key}": value for key, value in high.items()},
                    **{f"low_{key}": value for key, value in low.items()},
                    "high_profitable_before_cost": bool(
                        pd.notna(high["gross_annualized_return"])
                        and high["gross_annualized_return"] > 0
                    ),
                    "high_profitable_after_cost": bool(
                        pd.notna(high["net_annualized_return"])
                        and high["net_annualized_return"] > 0
                    ),
                    "low_profitable_before_cost": bool(
                        pd.notna(low["gross_annualized_return"])
                        and low["gross_annualized_return"] > 0
                    ),
                    "low_profitable_after_cost": bool(
                        pd.notna(low["net_annualized_return"])
                        and low["net_annualized_return"] > 0
                    ),
                }
            )
        return pd.DataFrame(rows)

    def sample_performance_test(
        self,
        *,
        ic: pd.DataFrame,
        neutralized_ic: pd.DataFrame,
        group_return: pd.DataFrame,
        tradable_top_quantile: pd.DataFrame,
        stat_long_short: pd.DataFrame,
        tradable_long_short: pd.DataFrame,
        oos_start: pd.Timestamp,
    ) -> pd.DataFrame:
        """Report chronological in-sample and out-of-sample metrics."""
        cfg = self.config
        rows: list[dict[str, Any]] = []
        group_spread = pd.DataFrame()
        if not group_return.empty:
            pivot = group_return.pivot_table(
                index=[cfg.date_col, "window"],
                columns="group",
                values="mean_forward_return",
                aggfunc="mean",
            )
            if not pivot.empty:
                group_spread = (
                    (pivot[pivot.columns.max()] - pivot[pivot.columns.min()])
                    .rename("top_bottom_return")
                    .reset_index()
                )

        for sample_name, start, end in (
            ("in_sample", None, oos_start),
            ("out_of_sample", oos_start, None),
        ):
            for window in cfg.forward_return_windows:
                row: dict[str, Any] = {
                    "sample": sample_name,
                    "oos_start_date": oos_start,
                    "window": int(window),
                }

                def subset(frame: pd.DataFrame) -> pd.DataFrame:
                    if frame.empty:
                        return frame
                    dates = pd.to_datetime(frame[cfg.date_col])
                    mask = dates < oos_start if end is not None else dates >= oos_start
                    return frame.loc[mask & frame["window"].eq(int(window))]

                ic_part = subset(ic)
                neutral_part = subset(neutralized_ic)
                group_part = subset(group_spread)
                tradable_top_quantile_part = subset(tradable_top_quantile)
                stat_part = subset(stat_long_short)
                tradable_part = subset(tradable_long_short)
                row.update(
                    {
                        "ic_mean": ic_part["ic"].mean() if not ic_part.empty else np.nan,
                        "rank_ic_mean": ic_part["rank_ic"].mean()
                        if not ic_part.empty
                        else np.nan,
                        "neutralized_ic_mean": neutral_part["neutralized_ic"].mean()
                        if not neutral_part.empty
                        else np.nan,
                        "neutralized_rank_ic_mean": neutral_part["neutralized_rank_ic"].mean()
                        if not neutral_part.empty
                        else np.nan,
                        "top_bottom_return_mean": group_part["top_bottom_return"].mean()
                        if not group_part.empty
                        else np.nan,
                    }
                )
                stat_returns = (
                    pd.to_numeric(stat_part["stat_return"], errors="coerce").dropna()
                    if not stat_part.empty
                    else pd.Series(dtype="float64")
                )
                tradable_returns = (
                    pd.to_numeric(tradable_part["net_return"], errors="coerce").dropna()
                    if not tradable_part.empty
                    else pd.Series(dtype="float64")
                )
                tradable_top_quantile_returns = (
                    pd.to_numeric(
                        tradable_top_quantile_part["net_return"],
                        errors="coerce",
                    ).dropna()
                    if not tradable_top_quantile_part.empty
                    else pd.Series(dtype="float64")
                )
                row.update(
                    {
                        "stat_period_return": (1.0 + stat_returns).prod() - 1.0
                        if not stat_returns.empty
                        else np.nan,
                        "stat_sharpe": _sharpe(
                            stat_returns,
                            return_horizon_days=int(window),
                        ),
                        "tradable_period_return": (1.0 + tradable_returns).prod() - 1.0
                        if not tradable_returns.empty
                        else np.nan,
                        "tradable_sharpe": _sharpe(tradable_returns),
                        "tradable_max_drawdown": _max_drawdown(
                            (1.0 + tradable_returns).cumprod()
                        )
                        if not tradable_returns.empty
                        else np.nan,
                        "tradable_top_quantile_period_return": (
                            (1.0 + tradable_top_quantile_returns).prod() - 1.0
                            if not tradable_top_quantile_returns.empty
                            else np.nan
                        ),
                        "tradable_top_quantile_sharpe": _sharpe(
                            tradable_top_quantile_returns
                        ),
                        "tradable_top_quantile_max_drawdown": _max_drawdown(
                            (1.0 + tradable_top_quantile_returns).cumprod()
                        )
                        if not tradable_top_quantile_returns.empty
                        else np.nan,
                    }
                )
                rows.append(row)
        return pd.DataFrame(rows)

    def _resolve_oos_start(self) -> pd.Timestamp:
        cfg = self.config
        dates = sorted(pd.to_datetime(self._prepared()[cfg.date_col].dropna().unique()))
        if not dates:
            raise ValueError("cannot create sample split without evaluation dates")
        if cfg.oos_start_date:
            return pd.Timestamp(cfg.oos_start_date)
        split_index = min(
            max(int(len(dates) * (1.0 - cfg.oos_fraction)), 1),
            len(dates) - 1,
        )
        return pd.Timestamp(dates[split_index])

    def _add_sample_split(self, frame: pd.DataFrame, oos_start: pd.Timestamp) -> pd.DataFrame:
        if frame.empty or self.config.date_col not in frame.columns:
            return frame
        out = frame.copy()
        out["sample"] = np.where(
            pd.to_datetime(out[self.config.date_col]) >= oos_start,
            "out_of_sample",
            "in_sample",
        )
        return out

    def turnover_test(self) -> pd.DataFrame:
        """Compute top-group membership turnover and factor rank autocorrelation."""
        df = self._prepared()
        cfg = self.config
        rows = []
        prev_top: Optional[set[str]] = None
        prev_ranks: Optional[pd.Series] = None
        for date, daily in df.groupby(cfg.date_col):
            valid = daily[[cfg.symbol_col, "factor_processed"]].dropna().copy()
            if len(valid) < cfg.groups:
                continue
            valid["group"] = self._assign_groups(valid["factor_processed"], cfg.groups)
            top_symbols = set(valid.loc[valid["group"] == cfg.groups, cfg.symbol_col])
            ranks = valid.set_index(cfg.symbol_col)["factor_processed"].rank(method="average")
            if prev_top is None:
                turnover = np.nan
                rank_autocorr = np.nan
            else:
                retained = len(prev_top & top_symbols)
                turnover = 1.0 - retained / len(prev_top) if prev_top else np.nan
                common = ranks.index.intersection(prev_ranks.index) if prev_ranks is not None else []
                comparable_ranks = ranks.loc[common]
                comparable_previous = prev_ranks.loc[common] if prev_ranks is not None else pd.Series(dtype=float)
                rank_autocorr = (
                    comparable_ranks.corr(comparable_previous, method="spearman")
                    if (
                        len(common) >= cfg.min_periods
                        and comparable_ranks.nunique() >= 2
                        and comparable_previous.nunique() >= 2
                    )
                    else _rank_autocorr(prev_top, top_symbols)
                )
            rows.append(
                {
                    cfg.date_col: date,
                    "top_count": len(top_symbols),
                    "turnover": turnover,
                    "rank_autocorr": rank_autocorr,
                }
            )
            prev_top = top_symbols
            prev_ranks = ranks
        out = pd.DataFrame(rows)
        if out.empty:
            return out
        out["avg_turnover"] = out["turnover"].mean()
        out["avg_rank_autocorr"] = out["rank_autocorr"].mean()
        return out

    def exposure_test(self) -> pd.DataFrame:
        """Test optional market-cap correlation and top-group industry distribution."""
        df = self._prepared()
        cfg = self.config
        rows = []
        if "_market_cap_lagged" not in df.columns:
            rows.append({"test": "market_cap_corr", "status": "skipped", "reason": "missing_market_cap"})
        else:
            for date, daily in df.groupby(cfg.date_col):
                pair = daily[["factor_processed", "_market_cap_lagged"]].dropna()
                corr = (
                    pair["factor_processed"].corr(pair["_market_cap_lagged"], method="spearman")
                    if len(pair) >= cfg.min_periods
                    else np.nan
                )
                rows.append({"test": "market_cap_corr", cfg.date_col: date, "value": corr, "status": "ok"})

        if not cfg.industry_col or cfg.industry_col not in df.columns:
            rows.append({"test": "top_group_industry", "status": "skipped", "reason": "missing_industry"})
        else:
            for date, daily in df.groupby(cfg.date_col):
                valid = daily[[cfg.symbol_col, "factor_processed", cfg.industry_col]].dropna().copy()
                if len(valid) < cfg.groups:
                    continue
                valid["group"] = self._assign_groups(valid["factor_processed"], cfg.groups)
                top = valid[valid["group"] == cfg.groups]
                counts = top[cfg.industry_col].value_counts(normalize=True)
                for industry, weight in counts.items():
                    rows.append(
                        {
                            "test": "top_group_industry",
                            cfg.date_col: date,
                            "industry": industry,
                            "weight": weight,
                            "status": "ok",
                        }
                    )
                industry_means = valid.groupby(cfg.industry_col)["factor_processed"].mean()
                for industry, value in industry_means.items():
                    rows.append(
                        {
                            "test": "industry_factor_mean",
                            cfg.date_col: date,
                            "industry": industry,
                            "value": value,
                            "status": "ok",
                        }
                    )
        return pd.DataFrame(rows)

    def _attach_market_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Attach point-in-time bull/bear/sideways labels without future data."""
        cfg = self.config
        out = df.copy()
        supplied = bool(cfg.market_regime_col and cfg.market_regime_col in out.columns)
        if supplied and cfg.market_regime_col:
            out["_market_regime"] = self._normalize_market_regime(
                out[cfg.market_regime_col]
            )
            out["_market_trailing_return"] = np.nan
            return out

        if cfg.daily_return_col not in out.columns:
            out["_market_regime"] = pd.Series(pd.NA, index=out.index, dtype="object")
            out["_market_trailing_return"] = np.nan
            return out

        eligible = out.loc[out["_eligible"]].copy()
        market_daily_return = (
            pd.to_numeric(eligible[cfg.daily_return_col], errors="coerce")
            .groupby(eligible[cfg.date_col])
            .mean()
            .sort_index()
        )
        trailing_return = (
            (1.0 + market_daily_return.clip(lower=-0.999999))
            .rolling(
                cfg.market_regime_lookback_days,
                min_periods=cfg.market_regime_min_periods,
            )
            .apply(np.prod, raw=True)
            .sub(1.0)
            .shift(1)
        )
        regime = pd.Series("sideways", index=trailing_return.index, dtype="object")
        regime.loc[trailing_return >= cfg.bull_return_threshold] = "bull"
        regime.loc[trailing_return <= cfg.bear_return_threshold] = "bear"
        regime.loc[trailing_return.isna()] = pd.NA
        out["_market_regime"] = out[cfg.date_col].map(regime)
        out["_market_trailing_return"] = out[cfg.date_col].map(trailing_return)
        return out

    @staticmethod
    def _normalize_market_regime(values: pd.Series) -> pd.Series:
        aliases = {
            "bull": "bull",
            "bullish": "bull",
            "牛": "bull",
            "牛市": "bull",
            "bear": "bear",
            "bearish": "bear",
            "熊": "bear",
            "熊市": "bear",
            "sideways": "sideways",
            "range": "sideways",
            "neutral": "sideways",
            "震荡": "sideways",
            "震荡市": "sideways",
        }
        normalized = values.astype("string").str.strip().str.lower().map(aliases)
        unknown = values.notna() & normalized.isna()
        if bool(unknown.any()):
            invalid = sorted(values.loc[unknown].astype(str).unique())
            raise ValueError(
                "unknown market regime values: " + ", ".join(invalid)
            )
        return normalized.astype("object")

    def _ensure_forward_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        if cfg.forward_return_col and cfg.forward_return_col in df.columns and len(cfg.forward_return_windows) == 1:
            df[self._return_col(cfg.forward_return_windows[0])] = pd.to_numeric(
                df[cfg.forward_return_col], errors="coerce"
            )
            return df
        if cfg.close_col not in df.columns:
            missing = [self._return_col(window) for window in cfg.forward_return_windows if self._return_col(window) not in df.columns]
            if missing:
                raise ValueError(
                    f"missing close column '{cfg.close_col}' and forward return columns: {', '.join(missing)}"
                )
            return df

        for window in cfg.forward_return_windows:
            ret_col = self._return_col(window)
            if ret_col in df.columns:
                df[ret_col] = pd.to_numeric(df[ret_col], errors="coerce")
                continue
            df[ret_col] = (
                df.groupby(cfg.symbol_col, group_keys=False)[cfg.close_col]
                .transform(lambda s: s.shift(-int(window)) / s - 1.0)
            )
        return df

    def _return_col(self, window: int) -> str:
        return forward_return_col(self.config.forward_return_prefix, int(window))

    def _prepared(self) -> pd.DataFrame:
        if self.data is None:
            return self.prepare_data()
        return self.data

    @staticmethod
    def _winsorize_by_date(
        values: pd.Series,
        dates: pd.Series,
        *,
        lower_q: float,
        upper_q: float,
    ) -> pd.Series:
        grouped = values.groupby(dates)
        lower = grouped.transform(lambda s: s.quantile(lower_q))
        upper = grouped.transform(lambda s: s.quantile(upper_q))
        return values.clip(lower=lower, upper=upper)

    @staticmethod
    def _zscore_by_date(values: pd.Series, dates: pd.Series) -> pd.Series:
        def _zscore(s: pd.Series) -> pd.Series:
            std = s.std(ddof=1)
            if not math.isfinite(float(std)) or std == 0:
                return pd.Series(np.nan, index=s.index)
            return (s - s.mean()) / std

        return values.groupby(dates).transform(_zscore)

    @staticmethod
    def _as_bool(values: pd.Series) -> pd.Series:
        if pd.api.types.is_bool_dtype(values):
            return values.fillna(False).astype(bool)
        if pd.api.types.is_numeric_dtype(values):
            return pd.to_numeric(values, errors="coerce").fillna(0).ne(0)
        normalized = values.astype(str).str.strip().str.lower()
        return normalized.isin({"1", "true", "t", "yes", "y"})

    @staticmethod
    def _assign_groups(values: pd.Series, groups: int) -> pd.Series:
        ranks = values.rank(method="first")
        try:
            labels = pd.qcut(ranks, q=groups, labels=False, duplicates="drop") + 1
        except ValueError:
            return pd.Series(np.nan, index=values.index)
        return labels.astype(float)

    def _build_summary(
        self,
        *,
        coverage_summary: pd.DataFrame,
        distribution_summary: pd.DataFrame,
        ic_summary: pd.DataFrame,
        group_summary: pd.DataFrame,
        top_n_summary: pd.DataFrame,
        tradable_top_n: pd.DataFrame,
        tradable_top_quantile: pd.DataFrame,
        stat_long_short: pd.DataFrame,
        tradable_long_short: pd.DataFrame,
        exposure: pd.DataFrame,
        neutralized_ic_summary: pd.DataFrame,
        oos_start: pd.Timestamp,
        row_count: int,
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = [
            {"section": "meta", "metric": "factor_name", "value": self.factor_name},
            {"section": "meta", "metric": "row_count", "value": row_count},
            {"section": "meta", "metric": "groups", "value": self.config.groups},
            {
                "section": "meta",
                "metric": "top_n_counts",
                "value": ",".join(str(value) for value in self.config.top_n_counts),
            },
            {
                "section": "meta",
                "metric": "factor_lag_days",
                "value": self.config.factor_lag_days,
            },
            {"section": "meta", "metric": "oos_start_date", "value": oos_start},
        ]
        for _, row in coverage_summary.iterrows():
            rows.append({"section": "coverage", "metric": row["metric"], "value": row["value"]})
        for _, row in distribution_summary.iterrows():
            rows.append({"section": "distribution", "metric": row["metric"], "value": row["value"]})
        for _, row in ic_summary.iterrows():
            window = int(row["window"])
            for metric in ("ic_mean", "ic_std", "icir", "ic_win_rate", "rank_ic_mean", "rank_icir", "rank_ic_win_rate"):
                rows.append({"section": "ic", "window": window, "metric": metric, "value": row.get(metric)})
        for _, row in group_summary.iterrows():
            rows.append(
                {
                    "section": "group_return",
                    "window": int(row["window"]),
                    "metric": "top_bottom_return",
                    "value": row.get("top_bottom_return"),
                }
            )
            rows.append(
                {
                    "section": "group_return",
                    "window": int(row["window"]),
                    "metric": "monotonic",
                    "value": int(bool(row.get("monotonic"))),
                }
            )
        for _, row in neutralized_ic_summary.iterrows():
            for metric in (
                "neutralized_ic_mean",
                "neutralized_icir",
                "neutralized_rank_ic_mean",
                "neutralized_rank_icir",
            ):
                rows.append(
                    {
                        "section": "neutralized_ic",
                        "window": int(row["window"]),
                        "metric": metric,
                        "value": row.get(metric),
                    }
                )
        for _, row in top_n_summary.iterrows():
            top_n = int(row["top_n"])
            window = int(row["window"])
            for metric in (
                "mean_forward_return",
                "annualized_return",
                "sharpe",
                "average_selected_count",
            ):
                rows.append(
                    {
                        "section": "top_n_return",
                        "window": window,
                        "top_n": top_n,
                        "metric": metric,
                        "value": row.get(metric),
                    }
                )
        if not tradable_top_n.empty:
            latest = tradable_top_n.sort_values(
                [self.config.date_col, "window", "top_n"]
            ).groupby(["window", "top_n"]).tail(1)
            for _, row in latest.iterrows():
                for metric in (
                    "tradable_top_n_cum_nav",
                    "annualized_return",
                    "max_drawdown",
                    "sharpe",
                    "selected_count",
                ):
                    rows.append(
                        {
                            "section": "tradable_top_n",
                            "window": int(row["window"]),
                            "top_n": int(row["top_n"]),
                            "metric": metric,
                            "value": row.get(metric),
                        }
                    )
        if not tradable_top_quantile.empty:
            latest = tradable_top_quantile.sort_values(
                [self.config.date_col, "window", "top_quantile"]
            ).groupby(["window", "top_quantile"]).tail(1)
            for _, row in latest.iterrows():
                for metric in (
                    "tradable_top_quantile_cum_nav",
                    "annualized_return",
                    "max_drawdown",
                    "sharpe",
                    "selected_count",
                ):
                    rows.append(
                        {
                            "section": "tradable_top_quantile",
                            "window": int(row["window"]),
                            "top_quantile": row["top_quantile"],
                            "metric": metric,
                            "value": row.get(metric),
                        }
                    )
        if not stat_long_short.empty:
            latest = stat_long_short.sort_values(
                [self.config.date_col, "window"]
            ).groupby("window").tail(1)
            for _, row in latest.iterrows():
                for metric in ("stat_cum_nav", "annualized_return", "max_drawdown", "sharpe"):
                    rows.append(
                        {
                            "section": "stat_long_short",
                            "window": int(row["window"]),
                            "metric": metric,
                            "value": row.get(metric),
                        }
                    )
        if not tradable_long_short.empty:
            latest = tradable_long_short.sort_values(
                [self.config.date_col, "window"]
            ).groupby("window").tail(1)
            for _, row in latest.iterrows():
                for metric in (
                    "tradable_cum_nav",
                    "annualized_return",
                    "max_drawdown",
                    "sharpe",
                ):
                    rows.append(
                        {
                            "section": "tradable_long_short",
                            "window": int(row["window"]),
                            "metric": metric,
                            "value": row.get(metric),
                        }
                    )
        skipped = exposure[exposure.get("status", pd.Series(dtype=str)).eq("skipped")] if not exposure.empty else pd.DataFrame()
        for _, row in skipped.iterrows():
            rows.append({"section": "exposure", "metric": row.get("test"), "value": row.get("reason")})
        return pd.DataFrame(rows)


def build_long_factor_frame_from_raw(
    raw_data: dict[str, pd.DataFrame],
    *,
    factor_name: str,
    factor_config: Optional[dict[str, Any]] = None,
    metadata: Optional[pd.DataFrame] = None,
    benchmark_data: Optional[pd.DataFrame] = None,
    style_factor_data: Optional[pd.DataFrame] = None,
    date_col: str = "date",
    symbol_col: str = "symbol",
    close_col: str = "close",
) -> pd.DataFrame:
    """Adapt the current per-symbol raw OHLCV CSV format into long factor format.

    Supported built-in factors:
    - momentum_Nd computes close / close.shift(N) - 1.
    - registered custom factors use ``factors.custom``.
    - alpha_001 through alpha_101 use ``factors.alpha101``.
    - gtja_001 through gtja_191 use ``factors.gtja191``.
    - brick tests the configured BrickChart strategy gate and score.
    - brick_growth tests the dense continuous BrickChart strength.
    """
    from factors.brick import brick_factor_to_long, is_brick_factor
    from factors.custom import custom_factor_to_long, is_custom_factor

    if is_custom_factor(factor_name):
        result = custom_factor_to_long(
            raw_data,
            factor_name,
            metadata=metadata,
        )
        rename_map = {
            "date": date_col,
            "symbol": symbol_col,
            "close": close_col,
        }
        return result.rename(columns=rename_map)

    if is_brick_factor(factor_name):
        result = brick_factor_to_long(
            raw_data,
            factor_name,
            config=factor_config,
            metadata=metadata,
        )
        rename_map = {
            "date": date_col,
            "symbol": symbol_col,
            "close": close_col,
        }
        return result.rename(columns=rename_map)

    if str(factor_name).lower().startswith("gtja"):
        from factors.gtja191 import gtja191_to_long

        result = gtja191_to_long(
            raw_data,
            factor_name,
            metadata=metadata,
            benchmark_data=benchmark_data,
            style_factor_data=style_factor_data,
        )
        rename_map = {
            "date": date_col,
            "symbol": symbol_col,
            "close": close_col,
        }
        return result.rename(columns=rename_map)

    if str(factor_name).lower().startswith("alpha"):
        from factors.alpha101 import alpha101_to_long

        result = alpha101_to_long(raw_data, factor_name, metadata=metadata)
        rename_map = {
            "date": date_col,
            "symbol": symbol_col,
            "close": close_col,
        }
        return result.rename(columns=rename_map)

    rows = []
    lookback = _parse_momentum_lookback(factor_name)
    for symbol, df in raw_data.items():
        if df is None or df.empty:
            continue
        frame = df.copy()
        frame.columns = [str(col).lower() for col in frame.columns]
        if date_col not in frame.columns or close_col not in frame.columns:
            continue
        frame[date_col] = pd.to_datetime(frame[date_col])
        frame[close_col] = pd.to_numeric(frame[close_col], errors="coerce")
        frame[symbol_col] = str(symbol).zfill(6)
        frame["factor_value"] = frame[close_col] / frame[close_col].shift(lookback) - 1.0
        keep = [date_col, symbol_col, "factor_value", close_col]
        keep.extend(
            column
            for column in (
                "volume",
                "amount",
                "turnover_value",
                "daily_return",
                "is_tradeable",
                "is_suspended",
                "is_limit_up",
                "is_limit_down",
                "is_st",
                "listing_age_days",
                "market_cap",
                "total_mv",
                "cap",
            )
            if column in frame.columns and column not in keep
        )
        rows.append(frame[keep])
    if not rows:
        return pd.DataFrame(columns=[date_col, symbol_col, "factor_value", close_col])
    result = pd.concat(rows, ignore_index=True)
    if metadata is not None and not metadata.empty:
        meta = metadata.copy()
        meta.columns = [str(column).lower() for column in meta.columns]
        meta_symbol = "symbol" if "symbol" in meta.columns else "ts_code"
        if meta_symbol in meta.columns:
            meta[symbol_col] = (
                meta[meta_symbol].astype(str).str.extract(r"(\d+)", expand=False).str.zfill(6)
            )
            available = [
                column for column in ("industry", "sector", "subindustry") if column in meta.columns
            ]
            if available:
                result = result.merge(
                    meta[[symbol_col, *available]].drop_duplicates(symbol_col),
                    on=symbol_col,
                    how="left",
                )
    if "market_cap" not in result.columns:
        for candidate in ("total_mv", "cap"):
            if candidate in result.columns:
                result["market_cap"] = pd.to_numeric(result[candidate], errors="coerce")
                break
    return result


def _parse_momentum_lookback(factor_name: str) -> int:
    if not factor_name.startswith("momentum_") or not factor_name.endswith("d"):
        raise ValueError("only built-in factors matching momentum_Nd are supported by the raw-data adapter")
    raw = factor_name.removeprefix("momentum_").removesuffix("d")
    lookback = int(raw)
    if lookback <= 0:
        raise ValueError("momentum lookback must be positive")
    return lookback
