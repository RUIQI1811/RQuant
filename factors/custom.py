"""User-defined factors for the standalone factor-research track."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

from factors.alpha101 import (
    Alpha101DataError,
    Alpha101Panels,
    build_alpha101_panels,
)
from factors.operators import covariance, delay, rank, replace_inf, safe_div


Panel = pd.DataFrame
CustomFactorPanels = Alpha101Panels
CUSTOM_FACTOR_NAMES = ("custom_001", "custom_002")
CUSTOM_FACTOR_ALIASES = {
    "custom001": "custom_001",
    "custom_1": "custom_001",
    "custom_return_turnover_cov_5d": "custom_001",
}


class CustomFactorError(ValueError):
    """Base error for custom-factor calculation failures."""


class CustomFactorDataError(CustomFactorError):
    """Raised when a custom factor's required input is unavailable."""


def normalize_custom_factor_name(factor_name: str | int) -> str:
    """Normalize supported aliases to the canonical ``custom_NNN`` form."""

    if isinstance(factor_name, int):
        number = factor_name
    else:
        name = str(factor_name).strip().lower().replace("-", "_")
        if name in CUSTOM_FACTOR_ALIASES:
            return CUSTOM_FACTOR_ALIASES[name]
        raw = name.removeprefix("custom_") if name.startswith("custom_") else name
        try:
            number = int(raw)
        except ValueError as exc:
            raise KeyError(f"unknown custom factor: {factor_name}") from exc
    normalized = f"custom_{number:03d}"
    if normalized not in CUSTOM_FACTOR_NAMES:
        raise KeyError(f"unknown custom factor: {factor_name}")
    return normalized


def is_custom_factor(factor_name: str) -> bool:
    """Return whether ``factor_name`` belongs to the custom-factor registry."""

    try:
        normalize_custom_factor_name(factor_name)
    except KeyError:
        return False
    return True


class CustomFactors:
    """Calculate registered user-defined factors on aligned market panels."""

    def __init__(self, data: CustomFactorPanels) -> None:
        self.d = data

    @property
    def names(self) -> tuple[str, ...]:
        return CUSTOM_FACTOR_NAMES

    def calculate(self, name: str | int) -> Panel:
        normalized = normalize_custom_factor_name(name)
        method = getattr(self, normalized, None)
        if method is None:
            raise KeyError(f"custom factor is not implemented: {normalized}")
        return replace_inf(method()).reindex(
            index=self.d.close.index,
            columns=self.d.close.columns,
        )

    def calculate_many(
        self,
        names: Iterable[str | int] | None = None,
        *,
        on_error: str = "raise",
    ) -> dict[str, Panel]:
        selected = CUSTOM_FACTOR_NAMES if names is None else tuple(
            normalize_custom_factor_name(name) for name in names
        )
        output: dict[str, Panel] = {}
        for name in selected:
            try:
                output[name] = self.calculate(name)
            except CustomFactorError:
                if on_error != "nan":
                    raise
                output[name] = self.d.close.copy() * np.nan
        return output

    def custom_001(self) -> Panel:
        """Negative rank of 5-day covariance between return and turnover ranks.

        ``turnover_value`` means traded amount.  The raw-data adapter uses the
        repository's existing point-in-time fallback: explicit turnover value,
        then ``amount * 1000``, then ``close * volume``.
        来自于 alpha13
        """

        if self.d.turnover_value is None:
            raise CustomFactorDataError("custom_001 requires turnover_value")
        daily_return = self.d.close / delay(self.d.close, 1) - 1.0
        return -rank(
            covariance(
                rank(daily_return),
                rank(self.d.turnover_value),
                5,
            )
        )

    def custom_002(self) -> Panel:
        """Cross-sectional rank of the close discount relative to VWAP."""

        return rank(safe_div(self.d.vwap - self.d.close, self.d.vwap))


def build_custom_factor_panels(
    raw_data: Mapping[str, pd.DataFrame],
    *,
    metadata: pd.DataFrame | Mapping[str, Mapping[str, object]] | None = None,
) -> CustomFactorPanels:
    """Build aligned panels for the custom-factor calculator."""

    try:
        return build_alpha101_panels(raw_data, metadata=metadata)
    except Alpha101DataError as exc:
        raise CustomFactorDataError(str(exc)) from exc


def custom_factor_to_long(
    raw_data: Mapping[str, pd.DataFrame],
    factor_name: str | int,
    *,
    metadata: pd.DataFrame | Mapping[str, Mapping[str, object]] | None = None,
) -> pd.DataFrame:
    """Calculate one custom factor and return FactorTester's long schema."""

    panels = build_custom_factor_panels(raw_data, metadata=metadata)
    name = normalize_custom_factor_name(factor_name)
    values = CustomFactors(panels).calculate(name)

    def stacked(panel: Panel, column: str) -> pd.Series:
        return (
            panel.rename_axis(index="date", columns="symbol")
            .stack(future_stack=True)
            .rename(column)
        )

    parts = [
        stacked(values, "factor_value"),
        stacked(panels.close, "close"),
        stacked(panels.volume, "volume"),
        stacked(panels.returns, "daily_return"),
        stacked(panels.close.notna().cumsum(), "listing_age_days"),
    ]
    for panel, column in (
        (panels.industry, "industry"),
        (panels.cap, "market_cap"),
        (panels.is_st, "is_st"),
        (panels.turnover_value, "turnover_value"),
    ):
        if panel is not None:
            parts.append(stacked(panel, column))

    result = pd.concat(parts, axis=1).reset_index()
    result["factor_value"] = result["factor_value"].replace(
        [np.inf, -np.inf], np.nan
    )
    return result


__all__ = [
    "CUSTOM_FACTOR_ALIASES",
    "CUSTOM_FACTOR_NAMES",
    "CustomFactorDataError",
    "CustomFactorError",
    "CustomFactorPanels",
    "CustomFactors",
    "build_custom_factor_panels",
    "custom_factor_to_long",
    "is_custom_factor",
    "normalize_custom_factor_name",
]
