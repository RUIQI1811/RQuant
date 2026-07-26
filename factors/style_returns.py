"""Construct explicit daily MKT/SMB/HML inputs for GTJA191 alpha 030."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from factors.external import load_research_context_file, research_context_signature
from strategies.preselect import load_raw_data


@dataclass(frozen=True)
class StyleFactorConfig:
    size_quantile: float = 0.5
    value_low_quantile: float = 0.3
    value_high_quantile: float = 0.7
    min_stocks_per_portfolio: int = 5
    start_date: str | None = None
    end_date: str | None = None

    def __post_init__(self) -> None:
        if not 0 < self.size_quantile < 1:
            raise ValueError("size_quantile must be in (0, 1)")
        if not 0 < self.value_low_quantile < self.value_high_quantile < 1:
            raise ValueError("value quantiles must satisfy 0 < low < high < 1")
        if self.min_stocks_per_portfolio <= 0:
            raise ValueError("min_stocks_per_portfolio must be positive")
        if self.start_date and self.end_date:
            if pd.Timestamp(self.start_date) > pd.Timestamp(self.end_date):
                raise ValueError("start_date must not be after end_date")


def build_style_factor_file(
    *,
    data_dir: str | Path,
    context_path: str | Path,
    output_file: str | Path = "data/context/style_factors.csv",
    manifest_path: str | Path | None = None,
    config: StyleFactorConfig | None = None,
    max_symbols: int | None = None,
) -> dict[str, object]:
    """Build daily 2x3 size/value returns from local point-in-time inputs."""

    settings = config or StyleFactorConfig()
    if max_symbols is not None and max_symbols <= 0:
        raise ValueError("max_symbols must be positive")
    data_root = Path(data_dir)
    symbols = None
    if max_symbols is not None:
        symbols = sorted(path.stem for path in data_root.glob("*.csv"))[:max_symbols]
    raw_data = load_raw_data(str(data_root), symbols=symbols)
    returns = _returns_from_raw(raw_data)
    context = load_research_context_file(context_path)
    if symbols is not None:
        context = context.loc[context["symbol"].isin(set(symbols))].copy()
    required = {"market_cap", "book_to_market"}
    missing = required.difference(context.columns)
    if missing:
        raise ValueError(
            "style-factor context missing columns: " + ", ".join(sorted(missing))
        )
    research = returns.merge(
        context[["date", "symbol", "market_cap", "book_to_market"]],
        on=["date", "symbol"],
        how="left",
        validate="one_to_one",
    )
    factors, audit = calculate_style_factor_returns(research, config=settings)

    output = Path(output_file).resolve()
    manifest = (
        Path(manifest_path).resolve()
        if manifest_path
        else output.with_suffix(output.suffix + ".manifest.json")
    )
    _atomic_write_csv(output, factors)
    payload = {
        "method": "daily_fama_french_2x3_value_weighted",
        "timing": (
            "close-to-close return on date t; size and book-to-market are the "
            "latest values available strictly before t"
        ),
        "mkt_definition": "lagged-market-cap-weighted raw stock return",
        "smb_definition": "mean(SL,SM,SH)-mean(BL,BM,BH)",
        "hml_definition": "mean(SH,BH)-mean(SL,BL)",
        "config": asdict(settings),
        "inputs": {
            "data_dir": str(data_root.resolve()),
            "data_signature": _data_signature(data_root),
            "context_path": str(Path(context_path).resolve()),
            "context_signature": research_context_signature(context_path),
            "max_symbols": max_symbols,
        },
        "audit": audit,
        "output": output.name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_write_json(manifest, payload)
    return {
        "output_file": output,
        "manifest_path": manifest,
        "row_count": len(factors),
        "audit": audit,
    }


def calculate_style_factor_returns(
    frame: pd.DataFrame,
    *,
    config: StyleFactorConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Calculate daily 2x3 factors with strictly lagged characteristics."""

    settings = config or StyleFactorConfig()
    required = {
        "date",
        "symbol",
        "daily_return",
        "market_cap",
        "book_to_market",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError("style-factor frame missing columns: " + ", ".join(sorted(missing)))
    work = frame[list(required)].copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work["symbol"] = work["symbol"].astype(str).str.zfill(6)
    for column in ("daily_return", "market_cap", "book_to_market"):
        work[column] = pd.to_numeric(work[column], errors="coerce")
    if work["date"].isna().any() or work.duplicated(["date", "symbol"]).any():
        raise ValueError("style-factor frame has invalid or duplicate date/symbol keys")
    work = work.sort_values(["symbol", "date"], kind="mergesort")
    for source, target in (
        ("market_cap", "formation_market_cap"),
        ("book_to_market", "formation_book_to_market"),
    ):
        work[target] = work.groupby("symbol", sort=False)[source].transform(
            lambda values: values.ffill().shift(1)
        )
    work = work.replace([np.inf, -np.inf], np.nan)
    eligible = work.dropna(
        subset=[
            "daily_return",
            "formation_market_cap",
            "formation_book_to_market",
        ]
    ).copy()
    eligible = eligible.loc[
        (eligible["formation_market_cap"] > 0)
        & (eligible["formation_book_to_market"] > 0)
    ]
    if settings.start_date:
        eligible = eligible.loc[eligible["date"] >= pd.Timestamp(settings.start_date)]
    if settings.end_date:
        eligible = eligible.loc[eligible["date"] <= pd.Timestamp(settings.end_date)]
    if eligible.empty:
        raise ValueError("no eligible rows remain for style-factor construction")

    grouped = eligible.groupby("date", sort=True)
    eligible["size_cutoff"] = grouped["formation_market_cap"].transform(
        lambda values: values.quantile(settings.size_quantile)
    )
    eligible["value_low_cutoff"] = grouped["formation_book_to_market"].transform(
        lambda values: values.quantile(settings.value_low_quantile)
    )
    eligible["value_high_cutoff"] = grouped["formation_book_to_market"].transform(
        lambda values: values.quantile(settings.value_high_quantile)
    )
    eligible["size_bucket"] = np.where(
        eligible["formation_market_cap"] <= eligible["size_cutoff"], "S", "B"
    )
    eligible["value_bucket"] = np.select(
        [
            eligible["formation_book_to_market"] <= eligible["value_low_cutoff"],
            eligible["formation_book_to_market"] >= eligible["value_high_cutoff"],
        ],
        ["L", "H"],
        default="M",
    )
    eligible["portfolio"] = eligible["size_bucket"] + eligible["value_bucket"]
    eligible["weighted_return"] = (
        eligible["daily_return"] * eligible["formation_market_cap"]
    )

    portfolio = (
        eligible.groupby(["date", "portfolio"], sort=True)
        .agg(
            weighted_return=("weighted_return", "sum"),
            total_weight=("formation_market_cap", "sum"),
            stock_count=("symbol", "nunique"),
        )
        .reset_index()
    )
    portfolio["portfolio_return"] = (
        portfolio["weighted_return"] / portfolio["total_weight"]
    )
    portfolio.loc[
        portfolio["stock_count"] < settings.min_stocks_per_portfolio,
        "portfolio_return",
    ] = np.nan
    six = portfolio.pivot(index="date", columns="portfolio", values="portfolio_return")
    required_portfolios = ["SL", "SM", "SH", "BL", "BM", "BH"]
    six = six.reindex(columns=required_portfolios)

    market = (
        eligible.groupby("date", sort=True)
        .agg(
            weighted_return=("weighted_return", "sum"),
            total_weight=("formation_market_cap", "sum"),
        )
    )
    output = pd.DataFrame(index=six.index)
    output["mkt"] = market["weighted_return"] / market["total_weight"]
    output["smb"] = six[["SL", "SM", "SH"]].mean(axis=1, skipna=False) - six[
        ["BL", "BM", "BH"]
    ].mean(axis=1, skipna=False)
    output["hml"] = six[["SH", "BH"]].mean(axis=1, skipna=False) - six[
        ["SL", "BL"]
    ].mean(axis=1, skipna=False)
    all_dates = int(output.index.nunique())
    output = output.replace([np.inf, -np.inf], np.nan).dropna().reset_index()
    output["date"] = pd.to_datetime(output["date"]).dt.strftime("%Y-%m-%d")
    audit = {
        "input_rows": int(len(frame)),
        "eligible_rows": int(len(eligible)),
        "candidate_date_count": all_dates,
        "output_date_count": int(len(output)),
        "dropped_incomplete_date_count": all_dates - int(len(output)),
    }
    return output[["date", "mkt", "smb", "hml"]], audit


def run_from_args(args: object) -> dict[str, object]:
    config = StyleFactorConfig(
        size_quantile=args.size_quantile,
        value_low_quantile=args.value_low_quantile,
        value_high_quantile=args.value_high_quantile,
        min_stocks_per_portfolio=args.min_stocks_per_portfolio,
        start_date=args.start,
        end_date=args.end,
    )
    return build_style_factor_file(
        data_dir=args.data,
        context_path=args.context,
        output_file=args.out,
        manifest_path=args.manifest,
        config=config,
        max_symbols=args.max_symbols,
    )


def _returns_from_raw(raw_data: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for raw_symbol, raw_frame in raw_data.items():
        frame = raw_frame.copy()
        frame.columns = [str(column).lower() for column in frame.columns]
        if not {"date", "close"}.issubset(frame.columns):
            continue
        close = pd.to_numeric(frame["close"], errors="coerce")
        rows.append(
            pd.DataFrame(
                {
                    "date": pd.to_datetime(frame["date"], errors="coerce"),
                    "symbol": str(raw_symbol).split(".", 1)[0].zfill(6),
                    "daily_return": close.pct_change(fill_method=None),
                }
            )
        )
    if not rows:
        raise ValueError("no usable close series found for style-factor construction")
    return pd.concat(rows, ignore_index=True).dropna(subset=["date"])


def _data_signature(data_dir: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted(data_dir.glob("*.csv"))
    if not paths:
        raise ValueError(f"market data directory contains no CSV files: {data_dir}")
    for path in paths:
        stat = path.stat()
        digest.update(path.name.encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return digest.hexdigest()


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)
