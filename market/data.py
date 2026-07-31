from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from domain.values import normalize_symbol


REQUIRED_DAILY_COLUMNS = ("date", "open", "high", "low", "close", "volume")
PRICE_COLUMNS = ("open", "high", "low", "close")


@dataclass(frozen=True)
class StockPoolConfig:
    top_m: int = 5000
    min_price: float = 1.0
    min_turnover: float = 0.0
    exclude_boards: tuple[str, ...] = ("gem", "star", "bj")
    require_tradeable: bool = True


def normalize_code(code: object) -> str:
    return normalize_symbol(code)


def board_of_code(code: str) -> str:
    code = normalize_code(code)
    if code.startswith(("300", "301")):
        return "gem"
    if code.startswith("688"):
        return "star"
    if code.startswith(("4", "8")):
        return "bj"
    return "main"


def limit_rate_for_code(code: str, *, is_st: bool = False) -> float:
    if is_st:
        return 0.05
    board = board_of_code(code)
    if board in {"gem", "star"}:
        return 0.20
    if board == "bj":
        return 0.30
    return 0.10


def clean_daily_frame(
    df: pd.DataFrame,
    *,
    code: str = "",
    end_date: Optional[str | pd.Timestamp] = None,
) -> pd.DataFrame:
    """Normalize one stock daily bar frame into the trading schema.

    The project currently stores qfq OHLCV CSVs. Optional Tushare columns such as
    pre_close, pct_chg, amount, adj_factor, is_st are preserved when available.
    Limit flags use pct_chg/pre_close when present and otherwise fall back to
    qfq close returns, which is an approximation.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=list(REQUIRED_DAILY_COLUMNS))

    out = df.copy()
    out.columns = [str(col).lower() for col in out.columns]
    missing = [col for col in REQUIRED_DAILY_COLUMNS if col not in out.columns]
    if missing:
        raise ValueError(f"{code or '<unknown>'} missing columns: {', '.join(missing)}")

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).drop_duplicates(subset="date", keep="last")
    out = out.sort_values("date").reset_index(drop=True)
    if end_date is not None:
        out = out[out["date"] <= pd.to_datetime(end_date)].reset_index(drop=True)

    numeric_cols = set(
        PRICE_COLUMNS + ("volume", "pre_close", "pct_chg", "amount", "adj_factor")
    )
    for col in numeric_cols.intersection(out.columns):
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=list(PRICE_COLUMNS))
    out = out[(out[list(PRICE_COLUMNS)] > 0).all(axis=1)].reset_index(drop=True)
    if out.empty:
        return out

    out["volume"] = pd.to_numeric(out["volume"], errors="coerce").fillna(0.0)
    out["is_suspended"] = out["volume"] <= 0
    out["prev_close"] = (
        pd.to_numeric(out["pre_close"], errors="coerce")
        if "pre_close" in out.columns
        else out["close"].shift(1)
    )

    if "pct_chg" in out.columns:
        pct_return = pd.to_numeric(out["pct_chg"], errors="coerce") / 100.0
    else:
        pct_return = out["close"] / out["prev_close"] - 1.0
    out["daily_return"] = pct_return.replace([np.inf, -np.inf], np.nan)

    base_limit_rate = limit_rate_for_code(code, is_st=False)
    if "is_st" in out.columns:
        point_in_time_is_st = out["is_st"].fillna(False).astype(bool)
        limit_rate = pd.Series(
            np.where(point_in_time_is_st, 0.05, base_limit_rate),
            index=out.index,
        )
    else:
        limit_rate = pd.Series(base_limit_rate, index=out.index)
    eps = 0.001
    out["limit_rate"] = limit_rate
    out["is_limit_up"] = out["daily_return"] >= limit_rate - eps
    out["is_limit_down"] = out["daily_return"] <= -limit_rate + eps
    out["is_tradeable"] = (
        ~out["is_suspended"]
        & out["open"].notna()
        & out["close"].notna()
        & (out["open"] > 0)
        & (out["close"] > 0)
    )

    if "amount" in out.columns:
        out["turnover_value"] = pd.to_numeric(out["amount"], errors="coerce").fillna(0.0) * 1000.0
    else:
        out["turnover_value"] = ((out["open"] + out["close"]) / 2.0 * out["volume"]).fillna(0.0)

    return out.set_index("date", drop=False)


def clean_market_data(
    data: Dict[str, pd.DataFrame],
    *,
    end_date: Optional[str | pd.Timestamp] = None,
) -> Dict[str, pd.DataFrame]:
    cleaned: Dict[str, pd.DataFrame] = {}
    for code, df in data.items():
        code_norm = normalize_code(code)
        clean = clean_daily_frame(df, code=code_norm, end_date=end_date)
        if not clean.empty:
            cleaned[code_norm] = clean
    return cleaned


def build_stock_pool_by_date(
    prepared: Dict[str, pd.DataFrame],
    *,
    config: StockPoolConfig,
    allowed_codes: Optional[Iterable[str]] = None,
) -> Dict[pd.Timestamp, list[str]]:
    frame = build_stock_pool_frame(
        prepared,
        config=config,
        allowed_codes=allowed_codes,
    )
    if frame.empty:
        return {}
    return {
        pd.to_datetime(date): daily["code"].tolist()
        for date, daily in frame.groupby("date", sort=False)
    }


def build_stock_pool_frame(
    prepared: Dict[str, pd.DataFrame],
    *,
    config: StockPoolConfig,
    allowed_codes: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    allowed = {normalize_code(code) for code in allowed_codes} if allowed_codes is not None else None
    excluded = set(config.exclude_boards or ())
    frames: list[pd.DataFrame] = []
    input_offset = 0

    for raw_code, df in prepared.items():
        code = normalize_code(raw_code)
        if allowed is not None and code not in allowed:
            continue
        if board_of_code(code) in excluded:
            continue
        if df is None or df.empty:
            continue

        close = pd.to_numeric(
            df["close"] if "close" in df.columns else pd.Series(np.nan, index=df.index),
            errors="coerce",
        )
        turnover_col = "turnover_n" if "turnover_n" in df.columns else "turnover_value"
        turnover = pd.to_numeric(
            df[turnover_col]
            if turnover_col in df.columns
            else pd.Series(0.0, index=df.index),
            errors="coerce",
        )
        tradeable = (
            df["is_tradeable"].astype(bool)
            if config.require_tradeable and "is_tradeable" in df.columns
            else pd.Series(True, index=df.index)
        )
        eligible = (
            tradeable
            & close.ge(config.min_price)
            & turnover.ge(config.min_turnover)
            & np.isfinite(close)
            & np.isfinite(turnover)
        )
        if not bool(eligible.any()):
            input_offset += len(df)
            continue
        selected_positions = np.flatnonzero(eligible.to_numpy())
        frames.append(
            pd.DataFrame(
                {
                    "date": pd.to_datetime(df.index[selected_positions]),
                    "turnover": turnover.iloc[selected_positions].to_numpy(dtype=float),
                    "code": code,
                    "_input_order": input_offset + selected_positions,
                }
            )
        )
        input_offset += len(df)

    if not frames:
        return pd.DataFrame(columns=["date", "turnover", "code"])

    ranked = pd.concat(frames, ignore_index=True).sort_values(
        ["date", "turnover", "_input_order"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    if config.top_m > 0:
        ranked = ranked.groupby("date", sort=False, as_index=False).head(config.top_m)
    return ranked[["date", "turnover", "code"]].reset_index(drop=True)
