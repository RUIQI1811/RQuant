"""Resumable Tushare point-in-time research-context fetcher."""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from market import fetch_kline


DAILY_BASIC_FIELDS = (
    "ts_code",
    "trade_date",
    "total_mv",
    "circ_mv",
    "turnover_rate",
    "volume_ratio",
    "pb",
)


logger = logging.getLogger(__name__)


def fetch_daily_basic_context(
    *,
    start: str,
    end: str | None = None,
    output_dir: str | Path = "data/context/daily_basic",
    manifest_path: str | Path | None = None,
    resume: bool = False,
    max_requests_per_minute: int = 180,
    workers: int = 8,
    max_dates: int | None = None,
    session: Any | None = None,
) -> dict[str, object]:
    """Fetch all-stock ``daily_basic`` rows once per open trading date.

    Tushare reports A-share ``total_mv``/``circ_mv`` in ten-thousand yuan.
    Canonical context columns are converted to yuan before being persisted.
    Each trading date is written atomically to its year partition so an
    interrupted multi-year run can resume without rewriting completed dates.
    """

    start_text = _normalize_date(start)
    end_text = _normalize_date(end or date.today().strftime("%Y%m%d"))
    if start_text > end_text:
        raise ValueError("start must not be after end")
    if max_requests_per_minute < 0:
        raise ValueError("max_requests_per_minute must be non-negative")
    if workers <= 0:
        raise ValueError("workers must be positive")
    if max_dates is not None and max_dates <= 0:
        raise ValueError("max_dates must be positive")

    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    manifest = (
        Path(manifest_path).resolve()
        if manifest_path
        else destination / "_context_manifest.json"
    )
    signature = {
        "source": "tushare_daily_basic_by_trade_date",
        "start": start_text,
        "end": end_text,
        "fields": list(DAILY_BASIC_FIELDS),
        "market_cap_unit": "yuan",
        "max_dates": max_dates,
    }
    previous = _load_manifest(manifest)
    if resume and previous:
        previous_signature = previous.get("signature")
        if previous_signature != signature:
            raise ValueError("context resume manifest does not match requested range/fields")

    api = session or _create_session()
    limiter = (
        fetch_kline.RequestRateLimiter(max_requests_per_minute)
        if max_requests_per_minute > 0
        else None
    )
    with fetch_kline._temporary_tushare_network_env():
        if limiter is not None:
            limiter.wait()
        calendar = api.trade_cal(
            exchange="",
            start_date=start_text,
            end_date=end_text,
            is_open="1",
            fields="cal_date,is_open",
        )
        trading_dates = _open_dates(calendar)
        if max_dates is not None:
            trading_dates = trading_dates[: int(max_dates)]

        completed = set(
            str(value) for value in previous.get("completed_dates", [])
        ) if resume else set()
        failures: dict[str, str] = {}
        fetched_dates: list[str] = []
        reused_dates: list[str] = []
        pending_dates: list[str] = []
        for trade_date in trading_dates:
            output_path = _date_output_path(destination, trade_date)
            if resume and trade_date in completed and _valid_date_file(
                output_path, trade_date
            ):
                reused_dates.append(trade_date)
                continue
            pending_dates.append(trade_date)

        def fetch_one(trade_date: str) -> str:
            output_path = _date_output_path(destination, trade_date)
            if limiter is not None:
                limiter.wait()
            raw = api.daily_basic(
                trade_date=trade_date,
                fields=",".join(DAILY_BASIC_FIELDS),
            )
            normalized = normalize_daily_basic(raw, trade_date=trade_date)
            _atomic_write_csv(output_path, normalized)
            return trade_date

        logger.info(
            "Starting daily_basic context fetch: requested=%d, pending=%d, reused=%d, workers=%d",
            len(trading_dates),
            len(pending_dates),
            len(reused_dates),
            workers,
        )
        processed = 0
        try:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(fetch_one, trade_date): trade_date
                    for trade_date in pending_dates
                }
                for future in as_completed(futures):
                    trade_date = futures[future]
                    processed += 1
                    try:
                        future.result()
                    except Exception as exc:  # provider failures are isolated by date
                        failures[trade_date] = f"{type(exc).__name__}: {exc}"
                    else:
                        completed.add(trade_date)
                        failures.pop(trade_date, None)
                        fetched_dates.append(trade_date)
                    _write_manifest(
                        manifest,
                        signature=signature,
                        requested_dates=trading_dates,
                        completed_dates=completed,
                        failures=failures,
                    )
                    if processed == len(pending_dates) or processed % 25 == 0:
                        logger.info(
                            "daily_basic progress: processed=%d/%d, complete=%d/%d, failed=%d",
                            processed,
                            len(pending_dates),
                            len(completed.intersection(trading_dates)),
                            len(trading_dates),
                            len(failures),
                        )
        except BaseException:
            _write_manifest(
                manifest,
                signature=signature,
                requested_dates=trading_dates,
                completed_dates=completed,
                failures=failures,
            )
            raise

    _write_manifest(
        manifest,
        signature=signature,
        requested_dates=trading_dates,
        completed_dates=completed,
        failures=failures,
    )
    failed_dates = tuple(sorted(failures))
    return {
        "ok": not failed_dates,
        "start": start_text,
        "end": end_text,
        "requested_date_count": len(trading_dates),
        "fetched_date_count": len(fetched_dates),
        "reused_date_count": len(reused_dates),
        "worker_count": int(workers),
        "failed_dates": list(failed_dates),
        "output_dir": str(destination),
        "manifest_path": str(manifest),
    }


def normalize_daily_basic(frame: pd.DataFrame | None, *, trade_date: str) -> pd.DataFrame:
    """Validate one provider response and convert market-cap units to yuan."""

    if frame is None or frame.empty:
        raise ValueError(f"daily_basic returned no rows for open date {trade_date}")
    missing = set(DAILY_BASIC_FIELDS).difference(frame.columns)
    if missing:
        raise ValueError("daily_basic missing fields: " + ", ".join(sorted(missing)))
    result = frame.loc[:, list(DAILY_BASIC_FIELDS)].copy()
    result["date"] = pd.to_datetime(result["trade_date"], format="%Y%m%d", errors="coerce")
    if result["date"].isna().any():
        raise ValueError("daily_basic contains invalid trade_date")
    expected = pd.Timestamp(trade_date)
    if not result["date"].eq(expected).all():
        raise ValueError("daily_basic response contains rows from another date")
    result["symbol"] = (
        result["ts_code"].astype(str).str.extract(r"^(\d{6})", expand=False)
    )
    if result["symbol"].isna().any():
        raise ValueError("daily_basic contains invalid ts_code")
    numeric = ("total_mv", "circ_mv", "turnover_rate", "volume_ratio", "pb")
    for column in numeric:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["market_cap"] = result["total_mv"] * 10_000.0
    result["circulating_market_cap"] = result["circ_mv"] * 10_000.0
    result["book_to_market"] = 1.0 / result["pb"].where(result["pb"] > 0)
    result = result[
        [
            "date",
            "symbol",
            "market_cap",
            "circulating_market_cap",
            "turnover_rate",
            "volume_ratio",
            "pb",
            "book_to_market",
        ]
    ]
    if result.duplicated(["date", "symbol"]).any():
        raise ValueError("daily_basic contains duplicate date/symbol rows")
    return result.sort_values("symbol", kind="mergesort").reset_index(drop=True)


def run_from_args(args: Any) -> dict[str, object]:
    return fetch_daily_basic_context(
        start=args.start,
        end=args.end,
        output_dir=args.out,
        manifest_path=args.manifest,
        resume=args.resume,
        max_requests_per_minute=args.max_requests_per_minute,
        workers=args.workers,
        max_dates=args.max_dates,
    )


def _create_session() -> Any:
    token = os.environ.get("TUSHARE_TOKEN") or fetch_kline._read_dotenv_value(
        "TUSHARE_TOKEN"
    )
    if not token:
        raise RuntimeError("TUSHARE_TOKEN is required for fetch-context")
    return fetch_kline.ts.pro_api(token)


def _normalize_date(value: object) -> str:
    parsed = pd.to_datetime(value, errors="raise")
    return pd.Timestamp(parsed).strftime("%Y%m%d")


def _open_dates(frame: pd.DataFrame | None) -> list[str]:
    if frame is None or frame.empty or "cal_date" not in frame.columns:
        raise ValueError("trade_cal returned no open dates")
    work = frame.copy()
    if "is_open" in work.columns:
        work = work.loc[pd.to_numeric(work["is_open"], errors="coerce").eq(1)]
    dates = pd.to_datetime(work["cal_date"], format="%Y%m%d", errors="coerce")
    if dates.isna().any():
        raise ValueError("trade_cal contains invalid cal_date")
    return sorted(dates.dt.strftime("%Y%m%d").unique().tolist())


def _date_output_path(destination: Path, trade_date: str) -> Path:
    return destination / trade_date[:4] / f"{trade_date}.csv"


def _valid_date_file(path: Path, trade_date: str) -> bool:
    if not path.exists():
        return False
    try:
        frame = pd.read_csv(path, dtype={"symbol": str})
    except (OSError, pd.errors.ParserError):
        return False
    required = {"date", "symbol", "market_cap"}
    if frame.empty or not required.issubset(frame.columns):
        return False
    dates = pd.to_datetime(frame["date"], errors="coerce")
    return bool(dates.notna().all() and dates.eq(pd.Timestamp(trade_date)).all())


def _load_manifest(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_manifest(
    path: Path,
    *,
    signature: dict[str, object],
    requested_dates: list[str],
    completed_dates: set[str],
    failures: dict[str, str],
) -> None:
    requested = set(requested_dates)
    complete = requested.issubset(completed_dates) and not failures
    payload = {
        "status": "complete" if complete else "partial",
        "signature": signature,
        "requested_date_count": len(requested_dates),
        "completed_dates": sorted(completed_dates.intersection(requested)),
        "failed_dates": failures,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_write_json(path, payload)


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)
