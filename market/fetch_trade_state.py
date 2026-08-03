"""Resumable daily Tushare price-limit and suspension context fetcher."""

from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from market import fetch_kline
from market.fetch_context import _date_output_path, _normalize_date, _open_dates


LIMIT_FIELDS = ("trade_date", "ts_code", "pre_close", "up_limit", "down_limit")
SUSPEND_FIELDS = ("ts_code", "trade_date", "suspend_timing", "suspend_type")


def fetch_trade_state_context(
    *,
    start: str,
    end: str | None = None,
    output_dir: str | Path = "data/context/trade_state",
    manifest_path: str | Path | None = None,
    resume: bool = False,
    max_requests_per_minute: int = 180,
    workers: int = 8,
    max_dates: int | None = None,
    include_suspensions: bool = True,
    expected_symbols_dir: str | Path | None = None,
    symbol_aliases: list[dict[str, str | None]] | None = None,
    session: Any | None = None,
) -> dict[str, object]:
    """Fetch exact daily limit prices and sparse suspension records."""

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
    expected_root = Path(expected_symbols_dir).resolve() if expected_symbols_dir else None
    expected_contract = (
        _expected_symbols_contract(expected_root) if expected_root is not None else None
    )
    normalized_aliases = _normalize_symbol_aliases(symbol_aliases or [])
    destination.mkdir(parents=True, exist_ok=True)
    manifest = (
        Path(manifest_path).resolve()
        if manifest_path
        else destination / "_context_manifest.json"
    )
    signature = {
        "source": "tushare_stk_limit_and_suspend_d_by_trade_date",
        "start": start_text,
        "end": end_text,
        "limit_fields": list(LIMIT_FIELDS),
        "suspend_fields": list(SUSPEND_FIELDS),
        "include_suspensions": bool(include_suspensions),
        "expected_symbols_dir": str(expected_root) if expected_root else None,
        "expected_symbols_contract": expected_contract,
        "symbol_aliases": normalized_aliases,
        "missing_limit_policy": "explicit_nullable_row_v1",
        "max_dates": max_dates,
    }
    previous = _load_manifest(manifest)
    if resume and previous and not _resume_signature_compatible(
        previous.get("signature"), signature
    ):
        raise ValueError("trade-state resume manifest does not match requested contract")

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
        unavailable_limits: dict[str, dict[str, object]] = (
            {
                str(key): dict(value)
                for key, value in (previous.get("unavailable_limits") or {}).items()
            }
            if resume
            else {}
        )
        fetched_dates: list[str] = []
        reused_dates: list[str] = []
        pending_dates: list[str] = []
        for trade_date in trading_dates:
            output_path = _date_output_path(destination, trade_date)
            if resume and trade_date in completed:
                reusable, unavailable = _prepare_resumable_date_file(
                    output_path, trade_date
                )
                if reusable:
                    reused_dates.append(trade_date)
                    if unavailable:
                        unavailable_limits[trade_date] = unavailable
                    else:
                        unavailable_limits.pop(trade_date, None)
                    continue
            pending_dates.append(trade_date)

        def fetch_one(trade_date: str) -> tuple[str, dict[str, object] | None]:
            if limiter is not None:
                limiter.wait()
            limits = api.stk_limit(
                trade_date=trade_date,
                fields=",".join(LIMIT_FIELDS),
            )
            suspensions: pd.DataFrame | None = None
            if include_suspensions:
                if limiter is not None:
                    limiter.wait()
                suspensions = api.suspend_d(
                    trade_date=trade_date,
                    suspend_type="S",
                    fields=",".join(SUSPEND_FIELDS),
                )
            normalized = normalize_trade_state(
                limits,
                suspensions,
                trade_date=trade_date,
                expected_symbols=(
                    _read_expected_symbols(expected_root, trade_date)
                    if expected_root is not None
                    else None
                ),
                symbol_aliases=normalized_aliases,
            )
            _atomic_write_csv(_date_output_path(destination, trade_date), normalized)
            unavailable = _unavailable_limit_summary(normalized)
            return trade_date, unavailable

        try:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(fetch_one, trade_date): trade_date
                    for trade_date in pending_dates
                }
                for future in as_completed(futures):
                    trade_date = futures[future]
                    try:
                        _, unavailable = future.result()
                    except Exception as exc:
                        failures[trade_date] = f"{type(exc).__name__}: {exc}"
                    else:
                        completed.add(trade_date)
                        failures.pop(trade_date, None)
                        fetched_dates.append(trade_date)
                        if unavailable:
                            unavailable_limits[trade_date] = unavailable
                        else:
                            unavailable_limits.pop(trade_date, None)
                    _write_manifest(
                        manifest,
                        signature=signature,
                        requested_dates=trading_dates,
                        completed_dates=completed,
                        failures=failures,
                        unavailable_limits=unavailable_limits,
                    )
        except BaseException:
            _write_manifest(
                manifest,
                signature=signature,
                requested_dates=trading_dates,
                completed_dates=completed,
                failures=failures,
                unavailable_limits=unavailable_limits,
            )
            raise

    _write_manifest(
        manifest,
        signature=signature,
        requested_dates=trading_dates,
        completed_dates=completed,
        failures=failures,
        unavailable_limits=unavailable_limits,
    )
    return {
        "ok": not failures and set(trading_dates).issubset(completed),
        "start": start_text,
        "end": end_text,
        "requested_date_count": len(trading_dates),
        "fetched_date_count": len(fetched_dates),
        "reused_date_count": len(reused_dates),
        "worker_count": int(workers),
        "failed_dates": sorted(failures),
        "dates_with_unavailable_limits": len(unavailable_limits),
        "unavailable_limit_row_count": sum(
            int(value.get("count", 0)) for value in unavailable_limits.values()
        ),
        "output_dir": str(destination),
        "manifest_path": str(manifest),
    }


def normalize_trade_state(
    limits: pd.DataFrame | None,
    suspensions: pd.DataFrame | None,
    *,
    trade_date: str,
    expected_symbols: set[str] | None = None,
    symbol_aliases: list[dict[str, str | None]] | None = None,
) -> pd.DataFrame:
    if limits is None or limits.empty:
        raise ValueError(f"stk_limit returned no rows for open date {trade_date}")
    missing = set(LIMIT_FIELDS).difference(limits.columns)
    if missing:
        raise ValueError("stk_limit missing fields: " + ", ".join(sorted(missing)))
    result = limits.loc[:, list(LIMIT_FIELDS)].copy()
    result["date"] = pd.to_datetime(
        result["trade_date"], format="%Y%m%d", errors="coerce"
    )
    if result["date"].isna().any() or not result["date"].eq(
        pd.Timestamp(trade_date)
    ).all():
        raise ValueError("stk_limit response contains an invalid or different date")
    result["symbol"] = result["ts_code"].astype(str).str.extract(
        r"^(\d{6})", expand=False
    )
    if result["symbol"].isna().any():
        raise ValueError("stk_limit contains invalid ts_code")
    result["symbol"] = _apply_symbol_aliases(
        result["symbol"], trade_date, symbol_aliases or []
    )
    for column in ("pre_close", "up_limit", "down_limit"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if result[["up_limit", "down_limit"]].isna().any().any():
        raise ValueError("stk_limit contains missing limit prices")
    if result.duplicated(["date", "symbol"]).any():
        raise ValueError("stk_limit contains duplicate date/symbol rows")
    if expected_symbols is not None:
        canonical_expected = set(
            _apply_symbol_aliases(
                pd.Series(sorted(_normalize_expected_symbols(expected_symbols))),
                trade_date,
                symbol_aliases or [],
            )
        )
        expected_frame = pd.DataFrame({"symbol": sorted(canonical_expected)})
        result = expected_frame.merge(
            result.drop(columns=["trade_date", "ts_code", "date"]),
            on="symbol",
            how="left",
            validate="one_to_one",
        )
        result.insert(0, "date", pd.Timestamp(trade_date))
    result["has_price_limit"] = result[["up_limit", "down_limit"]].notna().all(axis=1)

    suspended = pd.DataFrame(columns=["symbol", "suspend_timing"])
    if suspensions is not None and not suspensions.empty:
        missing = set(SUSPEND_FIELDS).difference(suspensions.columns)
        if missing:
            raise ValueError("suspend_d missing fields: " + ", ".join(sorted(missing)))
        suspended = suspensions.loc[:, list(SUSPEND_FIELDS)].copy()
        suspended_dates = pd.to_datetime(
            suspended["trade_date"], format="%Y%m%d", errors="coerce"
        )
        if suspended_dates.isna().any() or not suspended_dates.eq(
            pd.Timestamp(trade_date)
        ).all():
            raise ValueError("suspend_d response contains an invalid or different date")
        if not suspended["suspend_type"].astype(str).str.upper().eq("S").all():
            raise ValueError("suspend_d response contains a non-suspension row")
        suspended["symbol"] = suspended["ts_code"].astype(str).str.extract(
            r"^(\d{6})", expand=False
        )
        if suspended["symbol"].isna().any():
            raise ValueError("suspend_d contains invalid ts_code")
        suspended["symbol"] = _apply_symbol_aliases(
            suspended["symbol"], trade_date, symbol_aliases or []
        )
        suspended = (
            suspended.groupby("symbol", as_index=False, sort=False)["suspend_timing"]
            .agg(lambda values: ";".join(sorted({str(value) for value in values if pd.notna(value)})))
        )

    result = result.merge(suspended, on="symbol", how="left", validate="one_to_one")
    result["is_suspended"] = result["symbol"].isin(set(suspended["symbol"]))
    result["is_tradeable"] = ~result["is_suspended"]
    return result[
        [
            "date",
            "symbol",
            "pre_close",
            "up_limit",
            "down_limit",
            "has_price_limit",
            "is_suspended",
            "is_tradeable",
            "suspend_timing",
        ]
    ].sort_values("symbol", kind="mergesort").reset_index(drop=True)


def _prepare_resumable_date_file(
    path: Path, trade_date: str
) -> tuple[bool, dict[str, object] | None]:
    if not path.exists():
        return False, None
    try:
        frame = pd.read_csv(path, dtype={"symbol": str})
    except (OSError, pd.errors.ParserError):
        return False, None
    required = {
        "date",
        "symbol",
        "up_limit",
        "down_limit",
        "is_suspended",
    }
    if frame.empty or not required.issubset(frame.columns):
        return False, None
    dates = pd.to_datetime(frame["date"], errors="coerce")
    if not bool(dates.notna().all() and dates.eq(pd.Timestamp(trade_date)).all()):
        return False, None
    expected_has_limit = frame[["up_limit", "down_limit"]].notna().all(axis=1)
    changed = False
    if "has_price_limit" not in frame.columns:
        insert_at = frame.columns.get_loc("down_limit") + 1
        frame.insert(insert_at, "has_price_limit", expected_has_limit)
        changed = True
    elif not frame["has_price_limit"].astype(bool).eq(expected_has_limit).all():
        return False, None
    if changed:
        _atomic_write_csv(path, frame)
    return True, _unavailable_limit_summary(frame)


def _unavailable_limit_summary(frame: pd.DataFrame) -> dict[str, object] | None:
    unavailable = frame.loc[~frame["has_price_limit"].astype(bool), "symbol"].astype(str)
    if unavailable.empty:
        return None
    symbols = sorted(unavailable.str.zfill(6).unique().tolist())
    return {"count": len(symbols), "symbols": symbols[:20]}


def _normalize_expected_symbols(symbols: set[str]) -> set[str]:
    normalized = {str(value).strip().zfill(6) for value in symbols}
    invalid = sorted(value for value in normalized if not re.fullmatch(r"\d{6}", value))
    if invalid:
        raise ValueError("expected_symbols contains invalid symbols: " + ", ".join(invalid[:10]))
    if not normalized:
        raise ValueError("expected_symbols must not be empty")
    return normalized


def _normalize_symbol_aliases(
    aliases: list[dict[str, str | None]],
) -> list[dict[str, str | None]]:
    normalized: list[dict[str, str | None]] = []
    for raw in aliases:
        if not isinstance(raw, dict):
            raise ValueError("trade-state symbol aliases must be mappings")
        source = str(raw.get("source", "")).strip().upper()
        target = str(raw.get("target", "")).strip().upper()
        if not re.fullmatch(r"\d{6}", source) or not re.fullmatch(r"\d{6}", target):
            raise ValueError("trade-state alias source/target must be six digits")
        start = _normalize_optional_date(raw.get("start"))
        end = _normalize_optional_date(raw.get("end"))
        if start and end and start > end:
            raise ValueError("trade-state alias start must not be after end")
        normalized.append(
            {"source": source, "target": target, "start": start, "end": end}
        )
    return sorted(
        normalized,
        key=lambda item: (
            str(item["source"]),
            str(item["start"] or ""),
            str(item["end"] or ""),
            str(item["target"]),
        ),
    )


def _normalize_optional_date(value: object) -> str | None:
    if value is None or str(value).strip() == "":
        return None
    return _normalize_date(value)


def _apply_symbol_aliases(
    symbols: pd.Series,
    trade_date: str,
    aliases: list[dict[str, str | None]],
) -> pd.Series:
    result = symbols.astype(str).copy()
    for alias in _normalize_symbol_aliases(aliases):
        start = alias["start"]
        end = alias["end"]
        if (start is None or trade_date >= start) and (end is None or trade_date <= end):
            result = result.mask(result.eq(alias["source"]), str(alias["target"]))
    return result


def _resume_signature_compatible(previous: object, current: dict[str, object]) -> bool:
    if previous == current:
        return True
    if not isinstance(previous, dict):
        return False
    legacy = dict(current)
    legacy.pop("symbol_aliases", None)
    legacy.pop("missing_limit_policy", None)
    return previous == legacy


def _read_expected_symbols(root: Path, trade_date: str) -> set[str]:
    path = _date_output_path(root, trade_date)
    if not path.is_file():
        raise FileNotFoundError(f"daily_basic partition not found for {trade_date}: {path}")
    frame = pd.read_csv(path, dtype={"symbol": str}, usecols=["symbol"])
    symbols = set(frame["symbol"].astype(str).str.zfill(6))
    if not symbols:
        raise ValueError(f"daily_basic partition contains no symbols for {trade_date}")
    return symbols


def _expected_symbols_contract(root: Path) -> dict[str, object]:
    manifest_path = root / "_context_manifest.json"
    payload = _load_manifest(manifest_path)
    if not payload:
        raise FileNotFoundError(
            f"daily_basic manifest not found or unreadable: {manifest_path}"
        )
    if payload.get("status") != "complete":
        raise ValueError("daily_basic manifest must be complete before trade-state fetch")
    return {
        "signature": payload.get("signature"),
        "completed_dates": payload.get("completed_dates", []),
    }


def _create_session() -> Any:
    token = os.environ.get("TUSHARE_TOKEN") or fetch_kline._read_dotenv_value(
        "TUSHARE_TOKEN"
    )
    if not token:
        raise RuntimeError("TUSHARE_TOKEN is required for trade-state context")
    return fetch_kline.ts.pro_api(token)


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
    unavailable_limits: dict[str, dict[str, object]],
) -> None:
    requested = set(requested_dates)
    status = (
        "complete"
        if requested.issubset(completed_dates) and not failures
        else "partial"
    )
    payload = {
        "status": status,
        "signature": signature,
        "requested_date_count": len(requested_dates),
        "completed_dates": sorted(completed_dates.intersection(requested)),
        "failed_dates": {key: failures[key] for key in sorted(failures)},
        "unavailable_limits": {
            key: unavailable_limits[key]
            for key in sorted(unavailable_limits)
            if key in requested
        },
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
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)
