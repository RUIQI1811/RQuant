"""Fetch an auditable point-in-time benchmark series for GTJA191."""

from __future__ import annotations

import json
import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from market import fetch_kline


INDEX_DAILY_FIELDS = (
    "ts_code",
    "trade_date",
    "open",
    "close",
    "high",
    "low",
    "pre_close",
    "pct_chg",
    "vol",
    "amount",
)


def fetch_benchmark_index(
    *,
    start: str,
    end: str | None = None,
    index_code: str = "000300.SH",
    output_file: str | Path = "data/context/benchmark_000300.csv",
    manifest_path: str | Path | None = None,
    resume: bool = False,
    session: Any | None = None,
) -> dict[str, object]:
    """Fetch one complete Tushare ``index_daily`` range atomically.

    The output schema deliberately uses ``date, open, close`` so it can be
    passed directly to GTJA191 batch, correlation, and ML commands through
    ``--benchmark-file``. A signature-matching completed file is reusable;
    provider failures leave an explicit partial manifest and no false success.
    """

    start_text = _normalize_date(start)
    end_text = _normalize_date(end or date.today().strftime("%Y%m%d"))
    if start_text > end_text:
        raise ValueError("start must not be after end")
    normalized_code = str(index_code).strip().upper()
    if not normalized_code:
        raise ValueError("index_code must not be blank")

    output = Path(output_file).resolve()
    manifest = (
        Path(manifest_path).resolve()
        if manifest_path
        else output.with_suffix(output.suffix + ".manifest.json")
    )
    signature = {
        "source": "tushare_index_daily",
        "index_code": normalized_code,
        "start": start_text,
        "end": end_text,
        "fields": list(INDEX_DAILY_FIELDS),
    }
    previous = _load_manifest(manifest)
    if resume and previous:
        if previous.get("signature") != signature:
            raise ValueError("benchmark resume manifest does not match requested range/code")
        if previous.get("status") == "complete" and _valid_output(
            output,
            index_code=normalized_code,
            start=start_text,
            end=end_text,
        ):
            return _result(
                ok=True,
                signature=signature,
                output=output,
                manifest=manifest,
                row_count=int(previous.get("row_count", 0)),
                reused=True,
            )

    api = session or _create_session()
    try:
        with fetch_kline._temporary_tushare_network_env():
            raw = api.index_daily(
                ts_code=normalized_code,
                start_date=start_text,
                end_date=end_text,
                fields=",".join(INDEX_DAILY_FIELDS),
            )
        normalized = normalize_benchmark_index(
            raw,
            index_code=normalized_code,
            start=start_text,
            end=end_text,
        )
        _atomic_write_csv(output, normalized)
    except Exception as exc:
        _write_manifest(
            manifest,
            status="partial",
            signature=signature,
            row_count=0,
            error=f"{type(exc).__name__}: {exc}",
        )
        return _result(
            ok=False,
            signature=signature,
            output=output,
            manifest=manifest,
            row_count=0,
            reused=False,
            error=f"{type(exc).__name__}: {exc}",
        )

    _write_manifest(
        manifest,
        status="complete",
        signature=signature,
        row_count=len(normalized),
        error=None,
    )
    return _result(
        ok=True,
        signature=signature,
        output=output,
        manifest=manifest,
        row_count=len(normalized),
        reused=False,
    )


def normalize_benchmark_index(
    frame: pd.DataFrame | None,
    *,
    index_code: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    if frame is None or frame.empty:
        raise ValueError(f"index_daily returned no rows for {index_code}")
    missing = set(INDEX_DAILY_FIELDS).difference(frame.columns)
    if missing:
        raise ValueError("index_daily missing fields: " + ", ".join(sorted(missing)))
    result = frame.loc[:, list(INDEX_DAILY_FIELDS)].copy()
    codes = result["ts_code"].astype(str).str.upper()
    if not codes.eq(index_code).all():
        raise ValueError("index_daily response contains another index code")
    result["date"] = pd.to_datetime(
        result["trade_date"], format="%Y%m%d", errors="coerce"
    )
    if result["date"].isna().any():
        raise ValueError("index_daily contains invalid trade_date")
    start_date = pd.Timestamp(start)
    end_date = pd.Timestamp(end)
    if not result["date"].between(start_date, end_date).all():
        raise ValueError("index_daily response contains dates outside requested range")
    numeric = (
        "open",
        "close",
        "high",
        "low",
        "pre_close",
        "pct_chg",
        "vol",
        "amount",
    )
    for column in numeric:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if result[["open", "close"]].isna().any().any():
        raise ValueError("index_daily contains missing open/close values")
    result = result.rename(columns={"ts_code": "index_code", "vol": "volume"})
    result = result[
        [
            "date",
            "index_code",
            "open",
            "close",
            "high",
            "low",
            "pre_close",
            "pct_chg",
            "volume",
            "amount",
        ]
    ]
    if result.duplicated("date").any():
        raise ValueError("index_daily contains duplicate dates")
    return result.sort_values("date", kind="mergesort").reset_index(drop=True)


def run_from_args(args: Any) -> dict[str, object]:
    return fetch_benchmark_index(
        start=args.start,
        end=args.end,
        index_code=args.index_code,
        output_file=args.out,
        manifest_path=args.manifest,
        resume=args.resume,
    )


def _create_session() -> Any:
    token = os.environ.get("TUSHARE_TOKEN") or fetch_kline._read_dotenv_value(
        "TUSHARE_TOKEN"
    )
    if not token:
        raise RuntimeError("TUSHARE_TOKEN is required for fetch-benchmark")
    return fetch_kline.ts.pro_api(token)


def _normalize_date(value: object) -> str:
    parsed = pd.to_datetime(value, errors="raise")
    return pd.Timestamp(parsed).strftime("%Y%m%d")


def _valid_output(
    path: Path,
    *,
    index_code: str,
    start: str,
    end: str,
) -> bool:
    if not path.exists():
        return False
    try:
        frame = pd.read_csv(path)
        normalize_benchmark_index(
            frame.rename(
                columns={
                    "date": "trade_date",
                    "index_code": "ts_code",
                    "volume": "vol",
                }
            ).assign(
                trade_date=lambda value: pd.to_datetime(value["trade_date"]).dt.strftime(
                    "%Y%m%d"
                )
            ),
            index_code=index_code,
            start=start,
            end=end,
        )
    except (OSError, ValueError, KeyError, pd.errors.ParserError):
        return False
    return True


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
    status: str,
    signature: dict[str, object],
    row_count: int,
    error: str | None,
) -> None:
    payload = {
        "status": status,
        "signature": signature,
        "row_count": int(row_count),
        "error": error,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_write_json(path, payload)


def _result(
    *,
    ok: bool,
    signature: dict[str, object],
    output: Path,
    manifest: Path,
    row_count: int,
    reused: bool,
    error: str | None = None,
) -> dict[str, object]:
    return {
        "ok": ok,
        "index_code": signature["index_code"],
        "start": signature["start"],
        "end": signature["end"],
        "row_count": int(row_count),
        "reused": bool(reused),
        "output_file": str(output),
        "manifest_path": str(manifest),
        "error": error,
    }


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
