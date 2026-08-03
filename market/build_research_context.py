"""Build one point-in-time research context from separately audited sources."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from market.fetch_context import _date_output_path


def build_research_context(
    *,
    daily_basic_dir: str | Path = "data/context/daily_basic",
    industry_file: str | Path = "data/context/sw_industry_membership.csv",
    trade_state_dir: str | Path = "data/context/trade_state",
    output_dir: str | Path = "data/context/research",
    manifest_path: str | Path | None = None,
    resume: bool = False,
) -> dict[str, object]:
    """Join daily cap, interval industry, and exact trade-state data by date/symbol."""

    daily_root = Path(daily_basic_dir).resolve()
    industry_path = Path(industry_file).resolve()
    trade_root = Path(trade_state_dir).resolve()
    destination = Path(output_dir).resolve()
    manifest = (
        Path(manifest_path).resolve()
        if manifest_path
        else destination / "_context_manifest.json"
    )
    source_manifests = {
        "daily_basic": daily_root / "_context_manifest.json",
        "industry": industry_path.with_suffix(industry_path.suffix + ".manifest.json"),
        "trade_state": trade_root / "_context_manifest.json",
    }
    source_payloads = {
        name: _require_complete_manifest(path)
        for name, path in source_manifests.items()
    }
    daily_dates = [str(value) for value in source_payloads["daily_basic"].get("completed_dates", [])]
    trade_dates = set(
        str(value) for value in source_payloads["trade_state"].get("completed_dates", [])
    )
    if not daily_dates:
        raise ValueError("daily_basic manifest contains no completed dates")
    missing_trade_dates = sorted(set(daily_dates).difference(trade_dates))
    if missing_trade_dates:
        raise ValueError(
            "trade-state context is missing daily_basic dates: "
            + ", ".join(missing_trade_dates[:10])
        )
    if not industry_path.is_file():
        raise FileNotFoundError(f"industry membership file not found: {industry_path}")

    signature = {
        "source": "rquant_point_in_time_context_bundle",
        "context_schema_version": 2,
        "daily_basic_dir": str(daily_root),
        "industry_file": str(industry_path),
        "trade_state_dir": str(trade_root),
        "source_manifest_hashes": {
            name: _manifest_contract_hash(path) for name, path in source_manifests.items()
        },
        "source_manifest_hash_mode": "canonical_without_updated_at_v1",
        "requested_dates": daily_dates,
        "industry_mapping": {
            "sector": "SW L1",
            "industry": "SW L2",
            "subindustry": "SW L3",
            "out_date_semantics": "exclusive",
        },
    }
    previous = _load_manifest(manifest)
    if resume and previous and previous.get("signature") != signature:
        if _legacy_signature_can_rebuild(previous.get("signature"), signature):
            previous = {}
        else:
            raise ValueError("research-context resume manifest does not match source manifests")
    completed = set(
        str(value) for value in previous.get("completed_dates", [])
    ) if resume else set()
    failures: dict[str, str] = {}
    destination.mkdir(parents=True, exist_ok=True)

    membership = pd.read_csv(industry_path, dtype={"symbol": str})
    membership["symbol"] = membership["symbol"].astype(str).str.zfill(6)
    membership["in_date"] = pd.to_datetime(membership["in_date"], errors="coerce")
    membership["out_date"] = pd.to_datetime(membership["out_date"], errors="coerce")
    required_membership = {
        "symbol",
        "sector",
        "industry",
        "subindustry",
        "in_date",
        "out_date",
    }
    if not required_membership.issubset(membership.columns):
        raise ValueError("industry membership file is missing canonical columns")
    if membership["in_date"].isna().any():
        raise ValueError("industry membership file contains invalid in_date")

    reused_count = 0
    fetched_count = 0
    classified_rows = 0
    total_rows = 0
    for trade_date in daily_dates:
        output_path = _date_output_path(destination, trade_date)
        if resume and trade_date in completed and _valid_output(output_path, trade_date):
            reused_count += 1
            continue
        try:
            daily_path = _date_output_path(daily_root, trade_date)
            state_path = _date_output_path(trade_root, trade_date)
            daily = pd.read_csv(daily_path, dtype={"symbol": str})
            state = pd.read_csv(state_path, dtype={"symbol": str})
            normalized = _build_one_date(
                daily,
                state,
                membership,
                trade_date=trade_date,
            )
            _atomic_write_csv(output_path, normalized)
        except Exception as exc:
            failures[trade_date] = f"{type(exc).__name__}: {exc}"
            completed.discard(trade_date)
        else:
            completed.add(trade_date)
            failures.pop(trade_date, None)
            fetched_count += 1
            total_rows += len(normalized)
            classified_rows += int(normalized["industry"].notna().sum())
        _write_manifest(
            manifest,
            signature=signature,
            completed=completed,
            failures=failures,
            row_count=total_rows,
            classified_rows=classified_rows,
        )

    ok = set(daily_dates).issubset(completed) and not failures
    _write_manifest(
        manifest,
        signature=signature,
        completed=completed,
        failures=failures,
        row_count=total_rows,
        classified_rows=classified_rows,
    )
    return {
        "ok": ok,
        "requested_date_count": len(daily_dates),
        "built_date_count": fetched_count,
        "reused_date_count": reused_count,
        "failed_dates": sorted(failures),
        "row_count_written_this_run": total_rows,
        "classified_rows_written_this_run": classified_rows,
        "output_dir": str(destination),
        "manifest_path": str(manifest),
    }


def _build_one_date(
    daily: pd.DataFrame,
    state: pd.DataFrame,
    membership: pd.DataFrame,
    *,
    trade_date: str,
) -> pd.DataFrame:
    expected = pd.Timestamp(trade_date)
    for name, frame in (("daily_basic", daily), ("trade_state", state)):
        required = {"date", "symbol"}
        if frame.empty or not required.issubset(frame.columns):
            raise ValueError(f"{name} partition is empty or missing date/symbol")
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
        if frame["date"].isna().any() or not frame["date"].eq(expected).all():
            raise ValueError(f"{name} partition contains another date")
        if frame.duplicated(["date", "symbol"]).any():
            raise ValueError(f"{name} partition contains duplicate date/symbol rows")

    active = membership.loc[
        membership["in_date"].le(expected)
        & (membership["out_date"].isna() | membership["out_date"].gt(expected))
    ].copy()
    active = active.sort_values(
        ["symbol", "in_date", "l3_code"], kind="mergesort"
    ).drop_duplicates("symbol", keep="last")
    industry_columns = [
        "symbol",
        "sector",
        "industry",
        "subindustry",
        "l1_code",
        "l2_code",
        "l3_code",
    ]
    if active.empty:
        raise ValueError("no active SW membership exists for this date")
    result = daily.merge(
        active[industry_columns], on="symbol", how="left", validate="one_to_one"
    )
    state_payload = state.drop(columns="date")
    result = result.merge(
        state_payload,
        on="symbol",
        how="left",
        validate="one_to_one",
        suffixes=("", "__trade_state"),
    )
    if "has_price_limit" in result.columns:
        result["has_price_limit"] = result["has_price_limit"].eq(True)
    elif {"up_limit", "down_limit"}.issubset(result.columns):
        result["has_price_limit"] = result[["up_limit", "down_limit"]].notna().all(
            axis=1
        )
    if "is_suspended" in result.columns:
        result["is_suspended"] = result["is_suspended"].eq(True)
    else:
        result["is_suspended"] = False
    if "is_tradeable" in result.columns:
        supplied_tradeable = result["is_tradeable"]
        result["is_tradeable"] = supplied_tradeable.eq(True) | (
            supplied_tradeable.isna() & ~result["is_suspended"]
        )
    else:
        result["is_tradeable"] = ~result["is_suspended"]
    return result.sort_values("symbol", kind="mergesort").reset_index(drop=True)


def _require_complete_manifest(path: Path) -> dict[str, object]:
    payload = _load_manifest(path)
    if not payload:
        raise FileNotFoundError(f"source manifest not found or unreadable: {path}")
    if payload.get("status") != "complete":
        raise ValueError(f"source manifest is not complete: {path}")
    return payload


def _valid_output(path: Path, trade_date: str) -> bool:
    if not path.exists():
        return False
    try:
        frame = pd.read_csv(path, dtype={"symbol": str})
    except (OSError, pd.errors.ParserError):
        return False
    required = {
        "date",
        "symbol",
        "market_cap",
        "industry",
        "up_limit",
        "down_limit",
    }
    if frame.empty or not required.issubset(frame.columns):
        return False
    dates = pd.to_datetime(frame["date"], errors="coerce")
    return bool(dates.notna().all() and dates.eq(pd.Timestamp(trade_date)).all())


def _manifest_contract_hash(path: Path) -> str:
    payload = _load_manifest(path)
    if not payload:
        raise ValueError(f"source manifest is not valid JSON: {path}")
    canonical = dict(payload)
    canonical.pop("updated_at", None)
    encoded = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _legacy_signature_can_rebuild(
    previous: object, current: dict[str, object]
) -> bool:
    """Allow a one-time safe rebuild from the former timestamp-sensitive hash."""

    if not isinstance(previous, dict):
        return False
    legacy_core = dict(previous)
    current_core = dict(current)
    legacy_core.pop("source_manifest_hashes", None)
    legacy_core.pop("source_manifest_hash_mode", None)
    legacy_core.pop("context_schema_version", None)
    current_core.pop("source_manifest_hashes", None)
    current_core.pop("source_manifest_hash_mode", None)
    current_core.pop("context_schema_version", None)
    return legacy_core == current_core


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
    completed: set[str],
    failures: dict[str, str],
    row_count: int,
    classified_rows: int,
) -> None:
    requested = set(str(value) for value in signature["requested_dates"])
    status = "complete" if requested.issubset(completed) and not failures else "partial"
    payload = {
        "status": status,
        "signature": signature,
        "requested_date_count": len(requested),
        "completed_dates": sorted(completed.intersection(requested)),
        "failed_dates": {key: failures[key] for key in sorted(failures)},
        "row_count_written_this_run": int(row_count),
        "classified_rows_written_this_run": int(classified_rows),
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
