"""Resumable Tushare Shenwan point-in-time industry membership fetcher."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from market import fetch_kline


CLASSIFY_FIELDS = (
    "index_code",
    "industry_name",
    "parent_code",
    "level",
    "industry_code",
    "is_pub",
    "src",
)
MEMBER_FIELDS = (
    "l1_code",
    "l1_name",
    "l2_code",
    "l2_name",
    "l3_code",
    "l3_name",
    "ts_code",
    "name",
    "in_date",
    "out_date",
    "is_new",
)
EQUITY_TS_CODE_PATTERN = r"^\d{6}\.(?:SZ|SH|BJ)$"


def fetch_sw_industry_membership(
    *,
    output_file: str | Path = "data/context/sw_industry_membership.csv",
    classification_file: str | Path | None = None,
    manifest_path: str | Path | None = None,
    parts_dir: str | Path | None = None,
    source: str = "SW2021",
    resume: bool = False,
    max_requests_per_minute: int = 180,
    max_industries: int | None = None,
    session: Any | None = None,
) -> dict[str, object]:
    """Fetch complete current and historical membership for every SW L3 node.

    ``index_member_all`` is capped at 2,000 rows.  Querying every L3 industry
    separately keeps each request below that cap and makes truncation
    detectable.  One part file is committed only after both current and
    historical membership calls for that L3 node validate successfully.
    """

    normalized_source = str(source).strip().upper()
    if normalized_source not in {"SW2014", "SW2021"}:
        raise ValueError("source must be SW2014 or SW2021")
    if max_requests_per_minute < 0:
        raise ValueError("max_requests_per_minute must be non-negative")
    if max_industries is not None and max_industries <= 0:
        raise ValueError("max_industries must be positive")

    output = Path(output_file).resolve()
    classification_output = (
        Path(classification_file).resolve()
        if classification_file
        else output.with_name(f"{output.stem}_classification.csv")
    )
    manifest = (
        Path(manifest_path).resolve()
        if manifest_path
        else output.with_suffix(output.suffix + ".manifest.json")
    )
    parts = (
        Path(parts_dir).resolve()
        if parts_dir
        else output.parent / ".sw_industry_membership_parts"
    )
    parts.mkdir(parents=True, exist_ok=True)
    api = session or _create_session()
    limiter = (
        fetch_kline.RequestRateLimiter(max_requests_per_minute)
        if max_requests_per_minute > 0
        else None
    )

    with fetch_kline._temporary_tushare_network_env():
        classification_frames: list[pd.DataFrame] = []
        for level in ("L1", "L2", "L3"):
            if limiter is not None:
                limiter.wait()
            classification_raw = api.index_classify(
                level=level,
                src=normalized_source,
                fields=",".join(CLASSIFY_FIELDS),
            )
            if classification_raw is not None and len(classification_raw) >= 2000:
                raise ValueError(
                    f"index_classify reached the 2000-row provider cap for {level}; "
                    "completeness cannot be proven"
                )
            classification_frames.append(
                normalize_classification(
                    classification_raw,
                    source=normalized_source,
                    expected_level=level,
                )
            )
        classification = pd.concat(
            classification_frames, ignore_index=True, sort=False
        ).sort_values(["level", "index_code"], kind="mergesort")
        _atomic_write_csv(classification_output, classification.reset_index(drop=True))
        codes = classification.loc[
            classification["level"].astype(str).str.upper().eq("L3"), "index_code"
        ].tolist()
        if max_industries is not None:
            codes = codes[: int(max_industries)]

        signature = {
            "source": "tushare_index_member_all_by_l3",
            "classification_source": normalized_source,
            "classification_fields": list(CLASSIFY_FIELDS),
            "classification_file": str(classification_output),
            "membership_fields": list(MEMBER_FIELDS),
            "l3_codes": codes,
            "max_industries": max_industries,
            "invalid_member_policy": "exclude_non_equity_ts_code_v1",
        }
        previous = _load_manifest(manifest)
        if resume and previous and not _resume_signature_compatible(
            previous.get("signature"), signature
        ):
            raise ValueError(
                "industry resume manifest does not match classification/version"
            )
        completed = set(
            str(value) for value in previous.get("completed_l3_codes", [])
        ) if resume else set()
        reused: list[str] = []
        fetched: list[str] = []
        failures: dict[str, str] = {}
        excluded_members: dict[str, list[dict[str, str]]] = (
            {
                str(code): [dict(item) for item in items]
                for code, items in (previous.get("excluded_members") or {}).items()
            }
            if resume
            else {}
        )

        for code in codes:
            part_path = parts / f"{code.replace('.', '_')}.csv"
            if resume and code in completed and _valid_part(part_path, code):
                reused.append(code)
                continue
            try:
                frames: list[pd.DataFrame] = []
                for is_new in ("Y", "N"):
                    if limiter is not None:
                        limiter.wait()
                    raw = api.index_member_all(
                        l3_code=code,
                        is_new=is_new,
                        fields=",".join(MEMBER_FIELDS),
                    )
                    if raw is not None and len(raw) >= 2000:
                        raise ValueError(
                            f"index_member_all reached the 2000-row provider cap "
                            f"for {code}/{is_new}; completeness cannot be proven"
                        )
                    if raw is not None and not raw.empty:
                        frames.append(raw)
                if frames:
                    combined_raw = pd.concat(frames, ignore_index=True, sort=False)
                    excluded = _non_equity_member_records(combined_raw)
                    normalized = normalize_industry_membership(
                        combined_raw,
                        expected_l3_code=code,
                    )
                else:
                    excluded = []
                    normalized = _empty_membership_frame()
                _atomic_write_csv(part_path, normalized)
            except Exception as exc:
                failures[code] = f"{type(exc).__name__}: {exc}"
                completed.discard(code)
            else:
                completed.add(code)
                fetched.append(code)
                failures.pop(code, None)
                if excluded:
                    excluded_members[code] = excluded
                else:
                    excluded_members.pop(code, None)
            _write_manifest(
                manifest,
                signature=signature,
                completed=completed,
                failures=failures,
                row_count=0,
                excluded_members=excluded_members,
            )

    requested = set(codes)
    ok = requested.issubset(completed) and not failures
    row_count = 0
    if ok:
        combined = pd.concat(
            [
                pd.read_csv(
                    parts / f"{code.replace('.', '_')}.csv",
                    dtype={"symbol": str},
                )
                for code in codes
            ],
            ignore_index=True,
            sort=False,
        )
        combined["symbol"] = combined["symbol"].astype(str).str.zfill(6)
        combined = combined.drop_duplicates(
            ["symbol", "l3_code", "in_date", "out_date"], keep="last"
        ).sort_values(["symbol", "in_date", "l3_code"], kind="mergesort")
        _atomic_write_csv(output, combined.reset_index(drop=True))
        row_count = len(combined)
    _write_manifest(
        manifest,
        signature=signature,
        completed=completed,
        failures=failures,
        row_count=row_count,
        excluded_members=excluded_members,
    )
    return {
        "ok": ok,
        "classification_source": normalized_source,
        "industry_count": len(codes),
        "fetched_industry_count": len(fetched),
        "reused_industry_count": len(reused),
        "row_count": row_count,
        "failed_industries": sorted(failures),
        "excluded_member_count": sum(len(items) for items in excluded_members.values()),
        "output_file": str(output),
        "classification_file": str(classification_output),
        "manifest_path": str(manifest),
        "parts_dir": str(parts),
    }


def normalize_classification(
    frame: pd.DataFrame | None, *, source: str, expected_level: str = "L3"
) -> pd.DataFrame:
    if frame is None or frame.empty:
        raise ValueError(
            f"index_classify returned no {expected_level} rows for {source}"
        )
    required = {"index_code", "industry_name", "level"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError("index_classify missing fields: " + ", ".join(sorted(missing)))
    result = frame.copy()
    result["index_code"] = result["index_code"].astype(str).str.strip().str.upper()
    if result["index_code"].eq("").any() or result["index_code"].duplicated().any():
        raise ValueError("index_classify contains blank or duplicate index codes")
    if not result["level"].astype(str).str.upper().eq(expected_level.upper()).all():
        raise ValueError("index_classify response contains another level")
    return result.sort_values("index_code", kind="mergesort").reset_index(drop=True)


def normalize_industry_membership(
    frame: pd.DataFrame | None, *, expected_l3_code: str
) -> pd.DataFrame:
    if frame is None or frame.empty:
        raise ValueError(f"index_member_all returned no rows for {expected_l3_code}")
    missing = set(MEMBER_FIELDS).difference(frame.columns)
    if missing:
        raise ValueError("index_member_all missing fields: " + ", ".join(sorted(missing)))
    result = frame.loc[:, list(MEMBER_FIELDS)].copy()
    for column in ("l1_code", "l2_code", "l3_code", "ts_code", "is_new"):
        result[column] = result[column].astype(str).str.strip().str.upper()
    if not result["l3_code"].eq(expected_l3_code.upper()).all():
        raise ValueError("index_member_all response contains another L3 code")
    result = result.loc[
        result["ts_code"].str.fullmatch(EQUITY_TS_CODE_PATTERN, na=False)
    ].copy()
    if result.empty:
        return _empty_membership_frame()
    result["symbol"] = result["ts_code"].str.extract(r"^(\d{6})", expand=False)
    result["in_date"] = pd.to_datetime(
        result["in_date"], format="%Y%m%d", errors="coerce"
    )
    if result["in_date"].isna().any():
        raise ValueError("index_member_all contains invalid in_date")
    result["out_date"] = pd.to_datetime(
        result["out_date"].replace({"": pd.NA, "None": pd.NA, "nan": pd.NA}),
        format="%Y%m%d",
        errors="coerce",
    )
    invalid_interval = result["out_date"].notna() & result["out_date"].lt(
        result["in_date"]
    )
    if invalid_interval.any():
        raise ValueError("index_member_all contains out_date before in_date")
    result = result.rename(
        columns={
            "l1_name": "sector",
            "l2_name": "industry",
            "l3_name": "subindustry",
            "name": "stock_name",
        }
    )
    result["in_date"] = result["in_date"].dt.strftime("%Y-%m-%d")
    result["out_date"] = result["out_date"].dt.strftime("%Y-%m-%d")
    result = result[
        [
            "symbol",
            "ts_code",
            "sector",
            "industry",
            "subindustry",
            "l1_code",
            "l2_code",
            "l3_code",
            "in_date",
            "out_date",
            "is_new",
            "stock_name",
        ]
    ]
    return result.drop_duplicates().sort_values(
        ["symbol", "in_date", "l3_code"], kind="mergesort"
    ).reset_index(drop=True)


def _valid_part(path: Path, l3_code: str) -> bool:
    if not path.exists():
        return False
    try:
        frame = pd.read_csv(path, dtype={"symbol": str})
    except (OSError, pd.errors.ParserError):
        return False
    required = {"symbol", "l3_code", "in_date", "out_date"}
    if not required.issubset(frame.columns):
        return False
    return bool(
        frame.empty
        or frame["l3_code"].astype(str).str.upper().eq(l3_code.upper()).all()
    )


def _empty_membership_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "symbol",
            "ts_code",
            "sector",
            "industry",
            "subindustry",
            "l1_code",
            "l2_code",
            "l3_code",
            "in_date",
            "out_date",
            "is_new",
            "stock_name",
        ]
    )


def _non_equity_member_records(frame: pd.DataFrame) -> list[dict[str, str]]:
    """Return a compact audit trail for provider rows outside A-share identifiers."""

    ts_code = frame["ts_code"].astype(str).str.strip().str.upper()
    invalid = frame.loc[
        ~ts_code.str.fullmatch(EQUITY_TS_CODE_PATTERN, na=False),
        ["ts_code", "name", "in_date", "out_date"],
    ].copy()
    records: list[dict[str, str]] = []
    for row in invalid.itertuples(index=False):
        records.append(
            {
                "ts_code": "" if pd.isna(row.ts_code) else str(row.ts_code),
                "name": "" if pd.isna(row.name) else str(row.name),
                "in_date": "" if pd.isna(row.in_date) else str(row.in_date),
                "out_date": "" if pd.isna(row.out_date) else str(row.out_date),
            }
        )
    return records


def _resume_signature_compatible(previous: object, current: dict[str, object]) -> bool:
    if previous == current:
        return True
    if not isinstance(previous, dict):
        return False
    legacy = dict(current)
    legacy.pop("invalid_member_policy", None)
    return previous == legacy


def _create_session() -> Any:
    token = os.environ.get("TUSHARE_TOKEN") or fetch_kline._read_dotenv_value(
        "TUSHARE_TOKEN"
    )
    if not token:
        raise RuntimeError("TUSHARE_TOKEN is required for industry membership")
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
    completed: set[str],
    failures: dict[str, str],
    row_count: int,
    excluded_members: dict[str, list[dict[str, str]]],
) -> None:
    requested = set(str(value) for value in signature["l3_codes"])
    status = "complete" if requested.issubset(completed) and not failures else "partial"
    payload = {
        "status": status,
        "signature": signature,
        "completed_l3_codes": sorted(completed.intersection(requested)),
        "failed_industries": {key: failures[key] for key in sorted(failures)},
        "excluded_members": {
            key: excluded_members[key] for key in sorted(excluded_members)
        },
        "row_count": int(row_count),
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
