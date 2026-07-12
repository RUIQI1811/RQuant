"""Read-only environment, configuration, and local-data diagnostics for RQuant."""
from __future__ import annotations

import csv
import importlib
import importlib.metadata
import json
import os
import platform
import re
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from domain.reports import SystemDoctorResult


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLE_SIZE = 25
DEFAULT_MAX_MARKET_DATA_AGE_DAYS = 7
REQUIRED_CONFIGS: dict[str, tuple[str, ...]] = {
    "config/fetch_kline.yaml": ("start", "end", "stocklist", "out", "workers"),
    "config/factors.yaml": ("default_status", "factors"),
    "config/gtja191_factors.yaml": ("default_status", "factors"),
    "config/rules_preselect.yaml": ("global", "stock_pool"),
}
OPTIONAL_CONFIGS = ("config/dashboard.yaml", "config/gemini_review.yaml")
IMPORT_NAMES = {
    "google-genai": "google.genai",
    "protobuf": "google.protobuf",
    "pyyaml": "yaml",
    "scikit-learn": "sklearn",
}
SECRET_NAMES = ("TUSHARE_TOKEN", "GEMINI_API_KEY")
MARKET_COLUMNS = {"date", "open", "close", "high", "low", "volume"}


def run_system_doctor(
    *,
    project_root: str | Path = PROJECT_ROOT,
    data_dir: str | Path = "data/raw",
    output_path: str | Path | None = None,
    deep: bool = False,
    as_of_date: date | str | None = None,
    max_market_data_age_days: int | None = DEFAULT_MAX_MARKET_DATA_AGE_DAYS,
) -> SystemDoctorResult:
    """Run deterministic, read-only checks and optionally persist an atomic JSON report."""

    root = Path(project_root).resolve()
    dependencies = _check_dependencies(root)
    configs = _check_configs(root)
    secrets = _check_secrets(root)
    workflow_artifacts = _check_workflow_artifacts(root)
    market_data = _check_market_data(
        _resolve(root, data_dir),
        sample_size=None if deep else DEFAULT_SAMPLE_SIZE,
        as_of_date=_normalize_as_of_date(as_of_date),
        max_age_days=max_market_data_age_days,
    )
    sections = {
        "dependencies": dependencies,
        "configs": configs,
        "secrets": secrets,
        "workflow_artifacts": workflow_artifacts,
        "market_data": market_data,
    }
    error_count = sum(section["error_count"] for section in sections.values())
    warning_count = sum(section["warning_count"] for section in sections.values())
    status = "error" if error_count else ("warning" if warning_count else "ok")
    report: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_root": str(root),
        "status": status,
        "ok": error_count == 0,
        "summary": {
            "error_count": error_count,
            "warning_count": warning_count,
            "deep_scan": deep,
        },
        "runtime": {
            "status": "ok",
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        **sections,
    }
    if output_path is not None:
        resolved_output = _resolve(root, output_path)
        report["output_path"] = str(resolved_output)
        _atomic_write_json(resolved_output, report)
    return SystemDoctorResult(report)


def _check_dependencies(root: Path) -> dict[str, Any]:
    required_path = root / "requirements.txt"
    optional_path = root / "requirements-ml.txt"
    items: list[dict[str, Any]] = []
    errors: list[str] = []
    warnings: list[str] = []
    if not required_path.is_file():
        errors.append("missing requirements.txt")
        required_specs: list[tuple[str, str | None]] = []
    else:
        required_specs = _parse_requirements(required_path)
    optional_specs = _parse_requirements(optional_path) if optional_path.is_file() else []
    if not optional_path.is_file():
        warnings.append("missing requirements-ml.txt")

    required_names = {name.casefold() for name, _ in required_specs}
    optional_specs = [spec for spec in optional_specs if spec[0].casefold() not in required_names]
    for name, expected in required_specs:
        item = _check_distribution(name, expected=expected, required=True)
        items.append(item)
        if item["status"] == "error":
            errors.append(item["message"])
    for name, expected in optional_specs:
        item = _check_distribution(name, expected=expected, required=False)
        items.append(item)
        if item["status"] == "warning":
            warnings.append(item["message"])
    return _section(items=items, errors=errors, warnings=warnings)


def _parse_requirements(path: Path) -> list[tuple[str, str | None]]:
    specs: list[tuple[str, str | None]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or line.startswith(("-r", "--requirement")):
            continue
        match = re.match(r"^([A-Za-z0-9_.-]+)(?:\[[^]]+\])?(.*)$", line)
        if not match:
            continue
        name, constraint = match.groups()
        exact = constraint[2:].strip() if constraint.startswith("==") else None
        specs.append((name, exact or None))
    return specs


def _check_distribution(
    name: str,
    *,
    expected: str | None,
    required: bool,
) -> dict[str, Any]:
    normalized = name.casefold()
    module = IMPORT_NAMES.get(normalized, normalized.replace("-", "_"))
    try:
        installed_version = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        installed_version = None
    import_error = None
    try:
        importlib.import_module(module)
        importable = True
    except Exception as exc:
        importable = False
        import_error = _summarize_import_error(exc)
    version_ok = expected is None or installed_version == expected
    if installed_version is None:
        status = "error" if required else "warning"
        kind = "required" if required else "optional"
        message = f"{kind} dependency unavailable: {name}"
    elif not importable:
        status = "error" if required else "warning"
        kind = "required" if required else "optional"
        message = f"{kind} dependency import failed: {name} ({import_error})"
    elif not version_ok:
        status = "error" if required else "warning"
        kind = "required" if required else "optional"
        message = (
            f"{kind} dependency version mismatch: {name} "
            f"(expected {expected}, found {installed_version})"
        )
    else:
        status = "ok"
        message = f"dependency available: {name}"
    return {
        "name": name,
        "module": module,
        "required": required,
        "expected_version": expected,
        "installed_version": installed_version,
        "importable": importable,
        "import_error": import_error,
        "status": status,
        "message": message,
    }


def _summarize_import_error(exc: Exception) -> str:
    message = str(exc).casefold()
    if "libomp.dylib" in message:
        return "missing native library libomp.dylib"
    if "dll load failed" in message or "specified module could not be found" in message:
        return (
            "Windows native DLL load failed; activate the intended environment and "
            "reinstall the dependency"
        )
    return type(exc).__name__


def _check_configs(root: Path) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    errors: list[str] = []
    warnings: list[str] = []
    yaml_loader, yaml_error = _yaml_loader()
    for relative_path, required_keys in REQUIRED_CONFIGS.items():
        item = _check_yaml_config(
            root / relative_path,
            required_keys,
            yaml_loader,
            yaml_error,
            required=True,
        )
        items.append(item)
        if item["status"] == "error":
            errors.append(item["message"])
    for relative_path in OPTIONAL_CONFIGS:
        item = _check_yaml_config(
            root / relative_path,
            (),
            yaml_loader,
            yaml_error,
            required=False,
        )
        items.append(item)
        if item["status"] == "warning":
            warnings.append(item["message"])
    stocklist = _check_stocklist(root / "config/stocklist.csv")
    items.append(stocklist)
    if stocklist["status"] == "error":
        errors.append(stocklist["message"])
    return _section(items=items, errors=errors, warnings=warnings)


def _yaml_loader() -> tuple[Any | None, str | None]:
    try:
        import yaml
    except ImportError:
        return None, "PyYAML is unavailable"
    return yaml.safe_load, None


def _check_yaml_config(
    path: Path,
    required_keys: tuple[str, ...],
    loader: Any | None,
    loader_error: str | None,
    *,
    required: bool,
) -> dict[str, Any]:
    name = path.name
    failure_status = "error" if required else "warning"
    if not path.is_file():
        return {
            "path": str(path),
            "required": required,
            "status": failure_status,
            "message": f"missing {'required' if required else 'optional'} config: {name}",
        }
    if loader is None:
        return {
            "path": str(path),
            "required": required,
            "status": failure_status,
            "message": f"cannot parse {name}: {loader_error}",
        }
    try:
        payload = loader(path.read_text(encoding="utf-8"))
    except Exception as exc:  # PyYAML exposes several parser exception types.
        return {
            "path": str(path),
            "required": required,
            "status": failure_status,
            "message": f"invalid YAML config {name}: {type(exc).__name__}",
        }
    if not isinstance(payload, dict):
        return {
            "path": str(path),
            "required": required,
            "status": failure_status,
            "message": f"config must contain a mapping: {name}",
        }
    missing = [key for key in required_keys if key not in payload]
    if missing:
        return {
            "path": str(path),
            "required": required,
            "status": failure_status,
            "message": f"config {name} missing keys: {', '.join(missing)}",
        }
    value_error = _config_value_error(name, payload)
    if value_error:
        return {
            "path": str(path),
            "required": required,
            "status": failure_status,
            "message": f"config {name} has invalid value: {value_error}",
        }
    return {
        "path": str(path),
        "required": required,
        "status": "ok",
        "message": f"config valid: {name}",
    }


def _config_value_error(name: str, payload: dict[str, Any]) -> str | None:
    if name != "fetch_kline.yaml":
        return None
    workers = payload.get("workers")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        return "workers must be a positive integer"
    if "max_requests_per_minute" in payload:
        limit = payload["max_requests_per_minute"]
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            return "max_requests_per_minute must be a non-negative integer"
    return None


def _check_stocklist(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "path": str(path),
            "required": True,
            "status": "error",
            "message": "missing required config: stocklist.csv",
        }
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = set(reader.fieldnames or ())
            missing = sorted({"symbol", "ts_code"} - columns)
            row_count = 0
            invalid_symbols = 0
            for row in reader:
                row_count += 1
                if not re.fullmatch(r"\d{6}", str(row.get("symbol", "")).strip()):
                    invalid_symbols += 1
    except (OSError, csv.Error, UnicodeError) as exc:
        return {
            "path": str(path),
            "required": True,
            "status": "error",
            "message": f"cannot read stocklist.csv: {type(exc).__name__}",
        }
    if missing:
        message = f"stocklist.csv missing columns: {', '.join(missing)}"
        status = "error"
    elif row_count == 0:
        message = "stocklist.csv has no rows"
        status = "error"
    elif invalid_symbols:
        message = f"stocklist.csv has {invalid_symbols} non-six-digit symbols"
        status = "error"
    else:
        message = "stocklist.csv valid"
        status = "ok"
    return {
        "path": str(path),
        "required": True,
        "status": status,
        "message": message,
        "row_count": row_count,
        "invalid_symbol_count": invalid_symbols,
    }


def _check_secrets(root: Path) -> dict[str, Any]:
    dotenv_names = _dotenv_keys(root / ".env")
    items: list[dict[str, Any]] = []
    warnings: list[str] = []
    for name in SECRET_NAMES:
        if os.environ.get(name):
            source, configured = "environment", True
        elif name in dotenv_names:
            source, configured = ".env", True
        else:
            source, configured = None, False
        status = "ok" if configured else "warning"
        message = f"{name} configured" if configured else f"{name} not configured"
        items.append(
            {
                "name": name,
                "configured": configured,
                "source": source,
                "status": status,
                "message": message,
            }
        )
        if not configured:
            warnings.append(message)
    return _section(items=items, errors=[], warnings=warnings)


def _check_workflow_artifacts(root: Path) -> dict[str, Any]:
    candidates_path = root / "data/candidates/candidates_latest.json"
    if not candidates_path.is_file():
        return _section(items=[], errors=[], warnings=[])
    items: list[dict[str, Any]] = []
    errors: list[str] = []
    warnings: list[str] = []
    candidates = _read_artifact_json(candidates_path, "latest candidates", errors)
    if candidates is None:
        return _section(items=items, errors=errors, warnings=warnings)
    pick_date = str(candidates.get("pick_date", "")).strip()
    try:
        normalized_pick_date = date.fromisoformat(pick_date).isoformat()
    except ValueError:
        errors.append("latest candidates pick_date must use YYYY-MM-DD")
        return _section(items=items, errors=errors, warnings=warnings)
    if pick_date != normalized_pick_date:
        errors.append("latest candidates pick_date must use YYYY-MM-DD")
        return _section(items=items, errors=errors, warnings=warnings)
    raw_candidates = candidates.get("candidates")
    if not isinstance(raw_candidates, list):
        errors.append("latest candidates field candidates must be a list")
        return _section(items=items, errors=errors, warnings=warnings)
    codes: list[str] = []
    for candidate in raw_candidates:
        if not isinstance(candidate, dict):
            errors.append("latest candidates entries must be JSON objects")
            continue
        symbol = str(candidate.get("code", "")).strip()
        if symbol.isdigit() and len(symbol) <= 6:
            symbol = symbol.zfill(6)
        if not re.fullmatch(r"\d{6}", symbol):
            errors.append(f"latest candidates contains invalid symbol: {symbol!r}")
            continue
        codes.append(symbol)
    if len(codes) != len(set(codes)):
        errors.append("latest candidates contains duplicate symbols")
    items.append(
        {
            "type": "candidates",
            "path": str(candidates_path),
            "pick_date": pick_date,
            "candidate_count": len(codes),
            "status": "error" if errors else "ok",
        }
    )

    chart_dir = root / "data/kline" / pick_date
    chart_manifest_path = chart_dir / "export_manifest.json"
    expected_charts = {code: chart_dir / f"{code}_day.jpg" for code in codes}
    missing_charts = [code for code, path in expected_charts.items() if not path.is_file()]
    chart_manifest = (
        _read_artifact_json(chart_manifest_path, "chart export manifest", errors)
        if chart_manifest_path.is_file()
        else None
    )
    if chart_manifest is None:
        existing_count = sum(path.is_file() for path in expected_charts.values())
        if existing_count:
            warnings.append(
                "candidate charts exist without export_manifest.json; point-in-time boundary is unverified"
            )
        elif codes:
            warnings.append(f"candidate charts have not been exported for {pick_date}")
        chart_status = "warning" if codes else "ok"
    else:
        chart_status = str(chart_manifest.get("status"))
        if chart_status in {"partial", "interrupted"}:
            errors.append(f"chart export manifest status is {chart_status}")
        elif chart_status != "complete":
            errors.append(f"chart export manifest has unknown status: {chart_status}")
        if chart_manifest.get("pick_date") != pick_date:
            errors.append("chart export manifest pick_date does not match latest candidates")
        chart_failed_count = _artifact_count(
            chart_manifest.get("failed_count", 0),
            "chart export manifest failed_count",
            errors,
        )
        chart_success_count = _artifact_count(
            chart_manifest.get("success_count", -1),
            "chart export manifest success_count",
            errors,
        )
        if chart_failed_count != 0:
            errors.append("chart export manifest reports failed symbols")
        if chart_success_count != len(codes):
            errors.append("chart export manifest success_count does not match candidates")
        results = chart_manifest.get("results", {})
        if not isinstance(results, dict):
            errors.append("chart export manifest results must be an object")
        else:
            mismatched_dates = [
                code
                for code in codes
                if not isinstance(results.get(code), dict)
                or results[code].get("chart_end_date") != pick_date
            ]
            if mismatched_dates:
                errors.append(
                    f"chart point-in-time date mismatch for {len(mismatched_dates)} symbols"
                )
        if missing_charts:
            errors.append(f"missing {len(missing_charts)} candidate chart files")
    items.append(
        {
            "type": "charts",
            "directory": str(chart_dir),
            "manifest_path": str(chart_manifest_path),
            "manifest_exists": chart_manifest_path.is_file(),
            "manifest_status": chart_status,
            "expected_count": len(codes),
            "missing_count": len(missing_charts),
            "missing_codes": missing_charts,
            "status": "error" if any("chart" in error for error in errors) else (
                "warning" if any("chart" in warning for warning in warnings) else "ok"
            ),
        }
    )

    review_dir = root / "data/review" / pick_date
    suggestion_path = review_dir / "suggestion.json"
    review_manifest_path = review_dir / "run_manifest.json"
    suggestion = (
        _read_artifact_json(suggestion_path, "review suggestion", errors)
        if suggestion_path.is_file()
        else None
    )
    review_manifest = (
        _read_artifact_json(review_manifest_path, "review manifest", errors)
        if review_manifest_path.is_file()
        else None
    )
    if suggestion is not None:
        suggestion_status = suggestion.get("status")
        if suggestion_status in {"partial", "failed"}:
            errors.append(f"review suggestion status is {suggestion_status}")
        elif suggestion_status is None:
            warnings.append("review suggestion has no status; completeness is unverified")
        elif suggestion_status != "complete":
            errors.append(f"review suggestion has unknown status: {suggestion_status}")
        if suggestion.get("date") and suggestion.get("date") != pick_date:
            errors.append("review suggestion date does not match latest candidates")
        if review_manifest is None:
            warnings.append("review suggestion exists without run_manifest.json")
    if review_manifest is not None:
        review_status = review_manifest.get("status")
        if review_status in {"partial", "failed", "interrupted", "in_progress"}:
            errors.append(f"review manifest status is {review_status}")
        elif review_status != "complete":
            errors.append(f"review manifest has unknown status: {review_status}")
        if review_manifest.get("pick_date") != pick_date:
            errors.append("review manifest pick_date does not match latest candidates")
        review_failed_count = _artifact_count(
            review_manifest.get("failed_count", 0),
            "review manifest failed_count",
            errors,
        )
        if review_failed_count != 0:
            errors.append("review manifest reports failed symbols")
    items.append(
        {
            "type": "review",
            "directory": str(review_dir),
            "suggestion_exists": suggestion_path.is_file(),
            "manifest_exists": review_manifest_path.is_file(),
            "suggestion_status": suggestion.get("status") if suggestion else None,
            "manifest_status": review_manifest.get("status") if review_manifest else None,
            "status": "error" if any("review" in error for error in errors) else (
                "warning" if any("review" in warning for warning in warnings) else "ok"
            ),
        }
    )
    return _section(items=items, errors=errors, warnings=warnings)


def _read_artifact_json(
    path: Path,
    label: str,
    errors: list[str],
) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        errors.append(f"invalid {label}: {type(exc).__name__}")
        return None
    if not isinstance(payload, dict):
        errors.append(f"invalid {label}: root must be a JSON object")
        return None
    return payload


def _artifact_count(value: object, label: str, errors: list[str]) -> int | None:
    if isinstance(value, bool):
        errors.append(f"{label} must be an integer")
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        errors.append(f"{label} must be an integer")
        return None
    if parsed < 0:
        errors.append(f"{label} must be non-negative")
        return None
    return parsed


def _dotenv_keys(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    keys: set[str] = set()
    try:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.removeprefix("export ").strip()
            if key in SECRET_NAMES and value.strip().strip("'\""):
                keys.add(key)
    except (OSError, UnicodeError):
        return set()
    return keys


def _check_market_data(
    data_dir: Path,
    *,
    sample_size: int | None,
    as_of_date: date,
    max_age_days: int | None,
) -> dict[str, Any]:
    if max_age_days is not None and max_age_days < 0:
        raise ValueError("max_market_data_age_days must be non-negative or None")
    fetch_manifest, fetch_manifest_error, fetch_manifest_warning = _check_fetch_manifest(
        data_dir / "_fetch_manifest.json"
    )
    empty_extra = {
        "path": str(data_dir),
        "file_count": 0,
        "inspected_file_count": 0,
        "sampled": False,
        "date_min": None,
        "date_max": None,
        "data_age_days": None,
        "invalid_files": [],
        "fetch_manifest": fetch_manifest,
    }
    if not data_dir.is_dir():
        return _section(
            items=[],
            errors=[],
            warnings=[f"market data directory missing: {data_dir}"],
            extra=empty_extra,
        )
    files = sorted(data_dir.glob("*.csv"))
    if not files:
        return _section(
            items=[],
            errors=[fetch_manifest_error] if fetch_manifest_error else [],
            warnings=[
                f"no market CSV files found: {data_dir}",
                *([fetch_manifest_warning] if fetch_manifest_warning else []),
            ],
            extra=empty_extra,
        )

    inspected = files if sample_size is None else files[:sample_size]
    invalid_files: list[dict[str, str]] = []
    date_min: str | None = None
    date_max: str | None = None
    for path in inspected:
        result = _inspect_market_csv(path)
        if result["error"]:
            invalid_files.append({"file": path.name, "reason": str(result["error"])})
            continue
        if result["date_min"] is not None:
            date_min = min(date_min, result["date_min"]) if date_min else result["date_min"]
        if result["date_max"] is not None:
            date_max = max(date_max, result["date_max"]) if date_max else result["date_max"]

    invalid_names = {entry["file"] for entry in invalid_files}
    errors = [
        f"invalid market data file {entry['file']}: {entry['reason']}"
        for entry in invalid_files
    ]
    if fetch_manifest_error:
        errors.append(fetch_manifest_error)
    warnings: list[str] = []
    if fetch_manifest_warning:
        warnings.append(fetch_manifest_warning)
    data_age_days: int | None = None
    if date_max is not None:
        latest_date = date.fromisoformat(date_max)
        data_age_days = (as_of_date - latest_date).days
        if data_age_days < 0:
            errors.append(
                f"market data latest date {date_max} is after doctor as-of date {as_of_date.isoformat()}"
            )
        elif max_age_days is not None and data_age_days > max_age_days:
            warnings.append(
                f"market data may be stale: latest={date_max}, age={data_age_days} days, "
                f"threshold={max_age_days} days"
            )
    if sample_size is not None and len(files) > len(inspected):
        warnings.append(
            f"market data scan sampled {len(inspected)} of {len(files)} files; use --deep for all"
        )
    return _section(
        items=[
            {"path": str(path), "status": "error" if path.name in invalid_names else "ok"}
            for path in inspected
        ],
        errors=errors,
        warnings=warnings,
        extra={
            "path": str(data_dir),
            "file_count": len(files),
            "inspected_file_count": len(inspected),
            "sampled": sample_size is not None and len(files) > len(inspected),
            "date_min": date_min,
            "date_max": date_max,
            "data_age_days": data_age_days,
            "invalid_files": invalid_files,
            "fetch_manifest": fetch_manifest,
        },
    )


def _normalize_as_of_date(value: date | str | None) -> date:
    if value is None:
        return date.today()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value).strip())
    except ValueError as exc:
        raise ValueError(f"invalid as_of_date: {value!r}") from exc


def _check_fetch_manifest(
    path: Path,
) -> tuple[dict[str, Any] | None, str | None, str | None]:
    if not path.is_file():
        return None, None, None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, f"invalid fetch manifest: {type(exc).__name__}", None
    if not isinstance(payload, dict):
        return None, "invalid fetch manifest: root must be a JSON object", None
    summary = {
        "path": str(path),
        "status": payload.get("status"),
        "start": payload.get("start"),
        "end": payload.get("end"),
        "symbol_count": payload.get("symbol_count"),
        "completed_count": payload.get("completed_count"),
        "failed_count": payload.get("failed_count", 0),
        "pending_count": payload.get("pending_count", 0),
        "failed_codes": payload.get("failed_codes", []),
        "updated_at": payload.get("updated_at"),
    }
    status = summary["status"]
    failed_count = summary["failed_count"]
    pending_count = summary["pending_count"]
    if status in {"partial", "interrupted"} or failed_count:
        return (
            summary,
            f"fetch manifest is incomplete: status={status}, failed={failed_count}, pending={pending_count}",
            None,
        )
    if status == "in_progress":
        return summary, None, "fetch manifest reports an in-progress fetch"
    if status != "complete":
        return summary, f"fetch manifest has unknown status: {status}", None
    return summary, None, None


def _inspect_market_csv(path: Path) -> dict[str, str | None]:
    if not re.fullmatch(r"\d{6}", path.stem):
        return {"error": "filename is not a six-digit symbol", "date_min": None, "date_max": None}
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = sorted(MARKET_COLUMNS - set(reader.fieldnames or ()))
            if missing:
                return {
                    "error": f"missing columns {', '.join(missing)}",
                    "date_min": None,
                    "date_max": None,
                }
            first_date: str | None = None
            last_date: str | None = None
            row_count = 0
            for row in reader:
                row_count += 1
                raw_date = str(row.get("date", "")).strip()
                if not raw_date:
                    continue
                normalized = _normalize_date(raw_date)
                if normalized is None:
                    return {"error": "contains an invalid date", "date_min": None, "date_max": None}
                first_date = min(first_date, normalized) if first_date else normalized
                last_date = max(last_date, normalized) if last_date else normalized
            if row_count == 0:
                return {"error": "contains no data rows", "date_min": None, "date_max": None}
    except (OSError, csv.Error, UnicodeError) as exc:
        return {"error": f"cannot read file ({type(exc).__name__})", "date_min": None, "date_max": None}
    return {"error": None, "date_min": first_date, "date_max": last_date}


def _normalize_date(value: str) -> str | None:
    candidate = value[:10]
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(candidate, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def _section(
    *,
    items: list[dict[str, Any]],
    errors: list[str],
    warnings: list[str],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": "error" if errors else ("warning" if warnings else "ok"),
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
        "items": items,
    }
    if extra:
        payload.update(extra)
    return payload


def _resolve(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
