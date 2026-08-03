"""Governed fetch-data suite for the Tushare data RQuant actually consumes."""

from __future__ import annotations

import json
import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml

from market import fetch_benchmark, fetch_context, fetch_industry, fetch_kline
from market import fetch_trade_state
from market.build_research_context import build_research_context


DATASET_ORDER = (
    "bars",
    "daily_basic",
    "benchmark",
    "industry",
    "trade_state",
    "research_context",
)
DEFAULT_DATASETS = DATASET_ORDER
_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def fetch_tushare_research_data(
    *,
    config_path: str | Path = "config/fetch_kline.yaml",
    start: str | None = None,
    end: str | None = None,
    out_dir: str | Path | None = None,
    workers: int | None = None,
    max_requests_per_minute: int | None = None,
    max_symbols: int | None = None,
    max_dates: int | None = None,
    max_industries: int | None = None,
    bar_manifest_path: str | Path | None = None,
    suite_manifest_path: str | Path | None = None,
    log_path: str | Path | None = None,
    resume: bool = False,
    datasets: list[str] | tuple[str, ...] | None = None,
) -> dict[str, object]:
    """Run the complete A-share 2,000-point research-data contract.

    The scope is intentionally RQuant-specific: qfq bars plus adjustment
    factors, daily_basic, the configured benchmark, SW point-in-time industry
    membership, daily price limits/suspensions, and their merged context.  It
    does not silently expand into unrelated fund, futures, FX, or macro APIs
    that happen to share a Tushare point threshold.
    """

    resolved_config = _resolve(config_path)
    cfg = _load_config(resolved_config)
    suite_cfg = cfg.get("tushare_2000") or {}
    if not isinstance(suite_cfg, dict):
        raise ValueError("tushare_2000 config must be a mapping")
    requested = _normalize_datasets(datasets or suite_cfg.get("datasets") or DEFAULT_DATASETS)
    resolved_end = fetch_kline._normalize_request_date(
        end if end is not None else str(cfg.get("end", "today"))
    )
    bars_start = fetch_kline._normalize_request_date(
        start if start is not None else str(cfg.get("start", "20190101"))
    )
    context_start = fetch_kline._normalize_request_date(
        start
        if start is not None
        else str(suite_cfg.get("context_start", cfg.get("start", "20190101")))
    )
    if bars_start > resolved_end or context_start > resolved_end:
        raise ValueError("start must not be after end")
    resolved_workers = int(
        workers if workers is not None else cfg.get("workers", 8)
    )
    resolved_rate = int(
        max_requests_per_minute
        if max_requests_per_minute is not None
        else cfg.get(
            "max_requests_per_minute", fetch_kline.DEFAULT_MAX_REQUESTS_PER_MINUTE
        )
    )
    if resolved_workers <= 0:
        raise ValueError("workers must be positive")
    if resolved_rate < 0:
        raise ValueError("max_requests_per_minute must be non-negative")

    paths_cfg = suite_cfg.get("paths") or {}
    if not isinstance(paths_cfg, dict):
        raise ValueError("tushare_2000.paths must be a mapping")
    paths = {
        "bars": _resolve(out_dir if out_dir is not None else cfg.get("out", "data/raw")),
        "daily_basic": _resolve(paths_cfg.get("daily_basic", "data/context/daily_basic")),
        "benchmark": _resolve(paths_cfg.get("benchmark", "data/context/benchmark_000300.csv")),
        "industry": _resolve(paths_cfg.get("industry", "data/context/sw_industry_membership.csv")),
        "trade_state": _resolve(paths_cfg.get("trade_state", "data/context/trade_state")),
        "research_context": _resolve(paths_cfg.get("research_context", "data/context/research")),
    }
    suite_manifest = _resolve(
        suite_manifest_path
        if suite_manifest_path is not None
        else paths_cfg.get("suite_manifest", "data/context/_tushare_2000_manifest.json")
    )
    index_code = str(suite_cfg.get("benchmark_index", "000300.SH")).strip().upper()
    industry_source = str(suite_cfg.get("industry_source", "SW2021")).strip().upper()
    include_suspensions = bool(suite_cfg.get("include_suspensions", True))
    trade_state_symbol_aliases = suite_cfg.get("trade_state_symbol_aliases") or []
    if not isinstance(trade_state_symbol_aliases, list):
        raise ValueError("tushare_2000.trade_state_symbol_aliases must be a list")
    stages: dict[str, dict[str, object]] = {}
    started_at = datetime.now(timezone.utc).isoformat()

    signature = {
        "contract": "rquant_tushare_2000_v1",
        "datasets": list(requested),
        "bars_start": bars_start,
        "context_start": context_start,
        "end": resolved_end,
        "paths": {name: str(path) for name, path in paths.items()},
        "benchmark_index": index_code,
        "industry_source": industry_source,
        "include_suspensions": include_suspensions,
        "trade_state_symbol_aliases": trade_state_symbol_aliases,
        "max_symbols": max_symbols,
        "max_dates": max_dates,
        "max_industries": max_industries,
    }

    def checkpoint() -> None:
        complete = len(stages) == len(requested) and all(
            bool(result.get("ok")) for result in stages.values()
        )
        _atomic_write_json(
            suite_manifest,
            {
                "status": "complete" if complete else "partial",
                "signature": signature,
                "started_at": started_at,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "stages": stages,
            },
        )

    def run_stage(name: str, runner: Callable[[], Any]) -> None:
        try:
            raw_result = runner()
            result = _jsonable_mapping(raw_result)
            result.setdefault("ok", True)
        except Exception as exc:
            result = {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc)[:1000],
            }
        stages[name] = result
        checkpoint()

    checkpoint()
    if "bars" in requested:
        run_stage(
            "bars",
            lambda: fetch_kline.run_fetch(
                config_path=resolved_config,
                log_path=log_path,
                start=bars_start,
                end=resolved_end,
                out_dir=paths["bars"],
                workers=resolved_workers,
                max_requests_per_minute=resolved_rate,
                max_symbols=max_symbols,
                manifest_path=bar_manifest_path,
                resume=resume,
            ),
        )
    if "daily_basic" in requested:
        run_stage(
            "daily_basic",
            lambda: fetch_context.fetch_daily_basic_context(
                start=context_start,
                end=resolved_end,
                output_dir=paths["daily_basic"],
                resume=resume,
                max_requests_per_minute=resolved_rate,
                workers=resolved_workers,
                max_dates=max_dates,
            ),
        )
    if "benchmark" in requested:
        run_stage(
            "benchmark",
            lambda: fetch_benchmark.fetch_benchmark_index(
                start=context_start,
                end=resolved_end,
                index_code=index_code,
                output_file=paths["benchmark"],
                resume=resume,
            ),
        )
    if "industry" in requested:
        run_stage(
            "industry",
            lambda: fetch_industry.fetch_sw_industry_membership(
                output_file=paths["industry"],
                source=industry_source,
                resume=resume,
                max_requests_per_minute=resolved_rate,
                max_industries=max_industries,
            ),
        )
    if "trade_state" in requested:
        run_stage(
            "trade_state",
            lambda: fetch_trade_state.fetch_trade_state_context(
                start=context_start,
                end=resolved_end,
                output_dir=paths["trade_state"],
                resume=resume,
                max_requests_per_minute=resolved_rate,
                workers=resolved_workers,
                max_dates=max_dates,
                include_suspensions=include_suspensions,
                expected_symbols_dir=paths["daily_basic"],
                symbol_aliases=trade_state_symbol_aliases,
            ),
        )
    if "research_context" in requested:
        dependencies = ("daily_basic", "industry", "trade_state")
        failed_dependencies = [
            name
            for name in dependencies
            if name in stages and not bool(stages[name].get("ok"))
        ]
        if failed_dependencies:
            stages["research_context"] = {
                "ok": False,
                "error_type": "DependencyError",
                "error": "incomplete prerequisite stages: "
                + ", ".join(failed_dependencies),
            }
            checkpoint()
        else:
            run_stage(
                "research_context",
                lambda: build_research_context(
                    daily_basic_dir=paths["daily_basic"],
                    industry_file=paths["industry"],
                    trade_state_dir=paths["trade_state"],
                    output_dir=paths["research_context"],
                    resume=resume,
                ),
            )

    ok = len(stages) == len(requested) and all(
        bool(result.get("ok")) for result in stages.values()
    )
    checkpoint()
    failed_datasets = [name for name in requested if not stages.get(name, {}).get("ok")]
    return {
        "ok": ok,
        "contract": signature["contract"],
        "start": bars_start,
        "context_start": context_start,
        "end": resolved_end,
        "datasets": list(requested),
        "failed_datasets": failed_datasets,
        "stages": stages,
        "output_dir": str(paths["bars"]),
        "research_context_dir": str(paths["research_context"]),
        "manifest_path": str(suite_manifest),
    }


def run_from_args(args: Any) -> dict[str, object]:
    return fetch_tushare_research_data(
        config_path=args.config,
        start=args.start,
        end=args.end,
        out_dir=args.out,
        workers=args.workers,
        max_requests_per_minute=args.max_requests_per_minute,
        max_symbols=args.max_symbols,
        max_dates=getattr(args, "max_dates", None),
        max_industries=getattr(args, "max_industries", None),
        bar_manifest_path=getattr(args, "manifest", None),
        suite_manifest_path=getattr(args, "suite_manifest", None),
        log_path=args.log,
        resume=bool(args.resume),
        datasets=getattr(args, "datasets", None),
    )


def _normalize_datasets(values: Any) -> tuple[str, ...]:
    if isinstance(values, str):
        raw = values.replace(",", " ").split()
    else:
        raw = [str(value) for value in values]
    selected = set(raw)
    unknown = selected.difference(DATASET_ORDER)
    if unknown:
        raise ValueError("unknown fetch datasets: " + ", ".join(sorted(unknown)))
    if not selected:
        raise ValueError("at least one fetch dataset is required")
    return tuple(name for name in DATASET_ORDER if name in selected)


def _load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"fetch config not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("fetch config must contain a YAML mapping")
    return payload


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (_PROJECT_ROOT / path).resolve()


def _jsonable_mapping(value: Any) -> dict[str, object]:
    mapping = dict(value)
    return json.loads(json.dumps(mapping, default=_json_default))


def _json_default(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    raw = getattr(value, "value", None)
    return str(raw if raw is not None else value)


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)
