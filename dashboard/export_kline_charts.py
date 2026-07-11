"""Export point-in-time daily charts for the latest candidate run."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "dashboard") not in sys.path:
    sys.path.insert(0, str(ROOT / "dashboard"))

from components.charts import make_daily_chart  # noqa: E402


@dataclass(frozen=True)
class ChartExportConfig:
    bars: int = 120
    width: int = 1400
    height: int = 700

    def __post_init__(self) -> None:
        if self.bars < 0:
            raise ValueError("bars must be non-negative")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("chart width and height must be positive")


def run_export(
    *,
    candidates_path: str | Path = ROOT / "data/candidates/candidates_latest.json",
    raw_dir: str | Path = ROOT / "data/raw",
    output_dir: str | Path = ROOT / "data/kline",
    manifest_path: str | Path | None = None,
    config: ChartExportConfig = ChartExportConfig(),
    resume: bool = False,
) -> dict[str, object]:
    candidates_file = Path(candidates_path).resolve()
    raw_root = Path(raw_dir).resolve()
    output_root = Path(output_dir).resolve()
    codes, pick_date = _load_candidates(candidates_file)
    pick_timestamp = pd.Timestamp(pick_date)
    date_output = output_root / pick_date
    date_output.mkdir(parents=True, exist_ok=True)
    resolved_manifest = (
        Path(manifest_path).resolve()
        if manifest_path is not None
        else date_output / "export_manifest.json"
    )
    run_signature = _run_signature(candidates_file, pick_date, codes, config)
    previous_results: dict[str, dict] = {}
    if resume and resolved_manifest.is_file():
        previous = _load_manifest(resolved_manifest)
        if previous.get("run_signature") == run_signature:
            raw_results = previous.get("results", {})
            if isinstance(raw_results, dict):
                previous_results = {
                    str(code): result
                    for code, result in raw_results.items()
                    if isinstance(result, dict)
                }

    results: dict[str, dict] = {}
    failures: dict[str, dict[str, str]] = {}
    reused_count = 0
    rendered_count = 0
    started_at = datetime.now(timezone.utc).isoformat()

    def checkpoint(status: str) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": 1,
            "status": status,
            "run_signature": run_signature,
            "started_at": started_at,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "pick_date": pick_date,
            "candidate_count": len(codes),
            "success_count": len(results),
            "failed_count": len(failures),
            "rendered_count": rendered_count,
            "reused_count": reused_count,
            "failed_codes": sorted(failures),
            "failures": {code: failures[code] for code in sorted(failures)},
            "results": {code: results[code] for code in sorted(results)},
            "candidates_path": str(candidates_file),
            "raw_dir": str(raw_root),
            "output_dir": str(date_output),
            "settings": asdict(config),
        }
        _atomic_write_json(resolved_manifest, payload)
        return payload

    checkpoint("in_progress")
    try:
        for index, code in enumerate(codes, 1):
            raw_path = raw_root / f"{code}.csv"
            chart_path = date_output / f"{code}_day.jpg"
            try:
                frame = _load_point_in_time_raw(raw_path, code, pick_timestamp)
                signature = _chart_signature(frame, code, pick_date, config)
                previous = previous_results.get(code)
                if (
                    resume
                    and previous is not None
                    and previous.get("signature") == signature
                    and chart_path.is_file()
                    and previous.get("output_sha256") == _file_sha256(chart_path)
                ):
                    results[code] = previous
                    reused_count += 1
                    print(f"[{index}/{len(codes)}] {code} — 签名匹配，复用已有图表。")
                    checkpoint("in_progress")
                    continue

                if chart_path.exists():
                    archived = _archive_existing(chart_path)
                    print(f"[{index}/{len(codes)}] {code} — 旧图归档至 {archived.name}。")
                figure = make_daily_chart(
                    frame,
                    code,
                    bars=config.bars,
                    height=config.height,
                )
                _export_figure_atomic(
                    figure,
                    chart_path,
                    width=config.width,
                    height=config.height,
                )
                results[code] = {
                    "signature": signature,
                    "raw_path": str(raw_path),
                    "raw_point_in_time_sha256": _frame_sha256(frame),
                    "chart_end_date": frame["date"].max().date().isoformat(),
                    "row_count": len(frame),
                    "output_path": str(chart_path),
                    "output_sha256": _file_sha256(chart_path),
                }
                rendered_count += 1
                print(f"[{index}/{len(codes)}] {code} — 完成：{chart_path.name}")
            except Exception as exc:
                failures[code] = {
                    "error_type": type(exc).__name__,
                    "message": str(exc)[:500],
                }
                print(f"[{index}/{len(codes)}] {code} — 失败：{type(exc).__name__}: {exc}")
            checkpoint("in_progress")
    except BaseException:
        checkpoint("interrupted")
        raise

    status = "partial" if failures else "complete"
    manifest = checkpoint(status)
    print(
        f"\n图表导出结束：状态={status}，成功={len(results)}，失败={len(failures)}，"
        f"恢复={reused_count}。\n输出目录：{date_output}\nmanifest：{resolved_manifest}"
    )
    return {
        "ok": not failures,
        "status": status,
        "pick_date": pick_date,
        "candidate_count": len(codes),
        "rendered_count": rendered_count,
        "reused_count": reused_count,
        "failed_codes": sorted(failures),
        "output_dir": date_output,
        "manifest_path": resolved_manifest,
        "manifest": manifest,
    }


def _load_candidates(path: Path) -> tuple[list[str], str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read candidates JSON {path}: {type(exc).__name__}") from exc
    if not isinstance(payload, dict):
        raise ValueError("candidates JSON must contain an object")
    pick_date = str(payload.get("pick_date", "")).strip()
    try:
        parsed_date = pd.Timestamp(pick_date)
    except (TypeError, ValueError) as exc:
        raise ValueError("candidates pick_date must be a valid YYYY-MM-DD date") from exc
    if pick_date != parsed_date.date().isoformat():
        raise ValueError("candidates pick_date must use YYYY-MM-DD")
    raw_candidates = payload.get("candidates")
    if not isinstance(raw_candidates, list):
        raise ValueError("candidates field must be a list")
    codes: list[str] = []
    seen: set[str] = set()
    for candidate in raw_candidates:
        if not isinstance(candidate, dict):
            raise ValueError("each candidate must be a JSON object")
        code = _normalize_symbol(candidate.get("code"))
        if code in seen:
            raise ValueError(f"duplicate candidate code: {code}")
        seen.add(code)
        codes.append(code)
    return codes, pick_date


def _load_point_in_time_raw(path: Path, code: str, pick_date: pd.Timestamp) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"missing raw market CSV: {path}")
    try:
        frame = pd.read_csv(path)
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise ValueError(f"cannot read raw market CSV {path}: {type(exc).__name__}") from exc
    frame.columns = [str(column).lower() for column in frame.columns]
    required = {"date", "open", "high", "low", "close", "volume"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"raw market CSV {code} missing columns: {', '.join(missing)}")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    if frame["date"].isna().any():
        raise ValueError(f"raw market CSV {code} contains invalid dates")
    frame = frame.sort_values("date").drop_duplicates("date", keep="last")
    frame = frame.loc[frame["date"] <= pick_date].reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"raw market CSV {code} has no data on or before {pick_date.date()}")
    if frame["date"].max().normalize() != pick_date.normalize():
        raise ValueError(
            f"raw market CSV {code} has no bar on candidate pick_date {pick_date.date()}"
        )
    return frame


def _normalize_symbol(value: object) -> str:
    code = str(value if value is not None else "").strip()
    if code.isdigit() and len(code) <= 6:
        code = code.zfill(6)
    if not re.fullmatch(r"\d{6}", code):
        raise ValueError(f"candidate code must be a six-digit symbol: {value!r}")
    return code


def _run_signature(
    candidates_path: Path,
    pick_date: str,
    codes: list[str],
    config: ChartExportConfig,
) -> str:
    payload = {
        "schema_version": 1,
        "candidates_sha256": _file_sha256(candidates_path),
        "pick_date": pick_date,
        "codes": codes,
        "settings": asdict(config),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def _chart_signature(
    frame: pd.DataFrame,
    code: str,
    pick_date: str,
    config: ChartExportConfig,
) -> str:
    payload = {
        "schema_version": 1,
        "code": code,
        "pick_date": pick_date,
        "frame_sha256": _frame_sha256(frame),
        "settings": asdict(config),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def _frame_sha256(frame: pd.DataFrame) -> str:
    encoded = frame.to_csv(index=False, date_format="%Y-%m-%d").encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _export_figure_atomic(figure, path: Path, *, width: int, height: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        figure.write_image(
            str(temporary),
            format="jpg",
            width=width,
            height=height,
            scale=2,
        )
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError("chart renderer did not create a non-empty image")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _archive_existing(path: Path) -> Path:
    archive_dir = path.parent / ".stale"
    archive_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archived = archive_dir / f"{path.stem}_{timestamp}{path.suffix}"
    os.replace(path, archived)
    return archived


def _load_manifest(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export point-in-time candidate daily charts")
    parser.add_argument(
        "--candidates",
        default=str(ROOT / "data/candidates/candidates_latest.json"),
    )
    parser.add_argument("--raw-dir", default=str(ROOT / "data/raw"))
    parser.add_argument("--output-dir", default=str(ROOT / "data/kline"))
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--bars", type=int, default=120)
    parser.add_argument("--width", type=int, default=1400)
    parser.add_argument("--height", type=int, default=700)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse signature-matching non-empty chart files",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_export(
            candidates_path=args.candidates,
            raw_dir=args.raw_dir,
            output_dir=args.output_dir,
            manifest_path=args.manifest,
            config=ChartExportConfig(
                bars=args.bars,
                width=args.width,
                height=args.height,
            ),
            resume=args.resume,
        )
    except (OSError, ValueError) as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
