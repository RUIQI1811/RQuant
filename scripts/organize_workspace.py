from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ArchiveAction:
    source: Path
    destination: Path
    reason: str


@dataclass(frozen=True)
class CleanupReport:
    archived: tuple[dict[str, object], ...]
    removed_caches: tuple[str, ...]
    removed_empty_directories: tuple[str, ...]
    skipped: tuple[str, ...]


_CACHE_DIRECTORY_NAMES = {
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
}
_CACHE_FILE_NAMES = {".DS_Store"}
_ROOT_GENERATED_DIRECTORY_NAMES = {"build"}
_PROTECTED_OUTPUT_DIRECTORIES = {
    "data/backtest",
    "data/candidates",
    "data/context",
    "data/kline",
    "data/ml",
    "data/portfolio_backtest",
    "data/raw",
    "data/reports",
    "data/review",
    "data/runs",
}
_PROTECTED_EMPTY_SUBTREES = {
    "data/context",
    "data/ml",
    "data/raw",
    "data/runs",
}


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _directory_size(path: Path) -> int:
    total = 0
    for candidate in path.rglob("*"):
        if candidate.is_file() and not candidate.is_symlink():
            try:
                total += candidate.stat().st_size
            except FileNotFoundError:
                continue
    return total


def _is_old_enough(path: Path, minimum_age_days: int) -> bool:
    file_mtimes: list[float] = []
    for candidate in path.rglob("*"):
        if not candidate.is_file():
            continue
        try:
            file_mtimes.append(candidate.stat().st_mtime)
        except FileNotFoundError:
            continue
    latest_mtime = max(file_mtimes, default=path.stat().st_mtime)
    cutoff = datetime.now(timezone.utc) - timedelta(days=minimum_age_days)
    return datetime.fromtimestamp(latest_mtime, tz=timezone.utc) <= cutoff


def _has_meaningful_content(path: Path) -> bool:
    for candidate in path.rglob("*"):
        if not candidate.is_file():
            continue
        if candidate.name in _CACHE_FILE_NAMES:
            continue
        if candidate.name.endswith((".pyc", ".pyo")):
            continue
        return True
    return False


def _legacy_run_all_is_complete(path: Path) -> bool:
    manifests = sorted(path.glob("*/manifest.json"))
    if not manifests:
        return False
    for manifest in manifests:
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return False
        if payload.get("status") != "complete":
            return False
    return True


def discover_archive_actions(
    root: Path,
    *,
    minimum_age_days: int = 7,
) -> tuple[ArchiveAction, ...]:
    root = root.resolve()
    actions: list[ArchiveAction] = []
    data = root / "data"
    if data.is_dir():
        for source in sorted(data.iterdir()):
            if not source.is_dir() or source.name == "archive":
                continue
            if not _has_meaningful_content(source) or not _is_old_enough(
                source,
                minimum_age_days,
            ):
                continue
            if source.name.startswith("portfolio_backtest_"):
                actions.append(
                    ArchiveAction(
                        source=source,
                        destination=(
                            data / "archive" / "portfolio_backtests" / source.name
                        ),
                        reason="non-canonical historical portfolio backtest",
                    )
                )
            elif source.name.startswith("backtest_"):
                actions.append(
                    ArchiveAction(
                        source=source,
                        destination=data / "archive" / "signal_backtests" / source.name,
                        reason="non-canonical historical signal-return run",
                    )
                )

    legacy_run_all = root / "factor_report" / "factor_run_all"
    if legacy_run_all.is_dir() and _legacy_run_all_is_complete(legacy_run_all):
        actions.append(
            ArchiveAction(
                source=legacy_run_all,
                destination=(
                    root
                    / "factor_report"
                    / "archive"
                    / "legacy_workflows"
                    / "factor_run_all"
                ),
                reason="legacy monolithic factor-run-all output",
            )
        )
    return tuple(actions)


def discover_cache_paths(root: Path) -> tuple[Path, ...]:
    root = root.resolve()
    found: list[Path] = []
    for current, directory_names, file_names in os.walk(root, topdown=True):
        current_path = Path(current)
        if current_path == root / ".git":
            directory_names[:] = []
            continue

        kept_directories: list[str] = []
        for name in directory_names:
            candidate = current_path / name
            is_root_generated = current_path == root and (
                name in _ROOT_GENERATED_DIRECTORY_NAMES or name.endswith(".egg-info")
            )
            if name in _CACHE_DIRECTORY_NAMES or is_root_generated:
                found.append(candidate)
            elif name != ".git":
                kept_directories.append(name)
        directory_names[:] = kept_directories

        for name in file_names:
            if name in _CACHE_FILE_NAMES:
                found.append(current_path / name)
    return tuple(sorted(set(found)))


def discover_empty_directories(root: Path) -> tuple[Path, ...]:
    roots = [root / "data", root / "factor_report"]
    empty: list[Path] = []
    for output_root in roots:
        if not output_root.is_dir():
            continue
        for current, directory_names, file_names in os.walk(output_root, topdown=False):
            current_path = Path(current)
            relative_parts = current_path.relative_to(output_root).parts
            if current_path == output_root or "archive" in relative_parts:
                continue
            relative = _relative(current_path, root)
            if relative in _PROTECTED_OUTPUT_DIRECTORIES:
                continue
            if any(
                relative == subtree or relative.startswith(subtree + "/")
                for subtree in _PROTECTED_EMPTY_SUBTREES
            ):
                continue
            if not directory_names and not file_names:
                empty.append(current_path)
    return tuple(sorted(empty, key=lambda path: len(path.parts), reverse=True))


def _remove_paths(paths: Iterable[Path], root: Path) -> list[str]:
    removed: list[str] = []
    for path in paths:
        if not path.exists() and not path.is_symlink():
            continue
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()
        removed.append(_relative(path, root))
    return removed


def _prune_empty_directories(root: Path) -> list[str]:
    removed: list[str] = []
    for path in discover_empty_directories(root):
        try:
            path.rmdir()
        except OSError:
            continue
        removed.append(_relative(path, root))
    return removed


def _write_archive_index(root: Path, records: list[dict[str, object]]) -> None:
    if not records:
        return
    index_path = root / "data" / "archive" / "archive_index.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict[str, object]] = []
    if index_path.is_file():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            existing = [item for item in payload if isinstance(item, dict)]
    payload = existing + records
    temporary = index_path.with_suffix(index_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, index_path)


def organize_workspace(
    root: Path,
    *,
    apply: bool,
    minimum_age_days: int = 7,
) -> CleanupReport:
    root = root.resolve()
    if minimum_age_days < 0:
        raise ValueError("minimum_age_days must be non-negative")
    archive_actions = discover_archive_actions(
        root,
        minimum_age_days=minimum_age_days,
    )
    cache_paths = discover_cache_paths(root)
    empty_directories = discover_empty_directories(root)
    if not apply:
        return CleanupReport(
            archived=tuple(
                {
                    "source": _relative(action.source, root),
                    "destination": _relative(action.destination, root),
                    "reason": action.reason,
                }
                for action in archive_actions
            ),
            removed_caches=tuple(_relative(path, root) for path in cache_paths),
            removed_empty_directories=tuple(
                _relative(path, root) for path in empty_directories
            ),
            skipped=(),
        )

    removed_caches = _remove_paths(cache_paths, root)
    archived: list[dict[str, object]] = []
    skipped: list[str] = []
    archived_at = datetime.now(timezone.utc).isoformat()
    for action in archive_actions:
        if not action.source.exists():
            continue
        if action.destination.exists():
            skipped.append(
                f"{_relative(action.source, root)}: destination already exists"
            )
            continue
        size_bytes = _directory_size(action.source)
        action.destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(action.source), str(action.destination))
        archived.append(
            {
                "source": _relative(action.source, root),
                "destination": _relative(action.destination, root),
                "reason": action.reason,
                "size_bytes": size_bytes,
                "archived_at": archived_at,
            }
        )
    _write_archive_index(root, archived)
    removed_empty = _prune_empty_directories(root)
    return CleanupReport(
        archived=tuple(archived),
        removed_caches=tuple(removed_caches),
        removed_empty_directories=tuple(removed_empty),
        skipped=tuple(skipped),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit or organize local RQuant artifacts. The default is a read-only preview; "
            "use --apply to archive historical outputs and remove regenerable caches."
        )
    )
    parser.add_argument(
        "--project-root",
        default=str(ROOT),
        help="RQuant project root (default: repository containing this script)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the previewed archive and cache cleanup actions",
    )
    parser.add_argument(
        "--minimum-age-days",
        type=int,
        default=7,
        help="Archive named data experiments only after this many inactive days",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.project_root).expanduser()
    if not (root / "AGENTS.md").is_file() or not (root / "data").exists():
        print(f"not an RQuant project root: {root}", file=sys.stderr)
        return 2
    if args.minimum_age_days < 0:
        print("--minimum-age-days must be non-negative", file=sys.stderr)
        return 2
    report = organize_workspace(
        root,
        apply=args.apply,
        minimum_age_days=args.minimum_age_days,
    )
    payload = {"mode": "apply" if args.apply else "preview", **asdict(report)}
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if not report.skipped else 1


if __name__ == "__main__":
    raise SystemExit(main())
