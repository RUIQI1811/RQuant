"""Project discovery, logging, and auditable CLI run manifests."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from domain.artifacts import WorkflowResult


SCHEMA_VERSION = 1
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SENSITIVE_NAMES = ("token", "secret", "password", "api-key", "api_key", "credential")
_OUTPUT_FIELDS = ("output", "out", "manifest", "log", "log_dir")
_INPUT_FIELDS = (
    "config",
    "data",
    "metadata",
    "features",
    "labels",
    "signals",
    "candidates",
    "review",
    "signal_dir",
    "portfolio_dir",
    "benchmark_file",
    "style_factor_file",
    "factor_config",
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def generate_run_id() -> str:
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{uuid.uuid4().hex[:8]}"


def discover_project_root(explicit: str | os.PathLike[str] | None = None) -> Path:
    """Find a checkout root without depending on the caller's current directory."""

    if explicit:
        root = Path(explicit).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"project root does not exist or is not a directory: {root}")
        return root

    candidates = [Path.cwd(), Path(__file__).resolve().parents[1]]
    seen: set[Path] = set()
    for candidate in candidates:
        for path in (candidate, *candidate.parents):
            if path in seen:
                continue
            seen.add(path)
            if (path / "AGENTS.md").is_file() and (path / "scripts" / "quant_cli.py").is_file():
                return path.resolve()
    raise ValueError("cannot locate RQuant project root; pass --project-root PATH")


@dataclass(frozen=True)
class ProjectContext:
    project_root: Path
    runs_dir: Path
    run_id: str
    log_level: str = "INFO"

    @classmethod
    def create(
        cls,
        *,
        project_root: str | os.PathLike[str] | None = None,
        runs_dir: str | os.PathLike[str] | None = None,
        run_id: str | None = None,
        log_level: str = "INFO",
    ) -> "ProjectContext":
        root = discover_project_root(project_root)
        resolved_runs = Path(runs_dir).expanduser() if runs_dir else Path("data/runs")
        if not resolved_runs.is_absolute():
            resolved_runs = root / resolved_runs
        resolved_id = run_id or generate_run_id()
        if not _RUN_ID_RE.fullmatch(resolved_id):
            raise ValueError(
                "run id must start with an alphanumeric character and contain only "
                "letters, numbers, '.', '_' or '-' (maximum 128 characters)"
            )
        normalized_level = log_level.upper()
        if normalized_level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
            raise ValueError(f"invalid log level: {log_level}")
        return cls(root, resolved_runs.resolve(), resolved_id, normalized_level)

    @property
    def run_dir(self) -> Path:
        return self.runs_dir / self.run_id

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "run.json"

    @property
    def log_path(self) -> Path:
        return self.run_dir / "run.log"

    def resolve(self, value: str | os.PathLike[str]) -> Path:
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (self.project_root / path).resolve()


@dataclass
class CommandResult:
    """Stable result returned by the CLI dispatch boundary."""

    status: str = "complete"
    exit_code: int = 0
    outputs: dict[str, str] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    workflow: WorkflowResult[Any] | None = None


class _WarningCollector(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(self.format(record))


class _SecretRedactionFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = redact_text(record.getMessage())
        record.msg = message
        record.args = ()
        return True


class RunTracker:
    """Own one run directory and finalize its manifest on every exit path."""

    def __init__(self, context: ProjectContext, command: str, argv: Sequence[str]) -> None:
        self.context = context
        self.command = command
        self.argv = list(argv)
        self.started_at = utc_now()
        self.warning_collector = _WarningCollector()
        self._managed_handlers: list[logging.Handler] = []
        self._manifest: dict[str, Any] = {}

    def start(self) -> None:
        self.context.run_dir.mkdir(parents=True, exist_ok=False)
        self._configure_logging()
        git = _git_state(self.context.project_root)
        self._manifest = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.context.run_id,
            "command": self.command,
            "argv": redact_argv(self.argv),
            "status": "running",
            "exit_code": None,
            "started_at": self.started_at,
            "finished_at": None,
            "project_root": str(self.context.project_root),
            "runs_dir": str(self.context.runs_dir),
            "runtime": {
                "python_executable": sys.executable,
                "python_version": sys.version.split()[0],
                "git_commit": git["commit"],
                "git_dirty": git["dirty"],
            },
            "inputs": collect_path_evidence(self.argv, self.context, _INPUT_FIELDS),
            "outputs": {},
            "downstream_manifests": [],
            "summary": {},
            "warnings": [],
            "error": None,
        }
        atomic_write_json(self.context.manifest_path, self._manifest)

    def complete(self, result: CommandResult | None = None) -> None:
        result = result or CommandResult()
        output_evidence = collect_path_evidence(self.argv, self.context, _OUTPUT_FIELDS)
        for name, value in result.outputs.items():
            path = self.context.resolve(value)
            output_evidence[name] = path_evidence(path)
        self._manifest.update(
            status=result.status,
            exit_code=result.exit_code,
            finished_at=utc_now(),
            outputs=output_evidence,
            downstream_manifests=find_downstream_manifests(output_evidence),
            summary=_sanitize_json(result.summary),
            warnings=[redact_text(item) for item in dict.fromkeys(self.warning_collector.messages)],
        )
        atomic_write_json(self.context.manifest_path, self._manifest)
        self.close_logging()

    def fail(self, error: BaseException, exit_code: int, *, interrupted: bool = False) -> None:
        stack = "".join(traceback.format_tb(error.__traceback__))
        logging.getLogger(__name__).error(
            "run failed: %s: %s%s",
            type(error).__name__,
            error,
            f"\n{stack}" if stack else "",
        )
        self._manifest.update(
            status="interrupted" if interrupted else "failed",
            exit_code=exit_code,
            finished_at=utc_now(),
            outputs=collect_path_evidence(self.argv, self.context, _OUTPUT_FIELDS),
            warnings=[redact_text(item) for item in dict.fromkeys(self.warning_collector.messages)],
            error={"type": type(error).__name__, "message": redact_text(str(error))},
        )
        self._manifest["downstream_manifests"] = find_downstream_manifests(
            self._manifest["outputs"]
        )
        atomic_write_json(self.context.manifest_path, self._manifest)
        self.close_logging()

    def _configure_logging(self) -> None:
        root = logging.getLogger()
        root.setLevel(getattr(logging, self.context.log_level))
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(formatter)
        file_handler = logging.FileHandler(self.context.log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        self.warning_collector.setFormatter(formatter)
        for handler in (console, file_handler, self.warning_collector):
            handler.addFilter(_SecretRedactionFilter())
            root.addHandler(handler)
            self._managed_handlers.append(handler)

    def close_logging(self) -> None:
        root = logging.getLogger()
        for handler in self._managed_handlers:
            root.removeHandler(handler)
            handler.close()
        self._managed_handlers.clear()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def redact_argv(argv: Sequence[str]) -> list[str]:
    redacted: list[str] = []
    hide_next = False
    for token in argv:
        if hide_next:
            redacted.append("<redacted>")
            hide_next = False
            continue
        lowered = token.lower()
        if any(name in lowered for name in _SENSITIVE_NAMES):
            if "=" in token:
                redacted.append(token.split("=", 1)[0] + "=<redacted>")
            else:
                redacted.append(token)
                hide_next = token.startswith("-")
        else:
            redacted.append(redact_text(token))
    return redacted


def redact_text(value: str) -> str:
    value = re.sub(r"(?i)(sk-[A-Za-z0-9_-]{8,})", "<redacted>", value)
    value = re.sub(
        r"(?i)((?:token|api[_-]?key|secret|password)\s*[=:]\s*)[^\s,;]+",
        r"\1<redacted>",
        value,
    )
    return value


def collect_path_evidence(
    argv: Sequence[str], context: ProjectContext, field_names: Iterable[str]
) -> dict[str, dict[str, Any]]:
    wanted = {"--" + name.replace("_", "-"): name for name in field_names}
    evidence: dict[str, dict[str, Any]] = {}
    index = 0
    while index < len(argv):
        token = argv[index]
        option, separator, inline_value = token.partition("=")
        field = wanted.get(option)
        if field:
            if separator:
                value = inline_value
            elif index + 1 < len(argv) and not argv[index + 1].startswith("-"):
                index += 1
                value = argv[index]
            else:
                index += 1
                continue
            evidence[field] = path_evidence(context.resolve(value))
        index += 1
    return evidence


def path_evidence(path: Path) -> dict[str, Any]:
    item: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if path.is_file():
        stat = path.stat()
        item.update(type="file", size=stat.st_size, sha256=_sha256(path))
    elif path.is_dir():
        item.update(type="directory")
    else:
        item.update(type="missing")
    return item


def find_downstream_manifests(outputs: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    for item in outputs.values():
        path = Path(str(item.get("path", "")))
        candidates: list[Path] = []
        if path.is_file() and "manifest" in path.name.lower():
            candidates.append(path)
        elif path.is_dir():
            candidates.extend(
                candidate
                for name in ("manifest.json", "batch_manifest.json", "run_manifest.json")
                if (candidate := path / name).is_file()
            )
        for candidate in candidates:
            found[str(candidate.resolve())] = path_evidence(candidate.resolve())
    return list(found.values())


def load_run_manifests(
    runs_dir: Path, *, status: str | None = None, limit: int = 20
) -> list[dict[str, Any]]:
    if limit <= 0:
        raise ValueError("limit must be positive")
    manifests: list[dict[str, Any]] = []
    if not runs_dir.is_dir():
        return manifests
    for path in sorted(runs_dir.glob("*/run.json"), reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict) or (status and payload.get("status") != status):
            continue
        payload["_manifest_path"] = str(path)
        manifests.append(payload)
        if len(manifests) >= limit:
            break
    return manifests


def load_run_manifest(runs_dir: Path, run_id: str) -> dict[str, Any]:
    if not _RUN_ID_RE.fullmatch(run_id):
        raise ValueError("invalid run id")
    path = (runs_dir / run_id / "run.json").resolve()
    if path.parent.parent != runs_dir.resolve():
        raise ValueError("run id escapes runs directory")
    if not path.is_file():
        raise FileNotFoundError(f"run not found: {run_id}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"run manifest must contain an object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state(project_root: Path) -> dict[str, Any]:
    def run(*args: str) -> subprocess.CompletedProcess[str] | None:
        try:
            return subprocess.run(
                ["git", *args],
                cwd=project_root,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {
        "commit": commit.stdout.strip() if commit and commit.returncode == 0 else None,
        "dirty": bool(status.stdout.strip()) if status and status.returncode == 0 else None,
    }


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _sanitize_json(value: Any, *, key: str = "") -> Any:
    if any(name in key.lower() for name in _SENSITIVE_NAMES):
        return "<redacted>"
    safe = _json_safe(value)
    if isinstance(safe, str):
        return redact_text(safe)
    if isinstance(safe, dict):
        return {name: _sanitize_json(item, key=name) for name, item in safe.items()}
    if isinstance(safe, list):
        return [_sanitize_json(item) for item in safe]
    return safe
