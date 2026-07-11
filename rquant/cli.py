"""Governed public CLI for every RQuant research workflow."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .runtime import (
    CommandResult,
    ProjectContext,
    RunTracker,
    load_run_manifest,
    load_run_manifests,
)


@dataclass
class GlobalOptions:
    project_root: str | None = None
    runs_dir: str | None = None
    run_id: str | None = None
    log_level: str = "INFO"


def main(argv: Sequence[str] | None = None, *, backend=None, prog: str = "rquant") -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    try:
        options, command_argv = _extract_global_options(raw_argv)
        context = ProjectContext.create(
            project_root=options.project_root,
            runs_dir=options.runs_dir,
            run_id=options.run_id,
            log_level=options.log_level,
        )
    except ValueError as exc:
        print(f"rquant: error: {exc}", file=sys.stderr)
        return 2

    if not command_argv or command_argv == ["--help"] or command_argv == ["-h"]:
        _print_help(backend=backend, prog=prog)
        return 0
    if command_argv[0] == "runs":
        return _run_registry_command(context, command_argv[1:])
    if "--help" in command_argv or "-h" in command_argv:
        old_cwd = Path.cwd()
        try:
            os.chdir(context.project_root)
            backend = backend or _load_backend()
            backend.execute(command_argv, prog=prog)
        except SystemExit as exc:
            return exc.code if isinstance(exc.code, int) else 0
        finally:
            os.chdir(old_cwd)
        return 0

    command = command_argv[0]
    tracker = RunTracker(context, command, raw_argv)
    try:
        tracker.start()
    except FileExistsError:
        print(f"rquant: error: run id already exists: {context.run_id}", file=sys.stderr)
        return 2

    old_cwd = Path.cwd()
    try:
        os.chdir(context.project_root)
        backend = backend or _load_backend()
        result = backend.execute(command_argv, prog=prog)
        if result is None:
            result = CommandResult()
        tracker.complete(result)
        return result.exit_code
    except KeyboardInterrupt as exc:
        tracker.fail(exc, 130, interrupted=True)
        print("rquant: interrupted", file=sys.stderr)
        return 130
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        if code == 0:
            tracker.complete(CommandResult(exit_code=0))
        else:
            tracker.fail(exc, code)
        return code
    except BaseException as exc:
        tracker.fail(exc, 1)
        print(f"rquant: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        os.chdir(old_cwd)


def legacy_main(*, backend=None) -> int:
    print(
        "DEPRECATION: use 'python -m rquant' or the installed 'rquant' command; "
        "scripts/quant_cli.py remains a compatibility entrypoint.",
        file=sys.stderr,
    )
    return main(backend=backend, prog="scripts.quant_cli")


def _load_backend():
    from scripts import quant_cli

    return quant_cli


def _print_help(*, backend=None, prog: str = "rquant") -> None:
    backend = backend or _load_backend()
    backend.build_parser(prog=prog).print_help()
    print(
        "\nframework commands:\n"
        "  runs list [--status complete|failed] [--limit N]\n"
        "  runs show <run-id>\n\n"
        "global options (accepted before or after the command):\n"
        "  --project-root PATH\n"
        "  --runs-dir PATH\n"
        "  --run-id TEXT\n"
        "  --log-level DEBUG|INFO|WARNING|ERROR"
    )


def _extract_global_options(argv: Sequence[str]) -> tuple[GlobalOptions, list[str]]:
    options = GlobalOptions()
    remaining: list[str] = []
    names = {
        "--project-root": "project_root",
        "--runs-dir": "runs_dir",
        "--run-id": "run_id",
        "--log-level": "log_level",
    }
    index = 0
    while index < len(argv):
        token = argv[index]
        option, separator, inline_value = token.partition("=")
        attribute = names.get(option)
        if not attribute:
            remaining.append(token)
            index += 1
            continue
        if separator:
            value = inline_value
        else:
            if index + 1 >= len(argv):
                raise ValueError(f"{option} requires a value")
            index += 1
            value = argv[index]
        setattr(options, attribute, value)
        index += 1
    return options, remaining


def _run_registry_command(context: ProjectContext, argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="rquant runs", description="Inspect RQuant run manifests")
    sub = parser.add_subparsers(dest="runs_command", required=True)
    list_parser = sub.add_parser("list", help="List recent runs")
    list_parser.add_argument("--status", choices=("complete", "failed"), default=None)
    list_parser.add_argument("--limit", type=int, default=20)
    show_parser = sub.add_parser("show", help="Show one full run manifest")
    show_parser.add_argument("run_id")
    try:
        args = parser.parse_args(list(argv))
        if args.runs_command == "list":
            runs = load_run_manifests(
                context.runs_dir, status=args.status, limit=args.limit
            )
            if not runs:
                print("No matching RQuant runs.")
                return 0
            print(f"{'RUN ID':<30} {'STATUS':<12} {'COMMAND':<28} FINISHED")
            for run in runs:
                print(
                    f"{str(run.get('run_id', '')):<30} "
                    f"{str(run.get('status', 'unknown')):<12} "
                    f"{str(run.get('command', '')):<28} "
                    f"{run.get('finished_at') or '-'}"
                )
            return 0
        payload = load_run_manifest(context.runs_dir, args.run_id)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"rquant runs: error: {exc}", file=sys.stderr)
        return 2
