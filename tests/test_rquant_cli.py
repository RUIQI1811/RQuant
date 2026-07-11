import argparse
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

from rquant.cli import main
from rquant.runtime import CommandResult


class _FakeBackend:
    def __init__(self, result=None, error=None):
        self.result = result or CommandResult(summary={"ok": True})
        self.error = error
        self.calls = []

    def execute(self, argv, *, prog):
        self.calls.append((list(argv), prog, Path.cwd()))
        if self.error:
            raise self.error
        return self.result

    def build_parser(self, *, prog):
        return argparse.ArgumentParser(prog=prog)


class RQuantCliTest(unittest.TestCase):
    def test_governed_entry_removes_global_options_and_writes_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backend = _FakeBackend(CommandResult(outputs={"report": "report.json"}))
            code = main(
                [
                    "--project-root",
                    str(root),
                    "doctor",
                    "--deep",
                    "--runs-dir",
                    "audit/runs",
                    "--run-id",
                    "fixed-run",
                    "--log-level",
                    "warning",
                ],
                backend=backend,
            )

            self.assertEqual(code, 0)
            self.assertEqual(backend.calls[0][0], ["doctor", "--deep"])
            self.assertEqual(backend.calls[0][1], "rquant")
            self.assertEqual(backend.calls[0][2], root.resolve())
            manifest = root / "audit/runs/fixed-run/run.json"
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "complete")
            self.assertEqual(payload["command"], "doctor")

    def test_business_system_exit_is_recorded_before_exit_code_is_returned(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backend = _FakeBackend(error=SystemExit(2))
            with redirect_stderr(StringIO()):
                code = main(
                    [
                        "doctor",
                        "--project-root",
                        str(root),
                        "--run-id",
                        "partial-run",
                    ],
                    backend=backend,
                )

            self.assertEqual(code, 2)
            payload = json.loads(
                (root / "data/runs/partial-run/run.json").read_text(encoding="utf-8")
            )
            self.assertEqual(payload["status"], "failed")
            self.assertEqual(payload["exit_code"], 2)
            self.assertEqual(payload["error"]["type"], "SystemExit")

    def test_runs_list_and_show_do_not_create_new_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backend = _FakeBackend()
            self.assertEqual(
                main(
                    ["doctor", "--project-root", str(root), "--run-id", "source-run"],
                    backend=backend,
                ),
                0,
            )
            with redirect_stdout(StringIO()) as listed:
                code = main(
                    ["runs", "list", "--project-root", str(root)], backend=backend
                )
            self.assertEqual(code, 0)
            self.assertIn("source-run", listed.getvalue())

            with redirect_stdout(StringIO()) as shown:
                code = main(
                    ["runs", "show", "source-run", "--project-root", str(root)],
                    backend=backend,
                )
            self.assertEqual(code, 0)
            self.assertIn('"run_id": "source-run"', shown.getvalue())
            self.assertEqual(len(list((root / "data/runs").iterdir())), 1)

    def test_help_does_not_create_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backend = _FakeBackend()
            with redirect_stdout(StringIO()):
                code = main(["--help", "--project-root", str(root)], backend=backend)
            self.assertEqual(code, 0)
            self.assertFalse((root / "data/runs").exists())

    def test_subcommand_help_does_not_create_run(self):
        class HelpBackend(_FakeBackend):
            def execute(self, argv, *, prog):
                self.calls.append((list(argv), prog, Path.cwd()))
                raise SystemExit(0)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backend = HelpBackend()
            code = main(
                ["doctor", "--help", "--project-root", str(root)], backend=backend
            )
            self.assertEqual(code, 0)
            self.assertFalse((root / "data/runs").exists())


if __name__ == "__main__":
    unittest.main()
