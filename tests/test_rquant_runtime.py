import json
import logging
import tempfile
import unittest
from pathlib import Path

from rquant.runtime import (
    CommandResult,
    ProjectContext,
    RunTracker,
    load_run_manifest,
    load_run_manifests,
    redact_argv,
)


class RQuantRuntimeTest(unittest.TestCase):
    def test_context_resolves_relative_paths_under_explicit_project_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            context = ProjectContext.create(
                project_root=root,
                runs_dir="custom/runs",
                run_id="unit-run",
                log_level="debug",
            )

            self.assertEqual(context.project_root, root.resolve())
            self.assertEqual(context.runs_dir, (root / "custom/runs").resolve())
            self.assertEqual(context.resolve("data/raw"), (root / "data/raw").resolve())
            self.assertEqual(context.log_level, "DEBUG")

    def test_tracker_writes_atomic_complete_manifest_and_path_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "config.yaml"
            config.write_text("enabled: true\n", encoding="utf-8")
            output = root / "output"
            output.mkdir()
            downstream = output / "manifest.json"
            downstream.write_text('{"status": "complete"}\n', encoding="utf-8")
            context = ProjectContext.create(project_root=root, run_id="complete-run")
            tracker = RunTracker(
                context,
                "doctor",
                ["doctor", "--config", "config.yaml", "--output", "output"],
            )

            tracker.start()
            logging.getLogger("test").warning("diagnostic warning")
            tracker.complete(
                CommandResult(outputs={"report_dir": "output"}, summary={"ok": True})
            )

            payload = json.loads(context.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "complete")
            self.assertEqual(payload["exit_code"], 0)
            self.assertEqual(len(payload["inputs"]["config"]["sha256"]), 64)
            self.assertTrue(payload["outputs"]["report_dir"]["exists"])
            self.assertEqual(len(payload["downstream_manifests"]), 1)
            self.assertIn("diagnostic warning", payload["warnings"][0])
            self.assertTrue(context.log_path.is_file())

    def test_tracker_finalizes_failure_without_exposing_secret(self):
        with tempfile.TemporaryDirectory() as tmp:
            context = ProjectContext.create(project_root=tmp, run_id="failed-run")
            tracker = RunTracker(
                context,
                "fake",
                ["fake", "--api-key", "sk-supersecret123", "--token=plain-secret"],
            )
            tracker.start()
            tracker.fail(RuntimeError("token=sk-supersecret123 was rejected"), 2)

            raw = context.manifest_path.read_text(encoding="utf-8")
            payload = json.loads(raw)
            self.assertEqual(payload["status"], "failed")
            self.assertEqual(payload["exit_code"], 2)
            self.assertNotIn("supersecret", raw)
            self.assertIn("<redacted>", raw)

    def test_run_registry_filters_and_loads_manifests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for run_id, status in (("20260102-new", "failed"), ("20260101-old", "complete")):
                context = ProjectContext.create(project_root=root, run_id=run_id)
                tracker = RunTracker(context, "doctor", ["doctor"])
                tracker.start()
                if status == "complete":
                    tracker.complete()
                else:
                    tracker.fail(RuntimeError("boom"), 1)

            failed = load_run_manifests(root / "data/runs", status="failed", limit=10)
            self.assertEqual([item["run_id"] for item in failed], ["20260102-new"])
            loaded = load_run_manifest(root / "data/runs", "20260101-old")
            self.assertEqual(loaded["status"], "complete")

    def test_redact_argv_handles_separate_and_inline_values(self):
        redacted = redact_argv(
            ["doctor", "--api-key", "abc", "--password=def", "--config", "safe.yaml"]
        )
        self.assertEqual(redacted[2], "<redacted>")
        self.assertEqual(redacted[3], "--password=<redacted>")
        self.assertEqual(redacted[-1], "safe.yaml")
if __name__ == "__main__":
    unittest.main()
