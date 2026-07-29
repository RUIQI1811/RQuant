import json
import tempfile
import unittest
from pathlib import Path

from scripts.organize_workspace import organize_workspace


class OrganizeWorkspaceTest(unittest.TestCase):
    def _make_project(self, root: Path) -> None:
        (root / "AGENTS.md").write_text("test\n", encoding="utf-8")
        (root / "data" / "raw").mkdir(parents=True)
        (root / "factor_report").mkdir()

    def test_preview_does_not_change_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_project(root)
            historical = root / "data" / "portfolio_backtest_old"
            historical.mkdir()
            (historical / "summary.json").write_text("{}\n", encoding="utf-8")
            cache = root / "package" / ".DS_Store"
            cache.parent.mkdir(parents=True)
            cache.write_bytes(b"cache")

            report = organize_workspace(root, apply=False, minimum_age_days=0)

            self.assertTrue(historical.is_dir())
            self.assertTrue(cache.is_file())
            self.assertEqual(report.archived[0]["source"], "data/portfolio_backtest_old")
            self.assertIn("package/.DS_Store", report.removed_caches)

    def test_apply_archives_outputs_records_move_and_removes_only_caches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_project(root)
            canonical = root / "data" / "portfolio_backtest"
            canonical.mkdir()
            (canonical / "portfolio_summary.json").write_text("{}\n", encoding="utf-8")
            historical = root / "data" / "portfolio_backtest_experiment"
            historical.mkdir()
            (historical / "portfolio_summary.json").write_text("{}\n", encoding="utf-8")
            legacy = root / "factor_report" / "factor_run_all" / "alpha101"
            legacy.mkdir(parents=True)
            (legacy / "manifest.json").write_text(
                '{"status": "complete"}\n', encoding="utf-8"
            )
            cache = root / ".DS_Store"
            cache.write_bytes(b"cache")
            bytecode = root / "package" / "__pycache__" / "module.pyc"
            bytecode.parent.mkdir(parents=True)
            bytecode.write_bytes(b"offline cache")

            report = organize_workspace(root, apply=True, minimum_age_days=0)

            self.assertTrue(canonical.is_dir())
            self.assertFalse(historical.exists())
            self.assertTrue(
                (
                    root
                    / "data"
                    / "archive"
                    / "portfolio_backtests"
                    / historical.name
                    / "portfolio_summary.json"
                ).is_file()
            )
            self.assertTrue(
                (
                    root
                    / "factor_report"
                    / "archive"
                    / "legacy_workflows"
                    / "factor_run_all"
                    / "alpha101"
                    / "manifest.json"
                ).is_file()
            )
            self.assertFalse(cache.exists())
            self.assertTrue(bytecode.is_file())
            self.assertEqual(len(report.archived), 2)
            index = json.loads(
                (root / "data" / "archive" / "archive_index.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(len(index), 2)
            self.assertEqual(index[0]["source"], "data/portfolio_backtest_experiment")

    def test_apply_skips_archive_destination_collision(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_project(root)
            source = root / "data" / "backtest_smoke"
            source.mkdir()
            (source / "summary.json").write_text("{}\n", encoding="utf-8")
            destination = (
                root / "data" / "archive" / "signal_backtests" / "backtest_smoke"
            )
            destination.mkdir(parents=True)

            report = organize_workspace(root, apply=True, minimum_age_days=0)

            self.assertTrue(source.is_dir())
            self.assertTrue(destination.is_dir())
            self.assertEqual(len(report.skipped), 1)


if __name__ == "__main__":
    unittest.main()
