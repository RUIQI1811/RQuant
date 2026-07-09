import tempfile
import unittest
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from reports.gtja191_batch import (
    GTJA191BatchConfig,
    GTJA191BatchRunner,
    filter_gtja_selection_from_start,
    parse_gtja_selection,
)
from factors.gtja191 import GTJA191ExternalData, GTJA191_NAMES
from tests.test_gtja191 import _complete_panels


class GTJA191BatchTest(unittest.TestCase):
    def test_cli_help_exposes_all_factor_and_resume_controls(self):
        result = subprocess.run(
            [sys.executable, "scripts/test_gtja191_batch.py", "--help"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("--factors", result.stdout)
        self.assertIn("--start-factor", result.stdout)
        self.assertIn("--top-counts", result.stdout)
        self.assertIn("--no-progress", result.stdout)
        self.assertIn("--ignore-factor-config", result.stdout)
        self.assertIn("--benchmark-file", result.stdout)
        self.assertIn("leaderboard.csv", result.stdout)

    def test_parse_all_returns_ordered_191_names(self):
        self.assertEqual(parse_gtja_selection(("all",), ()), GTJA191_NAMES)

    def test_start_factor_keeps_requested_factors_from_registry_position(self):
        selected = parse_gtja_selection(("1-5",), ("3",))

        self.assertEqual(
            filter_gtja_selection_from_start(selected, "gtja_002"),
            ("gtja_002", "gtja_004", "gtja_005"),
        )

    def test_every_requested_factor_gets_one_terminal_status(self):
        panels = replace(_complete_panels(days=90), external=GTJA191ExternalData())
        with tempfile.TemporaryDirectory() as temp_dir:
            result = GTJA191BatchRunner(
                panels,
                factors=("gtja_001", "gtja_030", "gtja_191"),
                output_dir=Path(temp_dir),
                config=GTJA191BatchConfig(
                    windows=(1,),
                    groups=5,
                    min_periods=2,
                    min_listing_days=0,
                ),
            ).run()
            self.assertEqual(
                result.status["factor"].tolist(),
                ["gtja_001", "gtja_030", "gtja_191"],
            )
            statuses = result.status.set_index("factor")["status"]
            self.assertEqual(statuses.loc["gtja_030"], "missing_input")
            self.assertTrue((Path(temp_dir) / "batch_status.csv").exists())
            self.assertTrue((Path(temp_dir) / "leaderboard.csv").exists())

    def test_leaderboard_collects_prior_completed_gtja_reports(self):
        panels = replace(_complete_panels(days=90), external=GTJA191ExternalData())
        config = GTJA191BatchConfig(
            windows=(1,),
            groups=5,
            min_periods=2,
            min_listing_days=0,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            GTJA191BatchRunner(
                panels,
                factors=("gtja_001",),
                output_dir=Path(temp_dir),
                config=config,
                factor_statuses={"gtja_001": "watch", "gtja_002": "active"},
            ).run()

            result = GTJA191BatchRunner(
                panels,
                factors=("gtja_002",),
                output_dir=Path(temp_dir),
                config=config,
                factor_statuses={"gtja_001": "watch", "gtja_002": "active"},
            ).run()

            self.assertEqual(
                sorted(result.leaderboard["factor"].unique().tolist()),
                ["gtja_001", "gtja_002"],
            )
            statuses = result.leaderboard.drop_duplicates("factor").set_index("factor")[
                "factor_status"
            ]
            self.assertEqual(statuses.loc["gtja_001"], "watch")


class TopLevelGtja191BatchImportTests(unittest.TestCase):
    def test_gtja191_batch_imports_from_reports_package(self):
        import reports.gtja191_batch as gtja191_batch

        self.assertTrue(hasattr(gtja191_batch, "GTJA191BatchRunner"))


if __name__ == "__main__":
    unittest.main()
