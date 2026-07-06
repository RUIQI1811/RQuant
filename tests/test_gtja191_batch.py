import tempfile
import unittest
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from pipeline.gtja191_batch import (
    GTJA191BatchConfig,
    GTJA191BatchRunner,
    parse_gtja_selection,
)
from pipeline.factors.gtja191 import GTJA191ExternalData, GTJA191_NAMES
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
        self.assertIn("--ignore-factor-config", result.stdout)
        self.assertIn("--benchmark-file", result.stdout)

    def test_parse_all_returns_ordered_191_names(self):
        self.assertEqual(parse_gtja_selection(("all",), ()), GTJA191_NAMES)

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


if __name__ == "__main__":
    unittest.main()
