import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import print_candidates


class CandidatePrinterTest(unittest.TestCase):
    def test_format_candidate_codes_prints_all_codes_in_order(self):
        data = {
            "pick_date": "2026-06-04",
            "candidates": [
                {"code": "002371", "strategy": "b1", "close": 627.0},
                {"code": "002297", "strategy": "b1", "close": 20.78},
            ],
        }

        output = print_candidates.format_candidate_codes(data)

        self.assertEqual(output, "pick_date: 2026-06-04\n002371\n002297")


if __name__ == "__main__":
    unittest.main()
