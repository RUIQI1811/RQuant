import sys
import unittest
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from signals.factor_adapters import FactorSignalConfig, factor_frame_to_signal_frame


class FactorSignalTest(unittest.TestCase):
    def test_factor_frame_selects_top_n_each_day(self):
        data = pl.DataFrame(
            {
                "date": ["2026-01-01", "2026-01-01", "2026-01-01", "2026-01-02", "2026-01-02"],
                "symbol": ["000001", "000002", "000003", "000001", "000002"],
                "factor_value": [0.1, 0.3, 0.2, 0.5, 0.4],
            }
        )

        signals = factor_frame_to_signal_frame(
            data,
            config=FactorSignalConfig(source="factor_test", top_n=1),
        )

        self.assertEqual(signals["symbol"].to_list(), ["000002", "000001"])
        self.assertEqual(signals["source"].to_list(), ["factor_test", "factor_test"])
        self.assertEqual(signals["score"].to_list(), [0.3, 0.5])


if __name__ == "__main__":
    unittest.main()
