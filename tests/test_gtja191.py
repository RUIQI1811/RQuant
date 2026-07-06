import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.factors.gtja191 import (  # noqa: E402
    GTJA191ExternalData,
    GTJA191Panels,
    count,
    highday,
    lowday,
    normalize_gtja_name,
    regbeta,
    regresi,
    sma_cn,
    sumif,
    wma,
)


class GTJA191OperatorsTest(unittest.TestCase):
    def test_normalize_gtja_name_accepts_supported_aliases(self):
        self.assertEqual(normalize_gtja_name(1), "gtja_001")
        self.assertEqual(normalize_gtja_name("gtja1"), "gtja_001")
        self.assertEqual(normalize_gtja_name("gtja_191"), "gtja_191")

    def test_normalize_gtja_name_rejects_other_families_and_out_of_range(self):
        with self.assertRaises(KeyError):
            normalize_gtja_name("alpha_001")
        with self.assertRaises(KeyError):
            normalize_gtja_name(192)

    def test_sma_cn_uses_report_recursion(self):
        frame = pd.DataFrame({"a": [1.0, 4.0, 7.0]})
        expected = pd.DataFrame({"a": [1.0, 2.0, 11.0 / 3.0]})
        pd.testing.assert_frame_equal(sma_cn(frame, 3, 1), expected)

    def test_wma_uses_point_nine_distance_weights(self):
        frame = pd.DataFrame({"a": [1.0, 2.0, 4.0]})
        expected = (4.0 + 0.9 * 2.0 + 0.9**2) / (1.0 + 0.9 + 0.9**2)
        self.assertAlmostEqual(wma(frame, 3).iloc[-1, 0], expected)
        self.assertTrue(wma(frame, 3).iloc[:2, 0].isna().all())

    def test_highday_and_lowday_return_distance_from_current(self):
        frame = pd.DataFrame({"a": [3.0, 1.0, 2.0, 3.0]})
        self.assertEqual(highday(frame, 4).iloc[-1, 0], 0.0)
        self.assertEqual(lowday(frame, 4).iloc[-1, 0], 2.0)

    def test_regbeta_and_regresi_use_intercept(self):
        independent = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
        dependent = 2.0 * independent + 3.0
        self.assertAlmostEqual(
            regbeta(dependent, independent, 4).iloc[-1, 0],
            2.0,
        )
        self.assertAlmostEqual(
            regresi(dependent, independent, 4).iloc[-1, 0],
            0.0,
        )

    def test_count_and_sumif_require_full_windows(self):
        values = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
        condition = values > 2.0
        actual_count = count(condition, 3)
        actual_sum = sumif(values, 3, condition)
        self.assertTrue(actual_count.iloc[:2, 0].isna().all())
        self.assertEqual(actual_count.iloc[-1, 0], 2.0)
        self.assertEqual(actual_sum.iloc[-1, 0], 7.0)


class GTJA191PanelsTest(unittest.TestCase):
    def test_panels_keep_aligned_market_data_and_default_external_inputs(self):
        frame = pd.DataFrame(
            {"000001": [10.0, 11.0], "000002": [20.0, 21.0]},
            index=pd.date_range("2026-01-01", periods=2),
        )
        panels = GTJA191Panels(
            open=frame,
            close=frame,
            high=frame,
            low=frame,
            volume=frame,
            amount=frame,
            vwap=frame,
            returns=frame.pct_change(fill_method=None),
        )
        self.assertEqual(panels.close.columns.tolist(), ["000001", "000002"])
        self.assertEqual(panels.external, GTJA191ExternalData())


if __name__ == "__main__":
    unittest.main()
