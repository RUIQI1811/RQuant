import unittest

import numpy as np
import pandas as pd

import factors.alpha101 as alpha101
import factors.gtja191 as gtja191
from factors import operators


class SharedFactorOperatorsTest(unittest.TestCase):
    def test_window_uses_source_formula_floor_semantics(self):
        self.assertEqual(operators.window(0), 1)
        self.assertEqual(operators.window(2.49), 2)
        self.assertEqual(operators.window(2.5), 2)
        self.assertEqual(operators.window(3.92795), 3)

    def test_rank_is_daily_cross_sectional_percentile(self):
        frame = pd.DataFrame([[3.0, 1.0, 2.0]], columns=list("abc"))
        expected = pd.DataFrame([[1.0, 1 / 3, 2 / 3]], columns=list("abc"))
        pd.testing.assert_frame_equal(operators.rank(frame), expected)

    def test_time_series_operators_require_complete_windows(self):
        frame = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        self.assertTrue(np.isnan(operators.ts_sum(frame, 3).iloc[1, 0]))
        self.assertEqual(operators.ts_sum(frame, 3).iloc[2, 0], 6.0)
        self.assertEqual(operators.ts_argmax(frame, 3).iloc[2, 0], 3.0)

    def test_safe_div_masks_zero_and_near_zero_denominators(self):
        numerator = pd.DataFrame([[1.0, 1.0]], columns=["a", "b"])
        denominator = pd.DataFrame([[0.0, 1e-11]], columns=["a", "b"])
        actual = operators.safe_div(numerator, denominator)
        self.assertTrue(np.isnan(actual.loc[0, "a"]))
        self.assertEqual(actual.loc[0, "b"], 1e11)

    def test_alpha101_keeps_backward_compatible_operator_exports(self):
        for name in (
            "rank",
            "delay",
            "correlation",
            "covariance",
            "delta",
            "ts_rank",
            "ts_sum",
            "stddev",
        ):
            with self.subTest(name=name):
                self.assertIs(getattr(alpha101, name), getattr(operators, name))

    def test_gtja_reuses_common_core_but_keeps_family_specific_operators(self):
        self.assertIs(gtja191.rank, operators.rank)
        self.assertIs(gtja191._safe_div, operators.safe_div)
        self.assertTrue(callable(gtja191.sma_cn))
        self.assertTrue(callable(gtja191.wma))
        self.assertTrue(callable(gtja191.highday))
        self.assertFalse(hasattr(operators, "sma_cn"))
        self.assertFalse(hasattr(operators, "highday"))


if __name__ == "__main__":
    unittest.main()
