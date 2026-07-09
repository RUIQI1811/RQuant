import unittest

import pandas as pd

from labels.make_forward_return import make_forward_returns


class ForwardReturnLabelTests(unittest.TestCase):
    def test_forward_returns_are_future_targets(self):
        prices = pd.DataFrame(
            {
                "date": ["2026-01-01", "2026-01-02", "2026-01-03"],
                "symbol": ["000001", "000001", "000001"],
                "close": [10.0, 11.0, 12.1],
            }
        )

        result = make_forward_returns(prices, windows=(1, 2))

        self.assertEqual(result.loc[0, "symbol"], "000001")
        self.assertAlmostEqual(result.loc[0, "forward_return_1d"], 0.1)
        self.assertAlmostEqual(result.loc[0, "forward_return_2d"], 0.21)
        self.assertTrue(pd.isna(result.loc[2, "forward_return_1d"]))

    def test_duplicate_date_symbol_fails(self):
        prices = pd.DataFrame(
            {
                "date": ["2026-01-01", "2026-01-01"],
                "symbol": ["000001", "000001"],
                "close": [10.0, 11.0],
            }
        )

        with self.assertRaisesRegex(ValueError, "duplicate"):
            make_forward_returns(prices)


if __name__ == "__main__":
    unittest.main()
