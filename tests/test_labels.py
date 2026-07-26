import unittest

import polars as pl

from labels.make_forward_return import make_forward_returns, make_next_open_returns


class ForwardReturnLabelTests(unittest.TestCase):
    def test_next_open_returns_match_portfolio_entry_and_exit_timing(self):
        prices = pl.DataFrame(
            {
                "date": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"],
                "symbol": ["000001"] * 4,
                "open": [10.0, 11.0, 12.0, 13.2],
            }
        )

        result = make_next_open_returns(prices, windows=(1, 2))

        self.assertAlmostEqual(result.item(0, "next_open_return_1d"), 12.0 / 11.0 - 1.0)
        self.assertAlmostEqual(result.item(0, "next_open_return_2d"), 13.2 / 11.0 - 1.0)
        self.assertIsNone(result.item(2, "next_open_return_1d"))

    def test_forward_returns_are_future_targets(self):
        prices = pl.DataFrame(
            {
                "date": ["2026-01-01", "2026-01-02", "2026-01-03"],
                "symbol": ["000001", "000001", "000001"],
                "close": [10.0, 11.0, 12.1],
            }
        )

        result = make_forward_returns(prices, windows=(1, 2))

        self.assertEqual(result.item(0, "symbol"), "000001")
        self.assertAlmostEqual(result.item(0, "forward_return_1d"), 0.1)
        self.assertAlmostEqual(result.item(0, "forward_return_2d"), 0.21)
        self.assertIsNone(result.item(2, "forward_return_1d"))

    def test_duplicate_date_symbol_fails(self):
        prices = pl.DataFrame(
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
