import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import market.data as market_data
from market.data import StockPoolConfig, build_stock_pool_by_date, clean_daily_frame
from pipeline.pipeline_core import TopTurnoverPoolBuilder


class MarketDataTest(unittest.TestCase):
    def test_build_stock_pool_frame_preserves_stable_tie_order_and_filters(self):
        date = pd.Timestamp("2026-01-02")
        prepared = {
            "600002": pd.DataFrame(
                {"close": [10.0], "turnover_n": [100.0], "is_tradeable": [True]},
                index=[date],
            ),
            "600001": pd.DataFrame(
                {"close": [10.0], "turnover_n": [100.0], "is_tradeable": [True]},
                index=[date],
            ),
            "300001": pd.DataFrame(
                {"close": [20.0], "turnover_n": [200.0], "is_tradeable": [True]},
                index=[date],
            ),
            "600003": pd.DataFrame(
                {"close": [0.5], "turnover_n": [300.0], "is_tradeable": [True]},
                index=[date],
            ),
        }

        frame = market_data.build_stock_pool_frame(
            prepared,
            config=StockPoolConfig(top_m=2, min_price=1.0, exclude_boards=("gem",)),
            allowed_codes=("600002", "600001", "300001", "600003"),
        )

        self.assertEqual(frame["code"].tolist(), ["600002", "600001"])
        self.assertEqual(frame["date"].tolist(), [date, date])

    def test_clean_daily_frame_adds_tradeability_and_limit_flags(self):
        df = pd.DataFrame(
            {
                "date": ["2026-01-01", "2026-01-02", "2026-01-03"],
                "open": [10.0, 11.0, 11.0],
                "high": [10.0, 11.0, 11.0],
                "low": [10.0, 11.0, 11.0],
                "close": [10.0, 11.0, 9.9],
                "volume": [1000, 1000, 0],
                "pre_close": [9.9, 10.0, 11.0],
                "pct_chg": [1.0, 10.0, -10.0],
            }
        )

        clean = clean_daily_frame(df, code="600000")

        self.assertIn("is_tradeable", clean.columns)
        self.assertTrue(bool(clean.loc[pd.Timestamp("2026-01-02"), "is_limit_up"]))
        self.assertTrue(bool(clean.loc[pd.Timestamp("2026-01-03"), "is_limit_down"]))
        self.assertFalse(bool(clean.loc[pd.Timestamp("2026-01-03"), "is_tradeable"]))

    def test_build_stock_pool_by_date_filters_boards_and_tradeability(self):
        prepared = {
            "600000": clean_daily_frame(
                pd.DataFrame(
                    {
                        "date": ["2026-01-01"],
                        "open": [10.0],
                        "high": [10.0],
                        "low": [10.0],
                        "close": [10.0],
                        "volume": [1000],
                    }
                ),
                code="600000",
            ).assign(turnover_n=1000),
            "300001": clean_daily_frame(
                pd.DataFrame(
                    {
                        "date": ["2026-01-01"],
                        "open": [20.0],
                        "high": [20.0],
                        "low": [20.0],
                        "close": [20.0],
                        "volume": [1000],
                    }
                ),
                code="300001",
            ).assign(turnover_n=2000),
        }

        pool = build_stock_pool_by_date(
            prepared,
            config=StockPoolConfig(top_m=10, exclude_boards=("gem",)),
        )

        self.assertEqual(pool[pd.Timestamp("2026-01-01")], ["600000"])

    def test_top_turnover_pool_builder_uses_vectorized_filters(self):
        date = pd.Timestamp("2026-01-02")
        prepared = {
            "600002": pd.DataFrame(
                {"close": [10.0], "turnover_n": [100.0], "is_tradeable": [True]},
                index=[date],
            ),
            "600001": pd.DataFrame(
                {"close": [10.0], "turnover_n": [100.0], "is_tradeable": [True]},
                index=[date],
            ),
            "600003": pd.DataFrame(
                {"close": [0.5], "turnover_n": [300.0], "is_tradeable": [True]},
                index=[date],
            ),
            "600004": pd.DataFrame(
                {"close": [20.0], "turnover_n": [400.0], "is_tradeable": [False]},
                index=[date],
            ),
            "600005": pd.DataFrame(
                {"close": [20.0], "turnover_n": [float("nan")], "is_tradeable": [True]},
                index=[date],
            ),
        }

        pool = TopTurnoverPoolBuilder(top_m=2).build(prepared)

        self.assertEqual(pool[date], ["600002", "600001"])

    def test_st_limit_rate_uses_each_dates_point_in_time_flag(self):
        frame = pd.DataFrame(
            {
                "date": ["2026-01-01", "2026-01-02"],
                "open": [10.0, 10.6],
                "high": [10.0, 10.6],
                "low": [10.0, 10.6],
                "close": [10.0, 10.6],
                "volume": [1000, 1000],
                "pre_close": [10.0, 10.0],
                "pct_chg": [6.0, 6.0],
                "is_st": [False, True],
            }
        )

        clean = clean_daily_frame(frame, code="600000")

        self.assertAlmostEqual(clean.loc[pd.Timestamp("2026-01-01"), "limit_rate"], 0.10)
        self.assertAlmostEqual(clean.loc[pd.Timestamp("2026-01-02"), "limit_rate"], 0.05)
        self.assertFalse(bool(clean.loc[pd.Timestamp("2026-01-01"), "is_limit_up"]))
        self.assertTrue(bool(clean.loc[pd.Timestamp("2026-01-02"), "is_limit_up"]))


class TopLevelMarketImportTests(unittest.TestCase):
    def test_market_data_imports_from_top_level_package(self):
        import market.data as data

        self.assertTrue(hasattr(data, "build_stock_pool_by_date"))


if __name__ == "__main__":
    unittest.main()
