import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reports.factor_tester import (
    FactorTester,
    FactorTesterConfig,
    _max_drawdown,
    _sharpe,
)


def _sample_factor_frame():
    rows = []
    for date in ("2026-01-01", "2026-01-02"):
        for symbol, factor, ret in [
            ("000001", 1.0, 0.01),
            ("000002", 2.0, 0.02),
            ("000003", 3.0, 0.03),
            ("000004", 4.0, 0.04),
        ]:
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "factor_value": factor,
                    "close": 10.0,
                    "forward_return_1d": ret,
                }
            )
    return pd.DataFrame(rows)


def _test_config(**kwargs):
    values = {
        "min_listing_days": 0,
        "commission_rate": 0.0,
        "slippage_rate": 0.0,
        "stamp_tax_rate": 0.0,
    }
    values.update(kwargs)
    return FactorTesterConfig(**values)


class FactorTesterTest(unittest.TestCase):
    def test_ic_calculation_is_correct(self):
        tester = FactorTester(
            _sample_factor_frame(),
            factor_name="test_factor",
            config=_test_config(forward_return_windows=(1,), groups=2),
        )

        _, summary = tester.ic_test()

        self.assertAlmostEqual(summary.loc[0, "ic_mean"], 1.0)
        self.assertAlmostEqual(summary.loc[0, "ic_win_rate"], 1.0)

    def test_rank_ic_calculation_is_correct(self):
        tester = FactorTester(
            _sample_factor_frame(),
            factor_name="test_factor",
            config=_test_config(forward_return_windows=(1,), groups=2),
        )

        _, summary = tester.ic_test()

        self.assertAlmostEqual(summary.loc[0, "rank_ic_mean"], 1.0)
        self.assertAlmostEqual(summary.loc[0, "rank_ic_win_rate"], 1.0)

    def test_group_return_and_top_bottom_are_correct(self):
        tester = FactorTester(
            _sample_factor_frame(),
            factor_name="test_factor",
            config=_test_config(forward_return_windows=(1,), groups=2),
        )

        group_return, group_summary = tester.group_return_test()

        avg_by_group = group_return.groupby("group")["mean_forward_return"].mean()
        self.assertAlmostEqual(avg_by_group.loc[1], 0.015)
        self.assertAlmostEqual(avg_by_group.loc[2], 0.035)
        self.assertAlmostEqual(group_summary.loc[0, "top_bottom_return"], 0.02)
        self.assertTrue(bool(group_summary.loc[0, "monotonic"]))

    def test_top_n_return_uses_long_only_fixed_counts(self):
        tester = FactorTester(
            _sample_factor_frame(),
            factor_name="test_factor",
            config=_test_config(
                forward_return_windows=(1,),
                groups=2,
                top_n_counts=(1, 2, 3),
            ),
        )

        top_n_return, top_n_summary = tester.top_n_return_test()

        by_count = top_n_return.groupby("top_n")["mean_forward_return"].mean()
        self.assertAlmostEqual(by_count.loc[1], 0.04)
        self.assertAlmostEqual(by_count.loc[2], 0.035)
        self.assertAlmostEqual(by_count.loc[3], 0.03)
        summary = top_n_summary.set_index("top_n")
        self.assertAlmostEqual(summary.loc[1, "mean_forward_return"], 0.04)
        self.assertEqual(summary.loc[1, "average_selected_count"], 1.0)

    def test_long_short_nav_uses_daily_returns_and_staggered_holdings(self):
        rows = []
        dates = pd.date_range("2026-01-01", periods=6, freq="B")
        for date_index, date in enumerate(dates):
            for symbol, factor in [
                ("000001", 1.0),
                ("000002", 2.0),
                ("000003", 3.0),
                ("000004", 4.0),
            ]:
                rows.append(
                    {
                        "date": date,
                        "symbol": symbol,
                        "factor_value": factor,
                        "close": 10.0 * (1.05**date_index) if factor >= 3 else 10.0,
                    }
                )
        tester = FactorTester(
            pd.DataFrame(rows),
            factor_name="test_factor",
            config=_test_config(forward_return_windows=(2,), groups=2),
        )

        result = tester.tradable_long_short_test()

        self.assertEqual(result["date"].tolist(), dates[2:].tolist())
        self.assertEqual(result["active_cohorts"].tolist(), [1, 1, 1, 1])
        self.assertAlmostEqual(result.loc[0, "gross_return"], 0.025)
        self.assertAlmostEqual(result.loc[1, "gross_return"], 0.05)
        self.assertAlmostEqual(result.loc[1, "tradable_cum_nav"], 1.025 * 1.05)
        self.assertAlmostEqual(result.loc[3, "tradable_cum_nav"], 1.025 * 1.05**3)
        daily_returns = pd.Series([0.025, 0.05, 0.05, 0.05])
        expected_sharpe = daily_returns.mean() / daily_returns.std(ddof=1) * 252**0.5
        self.assertAlmostEqual(result.loc[0, "sharpe"], expected_sharpe)

    def test_factor_is_shifted_one_day_before_evaluation(self):
        tester = FactorTester(
            _sample_factor_frame(),
            factor_name="test_factor",
            config=_test_config(forward_return_windows=(1,), groups=2),
        )

        prepared = tester.prepare_data()
        symbol = prepared[prepared["symbol"].eq("000001")].sort_values("date")

        self.assertTrue(pd.isna(symbol.iloc[0]["factor_processed"]))
        self.assertEqual(symbol.iloc[1]["factor_processed"], symbol.iloc[0]["factor_raw"])

    def test_stat_nav_uses_forward_returns_but_tradable_nav_does_not(self):
        rows = []
        dates = pd.date_range("2026-01-01", periods=4, freq="B")
        for date in dates:
            for symbol, factor, forward_return in [
                ("000001", 1.0, 0.01),
                ("000002", 2.0, 0.02),
                ("000003", 3.0, 0.03),
                ("000004", 4.0, 0.04),
            ]:
                rows.append(
                    {
                        "date": date,
                        "symbol": symbol,
                        "factor_value": factor,
                        "close": 10.0,
                        "forward_return_5d": forward_return,
                    }
                )
        tester = FactorTester(
            pd.DataFrame(rows),
            factor_name="test_factor",
            config=_test_config(forward_return_windows=(5,), groups=2),
        )

        stat = tester.long_short_test()
        tradable = tester.tradable_long_short_test()

        self.assertIn("stat_cum_nav", stat.columns)
        self.assertNotIn("tradable_cum_nav", stat.columns)
        self.assertAlmostEqual(stat.iloc[-1]["stat_cum_nav"], 1.02**3)
        self.assertAlmostEqual(stat.iloc[0]["annualized_return"], 1.02 ** (252 / 5) - 1.0)
        self.assertIn("tradable_cum_nav", tradable.columns)
        self.assertNotIn("stat_cum_nav", tradable.columns)
        self.assertAlmostEqual(tradable.iloc[-1]["tradable_cum_nav"], 1.0)

    def test_tradable_nav_deducts_entry_and_exit_costs(self):
        rows = []
        dates = pd.date_range("2026-01-01", periods=3, freq="B")
        for date_index, date in enumerate(dates):
            for symbol, factor in [
                ("000001", 1.0),
                ("000002", 2.0),
                ("000003", 3.0),
                ("000004", 4.0),
            ]:
                close = 10.5 if date_index == 2 and factor >= 3 else 10.0
                rows.append(
                    {
                        "date": date,
                        "symbol": symbol,
                        "factor_value": factor,
                        "close": close,
                    }
                )
        tester = FactorTester(
            pd.DataFrame(rows),
            factor_name="test_factor",
            config=_test_config(
                forward_return_windows=(1,),
                groups=2,
                commission_rate=0.001,
            ),
        )

        result = tester.tradable_long_short_test()

        self.assertAlmostEqual(result.loc[0, "gross_return"], 0.05)
        self.assertAlmostEqual(result.loc[0, "transaction_cost"], 0.004)
        self.assertAlmostEqual(result.loc[0, "net_return"], 0.046)

    def test_limit_up_blocks_long_entry(self):
        rows = []
        dates = pd.date_range("2026-01-01", periods=3, freq="B")
        for date_index, date in enumerate(dates):
            for symbol, factor in [
                ("000001", 1.0),
                ("000002", 2.0),
                ("000003", 3.0),
                ("000004", 4.0),
            ]:
                rows.append(
                    {
                        "date": date,
                        "symbol": symbol,
                        "factor_value": factor,
                        "close": 10.0,
                        "is_limit_up": date_index == 1 and factor >= 3,
                    }
                )
        tester = FactorTester(
            pd.DataFrame(rows),
            factor_name="test_factor",
            config=_test_config(forward_return_windows=(1,), groups=2),
        )

        self.assertTrue(tester.tradable_long_short_test().empty)

    def test_st_new_stock_and_liquidity_filters_are_point_in_time(self):
        rows = []
        dates = pd.date_range("2026-01-01", periods=3, freq="B")
        for date in dates:
            for symbol, is_st in (("000001", False), ("000002", True)):
                rows.append(
                    {
                        "date": date,
                        "symbol": symbol,
                        "factor_value": 1.0,
                        "close": 10.0,
                        "volume": 100.0,
                        "is_st": is_st,
                    }
                )
        tester = FactorTester(
            pd.DataFrame(rows),
            factor_name="test_factor",
            config=_test_config(
                forward_return_windows=(1,),
                groups=2,
                min_listing_days=2,
                min_liquidity=500.0,
            ),
        )

        prepared = tester.prepare_data()
        first_date = prepared[prepared["date"].eq(dates[0])]
        second_date = prepared[prepared["date"].eq(dates[1])].set_index("symbol")
        _, status = tester.universe_filter_test()

        self.assertTrue(first_date["factor_processed"].isna().all())
        self.assertFalse(pd.isna(second_date.loc["000001", "factor_processed"]))
        self.assertTrue(pd.isna(second_date.loc["000002", "factor_processed"]))
        self.assertEqual(status.set_index("check").loc["st", "status"], "active")
        self.assertEqual(status.set_index("check").loc["liquidity", "status"], "active")

    def test_neutralized_ic_uses_industry_and_market_cap_controls(self):
        rows = []
        dates = pd.date_range("2026-01-01", periods=3, freq="B")
        factors = [1.0, 4.0, 2.0, 8.0, 5.0, 7.0]
        returns = [0.01, 0.05, 0.02, 0.08, 0.03, 0.07]
        caps = [10.0, 20.0, 50.0, 12.0, 25.0, 60.0]
        for date in dates:
            for index in range(6):
                rows.append(
                    {
                        "date": date,
                        "symbol": f"{index + 1:06d}",
                        "factor_value": factors[index],
                        "close": 10.0,
                        "forward_return_1d": returns[index],
                        "industry": "a" if index < 3 else "b",
                        "market_cap": caps[index],
                    }
                )
        tester = FactorTester(
            pd.DataFrame(rows),
            factor_name="test_factor",
            config=_test_config(forward_return_windows=(1,), groups=2, min_periods=3),
        )

        neutralized, summary = tester.neutralized_ic_test()

        valid = neutralized[neutralized["status"].eq("ok")]
        self.assertFalse(valid.empty)
        self.assertTrue(valid["controls"].eq("industry+log_market_cap").all())
        self.assertIn("neutralized_rank_ic_mean", summary.columns)

    def test_sharpe_uses_window_adjusted_annualization(self):
        returns = pd.Series([0.02, -0.01, 0.03])

        actual = _sharpe(returns, return_horizon_days=5)
        expected = returns.mean() / returns.std(ddof=1) * (252 / 5) ** 0.5

        self.assertAlmostEqual(actual, expected)

    def test_max_drawdown_includes_initial_nav(self):
        self.assertAlmostEqual(_max_drawdown(pd.Series([0.90, 0.95])), 0.10)

    def test_missing_values_are_ignored_in_ic(self):
        df = _sample_factor_frame()
        df.loc[0, "factor_value"] = None
        df.loc[1, "forward_return_1d"] = None
        tester = FactorTester(
            df,
            factor_name="test_factor",
            config=_test_config(forward_return_windows=(1,), groups=2, min_periods=2),
        )

        ic, summary = tester.ic_test()

        self.assertEqual(ic["ic"].notna().sum(), 1)
        self.assertAlmostEqual(summary.loc[0, "ic_mean"], 1.0)

    def test_constant_cross_section_produces_nan_ic_without_warning(self):
        df = _sample_factor_frame()
        df["factor_value"] = 1.0
        tester = FactorTester(
            df,
            factor_name="constant_factor",
            config=_test_config(forward_return_windows=(1,), groups=2),
        )

        ic, summary = tester.ic_test()

        self.assertTrue(ic["ic"].isna().all())
        self.assertTrue(ic["rank_ic"].isna().all())
        self.assertEqual(summary.loc[0, "count"], 0)


class TopLevelFactorTesterImportTests(unittest.TestCase):
    def test_factor_tester_imports_from_reports_package(self):
        import reports.factor_tester as factor_tester

        self.assertTrue(hasattr(factor_tester, "FactorTester"))


if __name__ == "__main__":
    unittest.main()
