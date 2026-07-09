import unittest

import pandas as pd

from backtest.benchmark_compare import align_portfolio_and_benchmark
from backtest.performance import annualized_return, max_drawdown, sharpe_ratio
from backtest.transaction_cost import calculate_buy_cost, calculate_sell_cost


class BacktestHelperTests(unittest.TestCase):
    def test_transaction_cost_direction(self):
        self.assertEqual(calculate_buy_cost(0, 0.0003, 5), 0.0)
        self.assertEqual(calculate_buy_cost(10000, 0.0003, 5), 5.0)
        self.assertAlmostEqual(calculate_sell_cost(10000, 0.0003, 5, 0.001, 0.00001), 15.1)

    def test_performance_helpers(self):
        self.assertAlmostEqual(max_drawdown([100.0, 120.0, 90.0]), -0.25)
        self.assertIsNotNone(annualized_return(0.1, 100))
        self.assertIsNotNone(sharpe_ratio([0.01, -0.005, 0.002]))

    def test_benchmark_alignment(self):
        portfolio = pd.DataFrame({"date": ["2026-01-01", "2026-01-02"], "ret": [0.01, 0.02]})
        benchmark = pd.DataFrame({"date": ["2026-01-02"], "ret": [0.015]})
        result = align_portfolio_and_benchmark(portfolio, benchmark)
        self.assertEqual(len(result), 1)
        self.assertEqual(str(result.loc[0, "date"].date()), "2026-01-02")


if __name__ == "__main__":
    unittest.main()
