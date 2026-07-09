import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reports.alpha101_batch import (
    Alpha101BatchConfig,
    Alpha101BatchRunner,
    build_forward_return_frame,
    build_leaderboard,
    parse_factor_selection,
)
from factors.alpha101 import Alpha101Panels
from factors.catalog import FactorCatalog, load_factor_catalog


def _sample_panels(*, with_cap: bool = False) -> Alpha101Panels:
    dates = pd.date_range("2025-01-01", periods=80, freq="B")
    symbols = [f"{number:06d}" for number in range(1, 11)]
    rng = np.random.default_rng(20260629)
    close = pd.DataFrame(
        20 + rng.normal(0, 0.15, (len(dates), len(symbols))).cumsum(axis=0),
        index=dates,
        columns=symbols,
    )
    open_ = close * (1 + rng.normal(0, 0.005, close.shape))
    high = pd.DataFrame(
        np.maximum(open_, close) * 1.01,
        index=dates,
        columns=symbols,
    )
    low = pd.DataFrame(
        np.minimum(open_, close) * 0.99,
        index=dates,
        columns=symbols,
    )
    volume = pd.DataFrame(
        rng.lognormal(12, 0.2, close.shape),
        index=dates,
        columns=symbols,
    )
    groups = pd.DataFrame(
        [["a"] * 5 + ["b"] * 5] * len(dates),
        index=dates,
        columns=symbols,
    )
    return Alpha101Panels(
        open=open_,
        close=close,
        high=high,
        low=low,
        volume=volume,
        vwap=(high + low + close) / 3,
        returns=close.pct_change(fill_method=None),
        cap=close * volume if with_cap else None,
        sector=groups,
        industry=groups,
        subindustry=groups,
    )


class Alpha101BatchCliTest(unittest.TestCase):
    def test_cli_help_uses_unified_factor_batch_entrypoint(self):
        result = subprocess.run(
            [sys.executable, "scripts/test_factor_batch.py", "--family", "alpha101", "--help"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("--family", result.stdout)
        self.assertIn("alpha101", result.stdout)
        self.assertIn("gtja191", result.stdout)
        self.assertIn("--list-factor-status", result.stdout)


class FactorSelectionTest(unittest.TestCase):
    def test_names_ranges_commas_and_exclusions(self):
        actual = parse_factor_selection(
            ["1-3", "alpha_010,alpha101"],
            ["2", "alpha_101"],
        )
        self.assertEqual(actual, ("alpha_001", "alpha_003", "alpha_010"))

    def test_descending_range_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "descending"):
            parse_factor_selection(["10-1"])

    def test_catalog_runs_active_before_watch_and_skips_disabled(self):
        catalog = FactorCatalog(
            statuses={
                "alpha_001": "watch",
                "alpha_002": "disabled",
                "alpha_003": "active",
            }
        )
        actual = catalog.select(("alpha_001", "alpha_002", "alpha_003"))
        self.assertEqual(actual, ("alpha_003", "alpha_001"))

    def test_catalog_loads_yaml_and_rejects_unknown_factor(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "factors.yaml"
            path.write_text(
                "default_status: watch\nfactors:\n  alpha_001: disabled\n",
                encoding="utf-8",
            )
            catalog = load_factor_catalog(path)
            self.assertEqual(catalog.status_for("alpha_001"), "disabled")
            self.assertEqual(catalog.status_for("alpha_002"), "watch")

            path.write_text(
                "default_status: disabled\n"
                "factors:\n"
                "  alpha_001:\n"
                "    final_score: 80\n"
                "    decision: active\n"
                "    useful_horizons: [10d, 20d]\n"
                "  alpha_002:\n"
                "    decision: component_only\n",
                encoding="utf-8",
            )
            catalog = load_factor_catalog(path)
            self.assertEqual(catalog.status_for("alpha_001"), "active")
            self.assertEqual(catalog.status_for("alpha_002"), "watch")

            path.write_text("factors:\n  alpha_999: active\n", encoding="utf-8")
            with self.assertRaisesRegex((KeyError, ValueError), "999"):
                load_factor_catalog(path)


class ForwardReturnFrameTest(unittest.TestCase):
    def test_evaluation_end_date_keeps_later_prices_for_forward_return(self):
        panels = _sample_panels()
        end_date = panels.close.index[10]
        frame = build_forward_return_frame(
            panels,
            (5,),
            start_date=str(end_date.date()),
            end_date=str(end_date.date()),
        )
        first_symbol = panels.close.columns[0]
        actual = frame.loc[frame["symbol"].eq(first_symbol), "forward_return_5d"].iloc[0]
        expected = panels.close[first_symbol].iloc[15] / panels.close[first_symbol].iloc[10] - 1
        self.assertAlmostEqual(actual, expected)
        row = frame.loc[frame["symbol"].eq(first_symbol)].iloc[0]
        self.assertAlmostEqual(row["close"], panels.close[first_symbol].iloc[10])
        self.assertAlmostEqual(
            row["daily_return"],
            panels.close[first_symbol].iloc[10] / panels.close[first_symbol].iloc[9] - 1,
        )
        expected_liquidity = (
            panels.turnover_value[first_symbol].iloc[:10].mean()
            if panels.turnover_value is not None
            else (panels.close[first_symbol] * panels.volume[first_symbol]).iloc[:10].mean()
        )
        self.assertAlmostEqual(row["avg_turnover_lagged"], expected_liquidity)


class Alpha101BatchRunnerTest(unittest.TestCase):
    def test_progress_setting_does_not_change_result_settings(self):
        hidden = Alpha101BatchConfig(windows=(1,), groups=5)
        visible = Alpha101BatchConfig(windows=(1,), groups=5, show_progress=True)

        self.assertEqual(hidden.result_settings(), visible.result_settings())

    def test_show_progress_wraps_factor_loop_and_reports_current_factor(self):
        class FakeProgress:
            def __init__(self, iterable):
                self.items = list(iterable)
                self.postfixes = []

            def __iter__(self):
                return iter(self.items)

            def set_postfix_str(self, value):
                self.postfixes.append(value)

        progress = FakeProgress([(1, "alpha_101")])
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("reports.alpha101_batch.tqdm", return_value=progress) as progress_factory:
                Alpha101BatchRunner(
                    _sample_panels(),
                    factors=("alpha_101",),
                    output_dir=temp_dir,
                    config=Alpha101BatchConfig(windows=(1,), groups=5, show_progress=True),
                    data_signature="data-v1",
                    implementation_signature="code-v1",
                ).run()

        progress_factory.assert_called_once()
        kwargs = progress_factory.call_args.kwargs
        self.assertEqual(kwargs["desc"], "因子批处理")
        self.assertEqual(kwargs["unit"], "因子")
        self.assertFalse(kwargs["disable"])
        self.assertEqual(progress.postfixes, ["alpha_101"])

    def test_checkpoint_rebuilds_full_leaderboard_only_once(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch(
                "reports.alpha101_batch.build_leaderboard",
                wraps=build_leaderboard,
            ) as builder:
                result = Alpha101BatchRunner(
                    _sample_panels(),
                    factors=("alpha_101", "alpha_001"),
                    output_dir=temp_dir,
                    config=Alpha101BatchConfig(windows=(1,), groups=5),
                    data_signature="data-v1",
                    implementation_signature="code-v1",
                ).run()

        full_rebuilds = [
            call
            for call in builder.call_args_list
            if tuple(call.args[1]) == ("alpha_101", "alpha_001")
        ]
        self.assertEqual(len(full_rebuilds), 1)
        self.assertEqual(set(result.leaderboard["factor"]), {"alpha_101", "alpha_001"})

    def test_failure_is_isolated_and_success_builds_leaderboard(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = Alpha101BatchRunner(
                _sample_panels(with_cap=False),
                factors=("alpha_101", "alpha_056"),
                output_dir=temp_dir,
                config=Alpha101BatchConfig(windows=(1,), groups=5),
                data_signature="data-v1",
                implementation_signature="code-v1",
            ).run()

            statuses = result.status.set_index("factor")["status"].to_dict()
            self.assertEqual(statuses["alpha_101"], "success")
            self.assertEqual(statuses["alpha_056"], "failed")
            self.assertEqual(result.failed_factors, ("alpha_056",))
            self.assertTrue((Path(temp_dir) / "alpha_101" / "summary.csv").exists())
            self.assertTrue((Path(temp_dir) / "logs" / "alpha_056.log").exists())
            self.assertEqual(result.leaderboard["factor"].unique().tolist(), ["alpha_101"])
            self.assertTrue((result.status["factor_status"] == "active").all())
            report_dir = Path(temp_dir) / "alpha_101"
            for filename in (
                "tradable_long_short.csv",
                "stat_long_short.csv",
                "top_n_return.csv",
                "top_n_summary.csv",
                "neutralized_ic.csv",
                "annual_performance.csv",
                "sample_performance.csv",
                "universe_filter.csv",
                "filter_status.csv",
            ):
                self.assertTrue((report_dir / filename).exists(), filename)
            stat = pd.read_csv(report_dir / "long_short.csv")
            tradable = pd.read_csv(report_dir / "tradable_long_short.csv")
            self.assertIn("stat_cum_nav", stat.columns)
            self.assertNotIn("tradable_cum_nav", stat.columns)
            self.assertIn("tradable_cum_nav", tradable.columns)
            self.assertNotIn("stat_cum_nav", tradable.columns)
            self.assertIn("oos_tradable_period_return", result.leaderboard.columns)
            self.assertIn("factor_status", result.leaderboard.columns)
            self.assertIn("top_1_mean_return", result.leaderboard.columns)
            self.assertIn("top_5_mean_return", result.leaderboard.columns)
            self.assertIn("top_10_mean_return", result.leaderboard.columns)
            samples = pd.read_csv(report_dir / "sample_performance.csv")
            self.assertEqual(set(samples["sample"]), {"in_sample", "out_of_sample"})
            summary = pd.read_csv(report_dir / "summary.csv")
            lag = summary.loc[summary["metric"].eq("factor_lag_days"), "value"].iloc[0]
            self.assertEqual(float(lag), 1.0)

    def test_matching_completed_factor_is_resumed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            kwargs = {
                "panels": _sample_panels(),
                "factors": ("alpha_101",),
                "output_dir": temp_dir,
                "config": Alpha101BatchConfig(windows=(1,), groups=5),
                "data_signature": "data-v1",
                "implementation_signature": "code-v1",
            }
            first = Alpha101BatchRunner(**kwargs).run()
            second = Alpha101BatchRunner(**kwargs).run()

            self.assertEqual(first.status.loc[0, "status"], "success")
            self.assertEqual(second.status.loc[0, "status"], "skipped")
            self.assertIn("matching completed report", second.status.loc[0, "message"])

    def test_leaderboard_excludes_reports_from_another_fingerprint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            Alpha101BatchRunner(
                _sample_panels(),
                factors=("alpha_101",),
                output_dir=temp_dir,
                config=Alpha101BatchConfig(windows=(1,), groups=5),
                data_signature="data-v1",
                implementation_signature="code-v1",
            ).run()

            leaderboard = build_leaderboard(
                temp_dir,
                ("alpha_101",),
                fingerprint="different-fingerprint",
            )
            self.assertTrue(leaderboard.empty)

    def test_watch_factors_are_ranked_after_active_factors(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = Alpha101BatchRunner(
                _sample_panels(),
                factors=("alpha_101", "alpha_001"),
                output_dir=temp_dir,
                config=Alpha101BatchConfig(windows=(1,), groups=5),
                data_signature="data-v1",
                implementation_signature="code-v1",
                factor_statuses={"alpha_101": "watch", "alpha_001": "active"},
            ).run()

            self.assertEqual(
                result.leaderboard[["factor", "factor_status"]].values.tolist(),
                [["alpha_001", "active"], ["alpha_101", "watch"]],
            )
            manifest = json.loads(
                (Path(temp_dir) / "batch_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest["factor_statuses"],
                {"alpha_101": "watch", "alpha_001": "active"},
            )


class TopLevelAlpha101BatchImportTests(unittest.TestCase):
    def test_alpha101_batch_imports_from_reports_package(self):
        import reports.alpha101_batch as alpha101_batch

        self.assertTrue(hasattr(alpha101_batch, "Alpha101BatchRunner"))


if __name__ == "__main__":
    unittest.main()
