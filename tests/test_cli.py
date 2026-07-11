import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

from scripts import quant_cli as cli


class CliTest(unittest.TestCase):
    def test_doctor_parser_exposes_read_only_diagnostics(self):
        args = cli.build_parser().parse_args(
            ["doctor", "--data", "custom/raw", "--output", "doctor.json", "--deep"]
        )

        self.assertEqual(args.command, "doctor")
        self.assertEqual(args.data, "custom/raw")
        self.assertEqual(args.output, "doctor.json")
        self.assertTrue(args.deep)
        self.assertEqual(args.max_data_age_days, 7)

    def test_doctor_dispatch_fails_only_for_required_errors(self):
        args = cli.build_parser().parse_args(["doctor"])
        warning_report = {
            "status": "warning",
            "ok": True,
            "runtime": {"python_executable": "/python"},
            "dependencies": {
                "status": "warning",
                "errors": [],
                "warnings": ["optional dependency unavailable: torch"],
            },
            "configs": {"status": "ok", "errors": [], "warnings": []},
            "secrets": {
                "status": "warning",
                "errors": [],
                "warnings": ["GEMINI_API_KEY not configured"],
            },
            "workflow_artifacts": {
                "status": "ok",
                "errors": [],
                "warnings": [],
            },
            "market_data": {
                "status": "ok",
                "inspected_file_count": 1,
                "file_count": 1,
                "errors": [],
                "warnings": [],
            },
            "summary": {"error_count": 0, "warning_count": 2},
        }
        with patch("reports.system_doctor.run_system_doctor", return_value=warning_report):
            with redirect_stdout(StringIO()):
                cli.cmd_doctor(args)

        error_report = {**warning_report, "status": "error", "ok": False}
        error_report["summary"] = {"error_count": 1, "warning_count": 0}
        with patch("reports.system_doctor.run_system_doctor", return_value=error_report):
            with redirect_stdout(StringIO()):
                with self.assertRaisesRegex(SystemExit, "1"):
                    cli.cmd_doctor(args)

    def test_research_entrypoints_are_registered_in_top_level_cli(self):
        parser = cli.build_parser()
        fetch_args = parser.parse_args(
            [
                "fetch-data",
                "--start",
                "2026-01-01",
                "--max-symbols",
                "5",
                "--max-requests-per-minute",
                "175",
                "--manifest",
                "fetch.json",
                "--resume",
            ]
        )
        factor_args = parser.parse_args(["factor-test", "--factor", "custom_002"])
        batch_args = parser.parse_args(["factor-batch", "--family", "gtja191"])

        self.assertEqual(fetch_args.command, "fetch-data")
        self.assertEqual(fetch_args.max_symbols, 5)
        self.assertEqual(fetch_args.max_requests_per_minute, 175)
        self.assertEqual(fetch_args.manifest, "fetch.json")
        self.assertTrue(fetch_args.resume)
        self.assertEqual(factor_args.factor, "custom_002")
        self.assertEqual(batch_args.family, "gtja191")
        self.assertEqual(batch_args.factor_config, None)

    def test_factor_test_dispatches_to_existing_runner(self):
        args = cli.build_parser().parse_args(["factor-test", "--factor", "custom_002"])
        with patch("scripts.test_factor.run_from_args", return_value="factor_report/custom_002") as runner:
            cli.cmd_factor_test(args)
        runner.assert_called_once_with(args)

    def test_fetch_command_returns_nonzero_for_partial_run(self):
        args = cli.build_parser().parse_args(["fetch-data"])
        partial = {
            "start": "20260101",
            "end": "20260102",
            "symbol_count": 2,
            "output_dir": "data/raw",
            "outcomes": {"created": 1},
            "manifest_path": "data/raw/_fetch_manifest.json",
            "failed_codes": ["000002"],
            "ok": False,
        }
        with patch("market.fetch_kline.run_from_args", return_value=partial):
            with redirect_stdout(StringIO()):
                with self.assertRaisesRegex(SystemExit, "2"):
                    cli.cmd_fetch_data(args)

    def test_factor_batch_dispatches_to_lifecycle_aware_runner(self):
        args = cli.build_parser().parse_args(["factor-batch", "--family", "alpha101"])
        with patch("scripts.test_factor_batch.run_from_args", return_value=0) as runner:
            cli.cmd_factor_batch(args)
        runner.assert_called_once_with(args)
    def test_make_ml_dataset_parser_exposes_lagged_factor_contract(self):
        args = cli.build_parser().parse_args(
            [
                "make-ml-dataset",
                "--factors",
                "alpha_101",
                "custom_002",
                "--target-windows",
                "1",
                "20",
                "--output",
                "data/ml/dataset",
            ]
        )

        self.assertEqual(args.command, "make-ml-dataset")
        self.assertEqual(args.factors, ["alpha_101", "custom_002"])
        self.assertEqual(args.target_windows, [1, 20])
        self.assertEqual(args.factor_lag_days, 1)
        self.assertEqual(args.label_mode, "next_open")

    def test_signal_backtest_parser_accepts_unified_signal_file(self):
        args = cli.build_parser().parse_args(
            [
                "signal-backtest",
                "--signals",
                "data/ml/ridge_20d/signals.csv",
                "--source",
                "model_ridge",
                "--max-positions",
                "10",
            ]
        )

        self.assertEqual(args.command, "signal-backtest")
        self.assertEqual(args.signals, "data/ml/ridge_20d/signals.csv")
        self.assertEqual(args.source, "model_ridge")
        self.assertEqual(args.hold_days, 20)
        self.assertEqual(args.output, "data/portfolio_backtest_signals")

    def test_signal_backtest_dispatches_to_unified_signal_runner(self):
        args = cli.build_parser().parse_args(
            [
                "signal-backtest",
                "--signals",
                "signals.csv",
                "--source",
                "model_ridge",
                "--hold-days",
                "5",
            ]
        )
        fake_result = type(
            "FakeResult",
            (),
            {
                "summary": {
                    "signal_source_filter": "model_ridge",
                    "start_date": "2025-01-01",
                    "end_date": "2026-01-01",
                    "signal_count": 10,
                    "hold_days": 5,
                    "total_return": 0.1,
                    "max_drawdown": 0.05,
                    "sharpe_ratio": 1.0,
                }
            },
        )()
        payload = {
            "result": fake_result,
            "summary_path": "summary.json",
            "trades_path": "trades.csv",
            "equity_curve_html_path": "equity.html",
        }

        with patch.object(cli, "run_signal_portfolio_backtest", return_value=payload) as runner:
            with redirect_stdout(StringIO()):
                cli.cmd_signal_backtest(args)

        self.assertEqual(runner.call_args.kwargs["signals_path"], "signals.csv")
        self.assertEqual(runner.call_args.kwargs["source"], "model_ridge")
        self.assertEqual(runner.call_args.kwargs["hold_days"], 5)

    def test_train_model_parser_exposes_purge_and_model_contract(self):
        args = cli.build_parser().parse_args(
            [
                "train-model",
                "--features",
                "features.csv",
                "--labels",
                "labels.csv",
                "--output",
                "data/ml/ridge",
                "--feature-cols",
                "alpha_040",
                "turnover_20d",
                "--target-col",
                "forward_return_20d",
                "--model",
                "ridge",
            ]
        )

        self.assertEqual(args.command, "train-model")
        self.assertEqual(args.feature_cols, ["alpha_040", "turnover_20d"])
        self.assertEqual(args.target_col, "forward_return_20d")
        self.assertIsNone(args.purge_days)
        self.assertEqual(args.train_size, 504)
        self.assertEqual(args.test_size, 21)
        self.assertEqual(args.lightgbm_n_jobs, 1)

    def test_factor_ensemble_parser_requires_explicit_auditable_components(self):
        args = cli.build_parser().parse_args(
            [
                "factor-ensemble-backtest",
                "--factors",
                "alpha_013",
                "alpha_040",
                "--weights",
                "0.25",
                "0.75",
                "--ascending-factors",
                "alpha_013",
                "--min-factor-coverage",
                "0.8",
            ]
        )

        self.assertEqual(args.factors, ["alpha_013", "alpha_040"])
        self.assertEqual(args.weights, [0.25, 0.75])
        self.assertEqual(args.ascending_factors, ["alpha_013"])
        self.assertEqual(args.min_factor_coverage, 0.8)
        self.assertEqual(args.hold_days, 20)
        self.assertEqual(args.output, "data/portfolio_backtest_factor_ensemble")

    def test_factor_ensemble_backtest_dispatches_to_ensemble_runner(self):
        args = cli.build_parser().parse_args(
            [
                "factor-ensemble-backtest",
                "--factors",
                "alpha_013",
                "alpha_040",
                "--weights",
                "1",
                "3",
                "--start",
                "2025-01-01",
                "--end",
                "2026-06-23",
            ]
        )
        fake_result = type(
            "FakeFactorResult",
            (),
            {
                "summary": {
                    "start_date": "2025-01-01",
                    "end_date": "2026-06-23",
                    "signal_count": 10,
                    "hold_days": 20,
                    "total_return": 0.1,
                    "max_drawdown": 0.05,
                    "sharpe_ratio": 1.0,
                    "realized_trade_count": 4,
                }
            },
        )()
        payload = {
            "result": fake_result,
            "summary_path": "summary.json",
            "trades_path": "trades.csv",
            "equity_curve_html_path": "equity.html",
        }

        with patch.object(
            cli,
            "run_rank_ensemble_portfolio_backtest",
            return_value=payload,
        ) as runner:
            with redirect_stdout(StringIO()):
                cli.cmd_factor_ensemble_backtest(args)

        kwargs = runner.call_args.kwargs
        self.assertEqual(kwargs["start_date"], "2025-01-01")
        self.assertEqual(kwargs["end_date"], "2026-06-23")
        self.assertEqual(kwargs["selection_config"].factors, ("alpha_013", "alpha_040"))
        self.assertEqual(kwargs["selection_config"].normalized_weights, (0.25, 0.75))

    def test_signal_returns_base_reuse_is_explicit_opt_in(self):
        parser = cli.build_parser()

        default_args = parser.parse_args(["signal-returns"])
        enabled_args = parser.parse_args(["signal-returns", "--reuse-base-preparation"])

        self.assertFalse(default_args.reuse_base_preparation)
        self.assertTrue(enabled_args.reuse_base_preparation)

    def test_cmd_signal_returns_forwards_base_reuse_flag(self):
        args = cli.build_parser().parse_args(
            ["signal-returns", "--reuse-base-preparation"]
        )
        fake_result = {
            "summary": {"total_signals": 0, "metrics": {}},
            "csv_path": "signals.csv",
            "summary_path": "summary.json",
            "summary_csv_path": "summary.csv",
        }

        with patch.object(cli, "run_signal_returns", return_value=fake_result) as runner:
            with redirect_stdout(StringIO()):
                cli.cmd_signal_returns(args)

        self.assertTrue(runner.call_args.kwargs["reuse_base_preparation"])

    def test_factor_backtest_parser_uses_twenty_day_holding_period(self):
        args = cli.build_parser().parse_args(["factor-backtest"])

        self.assertEqual(args.command, "factor-backtest")
        self.assertEqual(args.filter_factor, "alpha_077")
        self.assertEqual(args.rank_factor, "alpha_040")
        self.assertEqual(args.filter_top_quantile, 0.8)
        self.assertEqual(args.initial_cash, 10000000.0)
        self.assertEqual(args.top_n, 500)
        self.assertEqual(args.rank_start, 1)
        self.assertIsNone(args.rank_end)
        self.assertEqual(args.hold_days, 20)
        self.assertFalse(args.no_progress)
        self.assertEqual(args.output, "data/portfolio_backtest_alpha077_alpha040")

    def test_factor_backtest_dispatches_to_factor_runner(self):
        args = cli.build_parser().parse_args(
            [
                "factor-backtest",
                "--start",
                "2025-01-01",
                "--end",
                "2026-06-23",
                "--hold-days",
                "20",
            ]
        )
        fake_result = type(
            "FakeFactorResult",
            (),
            {
                "summary": {
                    "start_date": "2025-01-01",
                    "end_date": "2026-06-23",
                    "signal_count": 100,
                    "hold_days": 20,
                    "total_return": 0.1,
                    "max_drawdown": 0.05,
                    "sharpe_ratio": 1.0,
                    "realized_trade_count": 20,
                }
            },
        )()
        payload = {
            "result": fake_result,
            "summary_path": "summary.json",
            "trades_path": "trades.csv",
            "equity_curve_html_path": "equity.html",
        }

        with patch.object(
            cli,
            "run_filter_rank_portfolio_backtest",
            return_value=payload,
        ) as runner:
            with redirect_stdout(StringIO()):
                cli.cmd_factor_backtest(args)

        kwargs = runner.call_args.kwargs
        self.assertEqual(kwargs["start_date"], "2025-01-01")
        self.assertEqual(kwargs["end_date"], "2026-06-23")
        self.assertEqual(kwargs["hold_days"], 20)
        self.assertTrue(kwargs["show_progress"])
        self.assertEqual(kwargs["selection_config"].filter_factor, "alpha_077")
        self.assertEqual(kwargs["selection_config"].rank_factor, "alpha_040")

    def test_factor_select_parser_defaults_to_alpha77_filter_alpha40_rank(self):
        args = cli.build_parser().parse_args(["factor-select"])

        self.assertEqual(args.command, "factor-select")
        self.assertEqual(args.filter_factor, "alpha_077")
        self.assertEqual(args.rank_factor, "alpha_040")
        self.assertEqual(args.filter_top_quantile, 0.5)
        self.assertEqual(args.top_n, 10)
        self.assertEqual(args.rank_start, 1)
        self.assertIsNone(args.rank_end)
        self.assertEqual(args.factor_lag_days, 1)
        self.assertEqual(args.output, "data/factor_signals/alpha077_alpha040")

    def test_factor_backtest_parser_accepts_inclusive_rank_interval(self):
        args = cli.build_parser().parse_args(
            ["factor-backtest", "--rank-start", "200", "--rank-end", "500"]
        )

        self.assertEqual(args.rank_start, 200)
        self.assertEqual(args.rank_end, 500)

    def test_factor_backtest_can_disable_progress(self):
        args = cli.build_parser().parse_args(["factor-backtest", "--no-progress"])

        self.assertTrue(args.no_progress)

    def test_portfolio_backtest_parser_accepts_public_options(self):
        parser = cli.build_parser()

        args = parser.parse_args(
            [
                "portfolio-backtest",
                "--start",
                "2024-10-20",
                "--end",
                "2026-06-05",
                "--strategy",
                "brick",
                "--buy-mode",
                "next_open",
                "--hold-days",
                "3",
                "--initial-cash",
                "200000",
                "--commission-wan",
                "1.2",
                "--max-positions",
                "8",
                "--position-pct",
                "0.125",
                "--lot-size",
                "100",
                "--output",
                "data/portfolio_backtest_manual",
            ]
        )

        self.assertEqual(args.command, "portfolio-backtest")
        self.assertEqual(args.start, "2024-10-20")
        self.assertEqual(args.end, "2026-06-05")
        self.assertEqual(args.strategy, "brick")
        self.assertEqual(args.buy_mode, "next_open")
        self.assertEqual(args.hold_days, 3)
        self.assertEqual(args.initial_cash, 200000.0)
        self.assertEqual(args.commission_wan, 1.2)
        self.assertEqual(args.max_positions, 8)
        self.assertEqual(args.position_pct, 0.125)
        self.assertEqual(args.lot_size, 100)
        self.assertEqual(args.output, "data/portfolio_backtest_manual")

    def test_cmd_portfolio_backtest_dispatches_to_runner(self):
        parser = cli.build_parser()
        args = parser.parse_args(
            [
                "portfolio-backtest",
                "--config",
                "config/rules_preselect.yaml",
                "--data",
                "data/raw",
                "--start",
                "2024-10-20",
                "--end",
                "2026-06-05",
                "--strategy",
                "brick",
                "--buy-mode",
                "signal_close",
                "--hold-days",
                "1",
                "--initial-cash",
                "100000",
                "--commission-wan",
                "0.8",
                "--output",
                "data/portfolio_backtest_manual",
            ]
        )
        fake_result = type(
            "FakeResult",
            (),
            {
                "summary": {
                    "strategy": "brick",
                    "buy_mode": "signal_close",
                    "hold_days": 1,
                    "initial_cash": 100000.0,
                    "final_cash": 101000.0,
                    "total_return": 0.01,
                    "max_drawdown": 0.02,
                    "annualized_volatility": None,
                    "sharpe_ratio": None,
                    "trade_count": 2,
                }
            },
        )()

        with patch.object(
            cli,
            "run_portfolio_backtest",
            return_value={
                "result": fake_result,
                "trades_path": "trades.csv",
                "orders_path": "daily_trade_plan.csv",
                "orders_json_path": "daily_trade_plan.json",
                "positions_path": "open_positions.csv",
                "summary_path": "summary.json",
                "equity_curve_path": "equity.csv",
                "equity_curve_html_path": "equity.html",
            },
        ) as runner:
            with redirect_stdout(StringIO()):
                cli.cmd_portfolio_backtest(args)

        runner.assert_called_once_with(
            config_path="config/rules_preselect.yaml",
            data_dir="data/raw",
            start_date="2024-10-20",
            end_date="2026-06-05",
            output_dir="data/portfolio_backtest_manual",
            initial_cash=100000.0,
            strategy="brick",
            buy_mode="signal_close",
            hold_days=1,
            commission_wan=0.8,
            max_positions=10,
            position_pct=0.1,
            lot_size=100,
        )

    def test_research_report_parser_accepts_public_options(self):
        parser = cli.build_parser()

        args = parser.parse_args(
            [
                "research-report",
                "--signal-dir",
                "data/backtest_manual",
                "--portfolio-dir",
                "data/portfolio_backtest_manual",
                "--candidates",
                "data/candidates/candidates_latest.json",
                "--review",
                "data/review/2026-06-23/suggestion.json",
                "--output",
                "data/reports",
                "--allow-inconsistent",
            ]
        )

        self.assertEqual(args.command, "research-report")
        self.assertEqual(args.signal_dir, "data/backtest_manual")
        self.assertEqual(args.portfolio_dir, "data/portfolio_backtest_manual")
        self.assertEqual(args.candidates, "data/candidates/candidates_latest.json")
        self.assertEqual(args.review, "data/review/2026-06-23/suggestion.json")
        self.assertEqual(args.output, "data/reports")
        self.assertTrue(args.allow_inconsistent)


class TopLevelCliImportTests(unittest.TestCase):
    def test_quant_cli_imports(self):
        import scripts.quant_cli as quant_cli

        parser = quant_cli.build_parser()
        command_names = {action.dest for action in parser._actions}
        self.assertIn("command", command_names)


if __name__ == "__main__":
    unittest.main()
