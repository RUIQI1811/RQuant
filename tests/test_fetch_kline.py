import os
import json
import logging
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from market import fetch_kline
from domain.market import FetchResult


def _kline_frame(dates: list[str], closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "open": closes,
            "close": closes,
            "high": closes,
            "low": closes,
            "volume": [1000.0] * len(dates),
        }
    )


class FetchKlineRangeOverwriteTest(unittest.TestCase):
    def test_request_rate_limiter_evenly_spaces_request_starts(self):
        now = [100.0]
        sleeps: list[float] = []

        def sleep(seconds: float) -> None:
            sleeps.append(seconds)
            now[0] += seconds

        limiter = fetch_kline.RequestRateLimiter(
            60,
            clock=lambda: now[0],
            sleeper=sleep,
        )

        limiter.wait()
        limiter.wait()
        limiter.wait()

        self.assertEqual(sleeps, [1.0, 1.0])

    def test_request_rate_limiter_rejects_non_positive_limit(self):
        with self.assertRaisesRegex(ValueError, "must be positive"):
            fetch_kline.RequestRateLimiter(0)

    def test_get_kline_waits_before_provider_call(self):
        limiter = Mock()
        with patch.object(fetch_kline.ts, "pro_bar", return_value=pd.DataFrame()) as api:
            result = fetch_kline._get_kline_tushare(
                "000001",
                "20260101",
                "20260102",
                rate_limiter=limiter,
            )

        self.assertTrue(result.empty)
        limiter.wait.assert_called_once_with()
        api.assert_called_once()

    def test_run_fetch_applies_cli_overrides_without_real_network(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "fetch.yaml"
            output_dir = root / "raw"
            log_path = root / "fetch.log"
            config_path.write_text(
                "start: '20260101'\n"
                "end: '20260131'\n"
                "stocklist: config/stocklist.csv\n"
                f"out: '{output_dir}'\n"
                "workers: 8\n",
                encoding="utf-8",
            )

            with (
                patch.dict(os.environ, {"TUSHARE_TOKEN": "test-token"}),
                patch.object(fetch_kline.ts, "pro_api", return_value=object()) as api,
                patch.object(
                    fetch_kline,
                    "load_codes_from_stocklist",
                    return_value=["000001", "000002", "000003"],
                ),
                patch.object(
                    fetch_kline,
                    "fetch_one",
                    side_effect=["created", "updated"],
                ) as fetch_one,
            ):
                result = fetch_kline.run_fetch(
                    config_path=config_path,
                    log_path=log_path,
                    start="2026-02-01",
                    end="2026-02-05",
                    out_dir=output_dir,
                    workers=1,
                    max_requests_per_minute=120,
                    max_symbols=2,
                )
                manifest = json.loads(
                    Path(result["manifest_path"]).read_text(encoding="utf-8")
                )

                self.assertEqual(manifest["status"], "complete")
                self.assertEqual(manifest["max_requests_per_minute"], 120)
                self.assertEqual(manifest["completed"], {"000001": "created", "000002": "updated"})

        api.assert_called_once_with("test-token")
        self.assertEqual(result["start"], "20260201")
        self.assertEqual(result["end"], "20260205")
        self.assertEqual(result["symbol_count"], 2)
        self.assertEqual(result["max_requests_per_minute"], 120)
        self.assertEqual(result["outcomes"], {"created": 1, "updated": 1})
        self.assertTrue(result["ok"])
        self.assertIsInstance(result, FetchResult)
        self.assertEqual(fetch_one.call_count, 2)
        self.assertIsInstance(
            fetch_one.call_args_list[0].args[4],
            fetch_kline.RequestRateLimiter,
        )

    def test_requested_range_overwrites_only_that_part(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            _kline_frame(
                ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"],
                [10.0, 11.0, 12.0, 13.0],
            ).to_csv(
                output / "000001.csv", index=False
            )
            replacement = _kline_frame(["2026-01-02", "2026-01-03"], [11.0, 12.0])

            with patch.object(
                fetch_kline, "_get_kline_tushare", return_value=replacement
            ) as api:
                outcome = fetch_kline.fetch_one(
                    "000001", "20260102", "20260103", output
                )

            self.assertEqual(outcome, "overwritten")
            api.assert_called_once_with("000001", "20260102", "20260103")
            saved = pd.read_csv(output / "000001.csv")
            self.assertEqual(
                saved["date"].tolist(),
                ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"],
            )

    def test_existing_end_date_is_requested_and_overwritten(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            _kline_frame(["2026-01-01", "2026-01-02"], [10.0, 11.0]).to_csv(
                output / "000001.csv", index=False
            )
            requested_day = _kline_frame(["2026-01-02"], [11.0])

            with patch.object(
                fetch_kline, "_get_kline_tushare", return_value=requested_day
            ) as api:
                outcome = fetch_kline.fetch_one(
                    "000001", "20260102", "20260102", output
                )

            self.assertEqual(outcome, "overwritten")
            api.assert_called_once_with("000001", "20260102", "20260102")
            saved = pd.read_csv(output / "000001.csv")
            self.assertEqual(saved["date"].tolist(), ["2026-01-01", "2026-01-02"])

    def test_empty_update_preserves_existing_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            csv_path = output / "000001.csv"
            _kline_frame(["2026-01-01", "2026-01-02"], [10.0, 11.0]).to_csv(
                csv_path, index=False
            )
            original = csv_path.read_bytes()

            with patch.object(
                fetch_kline, "_get_kline_tushare", return_value=pd.DataFrame()
            ):
                outcome = fetch_kline.fetch_one(
                    "000001", "20260103", "20260103", output
                )

            self.assertEqual(outcome, "no_new_data")
            self.assertEqual(csv_path.read_bytes(), original)

    def test_changed_qfq_overlap_triggers_full_refresh(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            _kline_frame(["2026-01-01", "2026-01-02"], [10.0, 11.0]).to_csv(
                output / "000001.csv", index=False
            )
            incremental = _kline_frame(
                ["2026-01-02", "2026-01-03"], [10.0, 11.0]
            )
            refreshed = _kline_frame(
                ["2026-01-01", "2026-01-02", "2026-01-03"],
                [9.0, 10.0, 11.0],
            )

            with patch.object(
                fetch_kline,
                "_get_kline_tushare",
                side_effect=[incremental, refreshed],
            ) as api:
                outcome = fetch_kline.fetch_one(
                    "000001", "20260102", "20260103", output
                )

            self.assertEqual(outcome, "refreshed")
            self.assertEqual(
                [call.args for call in api.call_args_list],
                [
                    ("000001", "20260102", "20260103"),
                    ("000001", "20260101", "20260103"),
                ],
            )
            saved = pd.read_csv(output / "000001.csv")
            self.assertEqual(saved["close"].tolist(), [9.0, 10.0, 11.0])

    def test_exhausted_symbol_raises_with_last_error_and_no_fake_success(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(
                    fetch_kline,
                    "_get_kline_tushare",
                    side_effect=RuntimeError("provider unavailable"),
                ) as api,
                patch.object(fetch_kline.time, "sleep"),
            ):
                with self.assertRaises(fetch_kline.FetchExhaustedError) as raised:
                    fetch_kline.fetch_one(
                        "000001",
                        "20260101",
                        "20260102",
                        Path(temp_dir),
                    )

        self.assertEqual(api.call_count, 3)
        self.assertEqual(raised.exception.code, "000001")
        self.assertEqual(raised.exception.attempts, 3)
        self.assertIn("provider unavailable", str(raised.exception.last_error))

    def test_partial_run_checkpoints_failures_and_resume_retries_only_failed_codes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "fetch.yaml"
            output_dir = root / "raw"
            log_path = root / "fetch.log"
            config_path.write_text(
                "start: '20260101'\n"
                "end: '20260102'\n"
                "stocklist: config/stocklist.csv\n"
                f"out: '{output_dir}'\n"
                "workers: 1\n",
                encoding="utf-8",
            )
            failure = fetch_kline.FetchExhaustedError(
                "000002",
                3,
                RuntimeError("temporary provider failure"),
            )
            with (
                patch.dict(os.environ, {"TUSHARE_TOKEN": "test-token"}),
                patch.object(fetch_kline.ts, "pro_api", return_value=object()),
                patch.object(
                    fetch_kline,
                    "load_codes_from_stocklist",
                    return_value=["000001", "000002"],
                ),
                patch.object(
                    fetch_kline,
                    "fetch_one",
                    side_effect=["created", failure],
                ),
            ):
                first = fetch_kline.run_fetch(
                    config_path=config_path,
                    log_path=log_path,
                    out_dir=output_dir,
                    workers=1,
                )

            first_manifest = json.loads(
                Path(first["manifest_path"]).read_text(encoding="utf-8")
            )
            self.assertFalse(first["ok"])
            self.assertEqual(first["status"], "partial")
            self.assertEqual(first["failed_codes"], ["000002"])
            self.assertEqual(first_manifest["failures"]["000002"]["attempts"], 3)
            self.assertEqual(first_manifest["completed"], {"000001": "created"})

            with (
                patch.dict(os.environ, {"TUSHARE_TOKEN": "test-token"}),
                patch.object(fetch_kline.ts, "pro_api", return_value=object()),
                patch.object(
                    fetch_kline,
                    "load_codes_from_stocklist",
                    return_value=["000001", "000002"],
                ),
                patch.object(fetch_kline, "fetch_one", return_value="created") as retry,
            ):
                resumed = fetch_kline.run_fetch(
                    config_path=config_path,
                    log_path=log_path,
                    out_dir=output_dir,
                    workers=1,
                    resume=True,
                )

            self.assertTrue(resumed["ok"])
            self.assertEqual(resumed["status"], "complete")
            self.assertEqual(resumed["resumed_count"], 1)
            self.assertEqual(resumed["submitted_count"], 1)
            self.assertEqual(retry.call_args.args[0], "000002")
            self.assertEqual(resumed["outcomes"], {"created": 2})

    def test_logging_handler_closes_when_setup_fails_after_logging_starts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "fetch.yaml"
            log_path = root / "fetch.log"
            config_path.write_text("start: '20260101'\nend: '20260102'\n", encoding="utf-8")
            with (
                patch.dict(os.environ, {}, clear=True),
                patch.object(fetch_kline, "_read_dotenv_value", return_value=None),
            ):
                with self.assertRaisesRegex(ValueError, "TUSHARE_TOKEN"):
                    fetch_kline.run_fetch(config_path=config_path, log_path=log_path)

        self.assertFalse(
            any(
                getattr(handler, "_rquant_fetch_file", False)
                for handler in logging.getLogger().handlers
            )
        )

    def test_fetch_run_restores_proxy_environment_session_and_console_handler(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "fetch.yaml"
            output_dir = root / "raw"
            log_path = root / "fetch.log"
            config_path.write_text(
                "start: '20260101'\nend: '20260102'\n"
                "stocklist: config/stocklist.csv\nworkers: 1\n",
                encoding="utf-8",
            )
            sentinel = object()
            previous_pro = fetch_kline.pro
            self.addCleanup(setattr, fetch_kline, "pro", previous_pro)
            fetch_kline.pro = sentinel
            root_logger = logging.getLogger()
            handlers_before = tuple(root_logger.handlers)
            with (
                patch.dict(
                    os.environ,
                    {"TUSHARE_TOKEN": "test-token", "NO_PROXY": "existing.example"},
                    clear=True,
                ),
                patch.object(fetch_kline.ts, "pro_api", return_value=object()),
                patch.object(
                    fetch_kline,
                    "load_codes_from_stocklist",
                    return_value=["000001"],
                ),
                patch.object(fetch_kline, "fetch_one", return_value="created"),
            ):
                fetch_kline.run_fetch(
                    config_path=config_path,
                    log_path=log_path,
                    out_dir=output_dir,
                    workers=1,
                )
                self.assertEqual(os.environ["NO_PROXY"], "existing.example")
                self.assertNotIn("no_proxy", os.environ)

            self.assertIs(fetch_kline.pro, sentinel)
            self.assertEqual(tuple(root_logger.handlers), handlers_before)

    def test_import_does_not_monkeypatch_pandas_fillna(self):
        self.assertTrue(pd.DataFrame.fillna.__module__.startswith("pandas."))
        self.assertTrue(pd.Series.fillna.__module__.startswith("pandas."))


class TopLevelFetchImportTests(unittest.TestCase):
    def test_fetch_kline_imports_from_top_level_package(self):
        import market.fetch_kline as fetch_kline

        self.assertTrue(hasattr(fetch_kline, "main"))


if __name__ == "__main__":
    unittest.main()
