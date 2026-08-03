import json
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from market.build_research_context import build_research_context
from market.fetch_data import fetch_tushare_research_data
from market.fetch_industry import (
    fetch_sw_industry_membership,
    normalize_industry_membership,
)
from market.fetch_trade_state import fetch_trade_state_context, normalize_trade_state


class _IndustrySession:
    def __init__(self) -> None:
        self.member_calls: list[tuple[str, str]] = []

    def index_classify(self, **kwargs):
        level = kwargs["level"]
        if level != "L3":
            code = "801010.SI" if level == "L1" else "801016.SI"
            return pd.DataFrame(
                {
                    "index_code": [code],
                    "industry_name": ["农林牧渔" if level == "L1" else "种植业"],
                    "parent_code": ["0" if level == "L1" else "801010.SI"],
                    "level": [level],
                    "industry_code": ["110000" if level == "L1" else "110100"],
                    "is_pub": ["1"],
                    "src": ["SW2021"],
                }
            )
        return pd.DataFrame(
            {
                "index_code": ["850111.SI", "850112.SI"],
                "industry_name": ["种子", "粮食种植"],
                "parent_code": ["801016.SI", "801016.SI"],
                "level": ["L3", "L3"],
                "industry_code": ["110101", "110102"],
                "is_pub": ["1", "1"],
                "src": ["SW2021", "SW2021"],
            }
        )

    def index_member_all(self, *, l3_code: str, is_new: str, fields: str):
        self.member_calls.append((l3_code, is_new))
        if is_new == "N":
            return pd.DataFrame(columns=fields.split(","))
        symbol = "000001.SZ" if l3_code == "850111.SI" else "600000.SH"
        return pd.DataFrame(
            {
                "l1_code": ["801010.SI"],
                "l1_name": ["农林牧渔"],
                "l2_code": ["801016.SI"],
                "l2_name": ["种植业"],
                "l3_code": [l3_code],
                "l3_name": ["种子" if symbol.startswith("000001") else "粮食种植"],
                "ts_code": [symbol],
                "name": ["测试股"],
                "in_date": ["20200101"],
                "out_date": [None],
                "is_new": ["Y"],
            }
        )


class _TradeStateSession:
    def trade_cal(self, **_kwargs):
        return pd.DataFrame({"cal_date": ["20260102"], "is_open": [1]})

    def stk_limit(self, **_kwargs):
        return pd.DataFrame(
            {
                "trade_date": ["20260102", "20260102"],
                "ts_code": ["000001.SZ", "600000.SH"],
                "pre_close": [10.0, 20.0],
                "up_limit": [11.0, 22.0],
                "down_limit": [9.0, 18.0],
            }
        )

    def suspend_d(self, **_kwargs):
        return pd.DataFrame(
            {
                "ts_code": ["600000.SH"],
                "trade_date": ["20260102"],
                "suspend_timing": [None],
                "suspend_type": ["S"],
            }
        )


class FetchDataSuiteTest(unittest.TestCase):
    def test_industry_fetch_partitions_below_provider_cap_and_resumes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "industry.csv"
            session = _IndustrySession()
            result = fetch_sw_industry_membership(
                output_file=output,
                max_requests_per_minute=0,
                session=session,
            )

            self.assertTrue(result["ok"])
            self.assertEqual(result["industry_count"], 2)
            self.assertEqual(len(session.member_calls), 4)
            saved = pd.read_csv(output, dtype={"symbol": str})
            self.assertEqual(saved["symbol"].tolist(), ["000001", "600000"])
            self.assertEqual(saved["sector"].unique().tolist(), ["农林牧渔"])
            classifications = pd.read_csv(
                Path(result["classification_file"])
            )
            self.assertEqual(set(classifications["level"]), {"L1", "L2", "L3"})

            resumed_session = _IndustrySession()
            resumed = fetch_sw_industry_membership(
                output_file=output,
                resume=True,
                max_requests_per_minute=0,
                session=resumed_session,
            )
            self.assertTrue(resumed["ok"])
            self.assertEqual(resumed["reused_industry_count"], 2)
            self.assertEqual(resumed_session.member_calls, [])

    def test_industry_membership_excludes_audited_non_equity_identifiers(self):
        frame = _IndustrySession().index_member_all(
            l3_code="850111.SI",
            is_new="Y",
            fields="",
        )
        legacy = frame.iloc[[0]].copy()
        legacy["ts_code"] = "T00018.SH"
        legacy["name"] = "历史退市证券"
        normalized = normalize_industry_membership(
            pd.concat([frame, legacy], ignore_index=True),
            expected_l3_code="850111.SI",
        )

        self.assertEqual(normalized["symbol"].tolist(), ["000001"])
        self.assertNotIn("T00018.SH", normalized["ts_code"].tolist())

    def test_trade_state_fetches_limit_prices_and_suspension_flags(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "trade_state"
            result = fetch_trade_state_context(
                start="20260102",
                end="20260102",
                output_dir=destination,
                max_requests_per_minute=0,
                session=_TradeStateSession(),
            )

            self.assertTrue(result["ok"])
            saved = pd.read_csv(
                destination / "2026/20260102.csv", dtype={"symbol": str}
            )
            first = saved.set_index("symbol")
            self.assertFalse(bool(first.loc["000001", "is_suspended"]))
            self.assertTrue(bool(first.loc["600000", "is_suspended"]))
            self.assertEqual(first.loc["000001", "up_limit"], 11.0)
            self.assertNotIn("is_st", saved.columns)

    def test_trade_state_marks_legitimate_missing_limits_without_failing_date(self):
        session = _TradeStateSession()
        normalized = normalize_trade_state(
            session.stk_limit(),
            session.suspend_d(),
            trade_date="20260102",
            expected_symbols={"000001", "600000", "688033"},
        ).set_index("symbol")

        self.assertFalse(bool(normalized.loc["688033", "has_price_limit"]))
        self.assertTrue(bool(normalized.loc["688033", "is_tradeable"]))
        self.assertTrue(pd.isna(normalized.loc["688033", "up_limit"]))
        self.assertTrue(bool(normalized.loc["000001", "has_price_limit"]))

    def test_trade_state_maps_historical_provider_code_to_stable_symbol(self):
        limits = _TradeStateSession().stk_limit().iloc[[0]].copy()
        limits["ts_code"] = "000043.SZ"
        limits["trade_date"] = "20180102"
        normalized = normalize_trade_state(
            limits,
            pd.DataFrame(columns=["ts_code", "trade_date", "suspend_timing", "suspend_type"]),
            trade_date="20180102",
            expected_symbols={"000043", "001914"},
            symbol_aliases=[
                {
                    "source": "000043",
                    "target": "001914",
                    "start": "19940928",
                    "end": "20191215",
                }
            ],
        )

        self.assertEqual(normalized["symbol"].tolist(), ["001914"])
        self.assertEqual(normalized.loc[0, "up_limit"], 11.0)
        self.assertTrue(bool(normalized.loc[0, "has_price_limit"]))

    def test_builder_creates_directly_consumable_point_in_time_context(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            daily = root / "daily"
            trade = root / "trade"
            output = root / "research"
            (daily / "2026").mkdir(parents=True)
            (trade / "2026").mkdir(parents=True)
            pd.DataFrame(
                {
                    "date": ["2026-01-02", "2026-01-02", "2026-01-02"],
                    "symbol": ["000001", "000002", "600000"],
                    "market_cap": [100.0, 150.0, 200.0],
                    "pb": [1.0, 1.5, 2.0],
                }
            ).to_csv(daily / "2026/20260102.csv", index=False)
            pd.DataFrame(
                {
                    "date": ["2026-01-02", "2026-01-02"],
                    "symbol": ["000001", "600000"],
                    "up_limit": [11.0, 22.0],
                    "down_limit": [9.0, 18.0],
                    "is_suspended": [False, True],
                    "is_tradeable": [True, False],
                }
            ).to_csv(trade / "2026/20260102.csv", index=False)
            industry = root / "industry.csv"
            pd.DataFrame(
                {
                    "symbol": ["000001", "000002", "600000"],
                    "sector": ["金融", "金融", "金融"],
                    "industry": ["银行", "银行", "银行"],
                    "subindustry": ["股份行", "城商行", "股份行"],
                    "l1_code": ["L1", "L1", "L1"],
                    "l2_code": ["L2", "L2", "L2"],
                    "l3_code": ["L3", "L3B", "L3"],
                    "in_date": ["2020-01-01", "2020-01-01", "2020-01-01"],
                    "out_date": [None, None, None],
                }
            ).to_csv(industry, index=False)
            for path in (daily / "_context_manifest.json", trade / "_context_manifest.json"):
                path.write_text(
                    json.dumps({"status": "complete", "completed_dates": ["20260102"]}),
                    encoding="utf-8",
                )
            industry.with_suffix(".csv.manifest.json").write_text(
                json.dumps({"status": "complete"}), encoding="utf-8"
            )

            with warnings.catch_warnings():
                warnings.simplefilter("error", FutureWarning)
                result = build_research_context(
                    daily_basic_dir=daily,
                    industry_file=industry,
                    trade_state_dir=trade,
                    output_dir=output,
                )

            self.assertTrue(result["ok"])
            saved = pd.read_csv(output / "2026/20260102.csv", dtype={"symbol": str})
            self.assertEqual(saved["industry"].unique().tolist(), ["银行"])
            self.assertIn("up_limit", saved.columns)
            self.assertNotIn("is_st", saved.columns)
            missing_state = saved.set_index("symbol").loc["000002"]
            self.assertFalse(bool(missing_state["has_price_limit"]))
            self.assertFalse(bool(missing_state["is_suspended"]))
            self.assertTrue(bool(missing_state["is_tradeable"]))
            manifest = json.loads(
                (output / "_context_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["status"], "complete")

            daily_manifest = daily / "_context_manifest.json"
            daily_payload = json.loads(daily_manifest.read_text(encoding="utf-8"))
            daily_payload["updated_at"] = "2099-01-01T00:00:00+00:00"
            daily_manifest.write_text(json.dumps(daily_payload), encoding="utf-8")
            resumed = build_research_context(
                daily_basic_dir=daily,
                industry_file=industry,
                trade_state_dir=trade,
                output_dir=output,
                resume=True,
            )
            self.assertEqual(resumed["reused_date_count"], 1)
            self.assertEqual(resumed["built_date_count"], 0)

    def test_suite_marks_dependency_partial_and_writes_overall_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = root / "fetch.yaml"
            config.write_text(
                "start: '20200101'\n"
                "end: '20260102'\n"
                f"out: '{root / 'raw'}'\n"
                "workers: 1\n"
                "max_requests_per_minute: 0\n"
                "tushare_2000:\n"
                "  context_start: '20260102'\n"
                "  paths:\n"
                f"    daily_basic: '{root / 'daily'}'\n"
                f"    benchmark: '{root / 'benchmark.csv'}'\n"
                f"    industry: '{root / 'industry.csv'}'\n"
                f"    trade_state: '{root / 'trade'}'\n"
                f"    research_context: '{root / 'research'}'\n"
                f"    suite_manifest: '{root / 'suite.json'}'\n",
                encoding="utf-8",
            )
            complete = {"ok": True}
            partial = {"ok": False, "failed_dates": ["20260102"]}
            with (
                patch("market.fetch_data.fetch_kline.run_fetch", return_value=complete),
                patch(
                    "market.fetch_data.fetch_context.fetch_daily_basic_context",
                    return_value=complete,
                ),
                patch(
                    "market.fetch_data.fetch_benchmark.fetch_benchmark_index",
                    return_value=complete,
                ),
                patch(
                    "market.fetch_data.fetch_industry.fetch_sw_industry_membership",
                    return_value=complete,
                ),
                patch(
                    "market.fetch_data.fetch_trade_state.fetch_trade_state_context",
                    return_value=partial,
                ),
                patch("market.fetch_data.build_research_context") as builder,
            ):
                result = fetch_tushare_research_data(config_path=config)

            self.assertFalse(result["ok"])
            self.assertEqual(
                result["failed_datasets"], ["trade_state", "research_context"]
            )
            builder.assert_not_called()
            manifest = json.loads((root / "suite.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "partial")
            self.assertNotIn("token", json.dumps(manifest).lower())


if __name__ == "__main__":
    unittest.main()
