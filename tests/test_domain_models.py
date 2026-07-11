import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from backtest.portfolio import (
    FeeModel,
    PortfolioSettings,
    run_staggered_cohort_portfolio_from_prepared,
    write_portfolio_backtest_outputs,
)
from domain import (
    BacktestResult,
    BacktestSummary,
    Candidate,
    DateRange,
    EquityPoint,
    OrderResult,
    Signal,
    SignalBook,
    Symbol,
    Trade,
    WorkflowResult,
)


def _tradeable_frame(values):
    dates = pd.bdate_range("2026-01-02", periods=len(values))
    return pd.DataFrame(
        {
            "open": values,
            "close": values,
            "high": values,
            "low": values,
            "volume": [1000] * len(values),
            "is_tradeable": [True] * len(values),
            "is_limit_up": [False] * len(values),
            "is_limit_down": [False] * len(values),
        },
        index=dates,
    )


class DomainValueObjectTest(unittest.TestCase):
    def test_symbol_and_date_range_are_canonical(self):
        self.assertEqual(str(Symbol(1)), "000001")
        self.assertTrue(DateRange("2026-01-01", "2026-01-31").contains("2026-01-15"))
        with self.assertRaises(ValueError):
            Symbol("bad")
        with self.assertRaises(ValueError):
            DateRange("2026-02-01", "2026-01-01")

    def test_candidate_composes_the_canonical_signal(self):
        candidate = Candidate(
            code="1",
            date="2026-01-02",
            strategy="brick",
            close=10.0,
            turnover_n=1_000_000,
            brick_growth=2.5,
        )
        signal = candidate.to_signal(weight=0.1)

        self.assertEqual(candidate.symbol, "000001")
        self.assertEqual(signal.symbol, candidate.symbol)
        self.assertEqual(signal.source, candidate.source)
        self.assertEqual(signal.score, 2.5)
        self.assertEqual(signal.metadata["turnover_n"], 1_000_000)

    def test_signal_book_keeps_context_behind_legacy_code_view(self):
        signal = Signal(
            date="2026-01-02",
            symbol="1",
            source="model_ridge",
            score=0.9,
            metadata={"window": 3},
        )
        book = SignalBook([signal])

        self.assertEqual(book, {pd.Timestamp("2026-01-02"): ["000001"]})
        restored = book.signals_for("2026-01-02")[0]
        self.assertEqual(restored.source, "model_ridge")
        self.assertEqual(restored.metadata, {"window": 3})


class DomainExecutionTest(unittest.TestCase):
    def test_backtest_result_converts_legacy_dicts_to_typed_records(self):
        result = BacktestResult(
            initial_cash=100.0,
            final_cash=110.0,
            total_return=0.1,
            trades=[
                {
                    "signal_date": "2026-01-02",
                    "strategy": "factor",
                    "buy_mode": "next_open",
                    "hold_days": 1,
                    "code": "1",
                    "return": 0.1,
                }
            ],
            orders=[
                {
                    "date": "2026-01-05",
                    "code": "1",
                    "side": "buy",
                    "status": "filled",
                    "reason": "signal",
                    "signal_date": "2026-01-02",
                }
            ],
            positions=[],
            equity_curve=[{"date": "2026-01-05", "cash": 110, "total_return": 0.1}],
            summary={"strategy": "factor", "total_return": 0.1},
        )

        self.assertIsInstance(result.trades[0], Trade)
        self.assertIsInstance(result.orders[0], OrderResult)
        self.assertIsInstance(result.equity_curve[0], EquityPoint)
        self.assertIsInstance(result.summary, BacktestSummary)
        self.assertEqual(result.orders[0]["code"], "000001")
        self.assertEqual(result.trades[0]["return"], 0.1)

    def test_signal_context_reaches_orders_trades_and_positions(self):
        frame = _tradeable_frame([10.0, 11.0, 12.0, 13.0])
        signal = Signal(
            date=str(frame.index[0].date()),
            symbol="1",
            source="model_ridge",
            score=0.93,
            weight=1.0,
            metadata={"window_id": 7},
        )
        settings = PortfolioSettings(
            initial_cash=100_000,
            strategy="model_ridge",
            buy_mode="next_open",
            hold_days=1,
            fee_model=FeeModel(0.0, 0.0, 0.0),
            max_positions=1,
            position_pct=1.0,
        )

        result = run_staggered_cohort_portfolio_from_prepared(
            prepared={"000001": frame},
            picks_by_date=SignalBook([signal]),
            settings=settings,
            cohort_count=1,
        )

        buy = next(order for order in result.orders if order.side == "buy")
        trade = result.trades[0]
        self.assertEqual(buy.source, "model_ridge")
        self.assertEqual(buy.score, 0.93)
        self.assertEqual(buy.metadata["window_id"], 7)
        self.assertEqual(trade.source, "model_ridge")
        self.assertEqual(trade.score, 0.93)
        self.assertEqual(result.fills[0].source, "model_ridge")
        self.assertEqual(result.fills[0].symbol, "000001")

    def test_workflow_result_exposes_typed_artifacts_with_legacy_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "summary.json"
            artifact.write_text('{"ok": true}\n', encoding="utf-8")
            workflow = WorkflowResult.from_mapping(
                {"result": {"ok": True}, "summary_path": artifact}
            )

            self.assertEqual(workflow["summary_path"], artifact)
            self.assertTrue(workflow.artifacts["summary_path"].exists)
            self.assertEqual(len(workflow.artifacts["summary_path"].sha256), 64)

    def test_persisted_order_json_keeps_legacy_wire_shape(self):
        result = BacktestResult(
            initial_cash=100,
            final_cash=100,
            total_return=0,
            trades=[],
            orders=[
                OrderResult(
                    date="2026-01-05",
                    symbol="1",
                    side="buy",
                    status="blocked",
                    reason="limit_up",
                    signal_date="2026-01-02",
                    source="factor_alpha",
                    score=0.9,
                )
            ],
            positions=[],
            equity_curve=[],
            summary={"strategy": "factor", "buy_mode": "next_open", "hold_days": 1},
        )
        with tempfile.TemporaryDirectory() as tmp:
            outputs = write_portfolio_backtest_outputs(result, tmp)
            payload = json.loads(outputs["orders_json_path"].read_text(encoding="utf-8"))

        self.assertEqual(
            set(payload["orders"][0]),
            {
                "date",
                "code",
                "side",
                "status",
                "reason",
                "signal_date",
                "price",
                "shares",
                "cash_delta",
            },
        )


if __name__ == "__main__":
    unittest.main()
