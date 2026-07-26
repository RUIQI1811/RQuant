import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from signals.candidates import Candidate
from signals.schema import Signal, frame_to_signals, signals_to_frame
from signals.strategy_adapters import candidate_to_signal, candidates_to_signal_frame


class TopLevelSignalImportTests(unittest.TestCase):
    def test_signal_schema_imports_from_top_level_package(self):
        from signals.schema import Signal

        signal = Signal(
            date="2026-01-02",
            symbol="000001",
            signal_type="buy",
            source="unit_test",
            score=1.0,
            weight=0.1,
            metadata={"reason": "layout"},
        )

        self.assertEqual(signal.symbol, "000001")
        self.assertEqual(signal.signal_type, "buy")


class SignalSchemaTest(unittest.TestCase):
    def test_signal_round_trip_uses_stable_columns(self):
        signal = Signal(
            date="2026-06-23",
            symbol="1",
            source="factor_momentum_20d",
            score=0.5,
            weight=0.1,
            metadata={"factor_value": 0.5},
        )

        frame = signals_to_frame([signal])
        restored = frame_to_signals(frame)

        self.assertEqual(list(frame.columns), ["date", "symbol", "signal_type", "source", "score", "weight", "metadata"])
        self.assertEqual(frame.item(0, "symbol"), "000001")
        self.assertEqual(restored[0].symbol, "000001")
        self.assertEqual(restored[0].metadata["factor_value"], 0.5)

    def test_candidate_converts_to_strategy_signal(self):
        candidate = Candidate(
            code="002008",
            date="2026-06-23",
            strategy="brick",
            close=145.11,
            turnover_n=1000000.0,
            brick_growth=9.03,
        )

        signal = candidate_to_signal(candidate, default_weight=0.1)
        frame = candidates_to_signal_frame([candidate], default_weight=0.1)

        self.assertEqual(signal.source, "brick")
        self.assertEqual(signal.score, 9.03)
        self.assertEqual(signal.metadata["brick_growth"], 9.03)
        self.assertEqual(frame.item(0, "source"), "brick")


if __name__ == "__main__":
    unittest.main()
