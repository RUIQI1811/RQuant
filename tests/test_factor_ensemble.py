import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from factors.alpha101 import Alpha101Panels
from factors.ensemble import (
    RankEnsembleConfig,
    build_alpha101_rank_ensemble_frame,
    rank_factor_ensemble,
    write_rank_ensemble_reports,
)


class RankFactorEnsembleTest(unittest.TestCase):
    def test_weighted_percentiles_respect_explicit_lower_is_better_direction(self):
        rows = []
        for symbol, factor_a, factor_b in zip(
            [1, 2, 3, 4],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
        ):
            rows.extend(
                [
                    {
                        "date": "2026-06-23",
                        "symbol": symbol,
                        "factor": "factor_a",
                        "factor_value": factor_a,
                    },
                    {
                        "date": "2026-06-23",
                        "symbol": symbol,
                        "factor": "factor_b",
                        "factor_value": factor_b,
                    },
                ]
            )
        config = RankEnsembleConfig(
            factors=("factor_a", "factor_b"),
            weights=(0.75, 0.25),
            ascending_factors=("factor_b",),
            top_n=2,
            min_universe=2,
        )

        result = rank_factor_ensemble(pd.DataFrame(rows), config=config)

        self.assertEqual(result.selections["symbol"].tolist(), ["000004", "000003"])
        self.assertEqual(result.selections["rank_position"].tolist(), [1, 2])
        self.assertAlmostEqual(result.selections.loc[0, "ensemble_score"], 0.8125)
        self.assertAlmostEqual(result.selections.loc[1, "ensemble_score"], 0.6875)
        self.assertEqual(result.signals["weight"].to_list(), [0.5, 0.5])
        metadata = result.signals.item(0, "metadata")
        self.assertEqual(metadata["factors"], ["factor_a", "factor_b"])
        self.assertEqual(metadata["ascending_factors"], ["factor_b"])

    def test_missing_factor_uses_available_weight_only_after_coverage_gate(self):
        frame = pd.DataFrame(
            [
                {"date": "2026-06-23", "symbol": 1, "factor": "a", "factor_value": 1.0},
                {"date": "2026-06-23", "symbol": 1, "factor": "b", "factor_value": 1.0},
                {"date": "2026-06-23", "symbol": 1, "factor": "c", "factor_value": 1.0},
                {"date": "2026-06-23", "symbol": 2, "factor": "a", "factor_value": 2.0},
                {"date": "2026-06-23", "symbol": 2, "factor": "b", "factor_value": 2.0},
                {"date": "2026-06-23", "symbol": 3, "factor": "a", "factor_value": 3.0},
                {"date": "2026-06-23", "symbol": 3, "factor": "b", "factor_value": 3.0},
                {"date": "2026-06-23", "symbol": 3, "factor": "c", "factor_value": 3.0},
            ]
        )
        config = RankEnsembleConfig(
            factors=("a", "b", "c"),
            weights=(0.5, 0.3, 0.2),
            min_factor_coverage=0.8,
            top_n=3,
            min_universe=2,
        )

        result = rank_factor_ensemble(frame, config=config)

        by_symbol = result.selections.set_index("symbol")
        self.assertAlmostEqual(by_symbol.at["000002", "factor_coverage"], 0.8)
        self.assertEqual(by_symbol.at["000002", "available_factor_count"], 2)
        self.assertAlmostEqual(by_symbol.at["000002", "ensemble_score"], 2 / 3)

        strict = rank_factor_ensemble(
            frame,
            config=RankEnsembleConfig(
                factors=("a", "b", "c"),
                weights=(0.5, 0.3, 0.2),
                min_factor_coverage=1.0,
                top_n=3,
                min_universe=2,
            ),
        )
        self.assertNotIn("000002", strict.selections["symbol"].tolist())

    def test_factor_values_and_eligibility_are_lagged_before_combination(self):
        dates = pd.date_range("2026-01-01", periods=3, freq="B")
        symbols = ["000001", "000002"]
        close = pd.DataFrame([[10, 20], [11, 21], [12, 22]], index=dates, columns=symbols)
        panels = Alpha101Panels(
            open=close,
            close=close,
            high=close,
            low=close,
            volume=close * 100,
            vwap=close,
            returns=close.pct_change(fill_method=None),
            turnover_value=close * 1000,
        )
        factor_a = pd.DataFrame([[1, 2], [3, 4], [5, 6]], index=dates, columns=symbols)
        factor_b = factor_a * 10
        config = RankEnsembleConfig(
            factors=("alpha_013", "alpha_040"),
            factor_lag_days=1,
            min_listing_days=0,
            min_universe=2,
        )

        with patch(
            "factors.ensemble.Alpha101.calculate",
            side_effect=lambda name: factor_a if name == "alpha_013" else factor_b,
        ):
            frame, status = build_alpha101_rank_ensemble_frame(
                panels,
                config=config,
                dates=[dates[-1]],
            )

        values = frame.pivot(index="symbol", columns="factor", values="factor_value")
        self.assertEqual(values.loc["000001", "alpha_013"], 3.0)
        self.assertEqual(values.loc["000002", "alpha_040"], 40.0)
        self.assertEqual(frame.groupby("symbol")["reference_close"].first().tolist(), [11.0, 21.0])
        self.assertEqual(status.set_index("filter").at["is_st", "status"], "missing_input")

    def test_reports_preserve_component_audit_fields_and_json_metadata(self):
        frame = pd.DataFrame(
            [
                {"date": "2026-06-23", "symbol": 1, "factor": "a", "factor_value": 1.0},
                {"date": "2026-06-23", "symbol": 1, "factor": "b", "factor_value": 2.0},
                {"date": "2026-06-23", "symbol": 2, "factor": "a", "factor_value": 2.0},
                {"date": "2026-06-23", "symbol": 2, "factor": "b", "factor_value": 1.0},
            ]
        )
        config = RankEnsembleConfig(factors=("a", "b"), top_n=1, min_universe=2)
        result = rank_factor_ensemble(frame, config=config)

        with tempfile.TemporaryDirectory() as temp_dir:
            output = write_rank_ensemble_reports(result, temp_dir, config=config)
            payload = json.loads((output / "signals.json").read_text(encoding="utf-8"))
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            selections = pd.read_csv(output / "selections.csv", dtype={"symbol": str})

        self.assertIsInstance(payload["signals"][0]["metadata"]["factor_values"], dict)
        self.assertIn("factor_percentiles", selections.columns)
        self.assertEqual(manifest["strategy"], "rank_ensemble")
        self.assertEqual(manifest["settings"]["factors"], ["a", "b"])

    def test_config_rejects_ambiguous_factor_contracts(self):
        invalid_configs = [
            {"factors": ("a", "a")},
            {"factors": ("a", "b"), "weights": (1.0,)},
            {"factors": ("a",), "weights": (-1.0,)},
            {"factors": ("a",), "ascending_factors": ("missing",)},
            {"factors": ("a",), "min_factor_coverage": 0.0},
        ]
        for kwargs in invalid_configs:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                RankEnsembleConfig(**kwargs)


if __name__ == "__main__":
    unittest.main()
