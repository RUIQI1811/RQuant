import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from factors.scoring import (
    assign_decision,
    example_usage,
    parse_factor_metrics,
    score_all_factors,
    score_one_factor,
    update_factor_config,
)
from factors.catalog import load_factor_catalog


def _summary_frame(factor_name: str = "alpha_001") -> pd.DataFrame:
    rows = [
        {"section": "meta", "metric": "factor_name", "value": factor_name},
        {"section": "coverage", "metric": "avg_coverage", "value": 1.0},
        {"section": "coverage", "metric": "min_coverage", "value": 1.0},
    ]
    rank_ic = {1: 0.02, 5: 0.04, 10: 0.06, 20: 0.07}
    for window in (1, 5, 10, 20):
        values = {
            ("ic", "rank_ic_mean"): rank_ic[window],
            ("ic", "rank_icir"): 0.8,
            ("ic", "rank_ic_win_rate"): 0.75,
            ("neutralized_ic", "neutralized_rank_ic_mean"): 0.05,
            ("neutralized_ic", "neutralized_rank_icir"): 0.8,
            ("group_return", "top_bottom_return"): 0.012,
            ("group_return", "monotonic"): 1,
            ("stat_long_short", "annualized_return"): 0.15,
            ("stat_long_short", "max_drawdown"): 0.1,
            ("stat_long_short", "sharpe"): 1.0,
            ("stat_long_short", "stat_cum_nav"): 2.0,
            ("tradable_long_short", "annualized_return"): 0.15,
            ("tradable_long_short", "max_drawdown"): 0.0,
            ("tradable_long_short", "sharpe"): 1.5,
            ("tradable_long_short", "tradable_cum_nav"): 2.0,
        }
        for (section, metric), value in values.items():
            rows.append(
                {
                    "section": section,
                    "metric": metric,
                    "value": value,
                    "window": window,
                }
            )
    return pd.DataFrame(rows, columns=["section", "metric", "value", "window"])


class FactorScoringTest(unittest.TestCase):
    def test_perfect_factor_reaches_real_100_point_scale(self):
        result = score_one_factor(parse_factor_metrics(_summary_frame()))

        self.assertEqual(result["signal_score"], 35.0)
        self.assertEqual(result["tradability_score"], 35.0)
        self.assertEqual(result["robustness_score"], 20.0)
        self.assertEqual(result["penalty"], 0.0)
        self.assertEqual(result["final_score"], 100.0)
        self.assertEqual(result["decision"], "active")
        self.assertEqual(result["useful_horizons"], "10d,20d")

    def test_primary_scores_use_70_percent_20d_and_30_percent_10d(self):
        metrics = parse_factor_metrics(_summary_frame())
        for name in (
            "rank_ic_mean",
            "rank_icir",
            "neutralized_rank_ic_mean",
        ):
            metrics["windows"][10][name] = 0.0
        metrics["windows"][10]["rank_ic_win_rate"] = 0.5

        result = score_one_factor(metrics)

        self.assertEqual(result["signal_score"], 24.5)

    def test_monotonicity_loses_points_but_is_not_penalized_twice(self):
        metrics = parse_factor_metrics(_summary_frame())
        for window in (1, 5, 10, 20):
            metrics["windows"][window]["monotonic"] = 0.0

        result = score_one_factor(metrics)

        self.assertEqual(result["penalty"], 0.0)
        self.assertEqual(result["robustness_score"], 16.0)

    def test_collapse_ratio_is_only_used_for_positive_stat_sharpe(self):
        metrics = parse_factor_metrics(_summary_frame())
        metrics["windows"][20]["stat_sharpe"] = -1.0

        result = score_one_factor(metrics)

        self.assertEqual(result["penalty"], 0.0)
        self.assertNotIn("collapse_ratio", result["reason"])

    def test_coverage_and_severe_collapse_penalties_are_auditable(self):
        metrics = parse_factor_metrics(_summary_frame())
        metrics["coverage"]["avg_coverage"] = 0.4
        metrics["coverage"]["min_coverage"] = 0.4
        metrics["windows"][20]["stat_sharpe"] = 2.0
        metrics["windows"][20]["tradable_sharpe"] = 0.4

        result = score_one_factor(metrics)

        self.assertEqual(result["penalty"], -19.0)
        self.assertIn("tradable_collapse", result["reason"])

    def test_only_both_primary_negative_sharpes_force_disabled(self):
        metrics = parse_factor_metrics(_summary_frame())
        metrics["windows"][10]["tradable_sharpe"] = -0.1
        metrics["windows"][20]["tradable_sharpe"] = 0.5
        metrics["windows"][20]["tradable_annualized_return"] = 0.1
        self.assertEqual(assign_decision(90.0, metrics), "active")

        metrics["windows"][20]["tradable_sharpe"] = -0.1
        self.assertEqual(assign_decision(90.0, metrics), "disabled")

    def test_20d_hard_caps_and_missing_critical_metrics(self):
        metrics = parse_factor_metrics(_summary_frame())
        metrics["windows"][20]["tradable_sharpe"] = 0.2
        self.assertEqual(assign_decision(90.0, metrics), "component_only")

        metrics = parse_factor_metrics(_summary_frame())
        metrics["coverage"]["avg_coverage"] = 0.4
        self.assertEqual(assign_decision(90.0, metrics), "low_priority_watch")

        metrics = parse_factor_metrics(_summary_frame())
        metrics["windows"][20]["tradable_sharpe"] = np.nan
        self.assertEqual(assign_decision(90.0, metrics), "disabled")

    def test_score_all_factors_accepts_concatenated_summaries(self):
        first = _summary_frame("alpha_001").assign(factor_name="alpha_001")
        second = _summary_frame("alpha_002").assign(factor_name="alpha_002")

        result = score_all_factors(pd.concat([first, second], ignore_index=True))

        self.assertEqual(result["factor_name"].tolist(), ["alpha_001", "alpha_002"])
        self.assertEqual(result["final_score"].tolist(), [100.0, 100.0])

    def test_useful_horizons_only_consider_10d_and_20d(self):
        metrics = parse_factor_metrics(_summary_frame())
        metrics["windows"][10]["tradable_sharpe"] = 0.29

        result = score_one_factor(metrics)

        self.assertEqual(result["useful_horizons"], "20d")

    def test_update_factor_config_uses_scores_and_disables_unscored(self):
        scores = pd.DataFrame(
            [
                {
                    "factor_name": "alpha_001",
                    "final_score": 80.0,
                    "decision": "active",
                    "useful_horizons": "10d,20d",
                },
                {
                    "factor_name": "alpha_002",
                    "final_score": 50.0,
                    "decision": "component_only",
                    "useful_horizons": "20d",
                },
            ]
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "factors.yaml"
            path.write_text("default_status: active\nfactors: {}\n", encoding="utf-8")

            update_factor_config(scores, path, score_source="scores.csv")

            catalog = load_factor_catalog(path)
            self.assertEqual(catalog.status_for("alpha_001"), "active")
            self.assertEqual(catalog.status_for("alpha_002"), "watch")
            self.assertEqual(catalog.status_for("alpha_003"), "disabled")
            content = path.read_text(encoding="utf-8")
            self.assertIn("useful_horizons:", content)
            self.assertIn("- 20d", content)
            self.assertNotIn("factor_scoring:\n  factors:", content)

    def test_example_usage_only_writes_csv_when_requested(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report_dir = root / "alpha_001"
            report_dir.mkdir()
            _summary_frame().to_csv(report_dir / "summary.csv", index=False)

            result = example_usage(root)

            self.assertEqual(result.loc[0, "factor_name"], "alpha_001")
            self.assertFalse((root / "factor_scores.csv").exists())

            example_usage(root, root / "explicit_scores.csv")
            self.assertTrue((root / "explicit_scores.csv").exists())


if __name__ == "__main__":
    unittest.main()
