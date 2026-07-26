from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import yaml

from domain.artifacts import WorkflowResult
from reports.factor_research_pipeline import (
    load_factor_research_config,
    run_factor_research_pipeline,
)
from scripts.quant_cli import build_parser


class FactorResearchPipelineTests(unittest.TestCase):
    def _write_config(self, root: Path, *, ml_enabled: bool = True) -> Path:
        path = root / "run_all.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "version": 1,
                    "factor_library": {
                        "family": "alpha101",
                        "data": "data/raw",
                        "metadata": "config/stocklist.csv",
                        "factor_config": "config/factors.yaml",
                        "factors": ["alpha_001", "alpha_002"],
                    },
                    "evaluation": {
                        "windows": [1, 5, 10, 20],
                        "commission_rate": 0.0003,
                        "slippage_rate": 0.0005,
                        "stamp_tax_rate": 0.0005,
                        "market_cap_groups": 3,
                    },
                    "correlation": {
                        "high_correlation_threshold": 0.8,
                        "priority_window": 20,
                    },
                    "machine_learning": {
                        "enabled": ml_enabled,
                        "models": ["ridge", "elasticnet"],
                        "target_window": 20,
                        "label_mode": "next_open",
                        "window_mode": "calendar-years",
                        "train_years": 3,
                        "test_years": 1,
                        "run_backtests": True,
                    },
                    "execution": {
                        "output": str(root / "output"),
                        "require_classification": True,
                    },
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        return path

    def test_pipeline_composes_batch_dedup_and_long_only_3y_1y_ml(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = load_factor_research_config(self._write_config(root))
            calls: list[str] = []

            def batch_runner(args):
                calls.append("batch")
                self.assertTrue(args.no_progress)
                destination = Path(args.output)
                destination.mkdir(parents=True)
                pd.DataFrame(
                    {
                        "factor": ["alpha_001", "alpha_002"],
                        "status": ["success", "success"],
                    }
                ).to_csv(destination / "batch_status.csv", index=False)
                pd.DataFrame(
                    {
                        "factor": ["alpha_001", "alpha_002"],
                        "window": [20, 20],
                        "factor_category": ["price_behavior", "price_volume"],
                        "preferred_net_sharpe": [0.8, 0.3],
                        "preferred_profitable_after_cost": [True, True],
                    }
                ).to_csv(destination / "leaderboard.csv", index=False)
                pd.DataFrame({"factor": ["alpha_001", "alpha_002"]}).to_csv(
                    destination / "long_only_profitability.csv", index=False
                )
                pd.DataFrame({"factor": ["alpha_001"]}).to_csv(
                    destination / "profitable_long_only.csv", index=False
                )
                return 0

            def correlation_runner(args):
                calls.append("correlation")
                self.assertEqual(args.factors, ["alpha_001", "alpha_002"])
                self.assertEqual(args.priority_window, 20)
                self.assertEqual(args.priority_score_col, "preferred_net_sharpe")
                self.assertEqual(
                    args.eligibility_col, "preferred_profitable_after_cost"
                )
                destination = Path(args.output)
                destination.mkdir(parents=True)
                pd.DataFrame(
                    {
                        "factor_a": ["alpha_001"],
                        "factor_b": ["alpha_002"],
                        "spearman": [0.9],
                        "high_correlation": [True],
                    }
                ).to_csv(destination / "correlation_pairs.csv", index=False)
                pd.DataFrame({"factor": ["alpha_001"]}).to_csv(
                    destination / "deduplicated_factors.csv", index=False
                )
                pd.DataFrame({"factor": ["alpha_001"]}).to_csv(
                    destination / "ml_candidate_factors.csv", index=False
                )
                return 0

            def ml_runner(args):
                calls.append("ml")
                self.assertEqual(args.window_mode, "calendar-years")
                self.assertEqual(args.train_years, 3)
                self.assertEqual(args.test_years, 1)
                self.assertTrue(args.run_backtests)
                self.assertEqual(args.factors, [])
                destination = Path(args.output)
                destination.mkdir(parents=True)
                pd.DataFrame(
                    {"model": ["ridge", "elasticnet"], "net_sharpe": [0.4, 0.2]}
                ).to_csv(destination / "leaderboard.csv", index=False)
                pd.DataFrame({"model": ["ridge"]}).to_csv(
                    destination / "profitable_models.csv", index=False
                )
                return WorkflowResult.from_mapping(
                    {
                        "result": {"models": args.models},
                        "leaderboard_path": destination / "leaderboard.csv",
                    }
                )

            result = run_factor_research_pipeline(
                config,
                batch_runner=batch_runner,
                correlation_runner=correlation_runner,
                ml_runner=ml_runner,
            )

            self.assertEqual(calls, ["batch", "correlation", "ml"])
            self.assertEqual(result.result.ml_status, "complete")
            manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "complete")
            self.assertTrue(manifest["long_only_only"])
            summary = json.loads(Path(result["summary_path"]).read_text(encoding="utf-8"))
            self.assertFalse(summary["research_contract"]["short_positions"])
            self.assertEqual(summary["research_contract"]["top_side"], "buy_high_factor_values")
            self.assertEqual(summary["research_contract"]["bottom_side"], "buy_low_factor_values")
            self.assertEqual(summary["profitable_model_count"], 1)

    def test_no_after_cost_candidates_skips_ml_with_auditable_reason(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = load_factor_research_config(self._write_config(root))

            def batch_runner(args):
                destination = Path(args.output)
                destination.mkdir(parents=True)
                pd.DataFrame(
                    {"factor": ["alpha_001", "alpha_002"], "status": ["success", "success"]}
                ).to_csv(destination / "batch_status.csv", index=False)
                pd.DataFrame(
                    {
                        "factor": ["alpha_001", "alpha_002"],
                        "window": [20, 20],
                        "preferred_net_sharpe": [-0.1, -0.2],
                        "preferred_profitable_after_cost": [False, False],
                    }
                ).to_csv(destination / "leaderboard.csv", index=False)
                pd.DataFrame(columns=["factor"]).to_csv(
                    destination / "long_only_profitability.csv", index=False
                )
                pd.DataFrame(columns=["factor"]).to_csv(
                    destination / "profitable_long_only.csv", index=False
                )
                return 0

            def correlation_runner(args):
                destination = Path(args.output)
                destination.mkdir(parents=True)
                pd.DataFrame(columns=["factor_a", "factor_b"]).to_csv(
                    destination / "correlation_pairs.csv", index=False
                )
                pd.DataFrame({"factor": ["alpha_001", "alpha_002"]}).to_csv(
                    destination / "deduplicated_factors.csv", index=False
                )
                pd.DataFrame(columns=["factor"]).to_csv(
                    destination / "ml_candidate_factors.csv", index=False
                )
                return 0

            def unexpected_ml(_args):
                self.fail("ML must not run without after-cost profitable candidates")

            result = run_factor_research_pipeline(
                config,
                batch_runner=batch_runner,
                correlation_runner=correlation_runner,
                ml_runner=unexpected_ml,
            )
            self.assertEqual(result.result.ml_status, "skipped_no_profitable_candidates")
            self.assertTrue(result.warnings)

    def test_external_library_without_categories_writes_template_and_fails_preflight(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            factor_file = root / "external.csv"
            pd.DataFrame(
                {
                    "date": ["2024-01-02", "2024-01-02"],
                    "symbol": ["000001", "000002"],
                    "factor_a": [1.0, 2.0],
                    "factor_b": [2.0, 1.0],
                }
            ).to_csv(factor_file, index=False)
            config_path = root / "external.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "version": 1,
                        "factor_library": {
                            "family": "external",
                            "factor_file": str(factor_file),
                            "factors": ["all"],
                        },
                        "evaluation": {"windows": [20]},
                        "correlation": {"priority_window": 20},
                        "machine_learning": {
                            "enabled": False,
                            "target_window": 20,
                        },
                        "execution": {"output": str(root / "output")},
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            config = load_factor_research_config(config_path)
            with self.assertRaisesRegex(ValueError, "missing research categories"):
                run_factor_research_pipeline(config)
            template = root / "output/factor_classification_template.yaml"
            self.assertTrue(template.exists())
            payload = yaml.safe_load(template.read_text(encoding="utf-8"))
            self.assertEqual(payload["categories"]["factor_a"], "unclassified")

    def test_cli_registers_factor_run_all(self):
        args = build_parser(prog="rquant").parse_args(["factor-run-all"])
        self.assertEqual(args.command, "factor-run-all")
        self.assertEqual(args.config, "config/factor_research_run_all.yaml")

    def test_rejects_non_calendar_or_non_long_only_ml_contract(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = self._write_config(root)
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            payload["machine_learning"]["train_years"] = 2
            path.write_text(yaml.safe_dump(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "train_years must be 3"):
                load_factor_research_config(path)


if __name__ == "__main__":
    unittest.main()
