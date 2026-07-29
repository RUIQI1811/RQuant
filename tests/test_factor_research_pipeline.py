from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from reports.factor_research_pipeline import (
    build_stage_commands,
    load_factor_research_config,
    run_factor_research_pipeline,
)
from scripts.quant_cli import build_parser


class FactorResearchPipelineTests(unittest.TestCase):
    def _write_config(
        self,
        root: Path,
        *,
        family: str = "alpha101",
        factor_file: str | None = None,
        ml_enabled: bool = True,
        execution: dict[str, object] | None = None,
    ) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        path = root / "run_all.yaml"
        library: dict[str, object] = {
            "family": family,
            "data": "data/raw",
            "metadata": "config/stocklist.csv",
            "factor_config": "config/factors.yaml",
            "factors": ["alpha_001", "alpha_002"] if family == "alpha101" else ["all"],
            "exclude": [],
            "context_file": "data/context/daily_basic",
        }
        if factor_file is not None:
            library["factor_file"] = factor_file
            library["factor_config"] = str(root / "external_factors.yaml")
        path.write_text(
            yaml.safe_dump(
                {
                    "version": 1,
                    "factor_library": library,
                    "evaluation": {
                        "windows": [1, 5, 10, 20],
                        "groups": 10,
                        "commission_rate": 0.0003,
                        "slippage_rate": 0.0005,
                        "stamp_tax_rate": 0.0005,
                        "market_cap_groups": 3,
                        "fail_fast": True,
                    },
                    "correlation": {
                        "high_correlation_threshold": 0.8,
                        "priority_score_col": "preferred_gross_sharpe",
                        "priority_window": 20,
                    },
                    "machine_learning": {
                        "enabled": ml_enabled,
                        "models": ["ridge", "elasticnet"],
                        "target_window": 20,
                        "label_mode": "next_open",
                        "feature_transform": "rank",
                        "target_transform": "rank",
                        "window_mode": "calendar-years",
                        "train_years": 3,
                        "test_years": 1,
                        "signal_top_n": 10,
                        "run_backtests": True,
                    },
                    "execution": execution
                    or {"force": False, "require_classification": True},
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        return path

    def test_builds_three_governed_public_commands_with_native_stage_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = load_factor_research_config(self._write_config(root))
            commands = build_stage_commands(
                config,
                python_executable="/test/python",
                runs_dir=root / "runs",
                run_id_factory=lambda stage: f"child-{stage}",
            )

            self.assertEqual(
                [command.stage for command in commands],
                ["factor-batch", "factor-correlation", "fit-multifactor"],
            )
            self.assertEqual(config.batch_output, Path("factor_report/alpha101_batch"))
            self.assertEqual(
                config.correlation_output,
                Path("factor_report/alpha101_correlation"),
            )
            self.assertEqual(config.ml_output, Path("data/ml/alpha101_multifactor"))
            for command in commands:
                self.assertEqual(command.argv[:3], ("/test/python", "-m", "rquant"))
                self.assertEqual(
                    command.argv[command.argv.index("--run-id") + 1],
                    f"child-{command.stage}",
                )
                self.assertEqual(
                    command.argv[command.argv.index("--runs-dir") + 1],
                    str(root / "runs"),
                )
                stage_index = command.argv.index(command.stage)
                parsed = build_parser(prog="rquant").parse_args(
                    list(command.argv[stage_index:])
                )
                self.assertEqual(parsed.command, command.stage)

            batch = commands[0].argv
            self.assertIn("--fail-fast", batch)
            self.assertEqual(batch[batch.index("--profile") + 1], "core")
            correlation = commands[1].argv
            self.assertNotIn("--eligibility-col", correlation)
            self.assertEqual(
                correlation[correlation.index("--priority-score-col") + 1],
                "preferred_gross_sharpe",
            )
            self.assertEqual(
                correlation[correlation.index("--priority-file") + 1],
                "factor_report/alpha101_batch/leaderboard.csv",
            )
            ml = commands[2].argv
            self.assertEqual(
                ml[ml.index("--factor-selection-file") + 1],
                "factor_report/alpha101_correlation/deduplicated_factors.csv",
            )
            self.assertNotIn("ml_candidate_factors.csv", ml)
            self.assertIn("--run-backtests", ml)
            self.assertEqual(ml[ml.index("--window-mode") + 1], "calendar-years")
            self.assertEqual(ml[ml.index("--train-years") + 1], "3")
            self.assertEqual(ml[ml.index("--test-years") + 1], "1")

    def test_runs_each_command_once_and_returns_only_parent_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = load_factor_research_config(self._write_config(root))
            calls: list[tuple[str, ...]] = []

            def runner(argv):
                calls.append(tuple(argv))
                return 0

            result = run_factor_research_pipeline(
                config,
                command_runner=runner,
                python_executable="/test/python",
                runs_dir=root / "runs",
                run_id_factory=lambda stage: f"child-{stage}",
            )

            self.assertEqual(result.exit_code, 0)
            self.assertEqual(len(calls), 3)
            self.assertEqual(
                [stage["stage"] for stage in result.summary["stages"]],
                ["factor-batch", "factor-correlation", "fit-multifactor"],
            )
            self.assertEqual(
                result.summary["orchestration"], "governed_public_cli_commands"
            )
            self.assertFalse((root / "factor_report/factor_run_all").exists())
            self.assertIn("batch_run_manifest", result.outputs)
            self.assertIn("correlation_run_manifest", result.outputs)
            self.assertIn("ml_run_manifest", result.outputs)

    def test_parent_metadata_points_to_three_auditable_child_manifests(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runs = root / "runs"
            config = load_factor_research_config(self._write_config(root))

            def runner(argv):
                run_id = argv[argv.index("--run-id") + 1]
                runs_dir = Path(argv[argv.index("--runs-dir") + 1])
                manifest = runs_dir / run_id / "run.json"
                manifest.parent.mkdir(parents=True)
                manifest.write_text("{}\n", encoding="utf-8")
                return 0

            result = run_factor_research_pipeline(
                config,
                command_runner=runner,
                runs_dir=runs,
                run_id_factory=lambda stage: f"child-{stage}",
            )

            child_manifests = [
                Path(stage["run_manifest"])
                for stage in result.summary["stages"]
            ]
            self.assertEqual(len(child_manifests), 3)
            self.assertTrue(all(path.exists() for path in child_manifests))
            self.assertEqual(
                {path.parent.name for path in child_manifests},
                {
                    "child-factor-batch",
                    "child-factor-correlation",
                    "child-fit-multifactor",
                },
            )

    def test_failure_stops_later_commands(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = load_factor_research_config(self._write_config(root))
            calls: list[str] = []

            def runner(argv):
                stage = next(
                    value
                    for value in argv
                    if value in {"factor-batch", "factor-correlation", "fit-multifactor"}
                )
                calls.append(stage)
                return 7 if stage == "factor-correlation" else 0

            result = run_factor_research_pipeline(
                config,
                command_runner=runner,
                run_id_factory=lambda stage: f"child-{stage}",
            )

            self.assertEqual(calls, ["factor-batch", "factor-correlation"])
            self.assertEqual(result.status, "failed")
            self.assertEqual(result.exit_code, 7)
            self.assertEqual(result.summary["failed_stage"], "factor-correlation")
            self.assertNotIn("ml_output", result.outputs)

    def test_keyboard_interrupt_propagates_to_parent_tracker(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = load_factor_research_config(self._write_config(root))

            def interrupted(_argv):
                raise KeyboardInterrupt

            with self.assertRaises(KeyboardInterrupt):
                run_factor_research_pipeline(config, command_runner=interrupted)

    def test_skip_ml_and_force_propagation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = load_factor_research_config(
                self._write_config(root),
                force=True,
                skip_ml=True,
            )
            commands = build_stage_commands(
                config,
                run_id_factory=lambda stage: f"child-{stage}",
            )
            self.assertEqual(
                [command.stage for command in commands],
                ["factor-batch", "factor-correlation"],
            )
            self.assertIn("--force", commands[0].argv)
            self.assertNotIn("--force", commands[1].argv)

    def test_force_is_passed_to_batch_and_ml_only(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = load_factor_research_config(self._write_config(root), force=True)
            commands = build_stage_commands(
                config,
                run_id_factory=lambda stage: f"child-{stage}",
            )
            self.assertIn("--force", commands[0].argv)
            self.assertNotIn("--force", commands[1].argv)
            self.assertIn("--force", commands[2].argv)

    def test_library_specific_output_resolution(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            gtja = load_factor_research_config(
                self._write_config(root / "gtja", family="gtja191")
            )
            self.assertEqual(gtja.batch_output, Path("factor_report/gtja191_batch"))
            self.assertEqual(
                gtja.correlation_output,
                Path("factor_report/gtja191_correlation"),
            )
            self.assertEqual(gtja.ml_output, Path("data/ml/gtja191_multifactor"))

            factor_file = root / "My Factor Library.csv"
            external_path = self._write_config(
                root / "external",
                family="external",
                factor_file=str(factor_file),
            )
            external = load_factor_research_config(external_path)
            self.assertEqual(
                external.batch_output,
                Path("factor_report/my_factor_library_batch"),
            )
            self.assertEqual(
                external.correlation_output,
                Path("factor_report/my_factor_library_correlation"),
            )
            self.assertEqual(
                external.ml_output,
                Path("data/ml/my_factor_library_multifactor"),
            )

    def test_stage_output_overrides_and_legacy_output_compatibility(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            legacy_root = root / "legacy"
            legacy = load_factor_research_config(
                self._write_config(root / "legacy_config"),
                output=str(legacy_root),
            )
            self.assertEqual(legacy.batch_output, legacy_root / "batch")
            self.assertEqual(legacy.correlation_output, legacy_root / "correlation")
            self.assertEqual(legacy.ml_output, legacy_root / "ml")
            self.assertTrue(any("deprecated" in warning for warning in legacy.warnings))

            overridden = load_factor_research_config(
                self._write_config(root / "override_config"),
                batch_output=str(root / "batch"),
                correlation_output=str(root / "correlation"),
                ml_output=str(root / "ml"),
            )
            self.assertEqual(overridden.batch_output, root / "batch")
            self.assertEqual(overridden.correlation_output, root / "correlation")
            self.assertEqual(overridden.ml_output, root / "ml")

    def test_legacy_cost_gate_is_accepted_but_not_forwarded(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = self._write_config(root)
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            payload["correlation"][
                "eligibility_col"
            ] = "preferred_profitable_after_cost"
            path.write_text(yaml.safe_dump(payload), encoding="utf-8")
            config = load_factor_research_config(path)
            commands = build_stage_commands(
                config,
                run_id_factory=lambda stage: f"child-{stage}",
            )
            self.assertNotIn("--eligibility-col", commands[1].argv)
            self.assertTrue(any("ignored" in warning for warning in config.warnings))

    def test_cli_registers_factor_run_all_and_stage_output_overrides(self):
        args = build_parser(prog="rquant").parse_args(
            [
                "factor-run-all",
                "--batch-output",
                "factor_report/custom_batch",
                "--correlation-output",
                "factor_report/custom_correlation",
                "--ml-output",
                "data/ml/custom",
            ]
        )
        self.assertEqual(args.command, "factor-run-all")
        self.assertEqual(args.config, "config/factor_research_run_all.yaml")
        self.assertEqual(args.batch_output, "factor_report/custom_batch")
        self.assertEqual(args.correlation_output, "factor_report/custom_correlation")
        self.assertEqual(args.ml_output, "data/ml/custom")

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
