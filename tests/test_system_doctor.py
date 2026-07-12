import json
import os
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

from reports import system_doctor


class SystemDoctorTest(unittest.TestCase):
    def _build_project(self, root: Path) -> None:
        (root / "config").mkdir()
        (root / "data/raw").mkdir(parents=True)
        (root / "requirements.txt").write_text("numpy==1.0\nPyYAML\n", encoding="utf-8")
        (root / "requirements-ml.txt").write_text(
            "-r requirements.txt\ntorch\n",
            encoding="utf-8",
        )
        (root / "config/fetch_kline.yaml").write_text(
            "start: '20260101'\nend: '20260102'\nstocklist: config/stocklist.csv\n"
            "out: data/raw\nworkers: 1\nmax_requests_per_minute: 180\n",
            encoding="utf-8",
        )
        (root / "config/factors.yaml").write_text(
            "default_status: watch\nfactors: {}\n",
            encoding="utf-8",
        )
        (root / "config/gtja191_factors.yaml").write_text(
            "default_status: watch\nfactors: {}\n",
            encoding="utf-8",
        )
        (root / "config/rules_preselect.yaml").write_text(
            "global: {}\nstock_pool: {}\n",
            encoding="utf-8",
        )
        (root / "config/dashboard.yaml").write_text("server: {}\n", encoding="utf-8")
        (root / "config/gemini_review.yaml").write_text("model: test\n", encoding="utf-8")
        (root / "config/stocklist.csv").write_text(
            "ts_code,symbol,name\n000001.SZ,000001,Example\n",
            encoding="utf-8",
        )
        (root / "data/raw/000001.csv").write_text(
            "date,open,close,high,low,volume\n"
            f"{date.today().isoformat()},1,1,1,1,100\n",
            encoding="utf-8",
        )

    @staticmethod
    def _available_distribution(name, *, expected, required):
        return {
            "name": name,
            "module": name,
            "required": required,
            "expected_version": expected,
            "installed_version": expected or "1.0",
            "discoverable": True,
            "status": "ok",
            "message": f"dependency available: {name}",
        }

    def test_healthy_report_is_auditable_and_never_contains_secret_values(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._build_project(root)
            secret_token = "never-serialize-this-token"
            (root / ".env").write_text(
                f"TUSHARE_TOKEN={secret_token}\nGEMINI_API_KEY=another-secret\n",
                encoding="utf-8",
            )
            output = root / "doctor.json"
            with (
                patch.object(
                    system_doctor,
                    "_check_distribution",
                    side_effect=self._available_distribution,
                ),
                patch.dict(os.environ, {}, clear=True),
            ):
                report = system_doctor.run_system_doctor(
                    project_root=root,
                    output_path=output,
                    deep=True,
                )

            serialized = output.read_text(encoding="utf-8")
            persisted = json.loads(serialized)
            self.assertEqual(report["status"], "ok")
            self.assertTrue(report["ok"])
            self.assertNotIn(secret_token, serialized)
            self.assertNotIn("another-secret", serialized)
            self.assertEqual(persisted["output_path"], str(output))
            self.assertEqual(report["secrets"]["items"][0]["source"], ".env")
            self.assertEqual(report["market_data"]["date_max"], date.today().isoformat())
            self.assertEqual(report["market_data"]["data_age_days"], 0)

    def test_missing_optional_dependencies_and_secrets_are_warnings_only(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._build_project(root)

            def dependency(name, *, expected, required):
                if required:
                    return self._available_distribution(name, expected=expected, required=required)
                return {
                    "name": name,
                    "module": name,
                    "required": False,
                    "expected_version": expected,
                    "installed_version": None,
                    "discoverable": False,
                    "status": "warning",
                    "message": f"optional dependency unavailable: {name}",
                }

            with (
                patch.object(system_doctor, "_check_distribution", side_effect=dependency),
                patch.dict(os.environ, {}, clear=True),
            ):
                report = system_doctor.run_system_doctor(project_root=root, deep=True)

            self.assertEqual(report["status"], "warning")
            self.assertTrue(report["ok"])
            self.assertEqual(report["summary"]["error_count"], 0)
            self.assertGreaterEqual(report["summary"]["warning_count"], 3)

    def test_missing_required_config_is_an_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._build_project(root)
            (root / "config/factors.yaml").unlink()
            with (
                patch.object(
                    system_doctor,
                    "_check_distribution",
                    side_effect=self._available_distribution,
                ),
                patch.dict(
                    os.environ,
                    {"TUSHARE_TOKEN": "x", "GEMINI_API_KEY": "y"},
                    clear=True,
                ),
            ):
                report = system_doctor.run_system_doctor(project_root=root, deep=True)

            self.assertEqual(report["status"], "error")
            self.assertFalse(report["ok"])
            self.assertIn(
                "missing required config: factors.yaml",
                report["configs"]["errors"],
            )

    def test_invalid_fetch_rate_limit_is_a_config_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._build_project(root)
            fetch_config = root / "config/fetch_kline.yaml"
            fetch_config.write_text(
                fetch_config.read_text(encoding="utf-8").replace(
                    "max_requests_per_minute: 180",
                    "max_requests_per_minute: -1",
                ),
                encoding="utf-8",
            )
            with (
                patch.object(
                    system_doctor,
                    "_check_distribution",
                    side_effect=self._available_distribution,
                ),
                patch.dict(
                    os.environ,
                    {"TUSHARE_TOKEN": "x", "GEMINI_API_KEY": "y"},
                    clear=True,
                ),
            ):
                report = system_doctor.run_system_doctor(project_root=root, deep=True)

            self.assertEqual(report["status"], "error")
            self.assertIn(
                "config fetch_kline.yaml has invalid value: "
                "max_requests_per_minute must be a non-negative integer",
                report["configs"]["errors"],
            )

    def test_market_data_schema_and_six_digit_filename_are_enforced(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._build_project(root)
            (root / "data/raw/not-a-symbol.csv").write_text(
                "date,open,close,high,low,volume\n2026-01-01,1,1,1,1,100\n",
                encoding="utf-8",
            )
            with (
                patch.object(
                    system_doctor,
                    "_check_distribution",
                    side_effect=self._available_distribution,
                ),
                patch.dict(
                    os.environ,
                    {"TUSHARE_TOKEN": "x", "GEMINI_API_KEY": "y"},
                    clear=True,
                ),
            ):
                report = system_doctor.run_system_doctor(project_root=root, deep=True)

            self.assertEqual(report["status"], "error")
            self.assertEqual(
                report["market_data"]["invalid_files"][0]["file"],
                "not-a-symbol.csv",
            )

    def test_installed_native_dependency_must_actually_import(self):
        with (
            patch.object(system_doctor.importlib.metadata, "version", return_value="4.6.0"),
            patch.object(
                system_doctor.importlib,
                "import_module",
                side_effect=OSError("Library not loaded: @rpath/libomp.dylib"),
            ),
        ):
            result = system_doctor._check_distribution(
                "lightgbm",
                expected=None,
                required=False,
            )

        self.assertEqual(result["status"], "warning")
        self.assertFalse(result["importable"])
        self.assertEqual(result["import_error"], "missing native library libomp.dylib")
        self.assertIn("import failed", result["message"])

    def test_windows_native_dependency_error_has_actionable_summary(self):
        with (
            patch.object(system_doctor.importlib.metadata, "version", return_value="4.6.0"),
            patch.object(
                system_doctor.importlib,
                "import_module",
                side_effect=ImportError("DLL load failed while importing lightgbm"),
            ),
        ):
            result = system_doctor._check_distribution(
                "lightgbm",
                expected=None,
                required=False,
            )

        self.assertEqual(result["status"], "warning")
        self.assertIn("Windows native DLL load failed", result["import_error"])

    def test_partial_fetch_manifest_makes_existing_market_data_unhealthy(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._build_project(root)
            (root / "data/raw/_fetch_manifest.json").write_text(
                json.dumps(
                    {
                        "status": "partial",
                        "start": "20260101",
                        "end": "20260102",
                        "symbol_count": 2,
                        "completed_count": 1,
                        "failed_count": 1,
                        "pending_count": 0,
                        "failed_codes": ["000002"],
                    }
                ),
                encoding="utf-8",
            )
            with (
                patch.object(
                    system_doctor,
                    "_check_distribution",
                    side_effect=self._available_distribution,
                ),
                patch.dict(
                    os.environ,
                    {"TUSHARE_TOKEN": "x", "GEMINI_API_KEY": "y"},
                    clear=True,
                ),
            ):
                report = system_doctor.run_system_doctor(project_root=root, deep=True)

            self.assertFalse(report["ok"])
            self.assertEqual(report["market_data"]["status"], "error")
            self.assertEqual(
                report["market_data"]["fetch_manifest"]["failed_codes"],
                ["000002"],
            )
            self.assertIn("fetch manifest is incomplete", report["market_data"]["errors"][0])

    def test_stale_market_data_is_visible_but_not_a_structural_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._build_project(root)
            (root / "data/raw/000001.csv").write_text(
                "date,open,close,high,low,volume\n2026-07-01,1,1,1,1,100\n",
                encoding="utf-8",
            )
            with (
                patch.object(
                    system_doctor,
                    "_check_distribution",
                    side_effect=self._available_distribution,
                ),
                patch.dict(
                    os.environ,
                    {"TUSHARE_TOKEN": "x", "GEMINI_API_KEY": "y"},
                    clear=True,
                ),
            ):
                report = system_doctor.run_system_doctor(
                    project_root=root,
                    deep=True,
                    as_of_date="2026-07-11",
                    max_market_data_age_days=7,
                )

            self.assertTrue(report["ok"])
            self.assertEqual(report["market_data"]["status"], "warning")
            self.assertEqual(report["market_data"]["data_age_days"], 10)
            self.assertIn("market data may be stale", report["market_data"]["warnings"][0])

    def test_legacy_charts_and_review_are_flagged_as_unverified(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._build_project(root)
            candidates_dir = root / "data/candidates"
            chart_dir = root / "data/kline/2026-07-11"
            review_dir = root / "data/review/2026-07-11"
            candidates_dir.mkdir(parents=True)
            chart_dir.mkdir(parents=True)
            review_dir.mkdir(parents=True)
            (candidates_dir / "candidates_latest.json").write_text(
                json.dumps(
                    {
                        "pick_date": "2026-07-11",
                        "candidates": [{"code": "000001"}],
                    }
                ),
                encoding="utf-8",
            )
            (chart_dir / "000001_day.jpg").write_bytes(b"legacy-chart")
            (review_dir / "suggestion.json").write_text(
                json.dumps({"date": "2026-07-11", "recommendations": []}),
                encoding="utf-8",
            )
            with (
                patch.object(
                    system_doctor,
                    "_check_distribution",
                    side_effect=self._available_distribution,
                ),
                patch.dict(
                    os.environ,
                    {"TUSHARE_TOKEN": "x", "GEMINI_API_KEY": "y"},
                    clear=True,
                ),
            ):
                report = system_doctor.run_system_doctor(project_root=root, deep=True)

            self.assertTrue(report["ok"])
            self.assertEqual(report["workflow_artifacts"]["status"], "warning")
            self.assertEqual(report["workflow_artifacts"]["warning_count"], 3)
            self.assertTrue(
                any(
                    "point-in-time boundary is unverified" in warning
                    for warning in report["workflow_artifacts"]["warnings"]
                )
            )

    def test_partial_chart_manifest_is_a_doctor_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._build_project(root)
            candidates_dir = root / "data/candidates"
            chart_dir = root / "data/kline/2026-07-11"
            candidates_dir.mkdir(parents=True)
            chart_dir.mkdir(parents=True)
            (candidates_dir / "candidates_latest.json").write_text(
                json.dumps(
                    {
                        "pick_date": "2026-07-11",
                        "candidates": [{"code": "000001"}],
                    }
                ),
                encoding="utf-8",
            )
            (chart_dir / "export_manifest.json").write_text(
                json.dumps(
                    {
                        "status": "partial",
                        "pick_date": "2026-07-11",
                        "success_count": 0,
                        "failed_count": 1,
                        "results": {},
                    }
                ),
                encoding="utf-8",
            )
            with (
                patch.object(
                    system_doctor,
                    "_check_distribution",
                    side_effect=self._available_distribution,
                ),
                patch.dict(
                    os.environ,
                    {"TUSHARE_TOKEN": "x", "GEMINI_API_KEY": "y"},
                    clear=True,
                ),
            ):
                report = system_doctor.run_system_doctor(project_root=root, deep=True)

            self.assertFalse(report["ok"])
            self.assertEqual(report["workflow_artifacts"]["status"], "error")
            self.assertIn(
                "chart export manifest status is partial",
                report["workflow_artifacts"]["errors"],
            )


if __name__ == "__main__":
    unittest.main()
