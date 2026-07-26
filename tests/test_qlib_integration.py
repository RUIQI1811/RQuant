import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.signal_portfolio import run_signal_portfolio_backtest
from models.qlib_models import DoubleEnsembleModel
from training.qlib_dataset import build_qlib_dataset, normalize_qlib_scores
from training.train_walk_forward import WalkForwardTrainingConfig, run_walk_forward_training


QLIB_AVAILABLE = importlib.util.find_spec("qlib") is not None


def _model_frame(dates: pd.DatetimeIndex, symbols: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for day_index, date in enumerate(dates):
        for symbol_index, symbol in enumerate(symbols):
            features = {
                f"feature_{feature_index}": float(
                    np.sin((day_index + feature_index) / 3)
                    + symbol_index * 0.03 * (feature_index + 1)
                )
                for feature_index in range(8)
            }
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    **features,
                    "forward_return_1d": (
                        0.4 * features["feature_0"]
                        - 0.2 * features["feature_1"]
                        + 0.1 * features["feature_2"]
                    ),
                }
            )
    return pd.DataFrame(rows)


@unittest.skipUnless(QLIB_AVAILABLE, "pyqlib is optional")
class QlibIntegrationTests(unittest.TestCase):
    def test_dataset_is_time_aligned_and_preserves_six_digit_symbols(self):
        dates = pd.bdate_range("2026-01-02", periods=14)
        frame = _model_frame(dates, ("1", "000002"))

        bundle = build_qlib_dataset(
            train=frame.loc[frame["date"].isin(dates[:12])],
            test=frame.loc[frame["date"].isin(dates[12:])],
            feature_cols=("feature_0", "feature_1"),
            target_col="forward_return_1d",
            valid_ratio=0.25,
        )
        prepared = bundle.dataset.prepare(
            "test",
            col_set=["feature", "label"],
            data_key="learn",
        )

        self.assertEqual(prepared.index.names, ["datetime", "instrument"])
        self.assertEqual(set(prepared.index.get_level_values("instrument")), {"000001", "000002"})
        self.assertLess(bundle.train_end, bundle.valid_start)
        self.assertLess(bundle.valid_end, bundle.test_start)
        self.assertEqual(prepared.index.tolist(), bundle.test_index.tolist())

        reversed_scores = pd.Series(
            np.arange(len(bundle.test_index), dtype=float),
            index=bundle.test_index[::-1],
        )
        scores = normalize_qlib_scores(
            reversed_scores,
            expected_index=bundle.test_index,
        )
        self.assertEqual(scores.name, "score")
        self.assertEqual(scores.index.tolist(), bundle.test_index.tolist())

    def test_dataset_rejects_validation_test_overlap(self):
        dates = pd.bdate_range("2026-01-02", periods=10)
        frame = _model_frame(dates, ("000001", "000002"))

        with self.assertRaisesRegex(ValueError, "validation must end before"):
            build_qlib_dataset(
                train=frame.loc[frame["date"].isin(dates[:8])],
                test=frame.loc[frame["date"].isin(dates[7:])],
                feature_cols=("feature_0", "feature_1"),
                target_col="forward_return_1d",
            )

    def test_doubleensemble_trains_multiple_qlib_submodels(self):
        dates = pd.bdate_range("2026-01-02", periods=36)
        frame = _model_frame(dates, tuple(str(value).zfill(6) for value in range(1, 9)))
        feature_cols = tuple(f"feature_{value}" for value in range(8))
        bundle = build_qlib_dataset(
            train=frame.loc[frame["date"].isin(dates[:30])],
            test=frame.loc[frame["date"].isin(dates[30:])],
            feature_cols=feature_cols,
            target_col="forward_return_1d",
        )
        model = DoubleEnsembleModel(
            n_estimators=5,
            n_jobs=1,
            random_state=7,
            num_models=2,
        )

        model.fit(bundle.dataset)
        scores = normalize_qlib_scores(
            model.predict(bundle.dataset),
            expected_index=bundle.test_index,
        )

        self.assertEqual(len(model.model.ensemble), 2)
        self.assertEqual(len(model.model.sub_features), 2)
        self.assertTrue(np.isfinite(scores).all())

    def test_qlib_score_flows_through_unified_signal_and_rquant_backtest(self):
        market_dates = pd.bdate_range("2026-01-02", periods=22)
        training_dates = market_dates[:16]
        symbols = ("000001", "000002", "000003", "000004")
        frame = _model_frame(training_dates, symbols)
        feature_cols = tuple(f"feature_{value}" for value in range(8))

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            features_path = root / "features.csv"
            labels_path = root / "labels.csv"
            model_output = root / "model"
            raw_dir = root / "raw"
            backtest_output = root / "backtest"
            raw_dir.mkdir()
            frame[["date", "symbol", *feature_cols]].to_csv(features_path, index=False)
            frame[["date", "symbol", "forward_return_1d"]].to_csv(labels_path, index=False)

            for symbol_index, symbol in enumerate(symbols):
                day = np.arange(len(market_dates), dtype=float)
                close = 10.0 + symbol_index * 2.0 + day * (0.03 + symbol_index * 0.005)
                pd.DataFrame(
                    {
                        "date": market_dates,
                        "open": close * 1.001,
                        "close": close,
                        "high": close * 1.01,
                        "low": close * 0.99,
                        "volume": 1_000_000 + day * 1000,
                    }
                ).to_csv(raw_dir / f"{symbol}.csv", index=False)

            outputs = run_walk_forward_training(
                features_path=features_path,
                labels_path=labels_path,
                output_dir=model_output,
                config=WalkForwardTrainingConfig(
                    feature_cols=feature_cols,
                    target_col="forward_return_1d",
                    model="lightgbm",
                    train_size=10,
                    test_size=4,
                    purge_days=2,
                    signal_top_n=2,
                    lightgbm_estimators=5,
                    lightgbm_n_jobs=1,
                    qlib_valid_ratio=0.2,
                ),
            )
            windows = pd.read_csv(outputs["windows_path"])
            summary = json.loads(Path(outputs["summary_path"]).read_text(encoding="utf-8"))
            predictions = pd.read_csv(outputs["predictions_path"], dtype={"symbol": str})
            signals = pd.read_csv(outputs["signals_path"], dtype={"symbol": str})
            audit = json.loads(windows.loc[0, "backend_audit"])

            backtest_outputs = run_signal_portfolio_backtest(
                signals_path=outputs["signals_path"],
                data_dir=raw_dir,
                output_dir=backtest_output,
                source="model_lightgbm",
                start_date=str(market_dates[0].date()),
                end_date=str(market_dates[-1].date()),
                initial_cash=200_000,
                hold_days=2,
                commission_wan=0.8,
                max_positions=2,
                lot_size=100,
            )
            backtest_summary = json.loads(
                Path(backtest_outputs["summary_path"]).read_text(encoding="utf-8")
            )

        self.assertEqual(summary["model_backend"], "qlib")
        self.assertEqual(windows.loc[0, "model_backend"], "qlib")
        self.assertLess(pd.Timestamp(windows.loc[0, "purge_end"]), pd.Timestamp(windows.loc[0, "test_start"]))
        self.assertLess(pd.Timestamp(audit["valid_end"]), pd.Timestamp(audit["test_start"]))
        self.assertEqual(audit["index_names"], ["datetime", "instrument"])
        self.assertTrue(np.isfinite(predictions["score"]).all())
        self.assertEqual(set(signals["source"]), {"model_lightgbm"})
        self.assertTrue(signals["symbol"].str.fullmatch(r"\d{6}").all())
        self.assertEqual(backtest_summary["signal_source_filter"], "model_lightgbm")
        self.assertGreater(backtest_summary["signal_count"], 0)


if __name__ == "__main__":
    unittest.main()
