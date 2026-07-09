import unittest

import pandas as pd

from training.predict_score import scores_to_signals
from training.validation import WalkForwardWindow, build_walk_forward_windows, validate_feature_label_frame


class TrainingValidationTests(unittest.TestCase):
    def test_validate_feature_label_frame_rejects_duplicates(self):
        frame = pd.DataFrame(
            {
                "date": ["2026-01-01", "2026-01-01"],
                "symbol": ["000001", "000001"],
                "feature": [1.0, 2.0],
                "target": [0.1, 0.2],
            }
        )

        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_feature_label_frame(frame, feature_cols=("feature",), target_col="target")

    def test_walk_forward_windows_are_ordered(self):
        windows = build_walk_forward_windows(
            dates=pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"]),
            train_size=2,
            test_size=1,
        )

        self.assertEqual(
            windows[0],
            WalkForwardWindow(
                train_start=pd.Timestamp("2026-01-01"),
                train_end=pd.Timestamp("2026-01-02"),
                test_start=pd.Timestamp("2026-01-03"),
                test_end=pd.Timestamp("2026-01-03"),
            ),
        )

    def test_scores_to_signals_keeps_six_digit_symbols(self):
        scores = pd.DataFrame({"date": ["2026-01-02"], "symbol": ["1"], "score": [0.8]})
        signals = scores_to_signals(scores, source="ridge")

        self.assertEqual(signals.loc[0, "symbol"], "000001")
        self.assertEqual(signals.loc[0, "source"], "model_ridge")


if __name__ == "__main__":
    unittest.main()
