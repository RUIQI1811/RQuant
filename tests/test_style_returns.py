import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from factors.style_returns import (
    StyleFactorConfig,
    build_style_factor_file,
    calculate_style_factor_returns,
)
from factors.gtja191 import GTJA191, build_gtja191_panels
from strategies.preselect import load_raw_data


class StyleFactorReturnsTest(unittest.TestCase):
    def test_uses_prior_characteristics_and_forms_complete_2x3_returns(self):
        dates = pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"])
        rows = []
        symbol_index = 0
        for size, cap in (("S", 10.0), ("B", 100.0)):
            for value, bm, value_return in (
                ("L", 1.0, 0.00),
                ("M", 2.0, 0.015),
                ("H", 3.0, 0.03),
            ):
                symbol_index += 1
                symbol = f"{symbol_index:06d}"
                for date_index, date in enumerate(dates):
                    rows.append(
                        {
                            "date": date,
                            "symbol": symbol,
                            "daily_return": (
                                0.02 if size == "S" else 0.01
                            )
                            + value_return,
                            "market_cap": cap + date_index,
                            "book_to_market": bm,
                        }
                    )

        factors, audit = calculate_style_factor_returns(
            pd.DataFrame(rows),
            config=StyleFactorConfig(min_stocks_per_portfolio=1),
        )

        self.assertEqual(factors["date"].tolist(), ["2026-01-05", "2026-01-06"])
        self.assertTrue(np.allclose(factors["smb"], 0.01))
        self.assertTrue(np.allclose(factors["hml"], 0.03))
        self.assertEqual(audit["dropped_incomplete_date_count"], 0)

    def test_builds_auditable_file_from_raw_and_context(self):
        dates = pd.bdate_range("2026-01-02", periods=4)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_dir = root / "raw"
            context_path = root / "context.csv"
            output_path = root / "style.csv"
            data_dir.mkdir()
            context_rows = []
            for index, (cap, bm) in enumerate(
                (
                    (10.0, 1.0),
                    (11.0, 2.0),
                    (12.0, 3.0),
                    (100.0, 1.0),
                    (101.0, 2.0),
                    (102.0, 3.0),
                ),
                start=1,
            ):
                symbol = f"{index:06d}"
                close = 10.0 * np.cumprod(
                    [1.0, 1.01 + index * 0.001, 1.012 + index * 0.001, 1.011]
                )
                pd.DataFrame(
                    {
                        "date": dates,
                        "open": close,
                        "high": close * 1.01,
                        "low": close * 0.99,
                        "close": close,
                        "volume": 1_000_000,
                    }
                ).to_csv(data_dir / f"{symbol}.csv", index=False)
                for date in dates:
                    context_rows.append(
                        {
                            "date": date,
                            "symbol": symbol,
                            "market_cap": cap,
                            "book_to_market": bm,
                        }
                    )
            pd.DataFrame(context_rows).to_csv(context_path, index=False)

            result = build_style_factor_file(
                data_dir=data_dir,
                context_path=context_path,
                output_file=output_path,
                config=StyleFactorConfig(min_stocks_per_portfolio=1),
            )

            self.assertGreater(result["row_count"], 0)
            output = pd.read_csv(output_path)
            self.assertEqual(output.columns.tolist(), ["date", "mkt", "smb", "hml"])
            manifest = json.loads(
                Path(result["manifest_path"]).read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest["method"], "daily_fama_french_2x3_value_weighted"
            )
            self.assertIn("strictly before t", manifest["timing"])
            raw_data = load_raw_data(str(data_dir))
            panels = build_gtja191_panels(
                raw_data,
                style_factor_data=output,
            )
            gtja_030 = GTJA191(panels).calculate("gtja_030")
            self.assertEqual(gtja_030.shape, panels.close.shape)


if __name__ == "__main__":
    unittest.main()
