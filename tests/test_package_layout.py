import importlib
import unittest


class PackageLayoutTests(unittest.TestCase):
    def test_top_level_packages_import(self):
        for name in (
            "market",
            "signals",
            "strategies",
            "factors",
            "labels",
            "models",
            "training",
            "backtest",
            "reports",
        ):
            with self.subTest(name=name):
                self.assertIsNotNone(importlib.import_module(name))


if __name__ == "__main__":
    unittest.main()
