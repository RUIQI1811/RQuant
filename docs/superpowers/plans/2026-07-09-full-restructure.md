# Full Project Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure RQuant into top-level quant research packages for market data, factors, labels, models, training, signals, strategies, backtest, and reports while preserving the factor/custom-strategy boundary and existing research behavior.

**Architecture:** Create the new top-level package layout first, then migrate one boundary at a time while old `pipeline.*` imports remain thin wrappers until the new commands and tests cover the same workflows. Factor research, custom buy strategies, and ML scoring each feed the shared `signals` package and then the realistic `backtest` package.

**Tech Stack:** Python 3, pandas, numpy, unittest, argparse, YAML config files, existing Tushare/Gemini integration points, optional LightGBM and PyTorch model modules.

---

## Scope Check

The approved spec covers the full repository structure. Treat this as a master
plan made of independently testable phases. Each task must leave the repository
importable and runnable through either the new top-level package or the temporary
`pipeline` wrapper.

Do not remove the old `pipeline` wrappers until Task 12. If Task 12 is too risky
after validation, keep wrappers and document them as compatibility shims.

## File Structure

Create or migrate these packages:

- `market/`: data fetch, raw CSV loading, market data preparation, universe and tradability flags.
- `signals/`: stable shared signal schema and adapters for factor, model, and strategy signals.
- `strategies/`: custom buy rules and preselect workflow.
- `factors/`: factor calculators, registries, lifecycle scoring, correlation, factor signal conversion.
- `labels/`: forward-return label generation for ML and validation.
- `models/`: Ridge, ElasticNet, LightGBM, and MLP wrappers with a small common interface.
- `training/`: walk-forward splits, feature/label validation, model score prediction.
- `backtest/`: realistic portfolio engine and factor portfolio bridge.
- `reports/`: factor tester, batch runners, signal returns, and research report generation.
- `pipeline/`: temporary wrappers importing from the new packages.
- `tests/`: focused migration tests and preserved behavior tests.
- `docs/`: updated package map and command usage.

## Task 1: Baseline and Package Skeleton

**Files:**
- Create: `market/__init__.py`
- Create: `signals/__init__.py`
- Create: `strategies/__init__.py`
- Create: `factors/__init__.py`
- Create: `labels/__init__.py`
- Create: `models/__init__.py`
- Create: `training/__init__.py`
- Create: `backtest/__init__.py`
- Create: `reports/__init__.py`
- Create: `tests/test_package_layout.py`

- [ ] **Step 1: Record current status without changing it**

Run:

```bash
git status --short
python -c "import sys; print(sys.executable)"
```

Expected: `git status --short` may show the existing staged architecture work. Do not reset, checkout, or clean those files.

- [ ] **Step 2: Write the package import test**

Create `tests/test_package_layout.py`:

```python
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
```

- [ ] **Step 3: Run the package import test and verify it fails before package files exist**

Run:

```bash
python -m unittest tests.test_package_layout
```

Expected: FAIL with `ModuleNotFoundError` for the first missing package.

- [ ] **Step 4: Create package files**

Create each package `__init__.py` with this content, replacing the package name:

```python
"""Top-level package for RQuant market research components."""
```

Use package-specific one-line docstrings:

```python
"""Market data loading, normalization, and universe construction."""
"""Shared signal schema and signal adapters."""
"""Custom buy strategy rules and preselection workflows."""
"""Factor calculators, registries, scoring, and factor utilities."""
"""Forward-return and model-training label generation."""
"""Machine-learning model wrappers."""
"""Walk-forward training, validation, and prediction scoring."""
"""Portfolio construction, transaction costs, and performance analysis."""
"""IC, batch, signal-return, backtest, and research reports."""
```

- [ ] **Step 5: Run the package import test again**

Run:

```bash
python -m unittest tests.test_package_layout
```

Expected: PASS.

- [ ] **Step 6: Commit package skeleton**

Run:

```bash
git add market signals strategies factors labels models training backtest reports tests/test_package_layout.py
git commit -m "refactor: add top-level research packages"
```

## Task 2: Move Shared Signal Schema

**Files:**
- Move: `pipeline/signals/schema.py` -> `signals/schema.py`
- Move: `pipeline/factors/signals.py` -> `signals/factor_adapters.py`
- Move: `pipeline/strategies/adapters.py` -> `signals/strategy_adapters.py`
- Modify: `pipeline/signals/schema.py`
- Modify: `pipeline/factors/signals.py`
- Modify: `pipeline/strategies/adapters.py`
- Modify: `signals/__init__.py`
- Test: `tests/test_signal_schema.py`
- Test: `tests/test_factor_signals.py`

- [ ] **Step 1: Add a new import-path assertion to signal schema tests**

Append to `tests/test_signal_schema.py`:

```python
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
```

If `tests/test_signal_schema.py` uses a different class name, add this as a new class without editing unrelated tests.

- [ ] **Step 2: Run the new test and verify it fails**

Run:

```bash
python -m unittest tests.test_signal_schema
```

Expected: FAIL with `ModuleNotFoundError: No module named 'signals.schema'`.

- [ ] **Step 3: Move signal modules**

Run:

```bash
git mv pipeline/signals/schema.py signals/schema.py
git mv pipeline/factors/signals.py signals/factor_adapters.py
git mv pipeline/strategies/adapters.py signals/strategy_adapters.py
```

- [ ] **Step 4: Add wrapper modules for old imports**

Create `pipeline/signals/schema.py`:

```python
"""Compatibility wrapper for the top-level signal schema."""

from signals.schema import *  # noqa: F401,F403
```

Create `pipeline/factors/signals.py`:

```python
"""Compatibility wrapper for factor signal adapters."""

from signals.factor_adapters import *  # noqa: F401,F403
```

Create `pipeline/strategies/adapters.py`:

```python
"""Compatibility wrapper for custom strategy signal adapters."""

from signals.strategy_adapters import *  # noqa: F401,F403
```

- [ ] **Step 5: Export stable signal names**

Update `signals/__init__.py`:

```python
"""Shared signal schema and signal adapters."""

from .schema import Signal, SignalFrameError, signals_to_frame

__all__ = ["Signal", "SignalFrameError", "signals_to_frame"]
```

If the current schema exposes different names, export the actual public names from `signals/schema.py` and keep the test aligned with those names.

- [ ] **Step 6: Update moved imports**

Run:

```bash
rg -n "pipeline\\.signals|pipeline\\.factors\\.signals|pipeline\\.strategies\\.adapters" .
```

For production code outside `pipeline/` wrappers, replace old imports with:

```python
from signals.schema import Signal, signals_to_frame
from signals.factor_adapters import ...
from signals.strategy_adapters import ...
```

Do not edit generated files under `data/`, `factor_report/`, or `__pycache__/`.

- [ ] **Step 7: Run signal tests**

Run:

```bash
python -m unittest tests.test_signal_schema tests.test_factor_signals
```

Expected: PASS.

- [ ] **Step 8: Commit signal migration**

Run:

```bash
git add signals pipeline/signals pipeline/factors/signals.py pipeline/strategies/adapters.py tests/test_signal_schema.py tests/test_factor_signals.py
git commit -m "refactor: move shared signal layer"
```

## Task 3: Move Custom Strategy Layer

**Files:**
- Move: `pipeline/strategies/base.py` -> `strategies/base.py`
- Move: `pipeline/strategies/mbdsr.py` -> `strategies/mbdsr.py`
- Move: `pipeline/strategies/bdsr_macd_obv.py` -> `strategies/bdsr_macd_obv.py`
- Move: `pipeline/Selector.py` -> `strategies/selector.py`
- Move: `pipeline/select_stock.py` -> `strategies/preselect.py`
- Modify: `pipeline/strategies/base.py`
- Modify: `pipeline/strategies/mbdsr.py`
- Modify: `pipeline/strategies/bdsr_macd_obv.py`
- Modify: `pipeline/Selector.py`
- Modify: `pipeline/select_stock.py`
- Modify: `strategies/__init__.py`
- Test: `tests/test_mbdsr.py`
- Test: `tests/test_bdsr_macd_obv.py`

- [ ] **Step 1: Add top-level strategy import tests**

Append to `tests/test_mbdsr.py`:

```python
class TopLevelMbdsrImportTests(unittest.TestCase):
    def test_mbdsr_imports_from_top_level_package(self):
        import strategies.mbdsr as mbdsr

        self.assertTrue(hasattr(mbdsr, "calc_obv"))
```

Append to `tests/test_bdsr_macd_obv.py`:

```python
class TopLevelBdsrMacdObvImportTests(unittest.TestCase):
    def test_bdsr_macd_obv_imports_from_top_level_package(self):
        import strategies.bdsr_macd_obv as strategy

        self.assertTrue(hasattr(strategy, "BdsrMacdObvStrategy"))
```

If the class name differs, assert the actual public strategy class or public `run` function from `pipeline/strategies/bdsr_macd_obv.py`.

- [ ] **Step 2: Run strategy tests and verify top-level imports fail**

Run:

```bash
python -m unittest tests.test_mbdsr tests.test_bdsr_macd_obv
```

Expected: FAIL with missing `strategies.mbdsr` or `strategies.bdsr_macd_obv`.

- [ ] **Step 3: Move strategy modules**

Run:

```bash
git mv pipeline/strategies/base.py strategies/base.py
git mv pipeline/strategies/mbdsr.py strategies/mbdsr.py
git mv pipeline/strategies/bdsr_macd_obv.py strategies/bdsr_macd_obv.py
git mv pipeline/Selector.py strategies/selector.py
git mv pipeline/select_stock.py strategies/preselect.py
```

- [ ] **Step 4: Add old-path wrappers**

Create `pipeline/strategies/base.py`:

```python
"""Compatibility wrapper for custom strategy base classes."""

from strategies.base import *  # noqa: F401,F403
```

Create `pipeline/strategies/mbdsr.py`:

```python
"""Compatibility wrapper for mBDSR strategy utilities."""

from strategies.mbdsr import *  # noqa: F401,F403
```

Create `pipeline/strategies/bdsr_macd_obv.py`:

```python
"""Compatibility wrapper for BDSR/MACD/OBV strategy."""

from strategies.bdsr_macd_obv import *  # noqa: F401,F403
```

Create `pipeline/Selector.py`:

```python
"""Compatibility wrapper for the legacy selector module."""

from strategies.selector import *  # noqa: F401,F403
```

Create `pipeline/select_stock.py`:

```python
"""Compatibility wrapper for the preselect workflow."""

from strategies.preselect import *  # noqa: F401,F403
```

- [ ] **Step 5: Keep legacy Selector cache import support**

In `strategies/selector.py`, keep this alias near the imports:

```python
import sys

sys.modules.setdefault("Selector", sys.modules[__name__])
```

This preserves old Numba cache artifacts that reference top-level `Selector`.

- [ ] **Step 6: Update direct imports in production code**

Run:

```bash
rg -n "from (Selector|select_stock)|import (Selector|select_stock)|pipeline\\.strategies|pipeline\\.Selector|pipeline\\.select_stock" .
```

For production code outside wrappers, replace imports with:

```python
from strategies.preselect import run_preselect, resolve_preselect_output_dir
from strategies.selector import BrickChartSelector
from strategies.mbdsr import calc_obv
from strategies.bdsr_macd_obv import BdsrMacdObvStrategy
```

Use the actual public names found in the moved modules.

- [ ] **Step 7: Run strategy tests and CLI preselect help**

Run:

```bash
python -m unittest tests.test_mbdsr tests.test_bdsr_macd_obv
python -m pipeline.cli preselect --help
```

Expected: tests PASS and CLI help prints usage without import errors.

- [ ] **Step 8: Commit strategy migration**

Run:

```bash
git add strategies pipeline/strategies pipeline/Selector.py pipeline/select_stock.py tests/test_mbdsr.py tests/test_bdsr_macd_obv.py
git commit -m "refactor: move custom strategy layer"
```

## Task 4: Move Market Data Layer

**Files:**
- Move: `pipeline/fetch_kline.py` -> `market/fetch_kline.py`
- Move: `pipeline/market_data.py` -> `market/data.py`
- Move: `pipeline/pipeline_io.py` -> `market/io.py`
- Modify: `pipeline/fetch_kline.py`
- Modify: `pipeline/market_data.py`
- Modify: `pipeline/pipeline_io.py`
- Modify: `market/__init__.py`
- Test: `tests/test_market_data.py`
- Test: `tests/test_fetch_kline.py`

- [ ] **Step 1: Add top-level market import tests**

Append to `tests/test_market_data.py`:

```python
class TopLevelMarketImportTests(unittest.TestCase):
    def test_market_data_imports_from_top_level_package(self):
        import market.data as data

        self.assertTrue(hasattr(data, "build_stock_pool_by_date"))
```

Append to `tests/test_fetch_kline.py`:

```python
class TopLevelFetchImportTests(unittest.TestCase):
    def test_fetch_kline_imports_from_top_level_package(self):
        import market.fetch_kline as fetch_kline

        self.assertTrue(hasattr(fetch_kline, "main"))
```

If `market.fetch_kline` has no `main`, assert the current public fetch function from `pipeline/fetch_kline.py`.

- [ ] **Step 2: Run market tests and verify top-level imports fail**

Run:

```bash
python -m unittest tests.test_market_data tests.test_fetch_kline
```

Expected: FAIL on missing top-level market modules.

- [ ] **Step 3: Move market modules**

Run:

```bash
git mv pipeline/fetch_kline.py market/fetch_kline.py
git mv pipeline/market_data.py market/data.py
git mv pipeline/pipeline_io.py market/io.py
```

- [ ] **Step 4: Add old-path wrappers**

Create `pipeline/fetch_kline.py`:

```python
"""Compatibility wrapper for market data fetching."""

from market.fetch_kline import *  # noqa: F401,F403

if __name__ == "__main__":
    from market.fetch_kline import main

    main()
```

Create `pipeline/market_data.py`:

```python
"""Compatibility wrapper for market data preparation."""

from market.data import *  # noqa: F401,F403
```

Create `pipeline/pipeline_io.py`:

```python
"""Compatibility wrapper for market and candidate IO helpers."""

from market.io import *  # noqa: F401,F403
```

- [ ] **Step 5: Update direct imports**

Run:

```bash
rg -n "pipeline\\.market_data|pipeline\\.fetch_kline|pipeline_io|from market_data|from fetch_kline" .
```

Replace production imports with:

```python
from market.data import build_stock_pool_by_date
from market.fetch_kline import main
from market.io import save_candidates
```

Use the actual public helper names from `market/data.py`, `market/fetch_kline.py`, and `market/io.py`.

- [ ] **Step 6: Run market tests and fetch help path**

Run:

```bash
python -m unittest tests.test_market_data tests.test_fetch_kline
python -m market.fetch_kline --help
python -m pipeline.fetch_kline --help
```

Expected: tests PASS. Both module commands print help or fail only with the same existing argument behavior, not import errors.

- [ ] **Step 7: Commit market migration**

Run:

```bash
git add market pipeline/fetch_kline.py pipeline/market_data.py pipeline/pipeline_io.py tests/test_market_data.py tests/test_fetch_kline.py
git commit -m "refactor: move market data layer"
```

## Task 5: Move Factor Calculators and Factor Utilities

**Files:**
- Move: `pipeline/factors/alpha101.py` -> `factors/alpha101.py`
- Move: `pipeline/factors/gtja191.py` -> `factors/gtja191.py`
- Move: `pipeline/factors/brick.py` -> `factors/brick.py`
- Move: `pipeline/factors/base.py` -> `factors/base.py`
- Move: `pipeline/factors/catalog.py` -> `factors/catalog.py`
- Move: `pipeline/factors/filter_rank.py` -> `factors/filter_rank.py`
- Move: `pipeline/factor_correlation.py` -> `factors/correlation.py`
- Move: `pipeline/factor_scoring.py` -> `factors/scoring.py`
- Modify: `pipeline/factors/*.py`
- Modify: `pipeline/factor_correlation.py`
- Modify: `pipeline/factor_scoring.py`
- Test: `tests/test_alpha101.py`
- Test: `tests/test_gtja191.py`
- Test: `tests/test_brick_factor.py`
- Test: `tests/test_factor_filter_rank.py`
- Test: `tests/test_factor_correlation.py`
- Test: `tests/test_factor_scoring.py`

- [ ] **Step 1: Add top-level factor import tests**

Append to `tests/test_alpha101.py`:

```python
class TopLevelAlpha101ImportTests(unittest.TestCase):
    def test_alpha101_registry_imports_from_top_level_package(self):
        import factors.alpha101 as alpha101

        self.assertTrue(hasattr(alpha101, "Alpha101Panels"))
```

Append to `tests/test_gtja191.py`:

```python
class TopLevelGtja191ImportTests(unittest.TestCase):
    def test_gtja191_registry_imports_from_top_level_package(self):
        import factors.gtja191 as gtja191

        self.assertTrue(hasattr(gtja191, "GTJA191DataError"))
```

Append to `tests/test_brick_factor.py`:

```python
class TopLevelBrickFactorImportTests(unittest.TestCase):
    def test_brick_factor_imports_from_top_level_package(self):
        import factors.brick as brick

        self.assertTrue(hasattr(brick, "build_brick_factor_frame"))
```

If any asserted name differs, use the public name already asserted by the existing test file.

- [ ] **Step 2: Run factor tests and verify new imports fail**

Run:

```bash
python -m unittest tests.test_alpha101 tests.test_gtja191 tests.test_brick_factor
```

Expected: FAIL on missing top-level factor modules.

- [ ] **Step 3: Move factor modules**

Run:

```bash
git mv pipeline/factors/alpha101.py factors/alpha101.py
git mv pipeline/factors/gtja191.py factors/gtja191.py
git mv pipeline/factors/brick.py factors/brick.py
git mv pipeline/factors/base.py factors/base.py
git mv pipeline/factors/catalog.py factors/catalog.py
git mv pipeline/factors/filter_rank.py factors/filter_rank.py
git mv pipeline/factor_correlation.py factors/correlation.py
git mv pipeline/factor_scoring.py factors/scoring.py
```

- [ ] **Step 4: Add factor wrappers**

Create wrapper modules:

```python
"""Compatibility wrapper for the moved factor module."""

from factors.alpha101 import *  # noqa: F401,F403
```

Use this exact pattern for:

```text
pipeline/factors/alpha101.py      -> factors.alpha101
pipeline/factors/gtja191.py       -> factors.gtja191
pipeline/factors/brick.py         -> factors.brick
pipeline/factors/base.py          -> factors.base
pipeline/factors/catalog.py       -> factors.catalog
pipeline/factors/filter_rank.py   -> factors.filter_rank
pipeline/factor_correlation.py    -> factors.correlation
pipeline/factor_scoring.py        -> factors.scoring
```

- [ ] **Step 5: Update factor package exports**

Update `factors/__init__.py`:

```python
"""Factor calculators, registries, scoring, and factor utilities."""

from .base import FactorDataError

__all__ = ["FactorDataError"]
```

If `factors/base.py` does not expose `FactorDataError`, export the base exception or data container name used by that file and update the package import test to match.

- [ ] **Step 6: Update imports in moved factor modules**

Run:

```bash
rg -n "pipeline\\.factors|pipeline\\.factor_correlation|pipeline\\.factor_scoring|pipeline\\.market_data|pipeline\\.strategies|pipeline\\.signals" factors pipeline tests scripts
```

Replace production imports in moved modules with top-level imports:

```python
from factors.alpha101 import ...
from factors.gtja191 import ...
from factors.catalog import ...
from factors.filter_rank import ...
from market.data import ...
from signals.schema import ...
from signals.factor_adapters import ...
from strategies.selector import ...
```

Leave old imports only inside compatibility wrappers.

- [ ] **Step 7: Run factor test group**

Run:

```bash
python -m unittest \
  tests.test_alpha101 \
  tests.test_gtja191 \
  tests.test_brick_factor \
  tests.test_factor_filter_rank \
  tests.test_factor_correlation \
  tests.test_factor_scoring
```

Expected: PASS.

- [ ] **Step 8: Commit factor migration**

Run:

```bash
git add factors pipeline/factors pipeline/factor_correlation.py pipeline/factor_scoring.py tests/test_alpha101.py tests/test_gtja191.py tests/test_brick_factor.py tests/test_factor_filter_rank.py tests/test_factor_correlation.py tests/test_factor_scoring.py
git commit -m "refactor: move factor research modules"
```

## Task 6: Move Factor Testers and Batch Reports

**Files:**
- Move: `pipeline/factor_tester.py` -> `reports/factor_tester.py`
- Move: `pipeline/alpha101_batch.py` -> `reports/alpha101_batch.py`
- Move: `pipeline/gtja191_batch.py` -> `reports/gtja191_batch.py`
- Modify: `pipeline/factor_tester.py`
- Modify: `pipeline/alpha101_batch.py`
- Modify: `pipeline/gtja191_batch.py`
- Modify: `scripts/test_factor.py`
- Modify: `scripts/test_alpha101_batch.py`
- Modify: `scripts/test_gtja191_batch.py`
- Test: `tests/test_factor_tester.py`
- Test: `tests/test_alpha101_batch.py`
- Test: `tests/test_gtja191_batch.py`

- [ ] **Step 1: Add top-level report import tests**

Append to `tests/test_factor_tester.py`:

```python
class TopLevelFactorTesterImportTests(unittest.TestCase):
    def test_factor_tester_imports_from_reports_package(self):
        import reports.factor_tester as factor_tester

        self.assertTrue(hasattr(factor_tester, "FactorTester"))
```

Append to `tests/test_alpha101_batch.py`:

```python
class TopLevelAlpha101BatchImportTests(unittest.TestCase):
    def test_alpha101_batch_imports_from_reports_package(self):
        import reports.alpha101_batch as alpha101_batch

        self.assertTrue(hasattr(alpha101_batch, "run_alpha101_batch"))
```

Append to `tests/test_gtja191_batch.py`:

```python
class TopLevelGtja191BatchImportTests(unittest.TestCase):
    def test_gtja191_batch_imports_from_reports_package(self):
        import reports.gtja191_batch as gtja191_batch

        self.assertTrue(hasattr(gtja191_batch, "run_gtja191_batch"))
```

If a batch runner uses a different public function name, assert that exact name and keep scripts importing it.

- [ ] **Step 2: Run report tests and verify top-level imports fail**

Run:

```bash
python -m unittest tests.test_factor_tester tests.test_alpha101_batch tests.test_gtja191_batch
```

Expected: FAIL on missing `reports.factor_tester` or batch modules.

- [ ] **Step 3: Move report modules**

Run:

```bash
git mv pipeline/factor_tester.py reports/factor_tester.py
git mv pipeline/alpha101_batch.py reports/alpha101_batch.py
git mv pipeline/gtja191_batch.py reports/gtja191_batch.py
```

- [ ] **Step 4: Add report wrappers**

Create `pipeline/factor_tester.py`:

```python
"""Compatibility wrapper for factor report testing."""

from reports.factor_tester import *  # noqa: F401,F403
```

Create `pipeline/alpha101_batch.py`:

```python
"""Compatibility wrapper for Alpha101 batch reports."""

from reports.alpha101_batch import *  # noqa: F401,F403
```

Create `pipeline/gtja191_batch.py`:

```python
"""Compatibility wrapper for GTJA191 batch reports."""

from reports.gtja191_batch import *  # noqa: F401,F403
```

- [ ] **Step 5: Update scripts to use report modules**

In `scripts/test_factor.py`, replace factor tester imports with:

```python
from reports.factor_tester import FactorTester, FactorTesterConfig
```

In `scripts/test_alpha101_batch.py`, replace batch imports with:

```python
from reports.alpha101_batch import main
```

In `scripts/test_gtja191_batch.py`, replace batch imports with:

```python
from reports.gtja191_batch import main
```

If the scripts call a runner function rather than `main`, import the exact runner used by the current script and remove only stale `pipeline` imports.

- [ ] **Step 6: Run report and script smoke tests**

Run:

```bash
python -m unittest tests.test_factor_tester tests.test_alpha101_batch tests.test_gtja191_batch
python scripts/test_factor.py --list-factors
python scripts/test_alpha101_batch.py --help
python scripts/test_gtja191_batch.py --help
```

Expected: tests PASS. Scripts print factor lists or help without import errors.

- [ ] **Step 7: Commit report migration**

Run:

```bash
git add reports pipeline/factor_tester.py pipeline/alpha101_batch.py pipeline/gtja191_batch.py scripts/test_factor.py scripts/test_alpha101_batch.py scripts/test_gtja191_batch.py tests/test_factor_tester.py tests/test_alpha101_batch.py tests/test_gtja191_batch.py
git commit -m "refactor: move factor report runners"
```

## Task 7: Move Backtest Layer

**Files:**
- Move: `pipeline/portfolio_backtest.py` -> `backtest/portfolio.py`
- Move: `pipeline/factor_portfolio_backtest.py` -> `backtest/factor_portfolio.py`
- Create: `backtest/transaction_cost.py`
- Create: `backtest/performance.py`
- Create: `backtest/benchmark_compare.py`
- Modify: `pipeline/portfolio_backtest.py`
- Modify: `pipeline/factor_portfolio_backtest.py`
- Test: `tests/test_portfolio_backtest.py`
- Test: `tests/test_factor_portfolio_backtest.py`

- [ ] **Step 1: Add top-level backtest import tests**

Append to `tests/test_portfolio_backtest.py`:

```python
class TopLevelPortfolioBacktestImportTests(unittest.TestCase):
    def test_portfolio_backtest_imports_from_backtest_package(self):
        import backtest.portfolio as portfolio

        self.assertTrue(hasattr(portfolio, "run_portfolio_backtest"))
```

Append to `tests/test_factor_portfolio_backtest.py`:

```python
class TopLevelFactorPortfolioImportTests(unittest.TestCase):
    def test_factor_portfolio_imports_from_backtest_package(self):
        import backtest.factor_portfolio as factor_portfolio

        self.assertTrue(hasattr(factor_portfolio, "run_filter_rank_portfolio_backtest"))
```

- [ ] **Step 2: Run backtest tests and verify new imports fail**

Run:

```bash
python -m unittest tests.test_portfolio_backtest tests.test_factor_portfolio_backtest
```

Expected: FAIL on missing top-level backtest modules.

- [ ] **Step 3: Move backtest modules**

Run:

```bash
git mv pipeline/portfolio_backtest.py backtest/portfolio.py
git mv pipeline/factor_portfolio_backtest.py backtest/factor_portfolio.py
```

- [ ] **Step 4: Add backtest wrappers**

Create `pipeline/portfolio_backtest.py`:

```python
"""Compatibility wrapper for realistic portfolio backtests."""

from backtest.portfolio import *  # noqa: F401,F403
```

Create `pipeline/factor_portfolio_backtest.py`:

```python
"""Compatibility wrapper for factor portfolio backtests."""

from backtest.factor_portfolio import *  # noqa: F401,F403
```

- [ ] **Step 5: Create focused backtest helper modules**

Create `backtest/transaction_cost.py`:

```python
"""Transaction cost constants and helpers for A-share backtests."""

from __future__ import annotations


def calculate_buy_cost(notional: float, commission_rate: float, min_commission: float) -> float:
    commission = max(notional * commission_rate, min_commission) if notional > 0 else 0.0
    return float(commission)


def calculate_sell_cost(
    notional: float,
    commission_rate: float,
    min_commission: float,
    stamp_tax_rate: float,
    transfer_fee_rate: float,
) -> float:
    if notional <= 0:
        return 0.0
    commission = max(notional * commission_rate, min_commission)
    stamp_tax = notional * stamp_tax_rate
    transfer_fee = notional * transfer_fee_rate
    return float(commission + stamp_tax + transfer_fee)
```

Create `backtest/performance.py`:

```python
"""Performance metrics for portfolio equity curves."""

from __future__ import annotations

import math
from typing import Sequence


def annualized_return(total_return: float, trading_days: int) -> float | None:
    if trading_days <= 0:
        return None
    return float((1.0 + total_return) ** (252.0 / trading_days) - 1.0)


def max_drawdown(equity_values: Sequence[float]) -> float:
    peak = None
    worst = 0.0
    for value in equity_values:
        current = float(value)
        peak = current if peak is None else max(peak, current)
        if peak > 0:
            worst = min(worst, current / peak - 1.0)
    return float(worst)


def sharpe_ratio(daily_returns: Sequence[float]) -> float | None:
    values = [float(v) for v in daily_returns]
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    if variance <= 0:
        return None
    return float(mean / math.sqrt(variance) * math.sqrt(252.0))
```

Create `backtest/benchmark_compare.py`:

```python
"""Benchmark comparison helpers for report generation."""

from __future__ import annotations

import pandas as pd


def align_portfolio_and_benchmark(
    portfolio: pd.DataFrame,
    benchmark: pd.DataFrame,
    date_col: str = "date",
) -> pd.DataFrame:
    left = portfolio.copy()
    right = benchmark.copy()
    left[date_col] = pd.to_datetime(left[date_col])
    right[date_col] = pd.to_datetime(right[date_col])
    return left.merge(right, on=date_col, how="inner", suffixes=("_portfolio", "_benchmark"))
```

These helpers are added as stable homes first. Move duplicated logic from `backtest/portfolio.py` into them only when focused tests prove identical output.

- [ ] **Step 6: Add helper tests**

Create `tests/test_backtest_helpers.py`:

```python
import unittest

import pandas as pd

from backtest.benchmark_compare import align_portfolio_and_benchmark
from backtest.performance import annualized_return, max_drawdown, sharpe_ratio
from backtest.transaction_cost import calculate_buy_cost, calculate_sell_cost


class BacktestHelperTests(unittest.TestCase):
    def test_transaction_cost_direction(self):
        self.assertEqual(calculate_buy_cost(0, 0.0003, 5), 0.0)
        self.assertEqual(calculate_buy_cost(10000, 0.0003, 5), 5.0)
        self.assertAlmostEqual(calculate_sell_cost(10000, 0.0003, 5, 0.001, 0.00001), 15.1)

    def test_performance_helpers(self):
        self.assertAlmostEqual(max_drawdown([100.0, 120.0, 90.0]), -0.25)
        self.assertIsNotNone(annualized_return(0.1, 100))
        self.assertIsNotNone(sharpe_ratio([0.01, -0.005, 0.002]))

    def test_benchmark_alignment(self):
        portfolio = pd.DataFrame({"date": ["2026-01-01", "2026-01-02"], "ret": [0.01, 0.02]})
        benchmark = pd.DataFrame({"date": ["2026-01-02"], "ret": [0.015]})
        result = align_portfolio_and_benchmark(portfolio, benchmark)
        self.assertEqual(len(result), 1)
        self.assertEqual(str(result.loc[0, "date"].date()), "2026-01-02")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 7: Run backtest tests**

Run:

```bash
python -m unittest tests.test_portfolio_backtest tests.test_factor_portfolio_backtest tests.test_backtest_helpers
```

Expected: PASS.

- [ ] **Step 8: Commit backtest migration**

Run:

```bash
git add backtest pipeline/portfolio_backtest.py pipeline/factor_portfolio_backtest.py tests/test_portfolio_backtest.py tests/test_factor_portfolio_backtest.py tests/test_backtest_helpers.py
git commit -m "refactor: move backtest layer"
```

## Task 8: Move Remaining Report Modules

**Files:**
- Move: `pipeline/signal_returns.py` -> `reports/signal_returns.py`
- Move: `pipeline/research_report.py` -> `reports/research_report.py`
- Modify: `pipeline/signal_returns.py`
- Modify: `pipeline/research_report.py`
- Modify: `reports/__init__.py`
- Test: `tests/test_signal_returns.py`
- Test: `tests/test_research_report.py`

- [ ] **Step 1: Add top-level report import tests**

Append to `tests/test_signal_returns.py`:

```python
class TopLevelSignalReturnsImportTests(unittest.TestCase):
    def test_signal_returns_imports_from_reports_package(self):
        import reports.signal_returns as signal_returns

        self.assertTrue(hasattr(signal_returns, "run_signal_returns"))
```

Append to `tests/test_research_report.py`:

```python
class TopLevelResearchReportImportTests(unittest.TestCase):
    def test_research_report_imports_from_reports_package(self):
        import reports.research_report as research_report

        self.assertTrue(hasattr(research_report, "run_research_report"))
```

- [ ] **Step 2: Run report tests and verify new imports fail**

Run:

```bash
python -m unittest tests.test_signal_returns tests.test_research_report
```

Expected: FAIL on missing top-level report modules.

- [ ] **Step 3: Move report modules**

Run:

```bash
git mv pipeline/signal_returns.py reports/signal_returns.py
git mv pipeline/research_report.py reports/research_report.py
```

- [ ] **Step 4: Add report wrappers**

Create `pipeline/signal_returns.py`:

```python
"""Compatibility wrapper for signal return reports."""

from reports.signal_returns import *  # noqa: F401,F403
```

Create `pipeline/research_report.py`:

```python
"""Compatibility wrapper for combined research reports."""

from reports.research_report import *  # noqa: F401,F403
```

- [ ] **Step 5: Run report tests**

Run:

```bash
python -m unittest tests.test_signal_returns tests.test_research_report
```

Expected: PASS.

- [ ] **Step 6: Commit remaining report migration**

Run:

```bash
git add reports pipeline/signal_returns.py pipeline/research_report.py tests/test_signal_returns.py tests/test_research_report.py
git commit -m "refactor: move signal and research reports"
```

## Task 9: Add Labels Package

**Files:**
- Create: `labels/make_forward_return.py`
- Create: `tests/test_labels.py`

- [ ] **Step 1: Write forward-return label tests**

Create `tests/test_labels.py`:

```python
import unittest

import pandas as pd

from labels.make_forward_return import make_forward_returns


class ForwardReturnLabelTests(unittest.TestCase):
    def test_forward_returns_are_future_targets(self):
        prices = pd.DataFrame(
            {
                "date": ["2026-01-01", "2026-01-02", "2026-01-03"],
                "symbol": ["000001", "000001", "000001"],
                "close": [10.0, 11.0, 12.1],
            }
        )

        result = make_forward_returns(prices, windows=(1, 2))

        self.assertEqual(result.loc[0, "symbol"], "000001")
        self.assertAlmostEqual(result.loc[0, "forward_return_1d"], 0.1)
        self.assertAlmostEqual(result.loc[0, "forward_return_2d"], 0.21)
        self.assertTrue(pd.isna(result.loc[2, "forward_return_1d"]))

    def test_duplicate_date_symbol_fails(self):
        prices = pd.DataFrame(
            {
                "date": ["2026-01-01", "2026-01-01"],
                "symbol": ["000001", "000001"],
                "close": [10.0, 11.0],
            }
        )

        with self.assertRaisesRegex(ValueError, "duplicate"):
            make_forward_returns(prices)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run labels test and verify it fails**

Run:

```bash
python -m unittest tests.test_labels
```

Expected: FAIL with `ModuleNotFoundError` or missing `make_forward_returns`.

- [ ] **Step 3: Implement forward-return labels**

Create `labels/make_forward_return.py`:

```python
"""Forward-return label generation for ML training and factor reports."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def make_forward_returns(
    prices: pd.DataFrame,
    windows: Iterable[int] = (1, 5, 10, 20),
    date_col: str = "date",
    symbol_col: str = "symbol",
    close_col: str = "close",
) -> pd.DataFrame:
    required = {date_col, symbol_col, close_col}
    missing = sorted(required.difference(prices.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    frame = prices[[date_col, symbol_col, close_col]].copy()
    frame[symbol_col] = frame[symbol_col].astype(str).str.zfill(6)
    frame[date_col] = pd.to_datetime(frame[date_col])
    frame = frame.sort_values([symbol_col, date_col]).reset_index(drop=True)

    if frame.duplicated([date_col, symbol_col]).any():
        raise ValueError("duplicate date/symbol rows are not allowed")

    grouped = frame.groupby(symbol_col, sort=False)[close_col]
    for window in tuple(windows):
        if int(window) <= 0:
            raise ValueError("forward return windows must be positive")
        future = grouped.shift(-int(window))
        frame[f"forward_return_{int(window)}d"] = future / frame[close_col] - 1.0

    frame[date_col] = frame[date_col].dt.strftime("%Y-%m-%d")
    return frame.drop(columns=[close_col])
```

- [ ] **Step 4: Run labels test**

Run:

```bash
python -m unittest tests.test_labels
```

Expected: PASS.

- [ ] **Step 5: Commit labels package**

Run:

```bash
git add labels/make_forward_return.py tests/test_labels.py
git commit -m "feat: add forward return labels"
```

## Task 10: Add Model Interfaces

**Files:**
- Create: `models/base.py`
- Create: `models/linear_ridge.py`
- Create: `models/elasticnet.py`
- Create: `models/lightgbm_model.py`
- Create: `models/mlp_torch.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: Write model interface tests**

Create `tests/test_models.py`:

```python
import unittest

import numpy as np
import pandas as pd

from models.elasticnet import ElasticNetModel
from models.linear_ridge import RidgeModel


class ModelInterfaceTests(unittest.TestCase):
    def test_ridge_fit_predict(self):
        x = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [0.0, 1.0, 0.0, 1.0]})
        y = pd.Series([0.1, 0.2, 0.3, 0.4])
        model = RidgeModel(alpha=1.0)

        model.fit(x, y)
        predictions = model.predict(x)

        self.assertEqual(len(predictions), 4)
        self.assertTrue(np.isfinite(predictions).all())

    def test_elasticnet_fit_predict(self):
        x = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [0.0, 1.0, 0.0, 1.0]})
        y = pd.Series([0.1, 0.2, 0.3, 0.4])
        model = ElasticNetModel(alpha=0.1, l1_ratio=0.5)

        model.fit(x, y)
        predictions = model.predict(x)

        self.assertEqual(len(predictions), 4)
        self.assertTrue(np.isfinite(predictions).all())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run model tests and verify they fail**

Run:

```bash
python -m unittest tests.test_models
```

Expected: FAIL with missing model modules.

- [ ] **Step 3: Implement base protocol**

Create `models/base.py`:

```python
"""Shared model interface for supervised stock-score models."""

from __future__ import annotations

from typing import Protocol

import pandas as pd


class ScoreModel(Protocol):
    def fit(self, features: pd.DataFrame, target: pd.Series) -> "ScoreModel":
        ...

    def predict(self, features: pd.DataFrame) -> pd.Series:
        ...
```

- [ ] **Step 4: Implement Ridge and ElasticNet wrappers**

Create `models/linear_ridge.py`:

```python
"""Ridge regression score model."""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import Ridge


class RidgeModel:
    def __init__(self, alpha: float = 1.0) -> None:
        self.model = Ridge(alpha=alpha)

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "RidgeModel":
        self.model.fit(features, target)
        return self

    def predict(self, features: pd.DataFrame) -> pd.Series:
        values = self.model.predict(features)
        return pd.Series(values, index=features.index, name="score")
```

Create `models/elasticnet.py`:

```python
"""ElasticNet regression score model."""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import ElasticNet


class ElasticNetModel:
    def __init__(self, alpha: float = 1.0, l1_ratio: float = 0.5) -> None:
        self.model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "ElasticNetModel":
        self.model.fit(features, target)
        return self

    def predict(self, features: pd.DataFrame) -> pd.Series:
        values = self.model.predict(features)
        return pd.Series(values, index=features.index, name="score")
```

- [ ] **Step 5: Implement optional dependency wrappers**

Create `models/lightgbm_model.py`:

```python
"""LightGBM score model wrapper."""

from __future__ import annotations

import pandas as pd


class LightGBMModel:
    def __init__(self, **params: object) -> None:
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise ImportError("LightGBMModel requires lightgbm to be installed") from exc
        self.model = lgb.LGBMRegressor(**params)

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "LightGBMModel":
        self.model.fit(features, target)
        return self

    def predict(self, features: pd.DataFrame) -> pd.Series:
        values = self.model.predict(features)
        return pd.Series(values, index=features.index, name="score")
```

Create `models/mlp_torch.py`:

```python
"""Minimal PyTorch MLP score model with explicit dependency check."""

from __future__ import annotations


class TorchMLPModel:
    def __init__(self, *args: object, **kwargs: object) -> None:
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            raise ImportError("TorchMLPModel requires torch to be installed") from exc
        raise NotImplementedError("TorchMLPModel training is not wired into the first restructure pass")
```

- [ ] **Step 6: Run model tests**

Run:

```bash
python -m unittest tests.test_models
```

Expected: PASS if scikit-learn is installed. If scikit-learn is missing, record the interpreter path and missing package; do not claim model tests passed.

- [ ] **Step 7: Commit model interfaces**

Run:

```bash
git add models tests/test_models.py
git commit -m "feat: add supervised model interfaces"
```

## Task 11: Add Training and Prediction Scores

**Files:**
- Create: `training/validation.py`
- Create: `training/predict_score.py`
- Create: `training/train_walk_forward.py`
- Create: `tests/test_training.py`

- [ ] **Step 1: Write training validation tests**

Create `tests/test_training.py`:

```python
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
```

- [ ] **Step 2: Run training tests and verify they fail**

Run:

```bash
python -m unittest tests.test_training
```

Expected: FAIL with missing training modules.

- [ ] **Step 3: Implement validation helpers**

Create `training/validation.py`:

```python
"""Walk-forward validation helpers for model training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import pandas as pd


@dataclass(frozen=True)
class WalkForwardWindow:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def validate_feature_label_frame(
    frame: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    date_col: str = "date",
    symbol_col: str = "symbol",
) -> pd.DataFrame:
    required = set(feature_cols) | {target_col, date_col, symbol_col}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    result = frame.copy()
    result[symbol_col] = result[symbol_col].astype(str).str.zfill(6)
    result[date_col] = pd.to_datetime(result[date_col])
    if result.duplicated([date_col, symbol_col]).any():
        raise ValueError("duplicate date/symbol rows are not allowed")
    if result[list(feature_cols) + [target_col]].isna().any().any():
        raise ValueError("features and target must not contain missing values")
    return result.sort_values([date_col, symbol_col]).reset_index(drop=True)


def build_walk_forward_windows(
    dates: Iterable[pd.Timestamp],
    train_size: int,
    test_size: int,
) -> list[WalkForwardWindow]:
    ordered = sorted(pd.Timestamp(date) for date in set(dates))
    if train_size <= 0 or test_size <= 0:
        raise ValueError("train_size and test_size must be positive")
    windows: list[WalkForwardWindow] = []
    start = 0
    while start + train_size + test_size <= len(ordered):
        train = ordered[start : start + train_size]
        test = ordered[start + train_size : start + train_size + test_size]
        windows.append(
            WalkForwardWindow(
                train_start=train[0],
                train_end=train[-1],
                test_start=test[0],
                test_end=test[-1],
            )
        )
        start += test_size
    return windows
```

- [ ] **Step 4: Implement prediction score conversion**

Create `training/predict_score.py`:

```python
"""Convert model prediction scores into unified signals."""

from __future__ import annotations

import pandas as pd


def scores_to_signals(
    scores: pd.DataFrame,
    source: str,
    date_col: str = "date",
    symbol_col: str = "symbol",
    score_col: str = "score",
) -> pd.DataFrame:
    required = {date_col, symbol_col, score_col}
    missing = sorted(required.difference(scores.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    frame = scores[[date_col, symbol_col, score_col]].copy()
    frame[date_col] = pd.to_datetime(frame[date_col]).dt.strftime("%Y-%m-%d")
    frame[symbol_col] = frame[symbol_col].astype(str).str.zfill(6)
    frame["signal_type"] = "buy"
    frame["source"] = f"model_{source}"
    frame["weight"] = 0.0
    frame["metadata"] = "{}"
    return frame.rename(columns={score_col: "score"})[
        [date_col, symbol_col, "signal_type", "source", "score", "weight", "metadata"]
    ]
```

- [ ] **Step 5: Add train module entrypoint**

Create `training/train_walk_forward.py`:

```python
"""Walk-forward model training entrypoint."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a walk-forward stock score model")
    parser.add_argument("--features", required=True, help="Feature CSV path")
    parser.add_argument("--labels", required=True, help="Label CSV path")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--model", default="ridge", choices=("ridge", "elasticnet", "lightgbm"))
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(
        "walk-forward training CLI is scaffolded; wire data loading in the implementation pass "
        f"for model={args.model}, features={args.features}, labels={args.labels}, output={args.output}"
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run training tests**

Run:

```bash
python -m unittest tests.test_training
python -m training.train_walk_forward --help
```

Expected: tests PASS and help prints usage.

- [ ] **Step 7: Commit training package**

Run:

```bash
git add training tests/test_training.py
git commit -m "feat: add walk-forward training scaffolding"
```

## Task 12: CLI Routing and Pipeline Compatibility

**Files:**
- Create: `scripts/quant_cli.py`
- Modify: `pipeline/cli.py`
- Modify: `run_all.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Add CLI import tests**

Append to `tests/test_cli.py`:

```python
class TopLevelCliImportTests(unittest.TestCase):
    def test_quant_cli_imports(self):
        import scripts.quant_cli as quant_cli

        parser = quant_cli.build_parser()
        command_names = {action.dest for action in parser._actions}
        self.assertIn("command", command_names)
```

- [ ] **Step 2: Run CLI tests and verify new CLI import fails**

Run:

```bash
python -m unittest tests.test_cli
```

Expected: FAIL with missing `scripts.quant_cli`.

- [ ] **Step 3: Create new CLI dispatcher**

Create `scripts/quant_cli.py`:

```python
"""Top-level CLI dispatcher for RQuant research workflows."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RQuant research CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("fetch-data", help="Fetch and update market data")
    sub.add_parser("preselect", help="Run custom strategy preselection")
    sub.add_parser("signal-returns", help="Evaluate signal forward returns")
    sub.add_parser("portfolio-backtest", help="Run realistic portfolio backtest")
    sub.add_parser("research-report", help="Build combined research report")
    sub.add_parser("factor-test", help="Run single factor diagnostics")
    sub.add_parser("factor-batch-alpha101", help="Run Alpha101 batch diagnostics")
    sub.add_parser("factor-batch-gtja191", help="Run GTJA191 batch diagnostics")
    sub.add_parser("factor-score", help="Score factor batch outputs")
    sub.add_parser("factor-select", help="Create factor ranking signals")
    sub.add_parser("factor-backtest", help="Backtest factor ranking signals")
    sub.add_parser("make-labels", help="Create forward-return labels")
    sub.add_parser("train-model", help="Train walk-forward model")
    sub.add_parser("predict-score", help="Create model score signals")
    sub.add_parser("model-backtest", help="Backtest model score signals")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(
        f"command '{args.command}' is registered; wire command dispatch after module migration validation"
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Keep old CLI importable**

Edit `pipeline/cli.py` imports so migrated modules use top-level paths:

```python
from reports.research_report import DEFAULT_REPORT_OUTPUT_DIR, run_research_report
```

In lazy loaders, use:

```python
from reports.signal_returns import run_signal_returns as runner
from backtest.portfolio import run_portfolio_backtest as runner
from backtest.factor_portfolio import run_filter_rank_portfolio_backtest as runner
```

Keep existing argument parsing behavior unchanged in this task.

- [ ] **Step 5: Update run_all imports**

Run:

```bash
rg -n "pipeline\\.|from (select_stock|pipeline_io|Selector)" run_all.py
```

Replace imports with top-level packages:

```python
from market.fetch_kline import main as fetch_kline_main
from strategies.preselect import run_preselect
```

Use exact public functions already consumed by `run_all.py`.

- [ ] **Step 6: Run CLI checks**

Run:

```bash
python -m unittest tests.test_cli
python -m pipeline.cli --help
python scripts/quant_cli.py --help
```

Expected: tests PASS. Both help commands print usage without import errors.

- [ ] **Step 7: Commit CLI routing**

Run:

```bash
git add scripts/quant_cli.py pipeline/cli.py run_all.py tests/test_cli.py
git commit -m "refactor: add top-level CLI routing"
```

## Task 13: Documentation and Agent Guide

**Files:**
- Modify: `README.md`
- Modify: `docs/architecture.md`
- Modify: `AGENTS.md`

- [ ] **Step 1: Update README directory section**

Replace the old `pipeline`-centered directory summary with:

```markdown
## 2. 目录说明

- [market](market)：行情抓取、清洗、股票池和可交易状态。
- [factors](factors)：Alpha101、GTJA191、BrickChart 派生因子、因子注册、评分和相关性。
- [labels](labels)：forward return 和机器学习标签。
- [models](models)：Ridge、ElasticNet、LightGBM、MLP 等模型封装。
- [training](training)：walk-forward 切分、验证和预测分数生成。
- [signals](signals)：统一信号结构和因子/模型/策略信号适配。
- [strategies](strategies)：B1、brick、mBDSR、BDSR/MACD/OBV 等自定义买点。
- [backtest](backtest)：组合构建、交易成本、绩效和基准比较。
- [reports](reports)：IC、分层收益、批处理、信号收益、组合回测和研究报告。
- [scripts](scripts)：可重复执行的命令行入口。
- [agent](agent)：LLM 评审逻辑（Gemini）。
- [dashboard](dashboard)：看盘界面与图表导出。
- [config](config)：抓取、初选、因子生命周期、模型和 Gemini 复评配置。
- [data](data)：本地行情、标签和研究输出。
```

- [ ] **Step 2: Update architecture map**

In `docs/architecture.md`, replace the current mapping block with:

```markdown
## Current Mapping

- DataManager: `market/fetch_kline.py`, `market/data.py`, `market/io.py`
- UniverseBuilder: `market/data.py::build_stock_pool_by_date`
- Factor research: `factors/`, `reports/factor_tester.py`, `reports/alpha101_batch.py`, `reports/gtja191_batch.py`
- Label generation: `labels/make_forward_return.py`
- Model research: `models/`, `training/`
- Factor and model signals: `signals/`
- Custom strategy signals: `strategies/`
- BacktestEngine: `backtest/portfolio.py`, `backtest/factor_portfolio.py`
- Reporter: `reports/`
- Compatibility wrappers: `pipeline/`
```

- [ ] **Step 3: Update AGENTS architecture boundaries**

In `AGENTS.md`, replace `pipeline/`-specific boundary bullets with:

```markdown
1. **因子研究**
   - 因子计算、检验与排名信号放在 `factors/`、`reports/factor_tester.py` 和 `signals/`。
   - 因子 IC、Rank IC、分组收益等统计不得与自定义买点的评价逻辑混合。
2. **自定义买入策略**
   - B1、brick、mBDSR、BDSR/MACD/OBV 共振及新的明确买点规则放在 `strategies/`。
   - 旧 `pipeline/` 路径仅作为兼容包装，不能成为新业务逻辑的主要位置。
3. **机器学习研究**
   - forward return 与训练标签放在 `labels/`。
   - Ridge、ElasticNet、LightGBM、MLP 等模型放在 `models/`。
   - walk-forward、验证和预测分数放在 `training/`。
   - ML 分数只能通过 `signals/` 进入组合回测，不得反向污染因子计算或自定义买点。
```

- [ ] **Step 4: Run doc consistency searches**

Run:

```bash
rg -n "pipeline/factors|pipeline/strategies|pipeline/signals|pipeline/factor_tester|pipeline/portfolio_backtest|pipeline/research_report" README.md docs/architecture.md AGENTS.md
```

Expected: any remaining references are explicitly described as compatibility wrappers or historical paths.

- [ ] **Step 5: Commit docs**

Run:

```bash
git add README.md docs/architecture.md AGENTS.md
git commit -m "docs: document top-level research layout"
```

## Task 14: Full Validation and Compatibility Decision

**Files:**
- Modify only if validation reveals import drift in migrated modules.

- [ ] **Step 1: Run targeted validation**

Run:

```bash
python -m pipeline.cli --help
python scripts/quant_cli.py --help
python -m unittest tests.test_package_layout
python -m unittest tests.test_signal_schema
python -m unittest tests.test_factor_tester
python -m unittest tests.test_portfolio_backtest
python -m unittest tests.test_training
```

Expected: all listed tests PASS, unless dependency collection fails. If collection fails, record interpreter path and exact missing package.

- [ ] **Step 2: Run full unittest suite**

Run:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

Expected: PASS. If tests fail, fix only migration-related import or behavior regressions. Do not change factor formulas, strategy rules, fee semantics, or generated output schemas to make tests pass.

- [ ] **Step 3: Search for stale production imports**

Run:

```bash
rg -n "from pipeline\\.|import pipeline\\.|from Selector|import Selector|from select_stock|import select_stock" --glob '*.py'
```

Expected: stale imports appear only in compatibility wrappers or tests explicitly covering compatibility.

- [ ] **Step 4: Decide whether to keep wrappers**

If full validation passes and README documents new commands, keep `pipeline/` wrappers for one release cycle. Do not delete wrappers in the same migration unless the user explicitly asks for no compatibility shims.

- [ ] **Step 5: Commit validation fixes**

Run only if Step 1 or Step 2 required fixes:

```bash
git add .
git commit -m "fix: resolve restructure validation issues"
```

Do not use `git add .` if unrelated user files changed during implementation. In that case, stage only the migration files reported by `git status --short`.

## Self-Review Notes

- Spec coverage: Tasks 1-4 cover package layout, signals, strategies, and market data. Tasks 5-8 cover factors, reports, and backtest. Tasks 9-11 cover labels, models, and training. Tasks 12-14 cover CLI, docs, compatibility, and validation.
- Boundary coverage: factor research, custom strategies, ML labels, signal schema, tradable backtest constraints, six-digit symbols, and old wrapper policy are all explicitly tested or documented.
- Execution risk: current worktree has existing staged changes. Each task must check `git status --short` before staging and must not reset or overwrite unrelated staged files.
