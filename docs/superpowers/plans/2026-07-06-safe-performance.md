# Safe Performance Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align approved factor-backtest defaults and implement result-equivalent performance improvements.

**Architecture:** Keep the existing factor/custom-strategy split and public CLI/output schemas. Add small indexed/vectorized helpers inside existing modules, reuse existing base-preparation APIs, and cache Alpha101 leaderboard rows without changing checkpoint durability.

**Tech Stack:** Python 3.13, pandas, NumPy, unittest.

---

### Task 1: Align factor-backtest defaults

**Files:**
- Modify: `tests/test_cli.py`
- Modify: `pipeline/cli.py`
- Modify: `README.md`
- Modify: `docs/factor_strategies.md`

- [ ] Change the parser test to require `0.8`, `500`, and `10_000_000.0`.
- [ ] Run `python -m unittest tests.test_cli` and verify it fails on initial cash.
- [ ] Change only the factor-backtest initial-cash default to `10_000_000.0`.
- [ ] Update the two user documents to state the same defaults.
- [ ] Re-run `tests.test_cli` and verify it passes.

### Task 2: Index realistic portfolio dates and valuation

**Files:**
- Modify: `tests/test_portfolio_backtest.py`
- Modify: `backtest/portfolio.py`

- [ ] Add tests for a `_trading_bars_held` helper and a `_latest_close_on_or_before` helper, including missing calendar dates and dates before the first bar.
- [ ] Run the focused tests and verify the new helper imports fail.
- [ ] Implement the helpers using a date-to-position mapping and DatetimeIndex search.
- [ ] Use them in the realistic portfolio loop without changing order/trade output.
- [ ] Run `tests.test_portfolio_backtest` and verify it passes.

### Task 3: Vectorize stock-pool construction

**Files:**
- Modify: `tests/test_market_data.py`
- Modify: `market/data.py`

- [ ] Add a test for `build_stock_pool_frame` covering allowed symbols, board/tradeability/price/turnover filters, and stable tie order.
- [ ] Run the focused test and verify the new helper import fails.
- [ ] Implement a per-symbol filtered frame plus stable cross-sectional sort/group head.
- [ ] Make `build_stock_pool_by_date` convert the frame into the existing dictionary schema.
- [ ] Run `tests.test_market_data` and verify it passes.

### Task 4: Add opt-in multi-strategy base preparation reuse

**Files:**
- Modify: `tests/test_signal_returns.py`
- Modify: `tests/test_cli.py`
- Modify: `reports/signal_returns.py`
- Modify: `pipeline/cli.py`

- [ ] Add a parser test for an explicit `--reuse-base-preparation` flag and an integration-style mocked test asserting the default remains independent.
- [ ] Add a second test asserting the opt-in path prepares base data and the turnover pool once for two selectors while selector features run twice.
- [ ] Run the focused tests and verify the new function argument and CLI flag are missing.
- [ ] Add `reuse_base_preparation: bool = False`; keep the existing loop as the default and use `prepare_base_only` plus `apply_selector_features` only when true.
- [ ] Forward the CLI flag without changing existing command behavior.
- [ ] Run `tests.test_signal_returns` and verify output behavior remains green.

### Task 5: Cache Alpha101 leaderboard checkpoints

**Files:**
- Modify: `tests/test_alpha101_batch.py`
- Modify: `reports/alpha101_batch.py`

- [ ] Add a test that patches `build_leaderboard` and requires only one full disk rebuild during a two-factor run while still writing a leaderboard after each checkpoint.
- [ ] Run the test and verify current repeated rebuilding fails the assertion.
- [ ] Keep an in-memory leaderboard cache, update only the completed factor rows, and atomically write the combined sorted frame after each checkpoint.
- [ ] Preserve final `build_leaderboard` compatibility for standalone callers and resume runs.
- [ ] Run `tests.test_alpha101_batch` and verify it passes.

### Task 6: Documentation and full verification

**Files:**
- Modify: `README.md`

- [ ] Document the approved defaults and the opt-in base-reuse interface.
- [ ] Run `git diff --check` and correct whitespace only on touched lines.
- [ ] Run CLI help, all focused tests, and the full unittest suite in the stocktrade environment.
- [ ] Inspect `git diff --stat` and `git status --short` to confirm no unrelated files changed.
