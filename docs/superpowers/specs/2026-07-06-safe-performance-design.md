# Safe Performance Optimization Design

## Goal

Keep factor and custom-strategy behavior unchanged while making the main research
paths faster and aligning `factor-backtest` defaults to the approved values:
80% factor filter, 500 selected stocks, and CNY 10,000,000 initial cash.

## Scope

- Align CLI tests and documentation with the approved `factor-backtest` defaults.
- Replace repeated full-calendar scans in the realistic portfolio loop with a
  trading-date index.
- Replace repeated DataFrame filtering during position valuation with indexed
  lookup of the latest close on or before the valuation date.
- Vectorize daily stock-pool construction while preserving filtering, ranking,
  stable tie order, board exclusions, and six-digit symbols.
- Provide an opt-in interface for reusing base market-data preparation and the
  turnover pool across multiple selector strategies in `signal-returns`.
  Existing callers keep independent per-strategy preparation by default.
- Avoid rebuilding the Alpha101 leaderboard from every report after every factor;
  preserve atomic checkpoints and resumability.

## Compatibility Rules

- No factor formula, selector rule, signal date, trade constraint, fee, output
  column, or output path changes.
- `factor-select` keeps its existing 50% / top-10 defaults; only
  `factor-backtest` uses 80% / 500 / CNY 10,000,000.
- Pool ordering remains turnover descending with stable input order for ties.
- Multi-strategy reuse is enabled only through an explicit function argument or
  CLI flag; omitting it preserves current execution.
- Alpha101 batch checkpoints remain atomic and `leaderboard.csv` remains current
  after every completed, skipped, or failed factor.
- Existing uncommitted files are preserved; only task-related lines are edited.

## Verification

Each optimization gets a focused regression test before implementation. Run the
targeted tests after each change, then run the full suite with:

```bash
PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR=/tmp/rquant-numba \
  /opt/miniconda3/envs/stocktrade/bin/python -m unittest discover -s tests -p 'test_*.py'
```
