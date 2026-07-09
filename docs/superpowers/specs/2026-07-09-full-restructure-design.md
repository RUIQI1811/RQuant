# Full Project Restructure Design

## Goal

Restructure RQuant from a mostly `pipeline/`-centered layout into a
top-level quantitative research layout that matches the research flow:
market data, factors, labels, models, training, signals, strategies, backtest,
and reports.

The restructure must make the codebase easier to extend for machine-learning
research while preserving the project's existing boundaries:

- factor research stays separate from custom buy strategies;
- statistical diagnostics stay separate from tradable portfolio P&L;
- all tradable paths preserve point-in-time data, `shift(1)` factor timing,
  A-share trading constraints, fees, cash, lots, and blocked-order accounting;
- `symbol` remains a six-character string everywhere.

## Target Layout

```text
agent/
backtest/
config/
dashboard/
data/
  raw/
  processed/
  labels/
docs/
factors/
labels/
market/
models/
reports/
scripts/
signals/
strategies/
tests/
training/
```

Responsibilities:

- `market/`: Tushare fetching, raw CSV loading, daily bar normalization, stock
  pool construction, tradability flags, suspension detection, limit-up and
  limit-down state.
- `factors/`: factor calculators, factor registries, Alpha101, GTJA191,
  BrickChart-derived research factors, factor preprocessing, neutralization,
  lifecycle catalog, factor correlation, and factor-to-signal adapters.
- `labels/`: forward-return labels and future ML classification or regression
  labels. Labels are evaluation targets only; factor calculation must not
  depend on them.
- `models/`: model definitions for Ridge, ElasticNet, LightGBM, and PyTorch MLP.
  The first migration creates stable module boundaries and lightweight
  interfaces; it does not need to train production models in the same step.
- `training/`: walk-forward splits, validation, feature/label assembly,
  prediction score generation, model artifact metadata, and leakage checks.
- `signals/`: the shared signal schema and converters from factor scores,
  ML model scores, and custom strategy candidates.
- `strategies/`: explicit custom buy rules such as B1, brick, mBDSR, and
  BDSR/MACD/OBV resonance. These modules do not calculate IC or factor groups.
- `backtest/`: realistic portfolio construction, transaction costs, tradable
  staggered cohorts, performance metrics, and benchmark comparison.
- `reports/`: IC reports, layer-return reports, factor batch reports, backtest
  reports, and combined research reports.
- `scripts/`: thin CLI wrappers for repeatable research commands. Business
  logic belongs in packages, not scripts.

## Current-to-Target Mapping

```text
pipeline/fetch_kline.py                  -> market/fetch_kline.py
pipeline/market_data.py                  -> market/data.py
pipeline/select_stock.py                 -> strategies/preselect.py
pipeline/Selector.py                     -> strategies/selector.py
pipeline/strategies/*.py                 -> strategies/*.py
pipeline/signals/schema.py               -> signals/schema.py
pipeline/factors/*.py                    -> factors/*.py
pipeline/factor_tester.py                -> reports/factor_tester.py
pipeline/alpha101_batch.py               -> reports/alpha101_batch.py
pipeline/gtja191_batch.py                -> reports/gtja191_batch.py
pipeline/factor_correlation.py           -> factors/correlation.py
pipeline/factor_scoring.py               -> factors/scoring.py
pipeline/factor_portfolio_backtest.py    -> backtest/factor_portfolio.py
pipeline/portfolio_backtest.py           -> backtest/portfolio.py
pipeline/signal_returns.py               -> reports/signal_returns.py
pipeline/research_report.py              -> reports/research_report.py
```

New ML modules:

```text
labels/make_forward_return.py
models/linear_ridge.py
models/elasticnet.py
models/lightgbm_model.py
models/mlp_torch.py
training/train_walk_forward.py
training/validation.py
training/predict_score.py
```

## CLI Design

The public CLI remains command-line first and file-output first. The final
entrypoint should be a thin module under `scripts/` or a small root CLI package
that dispatches into top-level packages.

Planned command groups:

```text
fetch-data             -> market
preselect              -> strategies -> signals
signal-returns         -> reports
portfolio-backtest     -> backtest
research-report        -> reports
factor-test            -> factors + reports
factor-batch-alpha101  -> factors + reports
factor-batch-gtja191   -> factors + reports
factor-score           -> factors
factor-select          -> factors -> signals
factor-backtest        -> factors -> signals -> backtest
make-labels            -> labels
train-model            -> training + models
predict-score          -> training -> signals
model-backtest         -> training -> signals -> backtest
```

During migration, old `python -m pipeline.cli ...` commands may be kept as thin
wrappers if that reduces risk. If compatibility is intentionally dropped in a
specific cleanup phase, the deletion must happen only after the new commands,
tests, and docs are in place.

## Data Flow

Factor research:

```text
market/raw bars
-> market point-in-time universe
-> factors calculators
-> factors preprocessing and optional neutralization
-> reports IC / Rank IC / group returns / statistical NAV
-> signals factor ranking
-> backtest realistic tradable portfolio
```

Custom strategy research:

```text
market/raw bars
-> strategies explicit buy rules
-> signals candidate adapter
-> reports signal returns
-> backtest realistic tradable portfolio
-> reports research summary
```

Machine-learning research:

```text
market/raw bars
-> factors feature frame
-> labels forward-return targets
-> training walk-forward split
-> models fit and validate
-> training predict_score
-> signals model score adapter
-> backtest realistic tradable portfolio
-> reports model and portfolio summary
```

Labels and forward returns are targets for validation and training. They must not
enter factor calculation, stock-pool filtering, signal ranking, or tradable
execution before the point in time at which they would be observable.

## Migration Plan

The code migration should happen in phases so each phase leaves a runnable,
testable system.

1. Create the target packages with `__init__.py` files and move the lowest-risk
   shared interfaces first: `signals/`, `strategies/`, and `market/`.
2. Move factor modules and keep all factor-family public names stable:
   `momentum_Nd`, `alpha_001` through `alpha_101`, and `gtja_001` through
   `gtja_191`.
3. Move backtest and report modules after signal imports are stable.
4. Add the ML package skeleton with forward-return labels, model interfaces,
   walk-forward validation, and prediction-score output.
5. Update CLI wrappers and docs after each phase, then remove old `pipeline/`
   wrappers only if the new commands and tests cover the same workflows.

No phase should rename generated output columns without a migration reason.
Existing durable output roots stay stable unless a specific phase documents and
tests the new path.

## Error Handling

- Missing market columns should fail with a clear message naming the file,
  required column, and consuming module.
- Missing optional benchmark or style-factor inputs should mark the relevant
  factor as unavailable instead of fabricating current metadata.
- Empty signals, empty labels, insufficient factor coverage, and no-trade
  backtests should produce explicit empty outputs with counts and reasons.
- ML training must fail early when the feature/label join creates leakage,
  duplicate `(date, symbol)` rows, or unsorted walk-forward windows.
- External services remain integration-only. Unit tests must not call Tushare or
  Gemini.

## Testing Strategy

Every moved module needs import and behavior coverage before the old import path
is removed.

Focused tests:

- signal schema preserves `date, symbol, signal_type, source, score, weight,
  metadata`;
- six-digit symbols retain leading zeroes through market, factors, labels,
  signals, and backtest;
- factor values are lagged before IC, group returns, factor selection, and
  factor portfolio backtests;
- forward-return labels are available to reports and ML training but never to
  factor calculation or same-day signal selection;
- realistic portfolio constraints keep cash, lots, fees, T+1, suspension,
  limit-up buy blocks, and limit-down sell blocks;
- custom strategies still produce candidates through their own strategy path;
- factor batch runners still checkpoint and resume;
- CLI help loads without importing heavy optional dependencies.

Verification commands should remain unittest-based unless the project adopts a
new test runner:

```bash
python -m pipeline.cli --help
python -m unittest tests.test_cli
python -m unittest tests.test_factor_tester
python -m unittest tests.test_portfolio_backtest
python -m unittest discover -s tests -p 'test_*.py'
```

As migration advances, equivalent commands for the new CLI must be added to this
list before old commands are removed.

## Documentation Updates

The restructure changes module responsibility and public commands, so the same
implementation plan must update:

- `README.md`: practical workflows and copy-paste commands;
- `docs/architecture.md`: final package map and data flow;
- `AGENTS.md`: new package boundaries, while preserving the factor/custom
  strategy separation and verification rules.

The docs should state that this is a research and decision-support tool, not an
automatic trading system or profit guarantee.

## Out of Scope

- Live trading or broker integration.
- Rewriting factor formulas for performance unless tests prove identical output.
- Treating ML scores as guaranteed trading instructions.
- Rebuilding the dashboard UI.
- Deleting historical generated data under `data/` or `factor_report/`.
