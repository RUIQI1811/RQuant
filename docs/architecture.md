# RQuant Architecture

This project keeps three research paths and routes them into the same signal shape.

```text
DataManager
→ UniverseBuilder
→ Feature / Factor Layer
→ Signal Layer
   ├─ FactorSignalEngine      # single-factor, gate/rank, and rank ensembles
   ├─ ModelScoreEngine        # walk-forward model scores
   └─ StrategySignalEngine    # B1, brick, mBDSR, and BDSR/MACD/OBV rules
→ PortfolioConstructor
→ RiskManager
→ BacktestEngine
→ Reporter
```

## Current Mapping

- DataManager: `market/fetch_kline.py`, `market/data.py`, `market/io.py`
- Domain contracts: `domain/` owns cross-boundary values, signals, execution
  records, backtest results, workflow results, and artifact references.
- UniverseBuilder: `market/data.py::build_stock_pool_by_date`
- Factor research: `factors/`, `reports/factor_tester.py`, `reports/alpha101_batch.py`, `reports/gtja191_batch.py`, `reports/external_factor_batch.py`
- Label generation: `labels/make_forward_return.py`
- Model research: `models/`, `training/`
- Factor and model signals: `signals/`
- Custom strategy signals: `strategies/`
- BacktestEngine: `backtest/portfolio.py`, `backtest/factor_portfolio.py`
- Reporter: `reports/`
- System diagnostics: `reports/system_doctor.py` performs read-only dependency,
  configuration, secret-presence, and local market-data checks; `doctor` exposes it
  through the public CLI and only required failures produce a nonzero exit code.
- Public CLI: `scripts/quant_cli.py`; legacy fetch/factor scripts delegate to the same callable APIs
- Framework runtime: `rquant/` discovers the project root, initializes logging only
  for real command execution, records one atomic run manifest per invocation, and
  exposes `python -m rquant`, the installed `rquant` command, and read-only run lookup.
- Compatibility CLI: `scripts/quant_cli.py` retains the established parser and
  handler injection points while delegating executed scripts through the governed
  runtime. It remains compatible during migration but is no longer the preferred entry.
- Daily orchestration: `run_all.py` uses the public CLI for fetch and preselection,
  validates its 1-to-5 execution range, fails fast on subprocess or final-artifact
  errors, and keeps AI-reviewed candidates explicitly separate from trade advice.
- External review recovery: `agent/base_reviewer.py` fingerprints provider/model,
  prompt, date, symbol, and chart content before reusing per-symbol JSON. Writes are
  atomic, `run_manifest.json` is checkpointed after every symbol, and partial runs
  keep completed work but return nonzero.
- Point-in-time chart export: `dashboard/export_kline_charts.py` truncates every raw
  frame at candidate `pick_date`, requires an exact bar on that date, atomically
  writes each image, checkpoints signatures/hashes, and returns nonzero on partial
  export so future bars cannot leak into AI review.
- Market-fetch recovery: `market/fetch_kline.py` atomically checkpoints every symbol
  into `_fetch_manifest.json`. Resume accepts only the same date/output/universe
  signature, retries failed or pending symbols, and returns nonzero for partial runs.
- Point-in-time research context: `market/fetch_context.py` queries Tushare
  `daily_basic` once per open trading date, converts market-cap units to yuan, writes
  atomic date partitions, and blocks partial manifests from factor/ML consumers.
- Research-report consistency: `reports/research_report.py` requires valid core JSON,
  checks review dates/status plus signal/portfolio timing fields, fingerprints every
  input, and blocks inconsistent reports unless diagnostic override is explicit.
- Compatibility wrappers: legacy wrapper paths

## Factor Research Timing and NAV Boundaries

`reports/factor_research_pipeline.py` is a thin command-chain adapter exposed as
`factor-run-all`, not a fourth research implementation. It launches the governed
public `factor-batch`, `factor-correlation`, and `fit-multifactor --run-backtests`
commands in separate Python processes and stops at the first non-zero exit. Each
child therefore keeps its normal parser, `data/runs/<run-id>` audit record, resume
contract, and family-scoped output directory. The parent produces no research
manifest, summary, candidate table, factor metrics, model outputs, or portfolio
outputs of its own; its governed run record only lists child commands, run IDs,
exit codes, manifests, and output paths.

External classification validation belongs to the official `factor-batch`
boundary through `--require-classification`, where a missing-category template is
written before market-data loading. `factor-correlation` retains representatives
using same-horizon gross Sharpe, and `factor-run-all` passes its standard
`deduplicated_factors.csv` directly to `fit-multifactor`; factor-stage costs remain
reports rather than an ML gate. The ML command is fixed by the chain configuration
to three full calendar training years followed by one full test year and runs both
gross and cost-aware long-only portfolio backtests. `run_all.py` remains the
separate custom-buy daily orchestrator and does not import this factor chain.

`factor-batch` has explicit `core` and `full` report profiles. The default `core`
profile retains distribution, grouped returns, neutralized/cap/industry IC,
exposure, statistical long-short, and high-side tradable long-only results. It
does not execute market-regime IC, statistical TopN, low-side tradable portfolios,
tradable long-short, or per-date universe-detail stages. `full` preserves the
complete legacy report suite, while the single-factor `factor-test` entry remains
full by default. The selected profile is part of batch and per-factor manifests
and resume fingerprints, so outputs from different profiles cannot be reused.

Alpha101 batch research applies the lifecycle catalog before calculation. `active`
factors run first, `watch` factors remain in research but run and rank after active
factors, and `disabled` factors are omitted by default. This status changes research
scheduling and presentation only; it does not alter factor formulas or mix factor
evaluation with custom strategy logic.

GTJA191 is a second calculator family inside the same factor-research track. Its
public names are `gtja_001` through `gtja_191`, its resumable outputs live under
`factor_report/gtja191_batch/`, and factors that require benchmark or Fama-French
series fail explicitly when those point-in-time inputs are absent. GTJA values
enter the same FactorTester lag and report workflow as Alpha101, but neither family
is routed into the custom buy-strategy evaluation path.

GTJA lifecycle YAML may also define a `directions` mapping with multipliers limited
to `-1` and `1`. Direction is a post-formula research transform shared by batch
evaluation, daily cross-sectional correlation, and ML feature construction; it does
not rewrite the canonical expressions in `factors/gtja191.py`. The selected mapping
is recorded in batch manifests and fingerprints, while ML dataset manifests record
the effective direction for every feature.

`factors/operators.py` is the small shared operator kernel for Alpha101, GTJA191,
and registered custom factors. It owns only operators with identical tested
semantics across families, including cross-sectional rank, lag, rolling
correlation/covariance, common rolling reductions, safe division, and linear
decay. Family-specific definitions such as Chinese `SMA`, GTJA `WMA`,
`highday`/`lowday`, and regression operators remain in `factors/gtja191.py`.
Batch implementation fingerprints include the shared operator file so a future
operator change cannot silently reuse stale factor reports.

`factors/correlation.py` diagnoses redundancy inside the factor track. It
lags factor values by one trading day, calculates Spearman and Pearson correlations
within each daily stock cross-section, and averages those daily correlations across
the evaluation period. It does not infer correlations from summary metrics and does
not route factor values into the custom-strategy path.

`factors/external.py` is the canonical boundary for a user-supplied factor library.
It validates either wide `date, symbol, factor...` or long
`date, symbol, factor, factor_value` input, preserves six-digit symbols, rejects
duplicate primary keys, and leaves values unlagged. `reports/external_factor_batch.py`,
the external correlation adapter, and `training/build_dataset.py` each apply the same
mandatory one-day lag at their research boundary. Daily market fields are joined by
`date, symbol`; static metadata can fill classifications but never historical prices
or market capitalisation. External factors remain in the factor/ML tracks and cannot
enter custom buy-strategy code.

The same module validates an optional point-in-time research-context file containing
daily market cap, sector/industry, market-regime, or trade-state fields. It merges
only exact `date, symbol` observations into the per-symbol market frames used by
built-in factor batches, external batches, single-factor research, and ML dataset
construction. Dynamic classifications take precedence over static metadata and are
never forward/backward filled. Context-file content participates in resume/data
signatures, so hydrating or changing the file invalidates stale cached reports.

`reports/factor_tester.py` applies a one-trading-day lag to every factor before IC,
grouping, statistical NAV, or tradable portfolio construction. The original factor
is retained as `factor_raw`; evaluation uses `factor_lagged` / `factor_processed`.

The factor track exposes long-only A-share evaluation plus legacy long-short
diagnostics:

- `tradable_top_n_cum_nav` in `tradable_top_n.csv` buys only the highest ranked
  fixed-count buckets. It uses close-to-close daily returns, staggered holding
  sleeves, point-in-time universe filters, limit-up entry blocks, limit-down exit
  delays, commission, stamp tax, and slippage.
- `tradable_top_quantile_cum_nav` in `tradable_top_quantile.csv` applies the same
  long-only execution model to the highest factor quantile. This is the preferred
  generic tradability metric for A-share factor review because it does not assume
  shorting is available.

- `stat_cum_nav` in `stat_long_short.csv` compounds Top-Bottom `forward_return_h`
  observations. The compatibility duplicate `long_short.csv` is emitted only by
  the full profile. This is a statistical diagnostic only. Its annualization uses
  `252 / (N * h)` and its Sharpe uses `sqrt(252 / h)`.
- `tradable_cum_nav` in `tradable_long_short.csv` uses close-to-close daily returns,
  `h` staggered sleeves, point-in-time universe filters, directional limit rules,
  delayed exits, commission, stamp tax, and slippage. It is retained for
  long-short diagnostics and backwards-compatible reports, not as the primary
  A-share trading metric.

Forward returns must not feed any `tradable_*_cum_nav`. Missing historical ST or
market-cap inputs are reported as unavailable rather than reconstructed from
current metadata.

`factors/brick.py` exposes the existing BrickChart logic to this research
track without moving or changing the custom strategy. `brick` retains
`brick_growth` only on dates where the complete configured selector passes, while
`brick_growth` tests the dense continuous feature without the strategy filters.
Both are lagged by FactorTester in the same way as every other factor. This is a
cross-sectional factor diagnostic; the stock-pool and full strategy P&L remain in
the existing signal-return and portfolio-backtest paths.

`factors/custom.py` follows the same registry, calculator, panel-builder, and
long-format adapter shape as the Alpha101 and GTJA191 families. `custom_001`
computes the negative cross-sectional rank of the five-day rolling covariance
between ranked daily returns and ranked point-in-time turnover value;
`custom_002` ranks the close discount relative to point-in-time VWAP. Both are
routed only through
`FactorTester`; the tester applies the standard one-trading-day lag before IC,
grouping, or tradable long-only evaluation. It does not enter or modify the
custom buy-strategy path under `strategies/`.

`factors/filter_rank.py` composes factors without blending their raw
scales. The first factor defines a point-in-time cross-sectional gate; the second
factor ranks only the surviving stocks and can select either a top-N prefix or an
inclusive 1-based rank interval. The default `factor-select` command uses
lagged `alpha_077` for the top-50% gate and lagged `alpha_040` for the final top-10
ranking. It emits the unified signal schema plus daily universe and filter audit
files. `backtest/factor_portfolio.py` passes those ranked signals to the
realistic portfolio engine without routing them through custom selectors. This
remains in the factor track and is not a custom buy-point strategy.

`factors/ensemble.py` is the family-independent weighted-rank combiner. It accepts
long-format `date, symbol, factor, factor_value` rows, converts each daily factor
cross-section to an oriented percentile, applies explicit positive weights, and
renormalizes only across available factors after enforcing the configured weight-
coverage threshold. `build_alpha101_rank_ensemble_frame` is the current calculator
adapter: it calculates all requested Alpha101 components, applies the same factor
lag and point-in-time listing/liquidity/ST filters, and preserves raw component
values and percentiles in the signal metadata. `factor-ensemble-select` writes the
auditable signals; `factor-ensemble-backtest` routes the same scores into the strict
staggered-cohort portfolio engine. Other factor families must add adapters to this
combiner rather than duplicate its weighting semantics.

## Machine-Learning Timing Boundary

`training/multifactor.py` is the public multi-factor fitting orchestrator. It
builds one shared lagged factor matrix, optionally applies per-date rank or
z-score transforms, and runs every requested model against identical
walk-forward windows. Its leaderboard compares out-of-sample diagnostics only;
each model keeps a separate unified `signals.csv` for the constrained long-only
portfolio backtest. This prevents model comparison from silently changing the
feature sample, target, purge gap, or execution path.

`training/build_dataset.py` is the reproducible raw-data adapter for ML. It
calculates explicitly requested Alpha101, GTJA191, or registered custom factors and
can align explicitly requested columns from the validated external factor boundary,
shifts every feature by exactly one trading day, and separately creates labels
before date filtering. Its default target is aligned to portfolio execution:
next-open entry followed by an open-price exit after `N` holding bars. Close-to-close
returns remain an explicit diagnostic mode. The output manifest records raw-data
and implementation signatures plus per-column missing counts; GTJA external series
remain explicit inputs.

The ML universe is rebuilt on every trading date from finite close observations in
the raw per-symbol history. Factor values are masked by that exact-date universe
before the mandatory lag, and cross-sectional transforms preserve missing values.
Consequently, pre-listing dates and dates after a symbol's final observation cannot
enter that day's feature ranks, while a subsequently delisted symbol remains in its
earlier observed history. Features and labels are restricted to the same universe
keys, and the dataset manifest records annual daily-universe counts. This adapter
cannot reconstruct delisted symbols absent from `data/raw`; retaining their source
files is an input requirement for avoiding historical survivorship bias.

`factors/correlation.py` applies the same one-day-lagged, daily
cross-sectional Spearman/Pearson contract to Alpha101, GTJA191, and validated
external factor files. The CLI keeps each family's calculator and lifecycle
configuration separate and always emits `deduplicated_factors.csv`. A standalone
caller may additionally request an eligibility-derived `ml_candidate_factors.csv`;
the run-all chain deliberately does not, because all deduplicated representatives
enter downstream walk-forward fitting.

`market/fetch_benchmark.py` is the governed Tushare `index_daily` boundary for
GTJA factors that explicitly require benchmark open/close. It atomically writes
one validated date-ordered index file plus a signature manifest. This is kept
separate from MKT/SMB/HML style-factor inputs: an index series is not silently
relabelled as a Fama-French factor.

`factors/style_returns.py` constructs that separate style input locally. Daily
close-to-close stock returns are weighted and sorted only by the latest market
cap and book-to-market observations strictly preceding the return date. It
emits an explicit daily 2x3 MKT/SMB/HML file plus input signatures, methodology,
and incomplete-portfolio drop counts; GTJA030 consumes the file through its
existing external-data boundary.

`training/train_walk_forward.py` is the executable ML boundary. Feature and label
files are joined one-to-one on `date, symbol`; feature columns are always explicit.
Each rolling window contains training dates, a purge gap measured in trading dates,
and a disjoint test block. `next_open_return_Nd` uses the open at `t+N+1` and thus
requires a purge of at least `N+1` dates; close-to-close `forward_return_Nd` requires
at least `N`. Only test-block predictions enter `predictions.csv` and the unified
model `signals.csv`; training fitted values are never mixed into reported scores.

`training/qlib_dataset.py` is the only RQuant-to-Qlib data adapter. For each
walk-forward window it converts the already aligned feature and label rows into
Qlib's `datetime, instrument` MultiIndex, preserves six-digit instruments, and
chronologically reserves the tail of the training block as validation. Validation
must end before the disjoint test block; the outer RQuant purge gap is applied
before the adapter receives any rows. `models/qlib_models.py` binds `lightgbm`
directly to Qlib `LGBModel` and exposes Qlib `DEnsembleModel` as
`doubleensemble`; there is no native-LightGBM fallback or parity branch.

Every window writes predictions, metrics, and a local model artifact before its
manifest. Resume is allowed only when feature data, label data, configuration, and
implementation hashes match. Ridge and ElasticNet have tested NumPy fallbacks;
scikit-learn, Qlib, and Torch are optional installed backends. Qlib models expose
only out-of-sample `score` at the RQuant boundary, after which the existing model
signal adapter and constrained portfolio path are reused. The MLP
standardizes features and targets from each training window only and supports CPU,
MPS, or CUDA plus save/load. Model outputs do not modify factor calculators or
custom-strategy rules.

`backtest/signal_portfolio.py` is the public bridge from the stable signal schema
to the strict staggered-cohort engine. It can consume factor, model, or strategy
signal files without importing their calculators. It filters one explicit source,
orders candidates by score, caps the daily list, executes at the next trading-day
open, and preserves the existing fee, lot, suspension, limit, and T+1 behavior.
The current cohort engine is equal-weight only, so unequal daily signal weights are
rejected instead of silently discarded.

`fit-multifactor` uses that same public bridge by default after model fitting. It
runs each model's out-of-sample buy signals once with all fees zero and once with
configured A-share costs, then records both summaries in the ML leaderboard and
manifest. The ML handoff also aggregates total return, compounded annualized
return, the arithmetic mean of calendar-year annualized returns, per-year return
rows, and an after-cost multi-model equity chart. Each scenario retains its own
standard portfolio summary, yearly CSV, equity CSV, and HTML curve. Callers that
only need prediction diagnostics must opt out with `--skip-backtests`; this layer
does not introduce a short-signal path.

The factor portfolio uses strict staggered cohort slots: an `h`-day holding period
owns exactly `h` independently funded sleeves and schedules one sleeve per trading
day. A sleeve with a blocked exit cannot open a new cohort, so delayed exits never
increase the slot count or nominal capital budget beyond `h`.

## Unified Signal Schema

Both factor and custom-strategy tracks should emit:

```text
date, symbol, signal_type, source, score, weight, metadata
```

Examples:

```text
2026-06-23,600519,buy,factor_momentum_20d,0.87,0.10,{"factor_value":0.87}
2026-06-23,002008,buy,brick,9.03,0.10,{"brick_growth":9.03}
```

The canonical in-memory path is `Signal -> SignalBook -> BacktestResult ->
WorkflowResult`. `SignalBook` preserves provenance, score, weight, and metadata
through execution while exposing the former date-to-symbol-list view only as a
compatibility interface. Orders, trades, position snapshots, and equity points are
typed domain records rather than public dictionaries. Detailed contracts and wire
compatibility are documented in `docs/domain_model.md`.

## Runtime Governance Boundary

The `rquant/` package is framework glue, not a fourth research path. It may resolve
project-relative paths, dispatch CLI handlers, initialize console/file logging, and
record execution metadata. It must not calculate factors, labels, strategies,
predictions, portfolio orders, or performance metrics.

Every real public-CLI execution owns `data/runs/<run-id>/run.json` and `run.log`.
The run manifest records sanitized arguments, Python and Git identity, explicit
input fingerprints, declared outputs, downstream manifests, warnings, final status,
and exit code. Help and `runs list/show` are read-only and do not create runs.
Failures and interrupts finalize the manifest before returning the original nonzero
status. Existing domain-specific manifests remain authoritative for their detailed
resume signatures; the framework manifest references them rather than replacing them.

## Rules

- Factor research should stay in `factors/`, `reports/factor_tester.py`, and `signals/`.
- Custom buy rules should stay in `strategies/`.
- `bdsr_macd_obv` defines BDSR as RCI9 crossing above RCI26; MACD and OBV conditions are evaluated on the same completed daily bar.
- Factor filter/rank and rank-ensemble signals enter `backtest/portfolio.py` through the unified schema; custom selectors retain their adapter path.
- Walk-forward model scores enter only through `signals.csv`; purge gaps and out-of-sample window boundaries must remain auditable.
- Do not mix factor IC tests with custom buy-point strategy evaluation.
