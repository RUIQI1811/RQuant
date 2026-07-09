# RQuant Architecture

This project keeps two research tracks and routes both into the same signal shape.

```text
DataManager
→ UniverseBuilder
→ Feature / Factor Layer
→ Signal Layer
   ├─ FactorSignalEngine      # factor ranking / scoring
   └─ StrategySignalEngine    # B1, brick, mBDSR, and BDSR/MACD/OBV rules
→ PortfolioConstructor
→ RiskManager
→ BacktestEngine
→ Reporter
```

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

## Factor Research Timing and NAV Boundaries

Alpha101 batch research applies the lifecycle catalog before calculation. `active`
factors run first, `watch` factors remain in research but run and rank after active
factors, and `disabled` factors are omitted by default. This status changes research
scheduling and presentation only; it does not alter factor formulas or mix factor
evaluation with custom strategy logic.

GTJA191 is a second calculator family inside the same factor-research track. Its
public names are `gtja_001` through `gtja_191`, its resumable outputs live under
`factor_report/gtja191_batch/`, and factors that require benchmark or Fama-French
series fail explicitly when those point-in-time inputs are absent. GTJA values
enter the same FactorTester lag and report pipeline as Alpha101, but neither family
is routed into the custom buy-strategy evaluation path.

`factors/correlation.py` diagnoses redundancy inside the factor track. It
lags factor values by one trading day, calculates Spearman and Pearson correlations
within each daily stock cross-section, and averages those daily correlations across
the evaluation period. It does not infer correlations from summary metrics and does
not route factor values into the custom-strategy path.

`reports/factor_tester.py` applies a one-trading-day lag to every factor before IC,
grouping, statistical NAV, or tradable portfolio construction. The original factor
is retained as `factor_raw`; evaluation uses `factor_lagged` / `factor_processed`.

The factor track exposes two deliberately separate NAVs:

- `stat_cum_nav` in `long_short.csv` compounds Top-Bottom `forward_return_h`
  observations. It is a statistical diagnostic only. Its annualization uses
  `252 / (N * h)` and its Sharpe uses `sqrt(252 / h)`.
- `tradable_cum_nav` in `tradable_long_short.csv` uses close-to-close daily returns,
  `h` staggered sleeves, point-in-time universe filters, directional limit rules,
  delayed exits, commission, stamp tax, and slippage. Its risk metrics use the
  resulting daily net-return series.

Forward returns must not feed `tradable_cum_nav`. Missing historical ST or market-cap
inputs are reported as unavailable rather than reconstructed from current metadata.

`factors/brick.py` exposes the existing BrickChart logic to this research
track without moving or changing the custom strategy. `brick` retains
`brick_growth` only on dates where the complete configured selector passes, while
`brick_growth` tests the dense continuous feature without the strategy filters.
Both are lagged by FactorTester in the same way as every other factor. This is a
cross-sectional factor diagnostic; the stock-pool and full strategy P&L remain in
the existing signal-return and portfolio-backtest paths.

`factors/filter_rank.py` composes factors without blending their raw
scales. The first factor defines a point-in-time cross-sectional gate; the second
factor ranks only the surviving stocks and can select either a top-N prefix or an
inclusive 1-based rank interval. The default `factor-select` command uses
lagged `alpha_077` for the top-50% gate and lagged `alpha_040` for the final top-10
ranking. It emits the unified signal schema plus daily universe and filter audit
files. `backtest/factor_portfolio.py` passes those ranked signals to the
realistic portfolio engine without routing them through custom selectors. This
remains in the factor track and is not a custom buy-point strategy.

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

## Rules

- Factor research should stay in `factors/`, `reports/factor_tester.py`, and `signals/`.
- Custom buy rules should stay in `strategies/`.
- `bdsr_macd_obv` defines BDSR as RCI9 crossing above RCI26; MACD and OBV conditions are evaluated on the same completed daily bar.
- Factor filter/rank signals enter `backtest/portfolio.py` through the unified schema; custom selectors retain their adapter path.
- Do not mix factor IC tests with custom buy-point strategy evaluation.
