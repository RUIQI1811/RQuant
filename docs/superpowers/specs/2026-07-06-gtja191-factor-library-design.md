# GTJA191 Factor Library Design

## Goal

Implement the 191 short-horizon price-volume factors from the Guotai Junan
Alpha191 report as a second built-in factor family. Preserve the existing
WorldQuant Alpha101 API and formulas unchanged.

The feature remains inside the factor-research track. It does not alter B1,
BrickChart, mBDSR, BDSR/MACD/OBV, or any other custom buy strategy.

## Canonical sources

- Factor list and expressions: <https://bigquant.com/wiki/doc/Pyf0TYya6H>
- Original report and operator appendix:
  <https://guorn.com/static/upload/file/3/134065454575605.pdf>

The BigQuant list determines factor numbering. The original report appendix
determines operator semantics. Obvious transcription errors such as `DELAT`,
`HGIH`, and `COVIANCE` are corrected only when the intended operator is unique.
Every non-literal correction is recorded in a formula-notes mapping next to the
implementation and covered by a focused test. If an expression cannot be
resolved uniquely from the two canonical sources, its calculator raises an
explicit formula-data error rather than silently substituting a different
formula.

## Naming and compatibility

The existing WorldQuant factors keep their current names:

```text
alpha_001 ... alpha_101
```

The new Guotai Junan factors use a non-conflicting namespace:

```text
gtja_001 ... gtja_191
```

Accepted aliases for the new family are an integer passed directly to the
GTJA191 calculator, `gtja1`, `gtja_1`, and `gtja_001`. Generic repository entry
points require the `gtja` prefix so an Alpha101 name can never be routed to the
wrong formula family.

No existing Alpha101 public name, result directory, configuration entry, or
batch-resume fingerprint changes.

## Architecture

### Calculator and panels

Create `pipeline/factors/gtja191.py` with:

- `GTJA191_NAMES`: ordered names from `gtja_001` through `gtja_191`.
- `GTJA191Panels`: aligned date-by-symbol panels for open, high, low, close,
  volume, amount, VWAP, and returns, plus optional external inputs.
- `GTJA191ExternalData`: optional benchmark open/close series and MKT/SMB/HML
  series, aligned by trading date.
- `GTJA191`: registry-backed single-factor and multi-factor calculator.
- `build_gtja191_panels(...)`: adapter from the repository's per-symbol raw
  frames.
- `gtja191_to_long(...)`: adapter to FactorTester's long schema.
- `normalize_gtja_name(...)`: strict family-specific normalization.

The new module may import the already-tested generic panel operators from
`pipeline/factors/alpha101.py`. It must not add GTJA formulas to the Alpha101
class or change existing Alpha101 formula behavior. GTJA-specific operators
stay in `gtja191.py`; a broad operator-module refactor is outside this task.

### Repository routing

Update `pipeline.factor_tester.build_long_factor_frame_from_raw(...)` to route
only `gtja`-prefixed names to `gtja191_to_long(...)`. Existing `alpha`, Brick,
and momentum routes remain unchanged.

Update `scripts/test_factor.py --list-factors` to list GTJA191 names after the
existing families. A single factor runs through the existing command:

```bash
/opt/miniconda3/envs/stocktrade/bin/python scripts/test_factor.py \
  --factor gtja_001 \
  --windows 1 5 10 20 \
  --groups 10
```

The adapter output uses the existing fields `date`, `symbol`, `factor_value`,
`close`, `volume`, `daily_return`, `listing_age_days`, `industry`,
`turnover_value`, `market_cap`, and `is_st` when source data is available.
Symbols remain six-character strings.

### Batch research

Create a GTJA-specific resumable runner and CLI instead of making the
Alpha101 runner accept two incompatible registries:

- `pipeline/gtja191_batch.py`
- `scripts/test_gtja191_batch.py`
- `config/gtja191_factors.yaml`

The batch runner reuses the established checkpoint/report behavior: calculate
one factor at a time, write its FactorTester output atomically, update status
after every factor, isolate failures, support `--force`, `--fail-fast`, include
ranges, exclude ranges, and lifecycle filtering.

The same script must also provide one-command evaluation of the complete 191
factor family. This command selects every GTJA factor regardless of later
lifecycle settings:

```bash
/opt/miniconda3/envs/stocktrade/bin/python scripts/test_gtja191_batch.py \
  --data data/raw \
  --factors all \
  --ignore-factor-config \
  --windows 1 5 10 20 \
  --groups 10 \
  --output factor_report/gtja191_batch
```

`--factors` defaults to `all`, while `--ignore-factor-config` guarantees that
disabled or not-yet-scored factors are not omitted from a deliberate full
library evaluation. The runner builds aligned raw panels and reusable forward
returns once, then evaluates all selected formulas sequentially so it does not
hold 191 factor panels in memory. `batch_status.csv` contains one row for every
requested factor with `success`, `missing_input`, `formula_error`, `failed`, or
`skipped` status; therefore a full run cannot silently omit a factor.

Durable outputs are separate from Alpha101:

```text
factor_report/gtja191_batch/
  batch_manifest.json
  batch_status.csv
  leaderboard.csv
  logs/gtja_XXX.log
  gtja_XXX/
```

Resume fingerprints include the GTJA implementation, relevant adapter code,
data signature, external-data signature, and FactorTester settings.

## Operator semantics

All operators work on aligned wide panels with dates on rows and symbols on
columns.

- `RANK(A)` is an ascending percentile rank across symbols for each date.
- `DELAY`, `DELTA`, `SUM`, `MEAN`, `STD`, `CORR`, `COVIANCE`, `TSMIN`,
  `TSMAX`, `COUNT`, and `PROD` require a full lookback window.
- `TSRANK(A,n)` is the percentile rank of the current value within the last
  `n` observations.
- `SMA(A,n,m)` uses the report recursion
  `Y[t] = (m*A[t] + (n-m)*Y[t-1]) / n`, seeded with the first valid value.
- `WMA(A,n)` uses normalized weights proportional to `0.9**i`, where `i=0`
  is the current observation.
- `DECAYLINEAR(A,d)` uses normalized linear weights `1..d`, with the largest
  weight on the current observation.
- `REGBETA(A,B,n)` returns the rolling least-squares slope with an intercept.
- `REGRESI(A,B,n)` returns the current observation's residual from that
  rolling regression.
- `HIGHDAY` and `LOWDAY` return the distance in trading periods from the
  current date to the most recent maximum or minimum inside the window.
- `SUMIF`, `FILTER`, and all conditionals preserve missing observations.
- `SELF` in `gtja_143` is implemented as the explicitly recursive prior-day
  factor value.
- Zero denominators, invalid logarithms, insufficient history, and infinite
  results become `NaN`; factor code never backfills from future observations.

FactorTester applies the repository-wide one-trading-day lag after a raw GTJA
factor is calculated. Formula-internal `DELAY` calls are part of the formula and
do not replace that evaluation lag.

## External data and explicit failures

Most factors use only per-symbol daily price and volume data. These factors
require extra point-in-time inputs:

- `gtja_030`: MKT, SMB, and HML daily factor returns.
- `gtja_075`, `gtja_149`, `gtja_181`, `gtja_182`: benchmark index daily data.

Single-factor and batch CLIs accept optional paths for benchmark OHLC and
MKT/SMB/HML data. External rows are aligned by date without forward-looking
fills. When required data is absent, the factor raises `GTJA191DataError` with
the exact missing fields. Batch mode records `missing_input` for that factor and
continues unless `--fail-fast` was requested.

The implementation must not approximate benchmark returns with the stock-pool
average and must not synthesize MKT/SMB/HML from current-day cross-sectional
data.

## Error and ambiguity handling

The public page contains typographical and parenthesis errors. Corrections use
the following policy:

1. Prefer the original report expression over copied web text.
2. Apply an operator spelling correction only when one appendix operator is an
   exact contextual match.
3. Fix reversed variable descriptions by variable semantics: benchmark
   `OPEN` maps to index open and benchmark `CLOSE` maps to index close.
4. Record every corrected factor and its reason in `GTJA191_FORMULA_NOTES`.
5. If two materially different formulas remain plausible, raise an explicit
   formula error for that factor and keep batch execution auditable.

The source expressions for `gtja_159` and `gtja_181` remain unresolved under
this policy.  The former subtracts a multi-day sum of prices from one current
close; the latter mixes a stock-return deviation with squared benchmark-index
level deviations.  Both remain lifecycle `watch` entries for provenance, but
their calculators raise `GTJA191FormulaError` and batch execution records
`formula_error` until a new canonical source uniquely resolves them.

## Tests

Create `tests/test_gtja191.py` with four layers:

1. Operator examples with hand-computed expected values for recursive SMA,
   WMA, rolling regression, high/low day distance, conditional sums, and
   cumulative recursion.
2. Registry tests proving exactly 191 callable names and strict family name
   normalization.
3. Formula tests for representative categories: pure OHLCV, cross-sectional
   rank, rolling correlation, recursive indicator, regression, benchmark,
   Fama-French, and `SELF` recursion.
4. A complete-panel execution test proving every factor either returns an
   aligned panel or raises its documented explicit input/formula error. With
   complete external inputs, all resolved formulas must return an aligned
   panel.

Add integration assertions to existing focused tests for:

- `build_long_factor_frame_from_raw(..., factor_name="gtja_001")`;
- six-digit symbols and long-schema metadata;
- `scripts/test_factor.py --list-factors` exposure;
- batch range parsing, checkpointing, resume, isolated missing input, and
  output paths;
- unchanged Alpha101 routing and names.

Tests use generated multi-symbol data in temporary directories. They do not
read or overwrite `data/raw`, call Tushare, or run Gemini.

## Documentation

Update `README.md` with the single-factor command, batch commands, optional
external-data inputs, exact report paths, naming distinction, and known
`missing_input` behavior.

Update `docs/architecture.md` to show GTJA191 as a separate calculator family
inside the existing factor track. No custom-strategy documentation changes.

## Acceptance criteria

- `GTJA191_NAMES` exposes `gtja_001` through `gtja_191` without colliding with
  Alpha101.
- Every resolved formula has a callable method and returns an aligned panel on
  complete required inputs.
- Ambiguous or externally dependent formulas fail explicitly and are isolated
  by batch execution.
- FactorTester applies the existing one-day lag and preserves statistical NAV
  versus tradable NAV separation.
- Single-factor and resumable batch CLIs produce auditable outputs at the
  documented paths.
- One `scripts/test_gtja191_batch.py --factors all --ignore-factor-config`
  command evaluates all 191 registered factors through the same FactorTester
  settings and records one terminal status row per factor.
- Focused GTJA, FactorTester, CLI, batch, and Alpha101 regression tests pass.
- README and architecture documentation match actual commands and behavior.
- No secrets, user market data, or unrelated working-tree changes are included.
