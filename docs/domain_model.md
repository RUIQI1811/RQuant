# RQuant Unified Domain Model

RQuant only unifies objects that cross package boundaries. Factor panels,
walk-forward internals, selector state, and cohort implementation details remain
inside their own research contexts.

## Canonical Contracts

`domain/values.py` owns validated primitives:

- `Symbol`: six-digit security identifier; exchange suffixes and numeric inputs
  normalize at construction.
- `TradingDate` and `DateRange`: ISO trading dates and validated optional ranges.
- `SourceId`: factor, model, or custom-strategy provenance.

`domain/signals.py` owns research intent:

- `Signal`: the stable `date, symbol, signal_type, source, score, weight, metadata`
  record.
- `SignalBook`: date-indexed signals that retain the complete records through
  portfolio execution. Its mapping view still returns ranked symbol lists for
  compatibility.

`domain/execution.py` owns execution outcomes:

- `OrderIntent`, `OrderResult`, `Fill`, `Trade`, `Position`, `PositionSnapshot`,
  and `EquityPoint`.
- `BacktestSummary` and `BacktestResult`.

The records implement the old mapping access where needed, so expressions such as
`order["status"]` and `summary["total_return"]` remain valid. Portfolio CSV and JSON
wire formats keep their established columns and keys.

`domain/artifacts.py` owns workflow delivery:

- `ArtifactRef`: typed path, kind, existence, and on-demand SHA-256.
- `WorkflowStatus` and `WorkflowResult[T]`: canonical lifecycle state, domain result, artifacts, additional values, and
  warnings. It preserves legacy `outputs["summary_path"]` access.

ML workflows use the typed results in `domain/research.py`. The same module exposes
`FactorResearchPipelineResult` for the cross-stage factor run-all summary while the
individual factor panels, correlation matrices, model internals, and portfolio
cohorts remain in their original modules. Custom-strategy candidates use
`domain/selection.py` and compose a canonical `Signal` rather than redefining signal
identity.

## Boundary Flow

```text
factor / model / custom selector
        -> Signal
        -> SignalBook
        -> portfolio constructor
        -> OrderIntent
        -> OrderResult / Fill / Trade / PositionSnapshot / EquityPoint
        -> BacktestResult
        -> WorkflowResult[BacktestResult]
        -> CSV / JSON / HTML ArtifactRef
```

Signal `source`, `score`, `weight`, and `metadata` remain attached to buy orders,
positions, sell orders, and realized trades in memory. Existing persisted order
files deliberately retain their old schema; richer context can be exposed by a
future explicitly versioned output instead of silently changing durable files.

## Compatibility Imports

- `signals.schema.Signal` re-exports `domain.signals.Signal`.
- `signals.candidates.Candidate` and `CandidateRun` re-export the canonical
  selection models.
- `backtest.portfolio.PortfolioBacktestResult` is the canonical
  `domain.execution.BacktestResult`.

New cross-boundary code should import from `domain`. DataFrame conversion belongs
in adapters under `signals/`, report writers, or other I/O boundaries.
