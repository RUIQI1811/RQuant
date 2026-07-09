from __future__ import annotations

from typing import Iterable

import pandas as pd

from signals.candidates import Candidate, CandidateRun
from signals.schema import Signal, signals_to_frame


def candidate_to_signal(candidate: Candidate, *, default_weight: float | None = None) -> Signal:
    """Convert one strategy Candidate into the unified signal schema."""
    score = candidate.brick_growth
    metadata = dict(candidate.extra or {})
    metadata.update(
        {
            "close": candidate.close,
            "turnover_n": candidate.turnover_n,
            "strategy": candidate.strategy,
        }
    )
    if candidate.brick_growth is not None:
        metadata["brick_growth"] = candidate.brick_growth
    return Signal(
        date=candidate.date,
        symbol=candidate.code,
        signal_type="buy",
        source=candidate.strategy,
        score=score,
        weight=default_weight,
        metadata=metadata,
    )


def candidates_to_signal_frame(
    candidates: Iterable[Candidate],
    *,
    default_weight: float | None = None,
) -> pd.DataFrame:
    """Convert existing custom strategy candidates to unified signal DataFrame."""
    return signals_to_frame(
        candidate_to_signal(candidate, default_weight=default_weight)
        for candidate in candidates
    )


def candidate_run_to_signal_frame(
    run: CandidateRun,
    *,
    default_weight: float | None = None,
) -> pd.DataFrame:
    """Convert a CandidateRun archive into unified signal DataFrame."""
    return candidates_to_signal_frame(run.candidates, default_weight=default_weight)
