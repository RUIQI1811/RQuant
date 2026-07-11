from __future__ import annotations

from typing import Iterable

import pandas as pd

from domain.signals import SignalBook
from signals.candidates import Candidate, CandidateRun
from signals.schema import Signal, signals_to_frame


def candidate_to_signal(candidate: Candidate, *, default_weight: float | None = None) -> Signal:
    """Convert one strategy Candidate into the unified signal schema."""
    return candidate.to_signal(weight=default_weight)


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


def candidates_to_signal_book(
    candidates: Iterable[Candidate],
    *,
    default_weight: float | None = None,
) -> SignalBook:
    """Return canonical strategy signals without a DataFrame round trip."""
    return SignalBook(
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
