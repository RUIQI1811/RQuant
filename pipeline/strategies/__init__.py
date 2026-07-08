"""Compatibility package exports for custom strategy signal adapters."""

from signals.strategy_adapters import candidate_run_to_signal_frame, candidate_to_signal, candidates_to_signal_frame

__all__ = ["candidate_run_to_signal_frame", "candidate_to_signal", "candidates_to_signal_frame"]
