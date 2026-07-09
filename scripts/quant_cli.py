"""Top-level CLI dispatcher for RQuant research workflows."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RQuant research CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("fetch-data", help="Fetch and update market data")
    sub.add_parser("preselect", help="Run custom strategy preselection")
    sub.add_parser("signal-returns", help="Evaluate signal forward returns")
    sub.add_parser("portfolio-backtest", help="Run realistic portfolio backtest")
    sub.add_parser("research-report", help="Build combined research report")
    sub.add_parser("factor-test", help="Run single factor diagnostics")
    sub.add_parser("factor-batch-alpha101", help="Run Alpha101 batch diagnostics")
    sub.add_parser("factor-batch-gtja191", help="Run GTJA191 batch diagnostics")
    sub.add_parser("factor-score", help="Score factor batch outputs")
    sub.add_parser("factor-select", help="Create factor ranking signals")
    sub.add_parser("factor-backtest", help="Backtest factor ranking signals")
    sub.add_parser("make-labels", help="Create forward-return labels")
    sub.add_parser("train-model", help="Train walk-forward model")
    sub.add_parser("predict-score", help="Create model score signals")
    sub.add_parser("model-backtest", help="Backtest model score signals")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(
        f"command '{args.command}' is registered; wire command dispatch after module migration validation"
    )


if __name__ == "__main__":
    main()
