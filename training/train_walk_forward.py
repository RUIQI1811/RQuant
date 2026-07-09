"""Walk-forward model training entrypoint."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a walk-forward stock score model")
    parser.add_argument("--features", required=True, help="Feature CSV path")
    parser.add_argument("--labels", required=True, help="Label CSV path")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--model", default="ridge", choices=("ridge", "elasticnet", "lightgbm"))
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(
        "walk-forward training CLI is scaffolded; wire data loading in the implementation pass "
        f"for model={args.model}, features={args.features}, labels={args.labels}, output={args.output}"
    )


if __name__ == "__main__":
    main()
