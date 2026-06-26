"""
Print stock codes from a candidates JSON file.

Usage:
    python print_candidates.py
    python print_candidates.py data/candidates/candidates_2026-06-04.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_CANDIDATES_FILE = ROOT / "data" / "candidates" / "candidates_latest.json"


def load_candidates(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Candidates file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def format_candidate_codes(candidates_data: dict) -> str:
    lines = []
    pick_date = candidates_data.get("pick_date")
    if pick_date:
        lines.append(f"pick_date: {pick_date}")

    codes = [
        str(candidate["code"])
        for candidate in candidates_data.get("candidates", [])
        if candidate.get("code")
    ]
    lines.extend(codes)
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print codes from data/candidates/candidates_latest.json",
    )
    parser.add_argument(
        "candidates_file",
        nargs="?",
        default=str(DEFAULT_CANDIDATES_FILE),
        help="Candidates JSON file path",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    path = Path(args.candidates_file)
    if not path.is_absolute():
        path = ROOT / path

    try:
        output = format_candidate_codes(load_candidates(path))
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    if output:
        print(output)


if __name__ == "__main__":
    main()
