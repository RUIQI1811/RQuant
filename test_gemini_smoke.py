"""
Minimal Gemini API smoke test.

Usage:
    python test_gemini_smoke.py
    python test_gemini_smoke.py --model gemini-2.5-flash-lite
    python test_gemini_smoke.py --image-code 600483 --pick-date 2026-06-23
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from google import genai

ROOT = Path(__file__).resolve().parent


def load_dotenv(env_path: Path = ROOT / ".env") -> None:
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def test_text(model: str) -> None:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found. Export it or write it into .env.")

    client = genai.Client(api_key=api_key)
    print(f"[TEXT] model={model}")
    response = client.models.generate_content(
        model=model,
        contents="只回复 OK",
    )
    print(response.text)


def test_image(model: str, code: str, pick_date: str) -> None:
    sys.path.insert(0, str(ROOT / "agent"))

    from gemini_review import GeminiReviewer, load_config

    config = load_config(ROOT / "config" / "gemini_review.yaml")
    config["model"] = model
    reviewer = GeminiReviewer(config)

    day_chart = ROOT / "data" / "kline" / pick_date / f"{code}_day.jpg"
    if not day_chart.exists():
        raise FileNotFoundError(f"Chart not found: {day_chart}")

    print(f"[IMAGE] model={model} code={code} chart={day_chart}")
    result = reviewer.review_stock(code=code, day_chart=day_chart, prompt=reviewer.prompt)
    print(result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gemini API smoke test")
    parser.add_argument(
        "--model",
        default="gemini-3.5-flash",
        help="Gemini model name",
    )
    parser.add_argument(
        "--image-code",
        default=None,
        help="Optional stock code for one-image review test",
    )
    parser.add_argument(
        "--pick-date",
        default=None,
        help="Pick date for --image-code, e.g. 2026-06-23",
    )
    return parser


def main() -> None:
    load_dotenv()
    args = build_parser().parse_args()

    test_text(args.model)

    if args.image_code:
        if not args.pick_date:
            raise ValueError("--pick-date is required when --image-code is set")
        test_image(args.model, args.image_code.zfill(6), args.pick_date)


if __name__ == "__main__":
    main()
