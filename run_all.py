"""Fail-fast orchestration for RQuant's daily research-assistance workflow."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable


@dataclass(frozen=True)
class PipelineStep:
    number: int
    name: str
    command: tuple[str, ...] | None


class PipelineStepError(RuntimeError):
    def __init__(self, step: PipelineStep, return_code: int) -> None:
        self.step = step
        self.return_code = int(return_code)
        super().__init__(
            f"step {step.number} ({step.name}) returned exit code {self.return_code}"
        )


class PipelineResultError(RuntimeError):
    """Raised when the final review artifact is missing or malformed."""


def build_steps(
    *,
    python: str = PYTHON,
    root: Path = ROOT,
) -> tuple[PipelineStep, ...]:
    quant_cli = str(root / "scripts" / "quant_cli.py")
    return (
        PipelineStep(
            1,
            "拉取 K 线数据",
            (python, quant_cli, "fetch-data"),
        ),
        PipelineStep(
            2,
            "量化初选",
            (python, quant_cli, "preselect"),
        ),
        PipelineStep(
            3,
            "导出 K 线图",
            (python, str(root / "dashboard" / "export_kline_charts.py"), "--resume"),
        ),
        PipelineStep(
            4,
            "Gemini 图表复评",
            (python, str(root / "agent" / "gemini_review.py"), "--resume"),
        ),
        PipelineStep(5, "显示达到 AI 复评阈值的候选", None),
    )


def _run(step: PipelineStep, *, root: Path = ROOT) -> None:
    """Run one subprocess step and raise immediately on a nonzero return code."""

    if step.command is None:
        raise ValueError("subprocess step must define a command")
    print(f"\n{'=' * 60}")
    print(f"[步骤] {step.number}/5  {step.name}")
    print(f"  命令: {' '.join(step.command)}")
    print(f"{'=' * 60}")
    result = subprocess.run(list(step.command), cwd=str(root), check=False)
    if result.returncode != 0:
        raise PipelineStepError(step, result.returncode)


def _load_review_summary(*, root: Path = ROOT) -> tuple[str, Path, dict]:
    candidates_file = root / "data" / "candidates" / "candidates_latest.json"
    candidates = _load_json_object(candidates_file, "latest candidates")
    pick_date = str(candidates.get("pick_date", "")).strip()
    if not pick_date:
        raise PipelineResultError(
            f"{candidates_file} does not contain a non-empty pick_date"
        )

    suggestion_file = root / "data" / "review" / pick_date / "suggestion.json"
    suggestion = _load_json_object(suggestion_file, "Gemini review summary")
    status = suggestion.get("status")
    if status in {"partial", "failed"}:
        raise PipelineResultError(
            f"Gemini review summary is {status}, not complete: {suggestion_file}"
        )
    review_date = suggestion.get("date")
    if review_date and str(review_date) != pick_date:
        raise PipelineResultError(
            f"candidate pick_date {pick_date} does not match review date {review_date}"
        )
    recommendations = suggestion.get(
        "review_candidates",
        suggestion.get("recommendations", []),
    )
    if not isinstance(recommendations, list):
        raise PipelineResultError(
            f"{suggestion_file} field recommendations must be a list"
        )
    suggestion = {**suggestion, "recommendations": recommendations}
    return pick_date, suggestion_file, suggestion


def _load_json_object(path: Path, label: str) -> dict:
    if not path.is_file():
        raise PipelineResultError(f"missing {label}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PipelineResultError(
            f"cannot read {label} {path}: {type(exc).__name__}"
        ) from exc
    if not isinstance(payload, dict):
        raise PipelineResultError(f"{label} must contain a JSON object: {path}")
    return payload


def _print_review_candidates(*, root: Path = ROOT) -> None:
    """Print the latest AI-review candidates without presenting them as trade advice."""

    pick_date, suggestion_file, suggestion = _load_review_summary(root=root)
    recommendations: list[dict] = suggestion["recommendations"]
    min_score = suggestion.get("min_score_threshold", 0)
    total = suggestion.get("total_reviewed", 0)

    print(f"\n{'=' * 60}")
    print(f"  选股日期：{pick_date}")
    print(f"  复评总数：{total} 只   展示门槛：score ≥ {min_score}")
    print(f"{'=' * 60}")
    if not suggestion.get("status"):
        print("  [WARN] 旧版复评文件没有 status，无法验证是否完整。")
    if not recommendations:
        print("  暂无达到 AI 复评阈值的候选。")
    else:
        header = f"{'排名':>4}  {'代码':>8}  {'总分':>6}  {'信号':>10}  {'研判':>6}  备注"
        print(header)
        print("-" * len(header))
        for row in recommendations:
            if not isinstance(row, dict):
                raise PipelineResultError(
                    f"{suggestion_file} recommendations entries must be JSON objects"
                )
            rank = row.get("rank", "?")
            code = _normalize_display_symbol(row.get("code", "?"))
            score = row.get("total_score", "?")
            signal_type = row.get("signal_type", "")
            verdict = row.get("verdict", "")
            comment = row.get("comment", "")
            score_text = f"{score:.1f}" if isinstance(score, (int, float)) else str(score)
            print(
                f"{rank:>4}  {code:>8}  {score_text:>6}  "
                f"{signal_type:>10}  {verdict:>6}  {comment}"
            )
    print(f"\n复评结果：{suggestion_file}")
    print("提示：AI 复评仅用于研究辅助，不构成交易指令或收益保证。")


def _normalize_display_symbol(value: object) -> str:
    symbol = str(value).strip()
    return symbol.zfill(6) if symbol.isdigit() and len(symbol) <= 6 else symbol


def run_pipeline(
    *,
    start_from: int = 1,
    stop_after: int = 5,
    skip_fetch: bool = False,
    skip_review: bool = False,
    root: Path = ROOT,
    python: str = PYTHON,
) -> None:
    if not 1 <= start_from <= 5 or not 1 <= stop_after <= 5:
        raise ValueError("start_from and stop_after must be in [1, 5]")
    if start_from > stop_after:
        raise ValueError("start_from must not be greater than stop_after")

    for step in build_steps(python=python, root=root):
        if step.number < start_from or step.number > stop_after:
            continue
        if skip_fetch and step.number == 1:
            print("[跳过] 1/5  拉取 K 线数据（--skip-fetch）")
            continue
        if skip_review and step.number in (4, 5):
            print(f"[跳过] {step.number}/5  {step.name}（--skip-review）")
            continue
        if step.number == 5:
            print(f"\n[步骤] {step.number}/5  {step.name}")
            _print_review_candidates(root=root)
        else:
            _run(step, root=root)
    print("\n流程范围内的步骤已全部完成。")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RQuant 全流程自动运行脚本")
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="跳过步骤 1（行情下载）",
    )
    parser.add_argument(
        "--skip-review",
        action="store_true",
        help="跳过步骤 4 Gemini 复评和步骤 5 复评结果展示",
    )
    parser.add_argument(
        "--start-from",
        type=int,
        choices=range(1, 6),
        default=1,
        metavar="N",
        help="从第 N 步开始执行（1~5）",
    )
    parser.add_argument(
        "--stop-after",
        type=int,
        choices=range(1, 6),
        default=5,
        metavar="N",
        help="执行到第 N 步后停止（1~5）",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.start_from > args.stop_after:
        parser.error("--start-from must not be greater than --stop-after")
    try:
        run_pipeline(
            start_from=args.start_from,
            stop_after=args.stop_after,
            skip_fetch=args.skip_fetch,
            skip_review=args.skip_review,
        )
    except PipelineStepError as exc:
        print(f"[ERROR] {exc}; pipeline stopped", file=sys.stderr)
        return exc.return_code or 1
    except PipelineResultError as exc:
        print(f"[ERROR] {exc}; pipeline stopped", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
