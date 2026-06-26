"""
pipeline/cli.py
统一命令行入口。

用法：
  python -m pipeline.cli preselect
  python -m pipeline.cli preselect --date 2025-12-31
  python -m pipeline.cli preselect --config config/rules_preselect.yaml --data data/raw

子命令：
  preselect   运行量化初选，写入 data/candidates/
"""
from __future__ import annotations

import argparse
import datetime
import logging
import sys
from pathlib import Path

# 将 pipeline 目录加入 path（直接用 python cli.py 时需要）
sys.path.insert(0, str(Path(__file__).parent))

from select_stock import run_preselect, resolve_preselect_output_dir
from schemas import CandidateRun
from pipeline_io import save_candidates
from signal_returns import BUY_MODE_SIGNAL_CLOSE, VALID_BUY_MODES, DEFAULT_HORIZONS, run_signal_returns

# ── 日志配置 ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("cli")


def _add_log_file(log_dir: str, pick_date: str) -> None:
    """可选：追加文件日志到 data/logs/pipeline_YYYY-MM-DD.log。"""
    p = Path(log_dir)
    p.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(p / f"pipeline_{pick_date}.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logging.getLogger().addHandler(fh)


# =============================================================================
# preselect 子命令
# =============================================================================

def cmd_preselect(args: argparse.Namespace) -> None:
    logger.info("===== 量化初选开始 =====")

    pick_ts, candidates = run_preselect(
        config_path=args.config or None,
        data_dir=args.data or None,
        end_date=args.end_date or None,
        pick_date=args.date or None,
    )

    pick_date_str = pick_ts.strftime("%Y-%m-%d")
    run_date_str = datetime.date.today().isoformat()

    # 可选日志文件
    if args.log_dir:
        _add_log_file(args.log_dir, pick_date_str)

    run = CandidateRun(
        run_date=run_date_str,
        pick_date=pick_date_str,
        candidates=candidates,
        meta={
            "config": args.config,
            "data_dir": args.data,
            "total": len(candidates),
        },
    )

    resolved_output_dir = resolve_preselect_output_dir(
        config_path=args.config or None,
        output_dir=args.output or None,
    )

    paths = save_candidates(
        run,
        candidates_dir=resolved_output_dir,
    )

    logger.info("===== 初选完成 =====")
    logger.info("选股日期  : %s", pick_date_str)
    logger.info("候选数量  : %d 只", len(candidates))
    for key, path in paths.items():
        logger.info("%-8s → %s", key, path)

    # 终端摘要
    if candidates:
        print(f"\n{'代码':>8}  {'策略':>6}  {'收盘价':>8}  {'砖型增长':>10}")
        print("-" * 44)
        for c in candidates:
            bg = f"{c.brick_growth:.2f}x" if c.brick_growth is not None else "  —"
            print(f"{c.code:>8}  {c.strategy:>6}  {c.close:>8.2f}  {bg:>10}")
    else:
        print("\n(今日无候选股票)")


# =============================================================================
# CLI 解析
# =============================================================================

def _parse_horizons(value: str) -> tuple[int, ...]:
    horizons = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not horizons or any(h <= 0 for h in horizons):
        raise argparse.ArgumentTypeError("horizons must be positive integers, e.g. 1,5,10")
    return horizons


def _parse_strategies(value: str) -> tuple[str, ...]:
    strategies = tuple(part.strip().lower() for part in value.split(",") if part.strip())
    if not strategies:
        raise argparse.ArgumentTypeError("strategies must be strategy names, e.g. b1 or b1,brick")
    return strategies


def _parse_buy_mode(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in VALID_BUY_MODES:
        raise argparse.ArgumentTypeError(
            f"buy mode must be one of {', '.join(sorted(VALID_BUY_MODES))}"
        )
    return normalized


def cmd_signal_returns(args: argparse.Namespace) -> None:
    result = run_signal_returns(
        config_path=args.config or None,
        data_dir=args.data or None,
        start_date=args.start or None,
        end_date=args.end or None,
        output_dir=args.output or None,
        horizons=args.horizons,
        strategies=args.strategies,
        buy_mode=args.buy_mode,
    )

    summary = result["summary"]
    print("\nSignal return summary")
    print(f"signals: {summary['total_signals']}")
    for key, metrics in summary["metrics"].items():
        mean = metrics["mean_return"]
        median = metrics["median_return"]
        win_rate = metrics["win_rate"]
        if mean is None or median is None or win_rate is None:
            print(f"{key}: count=0")
        else:
            print(
                f"{key}: count={metrics['count']} "
                f"mean={mean:.4%} median={median:.4%} win_rate={win_rate:.2%}"
            )
    print(f"csv: {result['csv_path']}")
    print(f"summary: {result['summary_path']}")
    print(f"summary_csv: {result['summary_csv_path']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pipeline.cli",
        description="AgentTrader 量化初选 CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("preselect", help="运行量化初选")
    p.add_argument("--config", default=None, help="rules_preselect.yaml 路径")
    p.add_argument("--data",   default=None, help="CSV 数据目录（覆盖配置文件）")
    p.add_argument("--date",   default=None, help="选股基准日期 YYYY-MM-DD（默认最新）")
    p.add_argument("--end-date", dest="end_date", default=None,
                   help="数据截断日期（回测用）")
    p.add_argument("--output", default=None, help="候选输出目录（默认 data/candidates/）")
    p.add_argument("--log-dir", dest="log_dir", default=None,
                   help="流水日志目录（默认 data/logs/）")

    p = sub.add_parser("signal-returns", help="Evaluate future returns after selector signals")
    p.add_argument("--config", default=None, help="rules_preselect.yaml path")
    p.add_argument("--data", default=None, help="CSV data directory")
    p.add_argument("--start", default=None, help="Start signal date YYYY-MM-DD")
    p.add_argument("--end", default=None, help="End signal date YYYY-MM-DD")
    p.add_argument("--output", default=None, help="Output directory, default data/backtest")
    p.add_argument(
        "--horizons",
        type=_parse_horizons,
        default=DEFAULT_HORIZONS,
        help="Comma-separated holding horizons, default 1,5,10",
    )
    p.add_argument(
        "--strategies",
        type=_parse_strategies,
        default=None,
        help="Comma-separated strategies to backtest, e.g. b1, brick, or b1,brick",
    )
    p.add_argument(
        "--buy-mode",
        type=_parse_buy_mode,
        default=BUY_MODE_SIGNAL_CLOSE,
        help="Entry price mode: signal_close or next_open",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "preselect":
        cmd_preselect(args)
    elif args.command == "signal-returns":
        cmd_signal_returns(args)
    else:
        parser.print_help()
        sys.exit(1)
        
def test():
    """简单测试函数，验证 CLI 逻辑（不依赖外部数据）。"""
    class Args:
        command = "preselect"
        config = None
        data = None
        date = None
        end_date = None
        output = "./data/candidates"
        log_dir = "./data/logs"

    args = Args()
    cmd_preselect(args)


if __name__ == "__main__":
    main()  
    # test()
