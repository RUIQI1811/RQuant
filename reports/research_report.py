from __future__ import annotations

import datetime as dt
import json
import os
from html import escape
from pathlib import Path
from typing import Any, Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPORT_OUTPUT_DIR = Path("data") / "reports"


def _resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _fmt_percent(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.2%}"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_number(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def _candidate_summary(candidates_data: dict[str, Any], *, exists: bool) -> dict[str, Any]:
    candidates = candidates_data.get("candidates", [])
    if not isinstance(candidates, list):
        candidates = []

    by_strategy: dict[str, int] = {}
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        strategy = str(candidate.get("strategy") or "unknown")
        by_strategy[strategy] = by_strategy.get(strategy, 0) + 1

    return {
        "exists": exists,
        "run_date": candidates_data.get("run_date"),
        "pick_date": candidates_data.get("pick_date"),
        "count": len(candidates),
        "by_strategy": by_strategy,
    }


def _review_summary(review_data: dict[str, Any], *, exists: bool) -> dict[str, Any]:
    recommendations = review_data.get("recommendations", [])
    if not isinstance(recommendations, list):
        recommendations = []

    top_recommendations = []
    for item in recommendations[:10]:
        if not isinstance(item, dict):
            continue
        top_recommendations.append(
            {
                "rank": item.get("rank"),
                "code": item.get("code"),
                "total_score": item.get("total_score"),
                "verdict": item.get("verdict"),
                "signal_type": item.get("signal_type"),
                "comment": item.get("comment"),
            }
        )

    return {
        "exists": exists,
        "date": review_data.get("date"),
        "total_reviewed": review_data.get("total_reviewed", 0),
        "min_score_threshold": review_data.get("min_score_threshold"),
        "recommendation_count": len(recommendations),
        "top_recommendations": top_recommendations,
    }


def _signal_summary(signal_data: dict[str, Any], *, exists: bool) -> dict[str, Any]:
    return {
        "exists": exists,
        "run_date": signal_data.get("run_date"),
        "start_date": signal_data.get("start_date"),
        "end_date": signal_data.get("end_date"),
        "horizons": signal_data.get("horizons", []),
        "buy_mode": signal_data.get("buy_mode"),
        "total_signals": signal_data.get("total_signals", 0),
        "metrics": signal_data.get("metrics", {}),
    }


def _portfolio_summary(portfolio_data: dict[str, Any], *, exists: bool) -> dict[str, Any]:
    keys = [
        "run_date",
        "strategy",
        "buy_mode",
        "hold_days",
        "initial_cash",
        "final_cash",
        "total_return",
        "trade_count",
        "order_count",
        "open_position_count",
        "max_positions",
        "position_pct",
        "max_drawdown",
        "annualized_return_mean",
        "annualized_volatility",
        "sharpe_ratio",
        "commission_rate",
        "stamp_tax_rate",
        "transfer_fee_rate",
    ]
    summary = {key: portfolio_data.get(key) for key in keys}
    summary["exists"] = exists
    return summary


def build_research_summary(
    *,
    signal_dir: str | Path,
    portfolio_dir: str | Path,
    candidates_path: str | Path,
    review_path: Optional[str | Path] = None,
) -> dict[str, Any]:
    signal_summary_path = _resolve_path(signal_dir) / "signal_summary.json"
    portfolio_summary_path = _resolve_path(portfolio_dir) / "portfolio_summary.json"
    equity_curve_html_path = _resolve_path(portfolio_dir) / "equity_curve.html"
    daily_trade_plan_path = _resolve_path(portfolio_dir) / "daily_trade_plan.csv"
    resolved_candidates_path = _resolve_path(candidates_path)
    resolved_review_path = _resolve_path(review_path) if review_path else None

    candidates_data = _load_json(resolved_candidates_path)
    review_data = _load_json(resolved_review_path) if resolved_review_path else {}
    signal_data = _load_json(signal_summary_path)
    portfolio_data = _load_json(portfolio_summary_path)

    return {
        "run_date": dt.date.today().isoformat(),
        "sources": {
            "signal_summary": str(signal_summary_path),
            "portfolio_summary": str(portfolio_summary_path),
            "equity_curve_html": str(equity_curve_html_path) if equity_curve_html_path.exists() else None,
            "daily_trade_plan": str(daily_trade_plan_path) if daily_trade_plan_path.exists() else None,
            "candidates": str(resolved_candidates_path),
            "review": str(resolved_review_path) if resolved_review_path else None,
        },
        "candidates": _candidate_summary(
            candidates_data,
            exists=resolved_candidates_path.exists(),
        ),
        "review": _review_summary(
            review_data,
            exists=bool(resolved_review_path and resolved_review_path.exists()),
        ),
        "signal_returns": _signal_summary(
            signal_data,
            exists=signal_summary_path.exists(),
        ),
        "portfolio": _portfolio_summary(
            portfolio_data,
            exists=portfolio_summary_path.exists(),
        ),
    }


def _metrics_rows(metrics: dict[str, Any]) -> str:
    if not metrics:
        return "<tr><td colspan=\"5\">No signal metrics found.</td></tr>"

    rows = []
    for horizon, item in metrics.items():
        if not isinstance(item, dict):
            continue
        rows.append(
            "<tr>"
            f"<td>{escape(str(horizon))}</td>"
            f"<td>{escape(str(item.get('count', 0)))}</td>"
            f"<td>{escape(_fmt_percent(item.get('mean_return')))}</td>"
            f"<td>{escape(_fmt_percent(item.get('median_return')))}</td>"
            f"<td>{escape(_fmt_percent(item.get('win_rate')))}</td>"
            "</tr>"
        )
    return "\n".join(rows) or "<tr><td colspan=\"5\">No signal metrics found.</td></tr>"


def _recommendation_rows(recommendations: list[dict[str, Any]]) -> str:
    if not recommendations:
        return "<tr><td colspan=\"6\">No review recommendations found.</td></tr>"

    rows = []
    for item in recommendations:
        rows.append(
            "<tr>"
            f"<td>{escape(str(item.get('rank') or ''))}</td>"
            f"<td>{escape(str(item.get('code') or ''))}</td>"
            f"<td>{escape(_fmt_number(item.get('total_score'), 1))}</td>"
            f"<td>{escape(str(item.get('verdict') or ''))}</td>"
            f"<td>{escape(str(item.get('signal_type') or ''))}</td>"
            f"<td>{escape(str(item.get('comment') or ''))}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def render_research_html(summary: dict[str, Any], *, output_dir: str | Path) -> str:
    candidates = summary["candidates"]
    review = summary["review"]
    signal = summary["signal_returns"]
    portfolio = summary["portfolio"]
    sources = summary["sources"]

    equity_curve = sources.get("equity_curve_html")
    equity_link = "n/a"
    if equity_curve:
        rel = os.path.relpath(equity_curve, start=str(_resolve_path(output_dir)))
        equity_link = f"<a href=\"{escape(rel)}\">equity_curve.html</a>"
    daily_plan = sources.get("daily_trade_plan")
    daily_plan_link = "n/a"
    if daily_plan:
        rel = os.path.relpath(daily_plan, start=str(_resolve_path(output_dir)))
        daily_plan_link = f"<a href=\"{escape(rel)}\">daily_trade_plan.csv</a>"

    strategy_counts = ", ".join(
        f"{escape(str(name))}: {count}"
        for name, count in sorted(candidates.get("by_strategy", {}).items())
    ) or "n/a"

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>RQuant Research Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #1f2933; }}
    h1, h2 {{ margin-bottom: 8px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin: 16px 0 28px; }}
    .metric {{ border: 1px solid #d9e2ec; border-radius: 6px; padding: 12px; background: #f8fafc; }}
    .label {{ color: #52606d; font-size: 13px; }}
    .value {{ font-size: 22px; font-weight: 650; margin-top: 4px; }}
    table {{ width: 100%; border-collapse: collapse; margin: 12px 0 28px; }}
    th, td {{ border-bottom: 1px solid #d9e2ec; padding: 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f1f5f9; }}
    .muted {{ color: #627d98; }}
  </style>
</head>
<body>
  <h1>RQuant Research Report</h1>
  <p class="muted">Generated at {escape(str(summary.get("run_date") or ""))}</p>

  <h2>Run Overview</h2>
  <div class="grid">
    <div class="metric"><div class="label">Pick date</div><div class="value">{escape(str(candidates.get("pick_date") or "n/a"))}</div></div>
    <div class="metric"><div class="label">Candidates</div><div class="value">{escape(str(candidates.get("count", 0)))}</div></div>
    <div class="metric"><div class="label">Signals</div><div class="value">{escape(str(signal.get("total_signals", 0)))}</div></div>
    <div class="metric"><div class="label">Portfolio return</div><div class="value">{escape(_fmt_percent(portfolio.get("total_return")))}</div></div>
    <div class="metric"><div class="label">Max drawdown</div><div class="value">{escape(_fmt_percent(portfolio.get("max_drawdown")))}</div></div>
    <div class="metric"><div class="label">Sharpe</div><div class="value">{escape(_fmt_number(portfolio.get("sharpe_ratio"), 2))}</div></div>
  </div>

  <h2>Inputs</h2>
  <table>
    <tr><th>Item</th><th>Value</th></tr>
    <tr><td>Candidate strategies</td><td>{strategy_counts}</td></tr>
    <tr><td>Signal range</td><td>{escape(str(signal.get("start_date") or "n/a"))} to {escape(str(signal.get("end_date") or "n/a"))}</td></tr>
    <tr><td>Portfolio</td><td>{escape(str(portfolio.get("strategy") or "n/a"))} / {escape(str(portfolio.get("buy_mode") or "n/a"))} / hold {escape(str(portfolio.get("hold_days") or "n/a"))} bars</td></tr>
    <tr><td>Position sizing</td><td>max {escape(str(portfolio.get("max_positions") or "n/a"))} positions / {escape(_fmt_percent(portfolio.get("position_pct")))}</td></tr>
    <tr><td>Daily trade plan</td><td>{daily_plan_link}</td></tr>
    <tr><td>Equity curve</td><td>{equity_link}</td></tr>
  </table>

  <h2>Signal Returns</h2>
  <table>
    <tr><th>Horizon</th><th>Count</th><th>Mean</th><th>Median</th><th>Win rate</th></tr>
    {_metrics_rows(signal.get("metrics", {}))}
  </table>

  <h2>Portfolio Summary</h2>
  <table>
    <tr><th>Metric</th><th>Value</th></tr>
    <tr><td>Initial cash</td><td>{escape(_fmt_number(portfolio.get("initial_cash"), 2))}</td></tr>
    <tr><td>Final cash</td><td>{escape(_fmt_number(portfolio.get("final_cash"), 2))}</td></tr>
    <tr><td>Total return</td><td>{escape(_fmt_percent(portfolio.get("total_return")))}</td></tr>
    <tr><td>Trade count</td><td>{escape(str(portfolio.get("trade_count") or 0))}</td></tr>
    <tr><td>Order count</td><td>{escape(str(portfolio.get("order_count") or 0))}</td></tr>
    <tr><td>Open positions</td><td>{escape(str(portfolio.get("open_position_count") or 0))}</td></tr>
    <tr><td>Annualized volatility</td><td>{escape(_fmt_percent(portfolio.get("annualized_volatility")))}</td></tr>
    <tr><td>Sharpe ratio</td><td>{escape(_fmt_number(portfolio.get("sharpe_ratio"), 2))}</td></tr>
  </table>

  <h2>Review Recommendations</h2>
  <p class="muted">Review file present: {escape(str(review.get("exists", False)))}</p>
  <table>
    <tr><th>Rank</th><th>Code</th><th>Score</th><th>Verdict</th><th>Signal</th><th>Comment</th></tr>
    {_recommendation_rows(review.get("top_recommendations", []))}
  </table>
</body>
</html>
"""


def run_research_report(
    *,
    signal_dir: str | Path,
    portfolio_dir: str | Path,
    candidates_path: str | Path,
    review_path: Optional[str | Path] = None,
    output_dir: str | Path = DEFAULT_REPORT_OUTPUT_DIR,
) -> dict[str, Any]:
    resolved_output_dir = _resolve_path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    summary = build_research_summary(
        signal_dir=signal_dir,
        portfolio_dir=portfolio_dir,
        candidates_path=candidates_path,
        review_path=review_path,
    )
    json_path = resolved_output_dir / "research_report.json"
    html_path = resolved_output_dir / "research_report.html"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    html_path.write_text(
        render_research_html(summary, output_dir=resolved_output_dir),
        encoding="utf-8",
    )

    return {
        "summary": summary,
        "json_path": json_path,
        "html_path": html_path,
    }
