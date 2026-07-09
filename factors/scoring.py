"""Composite scoring for Alpha101 factor reports.

The primary score uses the 20-day horizon (70%) and the 10-day horizon
(30%).  The 1-day and 5-day horizons are used only by the five-point horizon
consistency diagnostic.

Run this module from the repository root to score every completed report::

    python -m factors.scoring
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import yaml

from factors.alpha101 import ALPHA101_NAMES


PRIMARY_WINDOWS = (10, 20)
CONSISTENCY_WINDOWS = (1, 5, 10, 20)
SCORE_COLUMNS = [
    "factor_name",
    "final_score",
    "signal_score",
    "tradability_score",
    "robustness_score",
    "penalty",
    "decision",
    "useful_horizons",
    "reason",
]

_DECISION_PRIORITY = {
    "disabled": 0,
    "component_only": 1,
    "low_priority_watch": 2,
    "watch": 3,
    "active": 4,
}


def parse_factor_metrics(df: pd.DataFrame) -> dict[str, object]:
    """Parse one FactorTester ``summary.csv`` into scoring metrics.

    ``df`` must contain the long-form columns ``section``, ``metric``,
    ``value`` and ``window``.  A constant ``factor_name`` column may be added
    by callers when several summary files are concatenated.
    """

    required = {"section", "metric", "value", "window"}
    missing_columns = required.difference(df.columns)
    if missing_columns:
        raise ValueError(
            "factor summary is missing columns: " + ", ".join(sorted(missing_columns))
        )
    if df.empty:
        raise ValueError("factor summary is empty")

    factor_name = _factor_name(df)
    metrics: dict[str, object] = {
        "factor_name": factor_name,
        "coverage": {
            "avg_coverage": _summary_value(df, "coverage", "avg_coverage"),
            "min_coverage": _summary_value(df, "coverage", "min_coverage"),
        },
        "windows": {},
    }

    windows: dict[int, dict[str, float]] = {}
    for window in CONSISTENCY_WINDOWS:
        windows[window] = {
            "rank_ic_mean": _summary_value(df, "ic", "rank_ic_mean", window),
            "rank_icir": _summary_value(df, "ic", "rank_icir", window),
            "rank_ic_win_rate": _summary_value(
                df, "ic", "rank_ic_win_rate", window
            ),
            "neutralized_rank_ic_mean": _summary_value(
                df, "neutralized_ic", "neutralized_rank_ic_mean", window
            ),
            "neutralized_rank_icir": _summary_value(
                df, "neutralized_ic", "neutralized_rank_icir", window
            ),
            "top_bottom_return": _summary_value(
                df, "group_return", "top_bottom_return", window
            ),
            "monotonic": _summary_value(df, "group_return", "monotonic", window),
            "stat_annualized_return": _summary_value(
                df, "stat_long_short", "annualized_return", window
            ),
            "stat_max_drawdown": _summary_value(
                df, "stat_long_short", "max_drawdown", window
            ),
            "stat_sharpe": _summary_value(
                df, "stat_long_short", "sharpe", window
            ),
            "stat_cum_nav": _summary_value(
                df, "stat_long_short", "stat_cum_nav", window
            ),
            "tradable_annualized_return": _summary_value(
                df, "tradable_long_short", "annualized_return", window
            ),
            "tradable_max_drawdown": _summary_value(
                df, "tradable_long_short", "max_drawdown", window
            ),
            "tradable_sharpe": _summary_value(
                df, "tradable_long_short", "sharpe", window
            ),
            "tradable_cum_nav": _summary_value(
                df, "tradable_long_short", "tradable_cum_nav", window
            ),
        }
    metrics["windows"] = windows
    return metrics


def score_one_factor(metrics: Mapping[str, object]) -> dict[str, object]:
    """Score one parsed factor and return an auditable result row."""

    windows = _windows(metrics)
    coverage = _coverage(metrics)

    signal_by_window = {
        window: _signal_score(windows[window]) for window in PRIMARY_WINDOWS
    }
    tradability_by_window = {
        window: _tradability_score(windows[window]) for window in PRIMARY_WINDOWS
    }
    signal_score = _combine_primary_windows(signal_by_window)
    tradability_score = _combine_primary_windows(tradability_by_window)

    coverage_score = _scaled_score(
        (_number(coverage.get("avg_coverage")) - 0.5) / 0.45,
        7.0,
    )
    horizon_score = _horizon_consistency_score(windows)
    top_bottom_score = _combine_primary_windows(
        {
            window: _scaled_score(
                _number(windows[window].get("top_bottom_return")) / 0.012,
                4.0,
            )
            for window in PRIMARY_WINDOWS
        }
    )
    monotonic_score = _combine_primary_windows(
        {
            window: 4.0
            if _number(windows[window].get("monotonic")) == 1.0
            else 0.0
            for window in PRIMARY_WINDOWS
        }
    )
    robustness_score = (
        coverage_score + horizon_score + top_bottom_score + monotonic_score
    )

    # The requested category weights sum to 90, so normalize the positive
    # subtotal to a real 100-point scale before applying deductions.
    positive_subtotal = signal_score + tradability_score + robustness_score
    normalized_base_score = positive_subtotal / 90.0 * 100.0
    deduction, penalty_reasons = _penalty(metrics)
    final_score = float(np.clip(normalized_base_score - deduction, 0.0, 100.0))
    decision, decision_reasons = _decision_details(final_score, metrics)
    horizons = _useful_horizons(metrics)

    missing_metrics = _missing_scoring_metrics(metrics)
    reason_parts = [
        f"base={normalized_base_score:.2f}",
        "20d/10d=70%/30%",
    ]
    reason_parts.extend(penalty_reasons)
    reason_parts.extend(decision_reasons)
    if missing_metrics:
        reason_parts.append("missing=" + ",".join(missing_metrics))

    return {
        "factor_name": str(metrics.get("factor_name", "unknown")),
        "final_score": round(final_score, 4),
        "signal_score": round(signal_score, 4),
        "tradability_score": round(tradability_score, 4),
        "robustness_score": round(robustness_score, 4),
        "penalty": round(-deduction, 4),
        "decision": decision,
        "useful_horizons": ",".join(horizons),
        "reason": "; ".join(reason_parts),
    }


def score_all_factors(df: pd.DataFrame) -> pd.DataFrame:
    """Score one or more concatenated FactorTester summary frames.

    For multiple factors, callers should add a ``factor_name`` source column
    before concatenation.  Sequential raw summaries are also supported when
    every block starts with its ``meta/factor_name`` row.
    """

    if df.empty:
        return pd.DataFrame(columns=SCORE_COLUMNS)

    rows = [score_one_factor(parse_factor_metrics(part)) for part in _factor_frames(df)]
    result = pd.DataFrame(rows, columns=SCORE_COLUMNS)
    return result.sort_values(
        ["final_score", "factor_name"], ascending=[False, True]
    ).reset_index(drop=True)


def assign_decision(score: float, metrics: Mapping[str, object]) -> str:
    """Assign the score tier, then apply coverage and tradability hard caps."""

    return _decision_details(float(score), metrics)[0]


def example_usage(
    report_root: str | Path = "factor_report/alpha101_batch",
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Score all ``alpha_*/summary.csv`` reports, optionally exporting CSV."""

    root = Path(report_root)
    frames: list[pd.DataFrame] = []
    for summary_path in sorted(root.glob("alpha_*/summary.csv")):
        frame = pd.read_csv(summary_path)
        frame["factor_name"] = summary_path.parent.name
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"no alpha_*/summary.csv reports found under {root}")

    scores = score_all_factors(pd.concat(frames, ignore_index=True))
    if output_path:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        scores.to_csv(destination, index=False)
        print(f"Scored {len(scores)} factors -> {destination}")
    else:
        print(f"Scored {len(scores)} factors")
    return scores


def update_factor_config(
    scores: pd.DataFrame,
    config_path: str | Path = "config/factors.yaml",
    *,
    score_source: str | Path = "factor_report/alpha101_batch/alpha_*/summary.csv",
) -> None:
    """Synchronize lifecycle status and score metadata to ``factors.yaml``.

    ``component_only`` and ``low_priority_watch`` remain ``watch`` lifecycle
    factors so future batch runs can re-evaluate them.  Factors without a
    completed score are disabled rather than inheriting an active default.
    """

    required = {"factor_name", "final_score", "decision", "useful_horizons"}
    missing = required.difference(scores.columns)
    if missing:
        raise ValueError("scores are missing columns: " + ", ".join(sorted(missing)))

    path = Path(config_path)
    existing = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    if existing is None:
        existing = {}
    if not isinstance(existing, dict):
        raise ValueError("factor config root must be a mapping")

    score_rows = {
        str(row["factor_name"]): row for _, row in scores.iterrows()
    }
    factor_entries: dict[str, dict[str, object]] = {}
    for factor_name in ALPHA101_NAMES:
        row = score_rows.get(factor_name)
        if row is None:
            factor_entries[factor_name] = {
                "decision": "unscored",
                "useful_horizons": [],
            }
            continue
        decision = str(row["decision"])
        horizons = [
            value.strip()
            for value in str(row["useful_horizons"]).split(",")
            if value.strip()
        ]
        factor_entries[factor_name] = {
            "final_score": round(float(row["final_score"]), 4),
            "decision": decision,
            "useful_horizons": horizons,
        }

    existing["default_status"] = "disabled"
    existing["factors"] = factor_entries
    existing["factor_scoring"] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": str(score_source),
        "primary_horizons": ["20d", "10d"],
        "horizon_weights": {"20d": 0.7, "10d": 0.3},
        "useful_horizon_rule": {
            "rank_ic_mean": "> 0",
            "rank_icir": "> 0",
            "tradable_sharpe": ">= 0.3",
            "tradable_annualized_return": ">= 0.03",
        },
        "lifecycle_mapping": {
            "active": "active",
            "watch": "watch",
            "low_priority_watch": "watch",
            "component_only": "watch",
            "disabled": "disabled",
            "unscored": "disabled",
        },
    }
    header = (
        "# Alpha101 factor lifecycle and generated composite-score metadata.\n"
        "# Regenerate with: python -m factors.scoring --update-config\n"
    )
    path.write_text(
        header + yaml.safe_dump(existing, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _signal_score(window_metrics: Mapping[str, object]) -> float:
    return sum(
        (
            _scaled_score(_number(window_metrics.get("rank_ic_mean")) / 0.06, 12.0),
            _scaled_score(_number(window_metrics.get("rank_icir")) / 0.8, 10.0),
            _scaled_score(
                (_number(window_metrics.get("rank_ic_win_rate")) - 0.5) / 0.25,
                5.0,
            ),
            _scaled_score(
                _number(window_metrics.get("neutralized_rank_ic_mean")) / 0.05,
                8.0,
            ),
        )
    )


def _tradability_score(window_metrics: Mapping[str, object]) -> float:
    return sum(
        (
            _scaled_score(
                _number(window_metrics.get("tradable_sharpe")) / 1.5,
                15.0,
            ),
            _scaled_score(
                _number(window_metrics.get("tradable_annualized_return")) / 0.15,
                8.0,
            ),
            _scaled_score(
                (0.35 - _number(window_metrics.get("tradable_max_drawdown")))
                / 0.35,
                7.0,
            ),
            _scaled_score(
                (_number(window_metrics.get("tradable_cum_nav")) - 1.0) / 1.0,
                5.0,
            ),
        )
    )


def _horizon_consistency_score(
    windows: Mapping[int, Mapping[str, object]],
) -> float:
    values = {
        window: _number(windows[window].get("rank_ic_mean"))
        for window in CONSISTENCY_WINDOWS
    }
    positive_count = sum(np.isfinite(value) and value > 0.0 for value in values.values())
    score = positive_count / len(CONSISTENCY_WINDOWS) * 3.0
    ordered_values = [values[window] for window in CONSISTENCY_WINDOWS]
    if all(np.isfinite(value) for value in ordered_values) and (
        values[20] > values[10] > values[5] > values[1]
    ):
        score += 2.0
    return min(score, 5.0)


def _useful_horizons(metrics: Mapping[str, object]) -> tuple[str, ...]:
    windows = _windows(metrics)
    useful: list[str] = []
    for window in PRIMARY_WINDOWS:
        values = windows[window]
        checks = (
            _number(values.get("rank_ic_mean")) > 0.0,
            _number(values.get("rank_icir")) > 0.0,
            _number(values.get("tradable_sharpe")) >= 0.3,
            _number(values.get("tradable_annualized_return")) >= 0.03,
        )
        if all(checks):
            useful.append(f"{window}d")
    return tuple(useful)


def _penalty(metrics: Mapping[str, object]) -> tuple[float, list[str]]:
    windows = _windows(metrics)
    coverage = _coverage(metrics)
    deduction = 0.0
    reasons: list[str] = []

    avg_coverage = _number(coverage.get("avg_coverage"))
    min_coverage = _number(coverage.get("min_coverage"))
    if np.isfinite(avg_coverage) and avg_coverage < 0.5:
        deduction += 8.0
        reasons.append("penalty:avg_coverage<0.5(-8)")
    elif np.isfinite(avg_coverage) and avg_coverage < 0.8:
        deduction += 4.0
        reasons.append("penalty:avg_coverage<0.8(-4)")
    if np.isfinite(min_coverage) and min_coverage < 0.5:
        deduction += 3.0
        reasons.append("penalty:min_coverage<0.5(-3)")

    stat_sharpe = _number(windows[20].get("stat_sharpe"))
    tradable_sharpe = _number(windows[20].get("tradable_sharpe"))
    if np.isfinite(stat_sharpe) and np.isfinite(tradable_sharpe):
        if stat_sharpe > 1.0 and tradable_sharpe < 0.5:
            deduction += 8.0
            reasons.append("penalty:tradable_collapse(-8)")
        elif stat_sharpe > 0.0:
            collapse_ratio = tradable_sharpe / stat_sharpe
            if collapse_ratio < 0.4:
                deduction += 5.0
                reasons.append("penalty:collapse_ratio<0.4(-5)")
            elif collapse_ratio < 0.6:
                deduction += 3.0
                reasons.append("penalty:collapse_ratio<0.6(-3)")

    # Monotonicity already contributes up to four positive points.  It is not
    # deducted again, which avoids double punishment for a binary diagnostic.
    return min(deduction, 20.0), reasons


def _decision_details(
    score: float,
    metrics: Mapping[str, object],
) -> tuple[str, list[str]]:
    if score >= 75.0:
        decision = "active"
    elif score >= 60.0:
        decision = "watch"
    elif score >= 45.0:
        decision = "component_only"
    else:
        decision = "disabled"

    windows = _windows(metrics)
    coverage = _coverage(metrics)
    avg_coverage = _number(coverage.get("avg_coverage"))
    sharpe_10d = _number(windows[10].get("tradable_sharpe"))
    sharpe_20d = _number(windows[20].get("tradable_sharpe"))
    return_20d = _number(windows[20].get("tradable_annualized_return"))
    reasons: list[str] = []

    critical = {
        "avg_coverage": avg_coverage,
        "tradable_sharpe_10d": sharpe_10d,
        "tradable_sharpe_20d": sharpe_20d,
        "tradable_annual_return_20d": return_20d,
    }
    missing_critical = [name for name, value in critical.items() if not np.isfinite(value)]
    if missing_critical:
        return "disabled", ["hard_rule:missing_critical_metrics"]

    # Only the two primary horizons together can force a full disable.  A bad
    # 1d/5d result never changes the decision tier.
    if sharpe_10d < 0.0 and sharpe_20d < 0.0:
        return "disabled", ["hard_rule:tradable_sharpe_10d_and_20d<0"]

    if avg_coverage < 0.5:
        capped = _cap_decision(decision, "low_priority_watch")
        if capped != decision:
            reasons.append("cap:avg_coverage<0.5=>low_priority_watch")
        decision = capped
    if sharpe_20d < 0.3:
        capped = _cap_decision(decision, "component_only")
        if capped != decision:
            reasons.append("cap:tradable_sharpe_20d<0.3=>component_only")
        decision = capped
    if return_20d < 0.03:
        capped = _cap_decision(decision, "component_only")
        if capped != decision:
            reasons.append("cap:tradable_return_20d<0.03=>component_only")
        decision = capped
    return decision, reasons


def _missing_scoring_metrics(metrics: Mapping[str, object]) -> list[str]:
    windows = _windows(metrics)
    missing: list[str] = []
    primary_names = (
        "rank_ic_mean",
        "rank_icir",
        "rank_ic_win_rate",
        "neutralized_rank_ic_mean",
        "top_bottom_return",
        "monotonic",
        "tradable_sharpe",
        "tradable_annualized_return",
        "tradable_max_drawdown",
        "tradable_cum_nav",
    )
    for window in PRIMARY_WINDOWS:
        for name in primary_names:
            if not np.isfinite(_number(windows[window].get(name))):
                missing.append(f"{name}_{window}d")
    for window in (1, 5):
        if not np.isfinite(_number(windows[window].get("rank_ic_mean"))):
            missing.append(f"rank_ic_mean_{window}d")
    if not np.isfinite(_number(_coverage(metrics).get("avg_coverage"))):
        missing.append("avg_coverage")
    return missing


def _combine_primary_windows(scores: Mapping[int, float]) -> float:
    return 0.7 * float(scores[20]) + 0.3 * float(scores[10])


def _scaled_score(raw_ratio: float, points: float) -> float:
    if not np.isfinite(raw_ratio):
        return 0.0
    return float(np.clip(raw_ratio, 0.0, 1.0) * points)


def _cap_decision(decision: str, maximum: str) -> str:
    if _DECISION_PRIORITY[decision] <= _DECISION_PRIORITY[maximum]:
        return decision
    return maximum


def _factor_frames(df: pd.DataFrame) -> list[pd.DataFrame]:
    if "factor_name" in df.columns:
        names = df["factor_name"].dropna().astype(str).str.strip()
        if names.empty:
            raise ValueError("factor_name column does not contain a factor name")
        ordered_names = names.drop_duplicates().tolist()
        return [
            df.loc[df["factor_name"].astype(str).str.strip().eq(name)].copy()
            for name in ordered_names
        ]

    required = {"section", "metric", "value"}
    if not required.issubset(df.columns):
        raise ValueError("cannot locate factor boundaries in the summary frame")
    marker = df["section"].eq("meta") & df["metric"].eq("factor_name")
    starts = np.flatnonzero(marker.to_numpy()).tolist()
    if not starts:
        raise ValueError("summary does not contain meta/factor_name")
    if len(starts) == 1:
        return [df.copy()]
    if starts[0] != 0:
        raise ValueError("rows before the first factor_name marker are ambiguous")
    boundaries = starts + [len(df)]
    return [df.iloc[start:end].copy() for start, end in zip(boundaries, boundaries[1:])]


def _factor_name(df: pd.DataFrame) -> str:
    if "factor_name" in df.columns:
        names = df["factor_name"].dropna().astype(str).str.strip().unique().tolist()
        if len(names) == 1 and names[0]:
            return names[0]
        if len(names) > 1:
            raise ValueError("parse_factor_metrics received more than one factor")
    mask = df["section"].eq("meta") & df["metric"].eq("factor_name")
    names = df.loc[mask, "value"].dropna().astype(str).str.strip().unique().tolist()
    if len(names) != 1 or not names[0]:
        raise ValueError("factor summary must contain exactly one factor_name")
    return names[0]


def _summary_value(
    df: pd.DataFrame,
    section: str,
    metric: str,
    window: int | None = None,
) -> float:
    mask = df["section"].eq(section) & df["metric"].eq(metric)
    if window is None:
        mask &= df["window"].isna()
    else:
        mask &= pd.to_numeric(df["window"], errors="coerce").eq(float(window))
    values = pd.to_numeric(df.loc[mask, "value"], errors="coerce")
    return float(values.iloc[-1]) if not values.empty else float("nan")


def _windows(metrics: Mapping[str, object]) -> Mapping[int, Mapping[str, object]]:
    windows = metrics.get("windows")
    if not isinstance(windows, Mapping):
        raise ValueError("metrics must contain a windows mapping")
    missing = set(CONSISTENCY_WINDOWS).difference(windows)
    if missing:
        raise ValueError(f"metrics are missing windows: {sorted(missing)}")
    return windows  # type: ignore[return-value]


def _coverage(metrics: Mapping[str, object]) -> Mapping[str, object]:
    coverage = metrics.get("coverage")
    if not isinstance(coverage, Mapping):
        raise ValueError("metrics must contain a coverage mapping")
    return coverage


def _number(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if np.isfinite(result) else float("nan")


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", default="factor_report/alpha101_batch")
    parser.add_argument("--output")
    parser.add_argument(
        "--update-config",
        action="store_true",
        help="sync scored lifecycle status and useful horizons to factors.yaml",
    )
    parser.add_argument("--factor-config", default="config/factors.yaml")
    args = parser.parse_args()

    scores = example_usage(args.report_root, args.output)
    if args.update_config:
        update_factor_config(
            scores,
            args.factor_config,
            score_source=Path(args.report_root) / "alpha_*/summary.csv",
        )
        print(f"Updated factor lifecycle -> {args.factor_config}")


if __name__ == "__main__":
    _main()
