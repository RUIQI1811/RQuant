"""Batch ``FactorTester`` runner for user-supplied factor panels."""

from __future__ import annotations

import json
import hashlib
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from factors.external import ExternalFactorFrame, merge_external_with_market_data
from reports.alpha101_batch import (
    build_leaderboard,
    write_long_only_profitability_reports,
)
from reports.factor_tester import FactorTester, FactorTesterConfig


@dataclass(frozen=True)
class ExternalFactorBatchResult:
    output_dir: Path
    status: pd.DataFrame
    leaderboard: pd.DataFrame

    @property
    def failed_factors(self) -> tuple[str, ...]:
        if self.status.empty:
            return ()
        return tuple(
            self.status.loc[self.status["status"].eq("failed"), "factor"].astype(str)
        )


def run_external_factor_batch(
    external: ExternalFactorFrame,
    raw_data: Mapping[str, pd.DataFrame],
    *,
    output_dir: str | Path,
    tester_config: FactorTesterConfig,
    metadata: pd.DataFrame | None = None,
    factors: Sequence[str] | None = None,
    factor_statuses: Mapping[str, str] | None = None,
    factor_categories: Mapping[str, str] | None = None,
    fail_fast: bool = False,
    force: bool = False,
    data_signature: str | None = None,
) -> ExternalFactorBatchResult:
    """Run the full single-factor report suite for every external factor."""

    selected = tuple(factors or external.factors)
    missing = set(selected).difference(external.factors)
    if missing:
        raise ValueError(
            "external batch missing factors: " + ", ".join(sorted(missing))
        )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    research_frame = merge_external_with_market_data(
        external,
        raw_data,
        metadata=metadata,
    )
    context_columns = [
        column
        for column in research_frame.columns
        if column not in set(external.factors)
    ]
    base_fingerprint = _base_fingerprint(
        external,
        tester_config=tester_config,
        data_signature=data_signature,
    )
    statuses: list[dict[str, object]] = []
    for factor in selected:
        started = time.perf_counter()
        factor_dir = destination / factor
        factor_fingerprint = hashlib.sha256(
            f"{base_fingerprint}\0{factor}".encode("utf-8")
        ).hexdigest()
        if (
            not force
            and data_signature is not None
            and _can_resume_factor(factor_dir, factor_fingerprint)
        ):
            statuses.append(
                {
                    "factor": factor,
                    "status": "success",
                    "factor_status": (factor_statuses or {}).get(factor, "active"),
                    "factor_category": (factor_categories or {}).get(
                        factor, "unclassified"
                    ),
                    "duration_seconds": round(time.perf_counter() - started, 3),
                    "report_dir": str(factor_dir.resolve()),
                    "resumed": True,
                    "message": "",
                }
            )
            continue
        try:
            factor_frame = research_frame[[*context_columns, factor]].rename(
                columns={factor: tester_config.factor_col}
            )
            report_dir = FactorTester(
                factor_frame,
                factor_name=factor,
                config=tester_config,
            ).write_reports(destination)
            _atomic_write_json(
                report_dir / "external_run_metadata.json",
                {
                    "factor": factor,
                    "fingerprint": factor_fingerprint,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            statuses.append(
                {
                    "factor": factor,
                    "status": "success",
                    "factor_status": (factor_statuses or {}).get(factor, "active"),
                    "factor_category": (factor_categories or {}).get(
                        factor, "unclassified"
                    ),
                    "duration_seconds": round(time.perf_counter() - started, 3),
                    "report_dir": str(report_dir.resolve()),
                    "resumed": False,
                    "message": "",
                }
            )
        except Exception as exc:
            statuses.append(
                {
                    "factor": factor,
                    "status": "failed",
                    "factor_status": (factor_statuses or {}).get(factor, "active"),
                    "factor_category": (factor_categories or {}).get(
                        factor, "unclassified"
                    ),
                    "duration_seconds": round(time.perf_counter() - started, 3),
                    "report_dir": "",
                    "resumed": False,
                    "message": f"{type(exc).__name__}: {exc}",
                }
            )
            if fail_fast:
                break

    status = pd.DataFrame(statuses)
    successful = tuple(
        status.loc[status["status"].eq("success"), "factor"].astype(str)
    )
    leaderboard = build_leaderboard(
        destination,
        successful,
        factor_statuses=factor_statuses,
        factor_categories=factor_categories,
    )
    _atomic_write_csv(destination / "batch_status.csv", status)
    _atomic_write_csv(destination / "leaderboard.csv", leaderboard)
    write_long_only_profitability_reports(destination, leaderboard)
    _atomic_write_json(
        destination / "manifest.json",
        {
            "source": "external_factor_file",
            "factor_file": str(external.source_path),
            "factor_layout": external.source_layout,
            "selected_factors": list(selected),
            "successful_factors": list(successful),
            "failed_factors": list(
                status.loc[status["status"].eq("failed"), "factor"].astype(str)
            ),
            "factor_statuses": dict(factor_statuses or {}),
            "factor_categories": dict(factor_categories or {}),
            "base_fingerprint": base_fingerprint,
            "data_signature": data_signature,
            "tester_config": asdict(tester_config),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "outputs": {
                "status": "batch_status.csv",
                "leaderboard": "leaderboard.csv",
                "long_only_profitability": "long_only_profitability.csv",
                "profitable_long_only": "profitable_long_only.csv",
            },
        },
    )
    return ExternalFactorBatchResult(destination, status, leaderboard)


def _base_fingerprint(
    external: ExternalFactorFrame,
    *,
    tester_config: FactorTesterConfig,
    data_signature: str | None,
) -> str:
    digest = hashlib.sha256()
    with external.source_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    payload = {
        "source_sha256": digest.hexdigest(),
        "source_layout": external.source_layout,
        "tester_config": asdict(tester_config),
        "data_signature": data_signature,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _can_resume_factor(factor_dir: Path, fingerprint: str) -> bool:
    required = (
        factor_dir / "summary.csv",
        factor_dir / "ic_summary.csv",
        factor_dir / "horizon_effectiveness.csv",
        factor_dir / "annual_long_only.csv",
    )
    metadata_path = factor_dir / "external_run_metadata.json"
    if not metadata_path.exists() or not all(path.exists() for path in required):
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return metadata.get("fingerprint") == fingerprint


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)
