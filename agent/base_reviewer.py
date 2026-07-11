"""Resumable and auditable base workflow for chart-review providers."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class ReviewRunIncomplete(RuntimeError):
    """Raised after partial artifacts are safely written for a run with failures."""

    def __init__(self, manifest: dict[str, Any]) -> None:
        self.manifest = manifest
        failed = manifest.get("failed_codes", [])
        super().__init__(
            f"review run {manifest.get('status')} with {len(failed)} failed codes: {failed}"
        )


class BaseReviewer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.prompt_path = Path(config["prompt_path"])
        self.prompt = self.load_prompt(self.prompt_path)
        self.kline_dir = Path(config["kline_dir"])
        self.output_dir = Path(config["output_dir"])
        self._chart_manifest_path: Path | None = None
        self._chart_manifest_sha256: str | None = None
        self._chart_results: dict[str, dict] = {}

    @staticmethod
    def load_prompt(prompt_path: Path) -> str:
        return prompt_path.read_text(encoding="utf-8")

    @staticmethod
    def load_candidates(path: Path) -> dict:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"cannot read candidates JSON {path}: {type(exc).__name__}"
            ) from exc
        if not isinstance(payload, dict):
            raise ValueError(f"candidates JSON must contain an object: {path}")
        return payload

    def find_chart_images(self, pick_date: str, code: str) -> Optional[Path]:
        date_dir = self.kline_dir / pick_date
        for suffix in (".jpg", ".png"):
            path = date_dir / f"{code}_day{suffix}"
            if path.is_file():
                return path
        return None

    @staticmethod
    def extract_json(text: str) -> dict:
        code_block = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if code_block:
            text = code_block.group(1)
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError(f"未能在模型输出中找到 JSON 对象:\n{text}")
        payload = json.loads(text[start:end])
        if not isinstance(payload, dict):
            raise ValueError("模型输出 JSON 必须是对象")
        return payload

    def review_stock(self, code: str, day_chart: Path, prompt: str) -> dict:
        """Subclasses call one provider and return a parsed JSON object."""
        raise NotImplementedError("子类必须实现 review_stock 方法")

    def generate_suggestion(
        self,
        pick_date: str,
        all_results: List[dict],
        min_score: float,
        *,
        failed_codes: List[str] | None = None,
    ) -> dict:
        failed = list(failed_codes or [])
        passed = [result for result in all_results if result["total_score"] >= min_score]
        excluded = [
            result["code"]
            for result in all_results
            if result["total_score"] < min_score
        ]
        passed.sort(key=lambda result: result["total_score"], reverse=True)
        review_candidates = [
            {
                "rank": index + 1,
                "code": result["code"],
                "verdict": result.get("verdict", ""),
                "total_score": result["total_score"],
                "signal_type": result.get("signal_type", ""),
                "comment": result.get("comment", ""),
            }
            for index, result in enumerate(passed)
        ]
        status = "complete" if not failed else ("partial" if all_results else "failed")
        return {
            "schema_version": 2,
            "status": status,
            "date": pick_date,
            "min_score_threshold": min_score,
            "total_reviewed": len(all_results),
            "failed_count": len(failed),
            "failed_codes": failed,
            "review_candidates": review_candidates,
            # Compatibility field for existing report readers.
            "recommendations": review_candidates,
            "excluded": excluded,
        }

    def run(self) -> dict[str, Any]:
        candidates_path = Path(self.config["candidates"])
        candidates_data = self.load_candidates(candidates_path)
        pick_date, candidates = self._validate_candidates(candidates_data)
        self._load_chart_export_manifest(
            pick_date,
            [candidate["code"] for candidate in candidates],
        )
        print(f"[INFO] pick_date={pick_date}，候选股票数={len(candidates)}")

        out_dir = self.output_dir / pick_date
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = out_dir / "run_manifest.json"
        suggestion_path = out_dir / "suggestion.json"
        all_results: List[dict] = []
        failed: list[dict[str, str]] = []
        reused_count = 0
        processed_count = 0

        self._write_manifest(
            manifest_path,
            self._manifest(
                pick_date=pick_date,
                candidates=candidates,
                status="in_progress",
                all_results=all_results,
                failed=failed,
                reused_count=reused_count,
                processed_count=processed_count,
                suggestion_path=suggestion_path,
            ),
        )

        for index, candidate in enumerate(candidates, 1):
            code = candidate["code"]
            out_file = out_dir / f"{code}.json"
            try:
                day_chart = self._validated_chart(pick_date, code)
            except (OSError, ValueError) as exc:
                print(f"[{index}/{len(candidates)}] {code} — 图表校验失败：{exc}")
                failed.append(
                    {
                        "code": code,
                        "reason": f"chart_validation: {type(exc).__name__}: {str(exc)[:300]}",
                    }
                )
                self._checkpoint_manifest(
                    manifest_path, pick_date, candidates, all_results, failed,
                    reused_count, processed_count, suggestion_path,
                )
                continue

            signature = self._review_signature(pick_date, code, day_chart)
            if self.config.get("skip_existing", False):
                result = self._load_reusable_result(out_file, signature, code)
                if result is not None:
                    print(f"[{index}/{len(candidates)}] {code} — 签名匹配，恢复已有结果。")
                    all_results.append(result)
                    reused_count += 1
                    self._checkpoint_manifest(
                        manifest_path, pick_date, candidates, all_results, failed,
                        reused_count, processed_count, suggestion_path,
                    )
                    continue
                if out_file.exists():
                    print(f"[{index}/{len(candidates)}] {code} — 已有结果无效或签名过期，重新复评。")
            elif out_file.exists():
                archived = _archive_existing_result(out_file)
                print(f"[{index}/{len(candidates)}] {code} — 强制重算，旧结果归档至 {archived}。")

            print(f"[{index}/{len(candidates)}] {code} — 正在复评 ...", end=" ", flush=True)
            requested = False
            try:
                requested = True
                raw_result = self.review_stock(code=code, day_chart=day_chart, prompt=self.prompt)
                result = self._validate_result(raw_result, code)
                result["_review_meta"] = {
                    "schema_version": 1,
                    "signature": signature,
                    "pick_date": pick_date,
                    "reviewer": self._reviewer_name,
                    "model": self.config.get("model"),
                    "prompt_sha256": self._prompt_sha256,
                    "chart_sha256": _file_sha256(day_chart),
                    "chart_export_manifest_sha256": self._chart_manifest_sha256,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                }
                _atomic_write_json(out_file, result)
                all_results.append(result)
                processed_count += 1
                print(
                    f"完成 — verdict={result.get('verdict', '?')}, "
                    f"score={result['total_score']}"
                )
            except Exception as exc:
                print(f"失败 — {type(exc).__name__}: {exc}")
                failed.append(
                    {
                        "code": code,
                        "reason": f"{type(exc).__name__}: {str(exc)[:300]}",
                    }
                )
            self._checkpoint_manifest(
                manifest_path, pick_date, candidates, all_results, failed,
                reused_count, processed_count, suggestion_path,
            )
            if requested and index < len(candidates):
                delay = float(self.config.get("request_delay", 5))
                if delay > 0:
                    time.sleep(delay)

        failed_codes = [item["code"] for item in failed]
        min_score = float(self.config.get("suggest_min_score", 4.0))
        suggestion = self.generate_suggestion(
            pick_date,
            all_results,
            min_score,
            failed_codes=failed_codes,
        )
        _atomic_write_json(suggestion_path, suggestion)
        manifest = self._manifest(
            pick_date=pick_date,
            candidates=candidates,
            status=suggestion["status"],
            all_results=all_results,
            failed=failed,
            reused_count=reused_count,
            processed_count=processed_count,
            suggestion_path=suggestion_path,
        )
        self._write_manifest(manifest_path, manifest)

        print(
            f"\n[INFO] 复评结束：状态={manifest['status']}，成功={len(all_results)}，"
            f"失败={len(failed_codes)}，恢复={reused_count}"
        )
        print(f"[INFO] 研究候选汇总：{suggestion_path}")
        print(f"[INFO] 运行清单：{manifest_path}")
        if failed_codes:
            raise ReviewRunIncomplete(manifest)
        return manifest

    @property
    def _reviewer_name(self) -> str:
        cls = type(self)
        return f"{cls.__module__}.{cls.__qualname__}"

    @property
    def _prompt_sha256(self) -> str:
        return hashlib.sha256(self.prompt.encode("utf-8")).hexdigest()

    def _review_signature(self, pick_date: str, code: str, chart: Path) -> str:
        payload = {
            "schema_version": 1,
            "reviewer": self._reviewer_name,
            "pick_date": pick_date,
            "code": code,
            "prompt_sha256": self._prompt_sha256,
            "chart_sha256": _file_sha256(chart),
            "chart_export_manifest_sha256": self._chart_manifest_sha256,
            "model": self.config.get("model"),
            "retry_models": list(self.config.get("retry_models") or []),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
        ).hexdigest()

    def _load_reusable_result(
        self,
        path: Path,
        signature: str,
        expected_code: str,
    ) -> dict | None:
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                return None
            metadata = payload.get("_review_meta")
            if not isinstance(metadata, dict) or metadata.get("signature") != signature:
                return None
            return self._validate_result(payload, expected_code)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError, TypeError):
            return None

    @staticmethod
    def _validate_result(result: object, expected_code: str) -> dict:
        if not isinstance(result, dict):
            raise ValueError("review result must be a JSON object")
        payload = dict(result)
        code = _normalize_symbol(payload.get("code", expected_code))
        if code != expected_code:
            raise ValueError(
                f"review result code mismatch: expected {expected_code}, got {code}"
            )
        score = payload.get("total_score")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError("review result total_score must be numeric")
        score = float(score)
        if not math.isfinite(score):
            raise ValueError("review result total_score must be finite")
        payload["code"] = code
        payload["total_score"] = score
        return payload

    @staticmethod
    def _validate_candidates(payload: dict) -> tuple[str, list[dict[str, Any]]]:
        pick_date = str(payload.get("pick_date", "")).strip()
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", pick_date):
            raise ValueError("candidates pick_date must use YYYY-MM-DD")
        raw_candidates = payload.get("candidates")
        if not isinstance(raw_candidates, list):
            raise ValueError("candidates field must be a list")
        candidates: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in raw_candidates:
            if not isinstance(item, dict):
                raise ValueError("each candidate must be a JSON object")
            code = _normalize_symbol(item.get("code"))
            if code in seen:
                raise ValueError(f"duplicate candidate code: {code}")
            seen.add(code)
            candidates.append({**item, "code": code})
        return pick_date, candidates

    def _checkpoint_manifest(
        self,
        path: Path,
        pick_date: str,
        candidates: list[dict[str, Any]],
        all_results: list[dict],
        failed: list[dict[str, str]],
        reused_count: int,
        processed_count: int,
        suggestion_path: Path,
    ) -> None:
        self._write_manifest(
            path,
            self._manifest(
                pick_date=pick_date,
                candidates=candidates,
                status="in_progress",
                all_results=all_results,
                failed=failed,
                reused_count=reused_count,
                processed_count=processed_count,
                suggestion_path=suggestion_path,
            ),
        )

    def _manifest(
        self,
        *,
        pick_date: str,
        candidates: list[dict[str, Any]],
        status: str,
        all_results: list[dict],
        failed: list[dict[str, str]],
        reused_count: int,
        processed_count: int,
        suggestion_path: Path,
    ) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "status": status,
            "pick_date": pick_date,
            "reviewer": self._reviewer_name,
            "model": self.config.get("model"),
            "prompt_sha256": self._prompt_sha256,
            "chart_export_manifest_path": str(self._chart_manifest_path),
            "chart_export_manifest_sha256": self._chart_manifest_sha256,
            "candidate_count": len(candidates),
            "success_count": len(all_results),
            "failed_count": len(failed),
            "failed_codes": [item["code"] for item in failed],
            "failures": failed,
            "reused_count": reused_count,
            "processed_count": processed_count,
            "suggestion_path": str(suggestion_path),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    def _load_chart_export_manifest(self, pick_date: str, codes: list[str]) -> None:
        path = self.kline_dir / pick_date / "export_manifest.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"cannot verify point-in-time charts from {path}: {type(exc).__name__}"
            ) from exc
        if not isinstance(payload, dict):
            raise ValueError("chart export manifest must contain a JSON object")
        if payload.get("status") != "complete":
            raise ValueError(
                f"chart export manifest must be complete, got {payload.get('status')!r}"
            )
        if payload.get("pick_date") != pick_date:
            raise ValueError("chart export manifest pick_date does not match candidates")
        if payload.get("failed_count", 0) != 0:
            raise ValueError("chart export manifest reports failed symbols")
        results = payload.get("results")
        if not isinstance(results, dict):
            raise ValueError("chart export manifest results must be an object")
        missing = sorted(set(codes) - set(results))
        if missing:
            raise ValueError(
                f"chart export manifest is missing {len(missing)} candidate symbols"
            )
        self._chart_manifest_path = path
        self._chart_manifest_sha256 = _file_sha256(path)
        self._chart_results = {
            code: result
            for code, result in results.items()
            if isinstance(result, dict)
        }

    def _validated_chart(self, pick_date: str, code: str) -> Path:
        metadata = self._chart_results.get(code)
        if not isinstance(metadata, dict):
            raise ValueError("symbol is absent from chart export manifest")
        if metadata.get("chart_end_date") != pick_date:
            raise ValueError(
                f"chart_end_date {metadata.get('chart_end_date')!r} does not match {pick_date}"
            )
        chart = self.find_chart_images(pick_date, code)
        if chart is None:
            raise ValueError("chart file is missing")
        expected_hash = metadata.get("output_sha256")
        if not isinstance(expected_hash, str) or _file_sha256(chart) != expected_hash:
            raise ValueError("chart file hash does not match export manifest")
        return chart

    @staticmethod
    def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
        _atomic_write_json(path, manifest)


def _normalize_symbol(value: object) -> str:
    symbol = str(value if value is not None else "").strip()
    if symbol.isdigit() and len(symbol) <= 6:
        symbol = symbol.zfill(6)
    if not re.fullmatch(r"\d{6}", symbol):
        raise ValueError(f"stock code must be a six-digit symbol: {value!r}")
    return symbol


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _archive_existing_result(path: Path) -> Path:
    archive_dir = path.parent / ".stale"
    archive_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archived = archive_dir / f"{path.stem}_{timestamp}{path.suffix}"
    os.replace(path, archived)
    return archived
