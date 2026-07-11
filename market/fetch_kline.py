from __future__ import annotations

import argparse
from contextlib import contextmanager
import datetime as dt
import hashlib
import json
import logging
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, List, Optional
import os

import numpy as np
import pandas as pd
import tushare as ts
import yaml
from tqdm import tqdm

from domain.market import FetchResult

# --------------------------- 全局日志配置 --------------------------- #
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_LOG_DIR = _PROJECT_ROOT / "data" / "logs"

def _read_dotenv_value(
    key_name: str,
    env_path: Path = _PROJECT_ROOT / ".env",
) -> str | None:
    """Read one non-empty .env value without mutating process environment."""
    if not env_path.exists():
        return None
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key == key_name and value:
            return value
    return None


@contextmanager
def _temporary_tushare_network_env():
    """Add Tushare bypass hosts for this run and restore both variables afterward."""

    names = ("NO_PROXY", "no_proxy")
    previous = {name: os.environ.get(name) for name in names}
    required = ("api.waditu.com", ".waditu.com", "waditu.com")
    for name in names:
        existing = [part.strip() for part in (previous[name] or "").split(",") if part.strip()]
        merged = existing + [host for host in required if host not in existing]
        os.environ[name] = ",".join(merged)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

def _resolve_cfg_path(path_like: str | Path, base_dir: Path = _PROJECT_ROOT) -> Path:
    """将配置中的路径统一解析为绝对路径：相对路径基于项目根目录。"""
    p = Path(path_like)
    return p if p.is_absolute() else (base_dir / p)

def _default_log_path() -> Path:
    today = dt.date.today().strftime("%Y-%m-%d")
    return _DEFAULT_LOG_DIR / f"fetch_{today}.log"

def setup_logging(log_path: Optional[Path] = None) -> logging.FileHandler:
    """初始化日志：同时输出到 stdout 和指定文件。"""
    if log_path is None:
        log_path = _default_log_path()
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d %(message)s"
    )
    if not any(
        isinstance(handler, logging.StreamHandler)
        and not isinstance(handler, logging.FileHandler)
        for handler in root.handlers
    ):
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(formatter)
        console._rquant_fetch_console = True  # type: ignore[attr-defined]
        root.addHandler(console)
    for handler in list(root.handlers):
        if getattr(handler, "_rquant_fetch_file", False):
            root.removeHandler(handler)
            handler.close()
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler._rquant_fetch_file = True  # type: ignore[attr-defined]
    root.addHandler(file_handler)
    return file_handler

logger = logging.getLogger("fetch_from_stocklist")

# --------------------------- 限流/封禁处理配置 --------------------------- #
COOLDOWN_SECS = 600
DEFAULT_MAX_REQUESTS_PER_MINUTE = 180
BAN_PATTERNS = (
    "访问频繁", "请稍后", "超过频率", "频繁访问",
    "too many requests", "429",
    "forbidden", "403",
    "max retries exceeded"
)

def _looks_like_ip_ban(exc: Exception) -> bool:
    msg = (str(exc) or "").lower()
    return any(pat in msg for pat in BAN_PATTERNS)

class RateLimitError(RuntimeError):
    """表示命中限流/封禁，需要长时间冷却后重试。"""
    pass


class FetchExhaustedError(RuntimeError):
    """One symbol exhausted all retries without a safely written result."""

    def __init__(self, code: str, attempts: int, last_error: Exception) -> None:
        self.code = code
        self.attempts = attempts
        self.last_error = last_error
        self.rate_limited = isinstance(last_error, RateLimitError) or _looks_like_ip_ban(last_error)
        super().__init__(
            f"{code} failed after {attempts} attempts: "
            f"{type(last_error).__name__}: {last_error}"
        )

def _cool_sleep(base_seconds: int) -> None:
    jitter = random.uniform(0.9, 1.2)
    sleep_s = max(1, int(base_seconds * jitter))
    logger.warning("疑似被限流/封禁，进入冷却期 %d 秒...", sleep_s)
    time.sleep(sleep_s)


class RequestRateLimiter:
    """Thread-safe, evenly spaced request-start limiter for Tushare calls."""

    def __init__(
        self,
        max_requests_per_minute: int,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_requests_per_minute <= 0:
            raise ValueError("max_requests_per_minute must be positive")
        self.max_requests_per_minute = int(max_requests_per_minute)
        self._interval = 60.0 / self.max_requests_per_minute
        self._clock = clock
        self._sleeper = sleeper
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def wait(self) -> None:
        """Reserve the next request slot, sleeping while holding the reservation lock."""

        with self._lock:
            now = self._clock()
            delay = self._next_allowed - now
            if delay > 0:
                self._sleeper(delay)
                now = self._clock()
            self._next_allowed = max(self._next_allowed, now) + self._interval

# --------------------------- 历史K线（Tushare 日线，固定qfq） --------------------------- #
pro: Optional[ts.pro_api] = None  # 模块级会话

def set_api(session) -> None:
    """由外部(比如GUI)注入已创建好的 ts.pro_api() 会话"""
    global pro
    pro = session
    

def _to_ts_code(code: str) -> str:
    """把6位code映射到标准 ts_code 后缀。"""
    code = str(code).zfill(6)
    if code.startswith(("60", "68", "9")):
        return f"{code}.SH"
    elif code.startswith(("4", "8")):
        return f"{code}.BJ"
    else:
        return f"{code}.SZ"

def _get_kline_tushare(
    code: str,
    start: str,
    end: str,
    *,
    rate_limiter: RequestRateLimiter | None = None,
) -> pd.DataFrame:
    ts_code = _to_ts_code(code)
    if rate_limiter is not None:
        rate_limiter.wait()
    try:
        df = ts.pro_bar(
            ts_code=ts_code,
            adj="qfq",
            start_date=start,
            end_date=end,
            freq="D",
            api=pro
        )
    except Exception as e:
        if _looks_like_ip_ban(e):
            raise RateLimitError(str(e)) from e
        raise

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.rename(columns={"trade_date": "date", "vol": "volume"}).copy()
    base_cols = ["date", "open", "close", "high", "low", "volume"]
    optional_cols = [col for col in ("pre_close", "change", "pct_chg", "amount") if col in df.columns]
    df = df[base_cols + optional_cols].copy()
    df["date"] = pd.to_datetime(df["date"])
    for c in ["open", "close", "high", "low", "volume", "pre_close", "change", "pct_chg", "amount"]:
        if c not in df.columns:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values("date").reset_index(drop=True)

def validate(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    # When old and newly fetched rows overlap, the newest API response wins.
    df = df.drop_duplicates(subset="date", keep="last").sort_values("date").reset_index(drop=True)
    if df["date"].isna().any():
        raise ValueError("存在缺失日期！")
    if (df["date"] > pd.Timestamp.today()).any():
        raise ValueError("数据包含未来日期，可能抓取错误！")
    return df


_BASE_KLINE_COLUMNS = ["date", "open", "close", "high", "low", "volume"]
_OPTIONAL_KLINE_COLUMNS = ["pre_close", "change", "pct_chg", "amount"]


def _empty_kline_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=_BASE_KLINE_COLUMNS + _OPTIONAL_KLINE_COLUMNS)


def _read_existing_kline(csv_path: Path) -> pd.DataFrame:
    """Read and normalize one existing CSV; raise if it cannot be reused safely."""

    existing = pd.read_csv(csv_path)
    existing.columns = [str(column).lower() for column in existing.columns]
    missing = set(_BASE_KLINE_COLUMNS).difference(existing.columns)
    if missing:
        raise ValueError(f"缺少必需列: {', '.join(sorted(missing))}")
    existing["date"] = pd.to_datetime(existing["date"], errors="coerce")
    for column in (_BASE_KLINE_COLUMNS + _OPTIONAL_KLINE_COLUMNS):
        if column != "date" and column in existing.columns:
            existing[column] = pd.to_numeric(existing[column], errors="coerce")
    return validate(existing)


def _ordered_kline_columns(df: pd.DataFrame) -> list[str]:
    preferred = _BASE_KLINE_COLUMNS + _OPTIONAL_KLINE_COLUMNS
    return [column for column in preferred if column in df.columns] + [
        column for column in df.columns if column not in preferred
    ]


def _qfq_history_changed(existing: pd.DataFrame, fetched: pd.DataFrame) -> bool:
    """Detect a changed qfq adjustment on overlapping dates."""

    if existing.empty or fetched.empty:
        return False
    price_columns = [column for column in ("open", "close", "high", "low") if column in fetched]
    overlap = existing[["date", *price_columns]].merge(
        fetched[["date", *price_columns]],
        on="date",
        how="inner",
        suffixes=("_old", "_new"),
    )
    if overlap.empty:
        return False
    return any(
        not np.allclose(
            overlap[f"{column}_old"],
            overlap[f"{column}_new"],
            rtol=1e-7,
            atol=1e-8,
            equal_nan=True,
        )
        for column in price_columns
    )


def _write_kline_atomic(df: pd.DataFrame, csv_path: Path) -> None:
    """Replace a CSV only after the complete merged frame has been written."""

    output = df.loc[:, _ordered_kline_columns(df)]
    temp_path = csv_path.with_suffix(f"{csv_path.suffix}.tmp")
    output.to_csv(temp_path, index=False)
    temp_path.replace(csv_path)

# --------------------------- 读取 stocklist.csv & 过滤板块 --------------------------- #

def _filter_by_boards_stocklist(df: pd.DataFrame, exclude_boards: set[str]) -> pd.DataFrame:
    ts = df["ts_code"].astype(str).str.upper()
    num = ts.str.extract(r"(\d{6})", expand=False).str.zfill(6)
    mask = pd.Series(True, index=df.index)

    if "gem" in exclude_boards:
        mask &= ~((ts.str.endswith(".SZ")) & num.str.startswith(("300", "301")))
    if "star" in exclude_boards:
        mask &= ~((ts.str.endswith(".SH")) & num.str.startswith(("688",)))
    if "bj" in exclude_boards:
        mask &= ~((ts.str.endswith(".BJ")) | num.str.startswith(("4", "8")))

    return df[mask].copy()


def load_codes_from_stocklist(stocklist_csv: Path, exclude_boards: set[str]) -> List[str]:
    df = pd.read_csv(stocklist_csv)    
    df = _filter_by_boards_stocklist(df, exclude_boards)
    codes = df["symbol"].astype(str).str.zfill(6).tolist()
    codes = list(dict.fromkeys(codes))  # 去重保持顺序
    logger.info("从 %s 读取到 %d 只股票（排除板块：%s）",
                stocklist_csv, len(codes), ",".join(sorted(exclude_boards)) or "无")
    return codes

# --------------------------- 单只抓取（指定区间覆盖，qfq 变化时全量刷新） --------------------------- #
def fetch_one(
    code: str,
    start: str,
    end: str,
    out_dir: Path,
    rate_limiter: RequestRateLimiter | None = None,
):
    csv_path = out_dir / f"{code}.csv"
    requested_start = pd.Timestamp(dt.datetime.strptime(start, "%Y%m%d").date())
    requested_end = pd.Timestamp(dt.datetime.strptime(end, "%Y%m%d").date())
    existing = pd.DataFrame()

    if csv_path.exists():
        try:
            existing = _read_existing_kline(csv_path)
        except Exception as exc:
            logger.warning("%s 现有 CSV 无法安全复用，将全量刷新：%s", code, exc)
            existing = pd.DataFrame()

    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            new_df = _get_kline(
                code,
                start,
                end,
                rate_limiter=rate_limiter,
            )
            if new_df.empty:
                if not existing.empty:
                    return "no_new_data"
                _write_kline_atomic(_empty_kline_frame(), csv_path)
                return "empty"

            new_df = validate(new_df)
            if not existing.empty and _qfq_history_changed(existing, new_df):
                logger.info("%s 检测到 qfq 历史价格变化，改为全量刷新", code)
                full_start = min(requested_start, existing["date"].min()).strftime("%Y%m%d")
                full_end = max(requested_end, existing["date"].max()).strftime("%Y%m%d")
                refreshed = validate(
                    _get_kline(
                        code,
                        full_start,
                        full_end,
                        rate_limiter=rate_limiter,
                    )
                )
                if refreshed.empty:
                    raise ValueError("qfq 全量刷新返回空数据")
                merged = refreshed
                outcome = "refreshed"
            elif not existing.empty:
                existing_dates = set(existing["date"])
                has_new_dates = any(date not in existing_dates for date in new_df["date"])
                outside_range = existing.loc[
                    (existing["date"] < requested_start) | (existing["date"] > requested_end)
                ]
                merged = pd.concat([outside_range, new_df], ignore_index=True, sort=False)
                outcome = "updated" if has_new_dates else "overwritten"
            else:
                merged = new_df
                outcome = "created"

            merged = validate(merged)
            _write_kline_atomic(merged, csv_path)
            return outcome
        except Exception as e:
            last_error = e
            if attempt >= 3:
                logger.error("%s 第 %d 次抓取失败，不再等待：%s", code, attempt, e)
                continue
            if _looks_like_ip_ban(e):
                logger.error(f"{code} 第 {attempt} 次抓取疑似被封禁，沉睡 {COOLDOWN_SECS} 秒")
                _cool_sleep(COOLDOWN_SECS)
            else:
                silent_seconds = 30 * attempt
                logger.info(f"{code} 第 {attempt} 次抓取失败，{silent_seconds} 秒后重试：{e}")
                time.sleep(silent_seconds)
    if last_error is None:
        last_error = RuntimeError("fetch ended without an outcome")
    logger.error("%s 三次抓取均失败！", code)
    raise FetchExhaustedError(code, 3, last_error)


def _get_kline(
    code: str,
    start: str,
    end: str,
    *,
    rate_limiter: RequestRateLimiter | None,
) -> pd.DataFrame:
    """Keep legacy three-argument mocks compatible when throttling is disabled."""

    if rate_limiter is None:
        return _get_kline_tushare(code, start, end)
    return _get_kline_tushare(
        code,
        start,
        end,
        rate_limiter=rate_limiter,
    )



# --------------------------- 配置加载 --------------------------- #
_CONFIG_PATH = Path(__file__).parent.parent / "config" / "fetch_kline.yaml"

def _load_config(config_path: Path = _CONFIG_PATH) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"找不到配置文件：{config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    logger.info("已加载配置文件：%s", config_path.resolve())
    return cfg


# --------------------------- 主入口 --------------------------- #
def run_fetch(
    *,
    config_path: str | Path = _CONFIG_PATH,
    log_path: Optional[str | Path] = None,
    start: str | None = None,
    end: str | None = None,
    out_dir: str | Path | None = None,
    workers: int | None = None,
    max_requests_per_minute: int | None = None,
    max_symbols: int | None = None,
    manifest_path: str | Path | None = None,
    resume: bool = False,
) -> FetchResult:
    global pro
    resolved_config = _resolve_cfg_path(config_path)
    cfg = _load_config(resolved_config)
    if log_path is None:
        cfg_log = cfg.get("log")
        resolved_log = _resolve_cfg_path(cfg_log) if cfg_log else _default_log_path()
    else:
        resolved_log = _resolve_cfg_path(log_path)
    root_logger = logging.getLogger()
    handlers_before = set(root_logger.handlers)
    previous_pro = pro
    file_handler = setup_logging(resolved_log)
    logger.info("日志文件：%s", resolved_log.resolve())
    try:
        with _temporary_tushare_network_env():
            return _execute_fetch(
                cfg=cfg,
                resolved_config=resolved_config,
                resolved_log=resolved_log,
                start=start,
                end=end,
                out_dir=out_dir,
                workers=workers,
                max_requests_per_minute=max_requests_per_minute,
                max_symbols=max_symbols,
                manifest_path=manifest_path,
                resume=resume,
            )
    finally:
        pro = previous_pro
        root_logger.removeHandler(file_handler)
        file_handler.close()
        for handler in list(root_logger.handlers):
            if (
                handler not in handlers_before
                and getattr(handler, "_rquant_fetch_console", False)
            ):
                root_logger.removeHandler(handler)
                handler.close()


def _execute_fetch(
    *,
    cfg: dict,
    resolved_config: Path,
    resolved_log: Path,
    start: str | None,
    end: str | None,
    out_dir: str | Path | None,
    workers: int | None,
    max_requests_per_minute: int | None,
    max_symbols: int | None,
    manifest_path: str | Path | None,
    resume: bool,
) -> dict[str, object]:
    ts_token = os.environ.get("TUSHARE_TOKEN") or _read_dotenv_value("TUSHARE_TOKEN")
    if not ts_token:
        raise ValueError(
            "请先设置环境变量 TUSHARE_TOKEN，或在项目根目录 .env 中写入 TUSHARE_TOKEN=你的token"
        )
    global pro
    pro = ts.pro_api(ts_token)

    raw_start = start if start is not None else str(cfg.get("start", "20190101"))
    raw_end = end if end is not None else str(cfg.get("end", "today"))
    resolved_start = _normalize_request_date(raw_start)
    resolved_end = _normalize_request_date(raw_end)
    if resolved_start > resolved_end:
        raise ValueError("start must not be after end")
    resolved_out = _resolve_cfg_path(
        out_dir if out_dir is not None else cfg.get("out", "./data")
    )
    resolved_out.mkdir(parents=True, exist_ok=True)

    stocklist_path = _resolve_cfg_path(cfg.get("stocklist", "./config/stocklist.csv"))
    exclude_boards = set(cfg.get("exclude_boards") or [])
    codes = load_codes_from_stocklist(stocklist_path, exclude_boards)
    if max_symbols is not None:
        if max_symbols <= 0:
            raise ValueError("max_symbols must be positive")
        codes = codes[:max_symbols]
    if not codes:
        raise ValueError("stocklist 为空或被过滤后无代码，请检查")
    resolved_workers = int(workers if workers is not None else cfg.get("workers", 8))
    if resolved_workers <= 0:
        raise ValueError("workers must be positive")
    resolved_max_requests_per_minute = int(
        max_requests_per_minute
        if max_requests_per_minute is not None
        else cfg.get("max_requests_per_minute", DEFAULT_MAX_REQUESTS_PER_MINUTE)
    )
    if resolved_max_requests_per_minute < 0:
        raise ValueError("max_requests_per_minute must be non-negative")
    rate_limiter = (
        RequestRateLimiter(resolved_max_requests_per_minute)
        if resolved_max_requests_per_minute
        else None
    )

    resolved_manifest = (
        _resolve_cfg_path(manifest_path)
        if manifest_path is not None
        else resolved_out / "_fetch_manifest.json"
    )
    run_signature = _fetch_signature(
        start=resolved_start,
        end=resolved_end,
        output_dir=resolved_out,
        codes=codes,
    )
    completed: dict[str, str] = {}
    resumed_count = 0
    if resume and resolved_manifest.is_file():
        previous = _load_fetch_manifest(resolved_manifest)
        if previous.get("run_signature") != run_signature:
            raise ValueError(
                "cannot resume fetch: manifest signature does not match dates, output, or stock universe"
            )
        previous_completed = previous.get("completed", {})
        if not isinstance(previous_completed, dict):
            raise ValueError("cannot resume fetch: manifest completed field is invalid")
        completed = {
            code: str(outcome)
            for code, outcome in previous_completed.items()
            if code in codes
        }
        resumed_count = len(completed)

    submitted_codes = [code for code in codes if code not in completed]
    remaining = set(submitted_codes)
    failures: dict[str, dict[str, object]] = {}
    started_at = dt.datetime.now(dt.timezone.utc).isoformat()

    def checkpoint(status: str) -> dict[str, object]:
        outcomes = _count_outcomes(completed)
        payload: dict[str, object] = {
            "schema_version": 1,
            "status": status,
            "run_signature": run_signature,
            "started_at": started_at,
            "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "config_path": str(resolved_config),
            "log_path": str(resolved_log),
            "start": resolved_start,
            "end": resolved_end,
            "output_dir": str(resolved_out),
            "stocklist_path": str(stocklist_path),
            "workers": resolved_workers,
            "max_requests_per_minute": resolved_max_requests_per_minute,
            "symbol_count": len(codes),
            "submitted_count": len(submitted_codes),
            "resumed_count": resumed_count,
            "completed_count": len(completed),
            "failed_count": len(failures),
            "pending_count": len(remaining),
            "outcomes": outcomes,
            "completed": dict(sorted(completed.items())),
            "failures": {code: failures[code] for code in sorted(failures)},
            "failed_codes": sorted(failures),
            "pending_codes": sorted(remaining),
        }
        _atomic_write_json(resolved_manifest, payload)
        return payload

    logger.info(
        "开始抓取 %d 支股票（提交=%d，恢复=%d） | 数据源:Tushare(日线,qfq) | "
        "日期:%s → %s | 排除:%s | 主动节流:%s",
        len(codes), len(submitted_codes), resumed_count,
        resolved_start, resolved_end, ",".join(sorted(exclude_boards)) or "无",
        (
            f"{resolved_max_requests_per_minute} 次/分钟"
            if resolved_max_requests_per_minute
            else "关闭"
        ),
    )
    checkpoint("in_progress")
    try:
        with ThreadPoolExecutor(max_workers=resolved_workers) as executor:
            futures = {
                executor.submit(
                    fetch_one,
                    code,
                    resolved_start,
                    resolved_end,
                    resolved_out,
                    rate_limiter,
                ): code
                for code in submitted_codes
            }
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="下载进度",
            ):
                code = futures[future]
                remaining.discard(code)
                try:
                    outcome = str(future.result())
                except Exception as exc:
                    failures[code] = _fetch_failure_payload(exc)
                    logger.error("%s 抓取失败：%s", code, exc)
                else:
                    completed[code] = outcome
                    failures.pop(code, None)
                checkpoint("in_progress")
    except BaseException:
        checkpoint("interrupted")
        raise

    status = "partial" if failures else "complete"
    manifest = checkpoint(status)
    outcomes = manifest["outcomes"]
    logger.info(
        "抓取结束：状态=%s，新建=%d，增量更新=%d，重叠日覆盖=%d，qfq全量刷新=%d，"
        "暂无新数据=%d，失败=%d | manifest=%s",
        status,
        outcomes.get("created", 0),
        outcomes.get("updated", 0),
        outcomes.get("overwritten", 0),
        outcomes.get("refreshed", 0),
        outcomes.get("no_new_data", 0) + outcomes.get("empty", 0),
        len(failures),
        resolved_manifest,
    )
    return FetchResult(
        config_path=resolved_config,
        log_path=resolved_log,
        start=resolved_start,
        end=resolved_end,
        output_dir=resolved_out,
        manifest_path=resolved_manifest,
        workers=resolved_workers,
        max_requests_per_minute=resolved_max_requests_per_minute,
        symbol_count=len(codes),
        submitted_count=len(submitted_codes),
        resumed_count=resumed_count,
        outcomes=dict(outcomes),
        failed_codes=tuple(sorted(failures)),
        ok=not failures,
    )


def _fetch_signature(
    *,
    start: str,
    end: str,
    output_dir: Path,
    codes: list[str],
) -> str:
    payload = {
        "schema_version": 1,
        "source": "tushare_daily_qfq",
        "start": start,
        "end": end,
        "output_dir": str(output_dir.resolve()),
        "codes": codes,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def _load_fetch_manifest(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"cannot resume fetch from manifest {path}: {type(exc).__name__}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(f"fetch manifest must contain a JSON object: {path}")
    return payload


def _fetch_failure_payload(exc: Exception) -> dict[str, object]:
    if isinstance(exc, FetchExhaustedError):
        return {
            "error_type": type(exc.last_error).__name__,
            "message": str(exc.last_error)[:500],
            "attempts": exc.attempts,
            "rate_limited": exc.rate_limited,
        }
    return {
        "error_type": type(exc).__name__,
        "message": str(exc)[:500],
        "attempts": None,
        "rate_limited": _looks_like_ip_ban(exc),
    }


def _count_outcomes(completed: dict[str, str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for outcome in completed.values():
        counts[outcome] = counts.get(outcome, 0) + 1
    return counts


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _normalize_request_date(value: str) -> str:
    raw = str(value).strip()
    if raw.lower() == "today":
        return dt.date.today().strftime("%Y%m%d")
    try:
        return pd.Timestamp(raw).strftime("%Y%m%d")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid request date: {value!r}") from exc


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", default=str(_CONFIG_PATH), help="fetch_kline YAML path")
    parser.add_argument("--start", default=None, help="YYYYMMDD, YYYY-MM-DD, or today")
    parser.add_argument("--end", default=None, help="YYYYMMDD, YYYY-MM-DD, or today")
    parser.add_argument("--out", default=None, help="Override output directory")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--max-requests-per-minute",
        type=int,
        default=None,
        help="Evenly throttle Tushare calls; 0 disables, default comes from YAML (180)",
    )
    parser.add_argument("--max-symbols", type=int, default=None, help="Smoke-test symbol limit")
    parser.add_argument("--log", default=None, help="Override log file path")
    parser.add_argument(
        "--manifest",
        default=None,
        help="Checkpoint JSON path; defaults to <out>/_fetch_manifest.json",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed symbols from a signature-matching manifest",
    )
    return parser


def run_from_args(args: argparse.Namespace) -> FetchResult:
    return run_fetch(
        config_path=args.config,
        log_path=args.log,
        start=args.start,
        end=args.end,
        out_dir=args.out,
        workers=args.workers,
        max_requests_per_minute=getattr(args, "max_requests_per_minute", None),
        max_symbols=args.max_symbols,
        manifest_path=getattr(args, "manifest", None),
        resume=bool(getattr(args, "resume", False)),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch and update local qfq daily bars")
    return add_arguments(parser)


def main(log_path: Optional[Path] = None):
    """Compatibility entrypoint used by existing imports and run_all.py."""
    return run_fetch(log_path=log_path)


def cli_main() -> None:
    result = run_from_args(build_parser().parse_args())
    print("Market data fetch complete")
    print(f"date range: {result['start']} to {result['end']}")
    print(f"symbols: {result['symbol_count']}")
    print(f"output: {result['output_dir']}")
    print(f"manifest: {result['manifest_path']}")
    if not result["ok"]:
        print(f"failed symbols: {result['failed_codes']}", file=sys.stderr)
        raise SystemExit(2)

if __name__ == "__main__":
    cli_main()
