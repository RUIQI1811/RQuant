from __future__ import annotations

import datetime as dt
import logging
import random
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional
import os

import numpy as np
import pandas as pd
import tushare as ts
import yaml
from tqdm import tqdm

warnings.filterwarnings("ignore")

# --------------------------- pandas 兼容补丁 --------------------------- #
# tushare 内部使用了 fillna(method='ffill'/'bfill')，在 pandas 2.2+ 中已移除该参数。
# 此补丁将旧式调用自动转发到 ffill()/bfill()，无需降级 pandas。
import pandas as _pd

_orig_fillna = _pd.DataFrame.fillna

def _patched_fillna(self, value=None, *, method=None, axis=None, inplace=False, limit=None, **kwargs):
    if method is not None:
        if method == "ffill":
            result = self.ffill(axis=axis, inplace=inplace, limit=limit)
        elif method == "bfill":
            result = self.bfill(axis=axis, inplace=inplace, limit=limit)
        else:
            raise ValueError(f"Unsupported fillna method: {method}")
        return result
    return _orig_fillna(self, value, axis=axis, inplace=inplace, limit=limit, **kwargs)

_pd.DataFrame.fillna = _patched_fillna  # type: ignore[method-assign]

_orig_series_fillna = _pd.Series.fillna

def _patched_series_fillna(self, value=None, *, method=None, axis=None, inplace=False, limit=None, **kwargs):
    if method is not None:
        if method == "ffill":
            result = self.ffill(axis=axis, inplace=inplace, limit=limit)
        elif method == "bfill":
            result = self.bfill(axis=axis, inplace=inplace, limit=limit)
        else:
            raise ValueError(f"Unsupported fillna method: {method}")
        return result
    return _orig_series_fillna(self, value, axis=axis, inplace=inplace, limit=limit, **kwargs)

_pd.Series.fillna = _patched_series_fillna  # type: ignore[method-assign]

# --------------------------- 全局日志配置 --------------------------- #
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_LOG_DIR = _PROJECT_ROOT / "data" / "logs"

def _load_dotenv(env_path: Path = _PROJECT_ROOT / ".env") -> None:
    """Load KEY=VALUE lines from .env without overriding existing env vars."""
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value

def _resolve_cfg_path(path_like: str | Path, base_dir: Path = _PROJECT_ROOT) -> Path:
    """将配置中的路径统一解析为绝对路径：相对路径基于项目根目录。"""
    p = Path(path_like)
    return p if p.is_absolute() else (base_dir / p)

def _default_log_path() -> Path:
    today = dt.date.today().strftime("%Y-%m-%d")
    return _DEFAULT_LOG_DIR / f"fetch_{today}.log"

def setup_logging(log_path: Optional[Path] = None) -> None:
    """初始化日志：同时输出到 stdout 和指定文件。"""
    if log_path is None:
        log_path = _default_log_path()
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, mode="a", encoding="utf-8"),
        ],
    )

logger = logging.getLogger("fetch_from_stocklist")

# --------------------------- 限流/封禁处理配置 --------------------------- #
COOLDOWN_SECS = 600
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

def _cool_sleep(base_seconds: int) -> None:
    jitter = random.uniform(0.9, 1.2)
    sleep_s = max(1, int(base_seconds * jitter))
    logger.warning("疑似被限流/封禁，进入冷却期 %d 秒...", sleep_s)
    time.sleep(sleep_s)

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

def _get_kline_tushare(code: str, start: str, end: str) -> pd.DataFrame:
    ts_code = _to_ts_code(code)
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

    for attempt in range(1, 4):
        try:
            new_df = _get_kline_tushare(code, start, end)
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
                refreshed = validate(_get_kline_tushare(code, full_start, full_end))
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
            if _looks_like_ip_ban(e):
                logger.error(f"{code} 第 {attempt} 次抓取疑似被封禁，沉睡 {COOLDOWN_SECS} 秒")
                _cool_sleep(COOLDOWN_SECS)
            else:
                silent_seconds = 30 * attempt
                logger.info(f"{code} 第 {attempt} 次抓取失败，{silent_seconds} 秒后重试：{e}")
                time.sleep(silent_seconds)
    logger.error("%s 三次抓取均失败，已跳过！", code)
    return "failed"



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
def main(log_path: Optional[Path] = None):
    # ---------- 读取 YAML 配置 ---------- #
    cfg = _load_config()

    # ---------- 日志路径（优先参数，其次 YAML，最后默认值） ---------- #
    if log_path is None:
        cfg_log = cfg.get("log")
        log_path = _resolve_cfg_path(cfg_log) if cfg_log else _default_log_path()
    setup_logging(log_path)
    logger.info("日志文件：%s", Path(log_path).resolve())

    # ---------- Tushare Token ---------- #
    os.environ["NO_PROXY"] = "api.waditu.com,.waditu.com,waditu.com"
    os.environ["no_proxy"] = os.environ["NO_PROXY"]
    _load_dotenv()
    ts_token = os.environ.get("TUSHARE_TOKEN")
    if not ts_token:
        raise ValueError("请先设置环境变量 TUSHARE_TOKEN，或在项目根目录 .env 中写入 TUSHARE_TOKEN=你的token")
    global pro
    pro = ts.pro_api(ts_token)

    # ---------- 日期解析 ---------- #
    raw_start = str(cfg.get("start", "20190101"))
    raw_end   = str(cfg.get("end",   "today"))
    start = dt.date.today().strftime("%Y%m%d") if raw_start.lower() == "today" else raw_start
    end   = dt.date.today().strftime("%Y%m%d") if raw_end.lower()   == "today" else raw_end

    out_dir = _resolve_cfg_path(cfg.get("out", "./data"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 从 stocklist.csv 读取股票池 ---------- #
    stocklist_path = _resolve_cfg_path(cfg.get("stocklist", "./config/stocklist.csv"))
    exclude_boards = set(cfg.get("exclude_boards") or [])
    codes = load_codes_from_stocklist(stocklist_path, exclude_boards)

    if not codes:
        logger.error("stocklist 为空或被过滤后无代码，请检查。")
        sys.exit(1)

    logger.info(
        "开始抓取 %d 支股票 | 数据源:Tushare(日线,qfq) | 日期:%s → %s | 排除:%s",
        len(codes), start, end, ",".join(sorted(exclude_boards)) or "无",
    )

    # ---------- 多线程增量抓取 ---------- #
    workers = int(cfg.get("workers", 8))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                fetch_one,
                code,
                start,
                end,
                out_dir,
            ): code
            for code in codes
        }
        outcomes: dict[str, int] = {}
        for future in tqdm(as_completed(futures), total=len(futures), desc="下载进度"):
            outcome = future.result()
            outcomes[outcome] = outcomes.get(outcome, 0) + 1

    logger.info(
        "全部任务完成：新建=%d，增量更新=%d，重叠日覆盖=%d，qfq全量刷新=%d，"
        "暂无新数据=%d，失败=%d | 输出=%s",
        outcomes.get("created", 0),
        outcomes.get("updated", 0),
        outcomes.get("overwritten", 0),
        outcomes.get("refreshed", 0),
        outcomes.get("no_new_data", 0) + outcomes.get("empty", 0),
        outcomes.get("failed", 0),
        out_dir.resolve(),
    )

if __name__ == "__main__":
    main()
