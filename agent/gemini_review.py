"""
gemini_review.py
~~~~~~~~~~~~~~~~
使用 Google Gemini 对候选股票进行图表分析评分。
继承自 BaseReviewer 基础架构。

用法：
    python agent/gemini_review.py
    python agent/gemini_review.py --config config/gemini_review.yaml

配置：
    默认读取 config/gemini_review.yaml。

环境变量：
    GEMINI_API_KEY  —— Google Gemini API Key（必填）

输出：
    ./data/review/{pick_date}/{code}.json   每支股票的评分 JSON
    ./data/review/{pick_date}/suggestion.json  研究候选汇总
    ./data/review/{pick_date}/run_manifest.json  可恢复运行清单
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types
import yaml

try:
    from .base_reviewer import BaseReviewer, ReviewRunIncomplete
except ImportError:  # Direct script execution: python agent/gemini_review.py
    from base_reviewer import BaseReviewer, ReviewRunIncomplete

# ────────────────────────────────────────────────
# 配置加载
# ────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG_PATH = _ROOT / "config" / "gemini_review.yaml"

DEFAULT_CONFIG: dict[str, Any] = {
    # 路径参数（相对路径默认基于项目根目录）
    "candidates": "data/candidates/candidates_latest.json",
    "kline_dir": "data/kline",
    "output_dir": "data/review",
    "prompt_path": "agent/prompt.md",
    # Gemini 模型参数
    "model": "gemini-3.5-flash",
    "retry_models": ["gemini-2.5-flash", "gemini-2.5-flash-lite"],
    "max_retries": 3,
    "retry_base_delay": 10,
    "request_delay": 5,
    "skip_existing": True,
    "suggest_min_score": 4.0,
}


def _read_dotenv_value(
    key_name: str,
    env_path: Path = _ROOT / ".env",
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


def _resolve_cfg_path(path_like: str | Path, base_dir: Path = _ROOT) -> Path:
    p = Path(path_like)
    return p if p.is_absolute() else (base_dir / p)


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    cfg_path = config_path or _DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"找不到配置文件：{cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    cfg = {**DEFAULT_CONFIG, **raw}

    # BaseReviewer 依赖这些路径字段为 Path 对象
    cfg["candidates"] = _resolve_cfg_path(cfg["candidates"])
    cfg["kline_dir"] = _resolve_cfg_path(cfg["kline_dir"])
    cfg["output_dir"] = _resolve_cfg_path(cfg["output_dir"])
    cfg["prompt_path"] = _resolve_cfg_path(cfg["prompt_path"])

    return cfg


class GeminiReviewer(BaseReviewer):
    def __init__(self, config):
        super().__init__(config)
        
        api_key = os.environ.get("GEMINI_API_KEY") or _read_dotenv_value("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "未找到 GEMINI_API_KEY，请设置环境变量或写入项目根目录 .env。"
            )
            
        self.client = genai.Client(api_key=api_key)

    @staticmethod
    def image_to_part(path: Path) -> types.Part:
        """将图片文件转为 Gemini Part 对象。"""
        suffix = path.suffix.lower()
        mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
        mime_type = mime_map.get(suffix, "image/jpeg")
        data = path.read_bytes()
        return types.Part.from_bytes(data=data, mime_type=mime_type)

    def review_stock(self, code: str, day_chart: Path, prompt: str) -> dict:
        """
        调用 Gemini API，对单支股票进行图表分析，返回解析后的 JSON 结果。
        """
        user_text = (
            f"股票代码：{code}\n\n"
            "以下是该股票的 **日线图**，请按照系统提示中的框架进行分析，"
            "并严格按照要求输出 JSON。"
        )

        parts: list[types.Part] = [
            types.Part.from_text(text="【日线图】"),
            self.image_to_part(day_chart),
            types.Part.from_text(text=user_text),
        ]

        primary_model = self.config.get("model", "gemini-3.5-flash")
        retry_models = self.config.get("retry_models") or []
        models = [primary_model] + [m for m in retry_models if m != primary_model]
        max_retries = int(self.config.get("max_retries", 3))
        retry_base_delay = float(self.config.get("retry_base_delay", 10))
        if max_retries <= 0:
            raise ValueError("max_retries must be positive")
        if retry_base_delay < 0:
            raise ValueError("retry_base_delay must be non-negative")

        last_error: Exception | None = None
        response = None
        for model in models:
            for attempt in range(1, max_retries + 1):
                try:
                    response = self.client.models.generate_content(
                        model=model,
                        contents=[types.Content(role="user", parts=parts)],
                        config=types.GenerateContentConfig(
                            system_instruction=prompt,
                            temperature=0.2,
                        ),
                    )
                    break
                except Exception as exc:
                    last_error = exc
                    if attempt >= max_retries:
                        print(f"\n[WARN] {code} model={model} 连续失败 {attempt} 次，尝试备用模型。")
                        break
                    delay = retry_base_delay * (2 ** (attempt - 1))
                    print(
                        f"\n[WARN] {code} model={model} 第 {attempt} 次失败，"
                        f"{delay:.0f}s 后重试：{exc}"
                    )
                    time.sleep(delay)
            if response is not None:
                break

        if response is None:
            raise RuntimeError(f"Gemini 调用失败（code={code}）：{last_error}")

        response_text = response.text
        if response_text is None:
            raise RuntimeError(f"Gemini 返回空响应，无法解析 JSON（code={code}）")

        result = self.extract_json(response_text)
        result["code"] = code  # 附加股票代码便于追溯
        return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gemini 图表复评")
    parser.add_argument(
        "--config",
        default=str(_DEFAULT_CONFIG_PATH),
        help="配置文件路径（默认 config/gemini_review.yaml）",
    )
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume",
        action="store_true",
        help="复用签名匹配的逐股结果（覆盖配置为 true）",
    )
    resume_group.add_argument(
        "--force",
        action="store_true",
        help="忽略已有逐股结果并重新复评（覆盖配置为 false）",
    )
    args = parser.parse_args(argv)

    try:
        config = load_config(Path(args.config))
        if args.resume:
            config["skip_existing"] = True
        elif args.force:
            config["skip_existing"] = False
        reviewer = GeminiReviewer(config)
        reviewer.run()
    except ReviewRunIncomplete as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
