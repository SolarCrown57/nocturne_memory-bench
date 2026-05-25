"""
LLM 记忆召回基准测试 — 配置管理。

config.json 是 LLM 配置的唯一来源（SSOT）。
如果 api_key 在 config.json 中未设置或为 "ENV"，则从环境变量回退。

配置加载链：
  config.json (SSOT) ← 环境变量 (回退)
"""

import json
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config.json"


# ── LLM 配置 ──

@dataclass
class LLMConfig:
    """单个 LLM 的配置"""
    provider: str                          # openai / anthropic / gemini / deepseek
    model: str                             # gpt-4o / claude-sonnet-4-20250514 / etc.
    api_key: str = ""                      # API 密钥（空字符串或 "ENV" = 从环境变量取）
    api_url: Optional[str] = None          # 自定义 endpoint
    temperature: float = 0.0               # benchmark 用 0 保证可复现性
    max_tokens: int = 4096
    enabled: bool = True                   # 是否启用该 LLM


def _env_key(provider: str) -> str:
    """获取 provider 对应的环境变量名。"""
    return {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
    }.get(provider.lower(), f"{provider.upper()}_API_KEY")


def _resolve_api_key(llm_config: dict) -> str:
    """解析 api_key：config.json 优先，空字符串或 'ENV' 时从环境变量回退。"""
    key = llm_config.get("api_key", "")
    provider = llm_config.get("provider", "")
    if not key or key.upper() == "ENV":
        key = os.environ.get(_env_key(provider), "")
    return key


# ── config.json 读写 ──

def _default_config() -> dict:
    """首次运行时自动生成的默认配置。"""
    return {
        "_comment": "LLM Memory Recall Benchmark 配置文件。api_key 留空或设为 'ENV' 则从环境变量读取。",
        "database_url": f"sqlite:///{PROJECT_ROOT / 'memory_bench.db'}",
        "context_variations": ["standard", "minimal", "verbose", "no_tool_hint"],
        "recall_k_values": [1, 3, 5, 10],
        "max_tool_calls_per_case": 20,
        "results_dir": str(PROJECT_ROOT / "results"),
        "llms": [
            {
                "provider": "openai",
                "model": "gpt-4o",
                "api_key": "ENV",
                "api_url": None,
                "temperature": 0.0,
                "max_tokens": 4096,
                "enabled": True,
            },
            {
                "provider": "anthropic",
                "model": "claude-sonnet-4-20250514",
                "api_key": "ENV",
                "api_url": None,
                "temperature": 0.0,
                "max_tokens": 4096,
                "enabled": True,
            },
            {
                "provider": "deepseek",
                "model": "deepseek-chat",
                "api_key": "ENV",
                "api_url": "https://api.deepseek.com/v1",
                "temperature": 0.0,
                "max_tokens": 4096,
                "enabled": True,
            },
            {
                "provider": "gemini",
                "model": "gemini-2.5-pro",
                "api_key": "ENV",
                "api_url": None,
                "temperature": 0.0,
                "max_tokens": 4096,
                "enabled": True,
            },
        ],
    }


def ensure_config_exists() -> dict:
    """确保 config.json 存在，不存在则自动创建默认配置。"""
    if not CONFIG_PATH.exists():
        config = _default_config()
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"[CONFIG] Created default config at {CONFIG_PATH}")
        return config
    return {}


def load_config() -> dict:
    """加载 config.json，如不存在则自动创建。"""
    ensure_config_exists()
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Benchmark 配置 ──

@dataclass
class BenchmarkConfig:
    """一次 benchmark 运行的完整配置，从 config.json 加载。"""

    # 有默认值以支持 memory/__init__.py 的惰性初始化
    database_url: str = field(default_factory=lambda: f"sqlite:///{PROJECT_ROOT / 'memory_bench.db'}")
    llms: list[LLMConfig] = field(default_factory=list)
    context_variations: list[str] = field(default_factory=lambda: ["standard"])
    recall_k_values: list[int] = field(default_factory=lambda: [1, 3, 5, 10])
    max_tool_calls_per_case: int = 20
    results_dir: str = field(default_factory=lambda: str(PROJECT_ROOT / "results"))

    @classmethod
    def from_config_json(cls) -> "BenchmarkConfig":
        """从 config.json 加载配置。"""
        raw = load_config()

        llms = []
        for item in raw.get("llms", []):
            if not item.get("enabled", True):
                continue
            llms.append(LLMConfig(
                provider=item["provider"],
                model=item["model"],
                api_key=_resolve_api_key(item),
                api_url=item.get("api_url"),
                temperature=item.get("temperature", 0.0),
                max_tokens=item.get("max_tokens", 4096),
            ))

        return cls(
            database_url=raw.get("database_url", f"sqlite:///{PROJECT_ROOT / 'memory_bench.db'}"),
            llms=llms,
            context_variations=raw.get("context_variations", ["standard"]),
            recall_k_values=raw.get("recall_k_values", [1, 3, 5, 10]),
            max_tool_calls_per_case=raw.get("max_tool_calls_per_case", 20),
            results_dir=raw.get("results_dir", str(PROJECT_ROOT / "results")),
        )
