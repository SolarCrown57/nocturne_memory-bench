"""
LLM 适配层 — 统一接口 + 多模型实现。

支持：
  - OpenAI (GPT-4o, GPT-4-turbo, etc.)
  - Anthropic (Claude Sonnet, Claude Opus)
  - Google (Gemini 2.5 Pro, etc.)
  - DeepSeek (deepseek-chat)
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """统一 LLM 响应格式"""
    text: str                          # 纯文本回复
    tool_calls: List[Dict[str, Any]]   # 工具调用列表
    finish_reason: str                 # "stop" | "tool_calls" | "length"
    usage: Dict[str, int]              # {"prompt_tokens": N, "completion_tokens": M}
    reasoning_content: Optional[str] = None  # DeepSeek V4+ 思维链


@dataclass
class Message:
    """统一消息格式"""
    role: str                          # "system" | "user" | "assistant" | "tool"
    content: str
    tool_call_id: Optional[str] = None  # tool 消息需要
    tool_calls: Optional[List[Dict]] = None  # assistant 消息需要
    reasoning_content: Optional[str] = None  # DeepSeek V4+ thinking mode 需要
    name: Optional[str] = None          # tool 消息的工具名


class BaseLLMAdapter(ABC):
    """LLM 适配器基类"""

    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 4096):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    def chat(
        self,
        messages: List[Message],
        tools: Optional[List[Dict]] = None,
    ) -> LLMResponse:
        """发送对话请求。"""
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """返回提供者名称（openai/anthropic/gemini/deepseek）"""
        ...


def build_adapter(config) -> BaseLLMAdapter:
    """根据配置构建 LLM 适配器。"""
    from .openai import OpenAIAdapter
    from .anthropic import AnthropicAdapter
    from .gemini import GeminiAdapter

    provider = config.provider.lower()
    if provider == "openai" or provider == "deepseek":
        return OpenAIAdapter(
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            api_key=config.api_key,
            base_url=config.api_url or ({"deepseek": "https://api.deepseek.com/v1"}.get(provider)),
        )
    elif provider == "anthropic":
        return AnthropicAdapter(
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            api_key=config.api_key,
        )
    elif provider == "gemini":
        return GeminiAdapter(
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            api_key=config.api_key,
        )
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
