"""Anthropic Claude 适配器。"""

from typing import List, Dict, Any, Optional
import anthropic

from .base import BaseLLMAdapter, LLMResponse, Message


class AnthropicAdapter(BaseLLMAdapter):
    """Anthropic Claude 系列适配器（支持 tool use）。"""

    def __init__(
        self, model: str, temperature: float = 0.0, max_tokens: int = 4096,
        api_key: str = "",
    ):
        super().__init__(model, temperature, max_tokens)
        self._client = anthropic.Anthropic(api_key=api_key)

    @property
    def provider_name(self) -> str:
        return "anthropic"

    def chat(
        self, messages: List[Message], tools: Optional[List[Dict]] = None,
    ) -> LLMResponse:
        # Anthropic: system message 单独提取
        system = ""
        api_messages = []
        for m in messages:
            if m.role == "system":
                system += m.content + "\n"
            else:
                api_messages.append({"role": m.role, "content": m.content})

        kwargs = {
            "model": self.model,
            "messages": api_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if system:
            kwargs["system"] = system.strip()
        if tools:
            kwargs["tools"] = tools

        response = self._client.messages.create(**kwargs)

        text = ""
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                text = block.text
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "arguments": block.input,  # Anthropic 直接返回 dict
                })

        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            finish_reason=response.stop_reason or "stop",
            usage={
                "prompt_tokens": response.usage.input_tokens if response.usage else 0,
                "completion_tokens": response.usage.output_tokens if response.usage else 0,
            },
        )
