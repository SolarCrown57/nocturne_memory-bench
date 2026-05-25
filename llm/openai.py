"""OpenAI-compatible LLM adapter (also works for DeepSeek)."""

from typing import List, Dict, Any, Optional
import json
from openai import OpenAI

from .base import BaseLLMAdapter, LLMResponse, Message


class OpenAIAdapter(BaseLLMAdapter):
    """OpenAI / DeepSeek 适配器。

    使用 OpenAI 兼容的 API 接口。
    DeepSeek 通过设置 base_url 使用。
    """

    def __init__(
        self, model: str, temperature: float = 0.0, max_tokens: int = 4096,
        api_key: str = "", base_url: Optional[str] = None,
    ):
        super().__init__(model, temperature, max_tokens)
        self._api_key = api_key
        self._client = OpenAI(
            api_key=api_key or "placeholder",
            base_url=base_url,
        )

    @property
    def provider_name(self) -> str:
        return "openai"

    def chat(
        self, messages: List[Message], tools: Optional[List[Dict]] = None,
    ) -> LLMResponse:
        api_messages = []
        for m in messages:
            msg = {"role": m.role}
            if m.role == "assistant" and m.tool_calls:
                msg["content"] = m.content or None
                # 确保 tool_calls 有 type 字段
                msg["tool_calls"] = [
                    {"id": tc["id"], "type": "function", "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc["arguments"], ensure_ascii=False)
                    }}
                    for tc in m.tool_calls
                ]
                if m.reasoning_content:
                    msg["reasoning_content"] = m.reasoning_content
            elif m.role == "tool":
                msg["tool_call_id"] = m.tool_call_id
                msg["content"] = m.content
            else:
                msg["content"] = m.content
            if m.name:
                msg["name"] = m.name
            api_messages.append(msg)

        kwargs = {
            "model": self.model,
            "messages": api_messages,
            "temperature": self.temperature,
        }
        if self.max_tokens and self.max_tokens > 0:
            kwargs["max_tokens"] = self.max_tokens

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = self._client.chat.completions.create(**kwargs)

        choice = response.choices[0]
        message = choice.message

        # 保存 reasoning_content (DeepSeek V4+ 需要)
        reasoning_content = getattr(message, 'reasoning_content', None)

        # 提取工具调用
        tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": json.loads(tc.function.arguments),
                })

        return LLMResponse(
            text=message.content or "",
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
            usage={
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            },
            reasoning_content=reasoning_content,
        )
