"""Google Gemini 适配器。"""

from typing import List, Dict, Any, Optional
from .base import BaseLLMAdapter, LLMResponse, Message


class GeminiAdapter(BaseLLMAdapter):
    """Google Gemini 系列适配器。

    注意：Gemini 的 function calling 格式与 OpenAI 不同，
    这里做了转换。如果 Gemini SDK 不可用，降级为纯文本模式。
    """

    def __init__(
        self, model: str, temperature: float = 0.0, max_tokens: int = 4096,
        api_key: str = "",
    ):
        super().__init__(model, temperature, max_tokens)
        self._api_key = api_key
        self._model = None
        self._init_client()

    def _init_client(self):
        try:
            import google.generativeai as genai
            genai.configure(api_key=self._api_key)
            self._model = genai.GenerativeModel(self.model)
        except ImportError:
            self._model = None  # 降级
        except Exception:
            self._model = None

    @property
    def provider_name(self) -> str:
        return "gemini"

    def chat(
        self, messages: List[Message], tools: Optional[List[Dict]] = None,
    ) -> LLMResponse:
        if self._model is None:
            return LLMResponse(
                text="[Gemini SDK not available - returning empty response]",
                tool_calls=[],
                finish_reason="stop",
                usage={"prompt_tokens": 0, "completion_tokens": 0},
            )

        # 转换消息格式
        import google.generativeai as genai

        # Gemini 使用不同的消息格式
        contents = []
        system_instruction = ""
        for m in messages:
            if m.role == "system":
                system_instruction = m.content
                continue
            role = "user" if m.role == "user" else "model"
            contents.append({"role": role, "parts": [m.content]})

        kwargs = {
            "contents": contents,
            "generation_config": genai.types.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            ),
        }

        if system_instruction:
            kwargs["system_instruction"] = system_instruction

        # Gemini 使用不同的工具格式
        if tools:
            gemini_tools = []
            for tool in tools:
                gemini_tools.append(genai.protos.Tool(
                    function_declarations=[genai.protos.FunctionDeclaration(
                        name=tool["name"],
                        description=tool["description"],
                        parameters=tool.get("input_schema", tool.get("parameters", {})),
                    )]
                ))
            kwargs["tools"] = gemini_tools

        try:
            response = self._model.generate_content(**kwargs)
            text = response.text if response.text else ""

            tool_calls = []
            # Gemini 的工具调用格式不同，简化处理
            if hasattr(response, 'candidates') and response.candidates:
                for candidate in response.candidates:
                    if hasattr(candidate, 'content') and candidate.content:
                        for part in candidate.content.parts:
                            if hasattr(part, 'function_call') and part.function_call:
                                tool_calls.append({
                                    "id": f"call_{len(tool_calls)}",
                                    "name": part.function_call.name,
                                    "arguments": dict(part.function_call.args),
                                })

            return LLMResponse(
                text=text,
                tool_calls=tool_calls,
                finish_reason="stop",
                usage={
                    "prompt_tokens": response.usage_metadata.prompt_token_count if hasattr(response, 'usage_metadata') else 0,
                    "completion_tokens": response.usage_metadata.candidates_token_count if hasattr(response, 'usage_metadata') else 0,
                },
            )
        except Exception as e:
            return LLMResponse(
                text=f"[Gemini error: {e}]",
                tool_calls=[],
                finish_reason="error",
                usage={"prompt_tokens": 0, "completion_tokens": 0},
            )
