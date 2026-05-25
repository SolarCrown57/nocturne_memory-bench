"""
模拟 AI Agent — 连接 LLM 和 MCP 工具。

负责:
  1. 构造系统提示 + 工具定义
  2. 发送用户查询
  3. LLM 返回工具调用 → 执行 → 反馈结果
  4. 循环直到 LLM 停止调用工具或达到最大轮次
  5. 记录所有工具调用序列
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from llm import BaseLLMAdapter, LLMResponse, Message
from memory.tools import TOOL_DEFINITIONS, TOOLS


# ── 上下文变体的系统提示 ──

CONTEXT_PROMPTS = {
    "minimal": """You have access to a memory system. Use the provided tools to recall relevant memories when needed.""",

    "standard": """你是一个 AI 助手，拥有长期记忆能力。

重要：在回答用户问题之前，你应该先检查记忆：
1. read_memory('system://boot') — 加载核心身份记忆
2. search_memory(query) — 搜索相关记忆
3. read_memory(uri) — 读取特定记忆

在回复中使用记忆中的信息。用中文回复。""",

    "verbose": """你是一个拥有长期记忆能力的 AI 助手。你的记忆存储在 Nocturne Memory 系统中，通过 MCP 工具接口访问。

你的记忆工具：
- read_memory(uri): 读取记忆。系统 URI: system://boot, system://index/<domain>
- search_memory(query, domain=None, limit=10): 全文搜索记忆
- create_memory / update_memory / delete_memory / add_alias

使用原则：
- 每次对话开始时先调用 read_memory('system://boot')
- 用户提问时，先用 search_memory 关键词搜索
- 找到 URI 后用 read_memory 查看详情
- 基于记忆内容回复，让用户知道你'记得'""",

    "no_tool_hint": """你是一个 AI 助手。用中文回答用户的问题。""",
}


@dataclass
class TurnRecord:
    """单轮对话的完整记录"""
    turn_index: int
    messages_summary: str = ""          # 发送给 LLM 的消息摘要
    llm_text: str = ""                  # LLM 返回的文本
    tool_calls: List[Dict] = field(default_factory=list)  # 本轮的 MCP 工具调用
    tool_results: List[str] = field(default_factory=list)  # 工具执行结果
    uris: List[str] = field(default_factory=list)          # 本轮涉及的 URI


@dataclass
class AgentRunResult:
    """Agent 单次查询的运行结果"""
    query: str
    tool_calls: List[Dict[str, Any]]    # 所有工具调用合集
    turns: int
    tokens_used: int
    final_response: str
    system_prompt: str = ""
    llm_responses: List[str] = field(default_factory=list)
    turn_records: List[TurnRecord] = field(default_factory=list)  # 逐轮详情


class MemoryAgent:
    """模拟使用 Nocturne Memory 的 AI Agent。"""

    def __init__(self, llm_adapter: BaseLLMAdapter, context_variation: str = "standard"):
        self.llm = llm_adapter
        self.context = CONTEXT_PROMPTS.get(context_variation, CONTEXT_PROMPTS["standard"])
        self.context_name = context_variation

    def _format_tools_for_llm(self) -> List[Dict]:
        """将 TOOLS 转换为 LLM 的 function calling 格式。"""
        tools = []
        for tool_def in TOOL_DEFINITIONS:
            tools.append({
                "type": "function",
                "function": {
                    "name": tool_def["name"],
                    "description": tool_def["description"],
                    "parameters": tool_def.get("parameters", {"type": "object", "properties": {}}),
                },
            })
        return tools

    def run_query(
        self, query: str, expected_recalls: Optional[List[str]] = None,
        max_tool_calls: int = 20,
    ) -> AgentRunResult:
        """执行一次查询。"""
        tools = self._format_tools_for_llm()
        all_tool_calls = []
        total_tokens = 0
        llm_responses = []
        turn_records = []

        messages = [
            Message(role="system", content=self.context),
            Message(role="user", content=query),
        ]

        for turn in range(max_tool_calls):
            turn_record = TurnRecord(turn_index=turn + 1)

            # 记录发给 LLM 的消息摘要
            msg_parts = []
            for m in messages:
                if m.role == "system":
                    msg_parts.append(f"[系统提示] ({len(m.content)} 字符)")
                elif m.role == "user":
                    msg_parts.append(f"[用户] {m.content[:80]}...")
                elif m.role == "tool":
                    msg_parts.append(f"[工具结果:{m.name}] ({len(m.content)} 字符)")
                elif m.role == "assistant":
                    msg_parts.append(f"[上轮回复] + {len(m.tool_calls or [])} 个工具调用")
            turn_record.messages_summary = " | ".join(msg_parts)

            # 调用 LLM
            response = self.llm.chat(messages=messages, tools=tools)
            total_tokens += sum(response.usage.values())

            turn_record.llm_text = response.text or ""

            if response.text:
                llm_responses.append(response.text)

            if not response.tool_calls:
                turn_records.append(turn_record)
                return AgentRunResult(
                    query=query, tool_calls=all_tool_calls,
                    turns=turn + 1, tokens_used=total_tokens,
                    final_response=response.text,
                    system_prompt=self.context, llm_responses=llm_responses,
                    turn_records=turn_records,
                )

            # 执行工具调用
            tool_call_records = []
            tool_results = []
            for tc in response.tool_calls:
                tool_name = tc["name"]
                tool_args = tc.get("arguments", {})
                tool_call_id = tc.get("id", f"call_{len(all_tool_calls)}")

                # 执行工具
                tool_func = TOOLS.get(tool_name, {}).get("function")
                if tool_func:
                    try:
                        tool_result = tool_func(**tool_args)
                    except Exception as e:
                        tool_result = f"[工具错误] {e}"

                    # 提取真正的 URI（仅 read_memory 等有明确 uri 参数的工具）
                    uri = tool_args.get("uri") or tool_args.get("target_uri") or tool_args.get("parent_uri") or ""
                    # search_memory 的 query 不是 URI，不对它做记录
                    # 如果 uri 看起来不像真实 URI（没有 ://），也跳过
                    if uri and "://" not in uri:
                        uri = ""

                    call_record = {
                        "call_index": len(all_tool_calls) + 1,
                        "name": tool_name, "arguments": tool_args,
                        "uri": uri if uri else None, "result": tool_result[:500],
                    }
                    all_tool_calls.append(call_record)
                    tool_call_records.append(call_record)
                    tool_results.append(tool_result)
                    if uri:
                        turn_record.uris.append(uri)
                else:
                    tool_result = f"[未知工具] {tool_name}"
                    tool_results.append(tool_result)

                # 加入对话
                messages.append(Message(
                    role="assistant", content="",
                    tool_calls=[tc], reasoning_content=response.reasoning_content,
                    tool_call_id=tool_call_id,
                ))
                messages.append(Message(
                    role="tool", content=tool_result,
                    tool_call_id=tool_call_id, name=tool_name,
                ))

            turn_record.tool_calls = tool_call_records
            turn_record.tool_results = tool_results
            turn_records.append(turn_record)

        return AgentRunResult(
            query=query, tool_calls=all_tool_calls,
            turns=max_tool_calls, tokens_used=total_tokens,
            final_response="[达到最大工具调用次数]",
            system_prompt=self.context, llm_responses=llm_responses,
            turn_records=turn_records,
        )
