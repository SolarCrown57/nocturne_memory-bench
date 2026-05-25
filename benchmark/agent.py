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

    "standard": """You are an AI assistant with long-term memory capabilities.

You have access to the following memory tools:
- read_memory: Read a memory by its URI
- search_memory: Search for memories by keywords
- create_memory: Create new memories
- update_memory: Update existing memories

When the user asks a question, FIRST check if any relevant memories exist by using search_memory or read_memory(system://boot). Then use the recalled memories to inform your response.

Important: Always use search_memory to find relevant memories before answering. Never guess URIs.""",

    "verbose": """You are an AI assistant with persistent long-term memory via the Nocturne Memory system.

Your memory is organized as a graph with URI-based addressing:
  - domain://path/to/memory (e.g., core://agent/identity)
  - system://boot — auto-loads your core identity memories
  - system://index/<domain> — lists all memories in a domain

Available tools:
1. read_memory(uri) — Read a memory by URI
2. search_memory(query, domain, limit) — Full-text search for memories
3. create_memory(parent_uri, content, name, priority, disclosure)
4. update_memory(uri, content, mode)
5. delete_memory(uri)
6. add_alias(target_uri, alias_uri, priority, disclosure)
7. manage_triggers(uri, add_triggers, remove_triggers)

Best practices:
- Start each session by reading system://boot to load your core identity
- Use search_memory with descriptive keywords before read_memory
- Each memory has a priority (1-10) and disclosure (trigger condition)
- Search is full-text, not semantic — use specific keywords
- After recalling memories, reference their content explicitly in your response""",

    "no_tool_hint": """You are an AI assistant. Respond to the user's questions naturally.""",
}


@dataclass
class AgentRunResult:
    """Agent 单次查询的运行结果"""
    query: str
    tool_calls: List[Dict[str, Any]]    # [{call_index, name, arguments, uri}]
    turns: int
    tokens_used: int
    final_response: str


class MemoryAgent:
    """模拟使用 Nocturne Memory 的 AI Agent。

    工作流:
      1. 构造对话（系统提示 + 工具定义 + 用户查询）
      2. 发送给 LLM
      3. 如果 LLM 返回工具调用 → 执行工具 → 将结果加入对话 → 回到步骤 2
      4. 如果 LLM 返回文本回复 → 结束
    """

    def __init__(self, llm_adapter: BaseLLMAdapter, context_variation: str = "standard"):
        self.llm = llm_adapter
        self.context = CONTEXT_PROMPTS.get(context_variation, CONTEXT_PROMPTS["standard"])
        self.context_name = context_variation

    def _format_tools_for_llm(self) -> List[Dict]:
        """将 TOOLS 转换为 LLM 的 function calling 格式。"""
        # TOOL_DEFINITIONS 需要包装为 OpenAI 标准格式
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
        """执行一次查询。

        Args:
            query: 用户查询文本
            expected_recalls: 期望召回的 URI 列表（仅用于日志，不影响执行）
            max_tool_calls: 最大工具调用次数

        Returns:
            包含工具调用序列的运行结果
        """
        tools = self._format_tools_for_llm()
        all_tool_calls = []
        total_tokens = 0

        messages = [
            Message(role="system", content=self.context),
            Message(role="user", content=query),
        ]

        for turn in range(max_tool_calls):
            # 调用 LLM
            response = self.llm.chat(messages=messages, tools=tools)
            total_tokens += sum(response.usage.values())

            if not response.tool_calls:
                # LLM 没有调用工具，结束
                return AgentRunResult(
                    query=query,
                    tool_calls=all_tool_calls,
                    turns=turn + 1,
                    tokens_used=total_tokens,
                    final_response=response.text,
                )

            # 执行工具调用
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
                        tool_result = f"[TOOL ERROR] {e}"

                    # 提取涉及到的 URI
                    uri = tool_args.get("uri", tool_args.get("target_uri", tool_args.get("parent_uri")))

                    all_tool_calls.append({
                        "call_index": len(all_tool_calls) + 1,
                        "name": tool_name,
                        "arguments": tool_args,
                        "uri": uri,
                        "result": tool_result[:500],  # 截断长结果
                    })
                else:
                    tool_result = f"[UNKNOWN TOOL] {tool_name}"

                # 将工具结果加入对话
                messages.append(Message(
                    role="assistant",
                    content="",
                    tool_calls=[tc],
                    reasoning_content=response.reasoning_content,
                    tool_call_id=tool_call_id,
                ))
                messages.append(Message(
                    role="tool",
                    content=tool_result,
                    tool_call_id=tool_call_id,
                    name=tool_name,
                ))

        # 达到最大调用次数
        return AgentRunResult(
            query=query,
            tool_calls=all_tool_calls,
            turns=max_tool_calls,
            tokens_used=total_tokens,
            final_response="[MAX TOOL CALLS REACHED]",
        )
