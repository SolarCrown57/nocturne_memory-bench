"""
测试运行器 — 编排 benchmark 的端到端执行流程。

流程：
1. 加载场景 → 初始化记忆库
2. 对每个 LLM × 上下文变体
3. 对每个测试用例 → 启动 MemoryAgent 模拟对话
4. 收集结果 → 调用 evaluator 计算 recall
5. 汇总
"""

import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from memory import reset_memory, get_graph_service, get_glossary_service, get_search_indexer
from llm import build_adapter, Message

from .scenarios import Scenario, TestCase, MemoryDefinition, load_scenarios
from .agent import MemoryAgent, AgentRunResult
from .evaluator import evaluate_run_result, RecallMetrics, aggregate_metrics, AggregatedMetrics

import config as cfg


@dataclass
class TestCaseResult:
    """单个测试用例的运行结果（供 evaluator 使用）"""
    test_case_id: str
    query: str
    expected_recalls: List[str]
    tool_calls: List[Dict[str, Any]]
    success: bool
    tokens_used: int = 0
    error: str = ""


@dataclass
class RunResult:
    """单次 (场景 × LLM × 上下文) 的运行结果"""
    scenario_name: str
    llm_provider: str
    llm_model: str
    context_variation: str
    test_case_results: List[TestCaseResult]
    total_tokens: int = 0
    total_time_seconds: float = 0


# ── 系统提示 (同 agent.py 的 CONTEXT_PROMPTS) ──

SYSTEM_PROMPTS = {
    "minimal": (
        "You are an AI assistant. You have access to memory tools to recall information. "
        "Use read_memory(uri) to read specific memories, search_memory(query) to find memories. "
        "Respond naturally."
    ),
    "standard": (
        "你是一个 AI 助手，拥有长期记忆能力。\n\n"
        "重要：在回答用户问题之前，你应该先检查记忆：\n"
        "1. read_memory('system://boot') — 加载核心身份记忆\n"
        "2. search_memory(query) — 搜索相关记忆\n"
        "3. read_memory(uri) — 读取特定记忆\n\n"
        "在回复中使用记忆中的信息。用中文回复。"
    ),
    "verbose": (
        "你是一个拥有长期记忆能力的 AI 助手。你的记忆存储在 Nocturne Memory 系统中，"
        "通过 MCP 工具接口访问。\n\n"
        "## 你的记忆工具\n"
        "- read_memory(uri): 读取记忆。系统 URI: system://boot, system://index/<domain>\n"
        "- search_memory(query, domain=None, limit=10): 全文搜索记忆\n"
        "- create_memory / update_memory / delete_memory / add_alias\n\n"
        "## 使用原则\n"
        "- 每次对话开始时先调用 read_memory('system://boot')\n"
        "- 用户提问时，先用 search_memory 搜索\n"
        "- 找到 URI 后用 read_memory 查看详情\n"
        "- 基于记忆内容回复，让用户知道你'记得'\n\n"
        "用中文回复。"
    ),
    "no_tool_hint": (
        "你是一个 AI 助手。用中文回答用户的问题。"
    ),
}


def run_benchmark(
    config: Optional[cfg.BenchmarkConfig] = None,
    scenario_names: Optional[List[str]] = None,
) -> List[RunResult]:
    """
    运行完整的 benchmark。

    Args:
        config: Benchmark 配置
        scenario_names: 要测试的场景名称列表（None = 全部）
    """
    if config is None:
        config = cfg.BenchmarkConfig.from_config_json()

    all_results: List[RunResult] = []
    scenarios = load_scenarios()

    for scenario in scenarios:
        if scenario_names and scenario.name not in scenario_names:
            continue

        print(f"\n{'='*60}")
        print(f"Scenario: {scenario.name} | Test cases: {len(scenario.test_cases)}")
        print(f"{'='*60}")

        for llm_cfg in config.llms:
            if not llm_cfg.api_key:
                print(f"  [SKIP]️ Skipping {llm_cfg.provider}/{llm_cfg.model} (no API key)")
                continue

            for ctx_var in config.context_variations:
                print(f"\n  >  {llm_cfg.provider}/{llm_cfg.model} [{ctx_var}]")

                run_result = _run_single(
                    config, scenario, llm_cfg, ctx_var,
                )
                all_results.append(run_result)

    return all_results


def _run_single(
    config: cfg.BenchmarkConfig,
    scenario: Scenario,
    llm_config: cfg.LLMConfig,
    context_variation: str,
) -> RunResult:
    """运行单个 (场景 × LLM × 上下文) 组合。"""
    start_time = time.time()
    total_tokens = 0

    # 重置并加载记忆
    reset_memory()
    _load_scenario_memories(scenario)

    # 创建 Agent
    adapter = build_adapter(llm_config)
    agent = MemoryAgent(adapter, context_variation)

    # 覆盖 agent 的系统提示为 runner 的提示
    agent.context = SYSTEM_PROMPTS.get(context_variation, SYSTEM_PROMPTS["standard"])
    agent.context_name = context_variation

    test_results: List[TestCaseResult] = []

    for tc in scenario.test_cases:
        print(f"    Test: {tc.id}...", end=" ", flush=True)

        try:
            agent_result = agent.run_query(
                query=tc.query,
                expected_recalls=tc.expected_recalls,
                max_tool_calls=config.max_tool_calls_per_case,
            )
            total_tokens += agent_result.tokens_used

            test_results.append(TestCaseResult(
                test_case_id=tc.id,
                query=tc.query,
                expected_recalls=tc.expected_recalls,
                tool_calls=agent_result.tool_calls,
                success=True,
                tokens_used=agent_result.tokens_used,
            ))
            recalled = [tc.get("uri") for tc in agent_result.tool_calls if tc.get("uri")]
            print(f"[OK] ({agent_result.turns} turns, {len(recalled)} URIs recalled)")

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[FAIL] {e}")
            test_results.append(TestCaseResult(
                test_case_id=tc.id,
                query=tc.query,
                expected_recalls=tc.expected_recalls,
                tool_calls=[],
                success=False,
                error=str(e),
            ))

    return RunResult(
        scenario_name=scenario.name,
        llm_provider=llm_config.provider,
        llm_model=llm_config.model,
        context_variation=context_variation,
        test_case_results=test_results,
        total_tokens=total_tokens,
        total_time_seconds=time.time() - start_time,
    )


def _load_scenario_memories(scenario: Scenario):
    """加载场景记忆到数据库（按深度排序，确保父节点先创建）。"""
    graph = get_graph_service()
    glossary = get_glossary_service()

    # 展平记忆树 + 深度排序
    flat_memories = []
    for mem in scenario.memories:
        _flatten_memories(mem, flat_memories, scenario.domain)

    flat_memories.sort(key=lambda m: m[0].count("/"))  # 深度小的先创建

    # 加载记忆并收集 node_id + URI 用于后续索引
    node_ids = []
    for full_uri, content, priority, disclosure in flat_memories:
        nid = _create_memory_safe(graph, full_uri, content, priority, disclosure)
        if nid:
            node_ids.append((nid, full_uri, content, disclosure))

    # 为所有记忆建立 FTS5 搜索索引
    search = get_search_indexer()
    for nid, uri, content, disclosure in node_ids:
        search.index_memory(node_id=nid, uri=uri, content=content, disclosure=disclosure)

    # 加载豆辞典
    if hasattr(scenario, 'glossary') and scenario.glossary:
        for keyword, uri in scenario.glossary.items():
            glossary.add_keyword(keyword, uri)


def _flatten_memories(mem_def, result: list, domain: str, parent_prefix: str = ""):
    """将树形记忆展平为 (full_uri, content, priority, disclosure) 列表。"""
    uri = mem_def.uri
    if parent_prefix:
        uri = f"{parent_prefix}/{uri}"
    full_uri = f"{domain}://{uri}"

    result.append((full_uri, mem_def.content, mem_def.priority, mem_def.disclosure))

    for child in mem_def.children:
        _flatten_memories(child, result, domain, parent_prefix=uri)


def _create_memory_safe(graph, uri: str, content: str, priority: float, disclosure: str) -> str:
    """安全创建记忆节点（直接 SQL，绕过 create_memory 的父节点检查）。"""
    from memory.graph import parse_uri
    from memory.models import Node, Memory as MemoryModel, Edge, Path as PathModel, new_uuid

    domain, path = parse_uri(uri)
    parts = path.split("/")

    # 确保所有中间父节点存在
    prev_node_id = None
    for i, part in enumerate(parts):
        current_path = "/".join(parts[:i+1])
        current_uri = f"{domain}://{current_path}"

        # 检查是否已存在
        existing = graph.read_memory(current_uri)
        if existing:
            prev_node_id = existing["node_id"]
            continue

        session = graph._session()
        try:
            node_id = new_uuid()
            node = Node(id=node_id, label=part)
            session.add(node)
            mem = MemoryModel(id=new_uuid(), node_id=node_id, content=content if i == len(parts)-1 else "")
            session.add(mem)

            # 创建 Edge（连接父节点）
            if prev_node_id:
                edge = Edge(id=new_uuid(), parent_node_id=prev_node_id,
                           child_node_id=node_id, priority=priority,
                           disclosure=disclosure if i == len(parts)-1 else None)
                session.add(edge)

                # 创建 Path（URI 路由缓存）
                path_record = PathModel(id=new_uuid(), domain=domain,
                                        path_string=current_path, edge_id=edge.id)
                session.add(path_record)

            session.commit()
            prev_node_id = node_id
        finally:
            session.close()

    return prev_node_id or ""
