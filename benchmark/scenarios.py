"""
场景定义 + JSON 加载器。

场景 JSON 格式：
{
  "name": "场景名",
  "domain": "core",
  "memories": [
    {"uri": "agent/identity", "content": "...", "priority": 10.0, "disclosure": "...", "children": [...]}
  ],
  "glossary": {"keyword": "target_uri"},
  "test_cases": [
    {"id": "tc_01", "query": "用户查询", "expected_recalls": ["uri1", "uri2"], "context": "standard"}
  ]
}
"""

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class MemoryDefinition:
    """单条记忆定义，支持递归子节点。"""
    uri: str
    content: str
    priority: float = 1.0
    disclosure: str = ""
    children: List["MemoryDefinition"] = field(default_factory=list)


@dataclass
class TestCase:
    """单个测试用例。"""
    id: str
    query: str
    expected_recalls: List[str]          # ground truth URI 列表
    context_variation: str = "standard"   # 上下文变体
    min_recall: int = 1                  # 最低召回数


@dataclass
class Scenario:
    """完整测试场景。"""
    name: str
    domain: str
    description: str = ""
    memories: List[MemoryDefinition] = field(default_factory=list)
    glossary: Dict[str, str] = field(default_factory=dict)
    test_cases: List[TestCase] = field(default_factory=list)


def load_scenario(filepath: str) -> Scenario:
    """从 JSON 文件加载场景。"""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    memories = [_parse_memory(m) for m in data.get("memories", []) if m.get("uri")]

    test_cases = []
    for tc in data.get("test_cases", []):
        test_cases.append(TestCase(
            id=tc["id"],
            query=tc["query"],
            expected_recalls=tc["expected_recalls"],
            context_variation=tc.get("context", tc.get("context_variation", "standard")),
            min_recall=tc.get("min_recall", tc.get("min_tool_calls", 1)),
        ))

    return Scenario(
        name=data["name"],
        domain=data.get("domain", "core"),
        description=data.get("description", ""),
        memories=memories,
        glossary=data.get("glossary", {}),
        test_cases=test_cases,
    )


def load_scenarios(scenarios_dir: str = "data/scenarios") -> List[Scenario]:
    """加载目录下所有 JSON 场景文件。"""
    base = Path(scenarios_dir)
    if not base.exists():
        base = Path(__file__).resolve().parent.parent / "data" / "scenarios"

    if not base.exists():
        return []

    scenarios = []
    for filepath in sorted(base.glob("*.json")):
        scenario = load_scenario(str(filepath))
        if scenario.memories:  # 跳过空场景
            scenarios.append(scenario)
    return scenarios


def _parse_memory(mem_data: dict) -> MemoryDefinition:
    """递归解析记忆树。"""
    return MemoryDefinition(
        uri=mem_data["uri"],
        content=mem_data["content"],
        priority=mem_data.get("priority", 1.0),
        disclosure=mem_data.get("disclosure", ""),
        children=[_parse_memory(c) for c in mem_data.get("children", []) if c.get("uri")],
    )
