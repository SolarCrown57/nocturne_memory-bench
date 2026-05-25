"""
Recall@k + MRR 评估引擎。

从工具调用序列中提取 URI 召回顺序，与 ground truth 对比，
计算标准的信息检索指标。

召回 URI 来源：
  - tool_calls[].uri（read_memory 等直接传 URI 的工具）
  - 搜索结果文本（search_memory 返回的 URI 列表）
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


# 从搜索结果文本中提取 URI 的正则
_URI_PATTERN = re.compile(r'(\w+://[\w/._-]+)')


@dataclass
class RecallMetrics:
    """单次测试的 recall 指标"""
    test_case_id: str
    expected: List[str]
    recalled: List[str]
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr: float
    hits: List[str]
    misses: List[str]
    first_hit_rank: Optional[int]


@dataclass
class AggregatedMetrics:
    """聚合后的评估指标"""
    total_cases: int
    avg_recall_at_1: float
    avg_recall_at_3: float
    avg_recall_at_5: float
    avg_recall_at_10: float = 0.0
    avg_mrr: float = 0.0
    detail: List[RecallMetrics] = field(default_factory=list)
    llm_provider: str = ""
    llm_model: str = ""
    scenario_name: str = ""
    context_variation: str = ""
    total_tokens: int = 0
    total_time_seconds: float = 0.0
    avg_tool_calls_per_query: float = 0.0
    avg_tokens_per_query: float = 0.0


def _extract_uris_from_tool_call(tc: Dict[str, Any]) -> List[str]:
    """
    从单个工具调用中提取所有涉及的 URI。

    来源：
      1. tc["uri"] — read_memory / create_memory 等的直接 URI
      2. tc["result"] 文本中的 URI — search_memory 返回的列表
      3. tc["arguments"]["uri"] / ["target_uri"] / ["parent_uri"]
    """
    uris = []

    # 直接 URI 字段
    direct = tc.get("uri")
    if direct:
        uris.append(direct)

    # 参数中的 URI
    args = tc.get("arguments", {})
    for key in ("uri", "target_uri", "parent_uri", "alias_uri"):
        if args.get(key):
            uris.append(args[key])

    # 搜索结果文本中的 URI
    result_text = tc.get("result", "")
    if result_text:
        found = _URI_PATTERN.findall(result_text)
        uris.extend(found)

    # 去重 + 保序（排除 system:// 和 node://）
    seen = set()
    ordered = []
    for u in uris:
        uri = u.rstrip(".,;:)")
        if uri not in seen and not uri.startswith(("system://", "node://")):
            seen.add(uri)
            ordered.append(uri)

    return ordered


def evaluate_run_result(
    test_case_id: str,
    expected_recalls: List[str],
    tool_calls: List[Dict[str, Any]],
) -> RecallMetrics:
    """
    评估一个测试用例的 recall 表现。

    Args:
        test_case_id: 测试用例 ID
        expected_recalls: 期望召回的 ground truth URI 列表
        tool_calls: 按时间顺序的工具调用列表

    Returns:
        RecallMetrics 包含各项 IR 指标
    """
    # 从所有工具调用中提取 URI（按时间顺序）
    recalled_uris: List[str] = []
    for tc in tool_calls:
        for uri in _extract_uris_from_tool_call(tc):
            if uri not in recalled_uris:
                recalled_uris.append(uri)

    # 计算每个 expected URI 在 recalled_uris 中的排名（1-based）
    ranks: Dict[str, float] = {}
    for uri in expected_recalls:
        try:
            ranks[uri] = recalled_uris.index(uri) + 1
        except ValueError:
            ranks[uri] = float("inf")

    # 最佳排名
    finite_ranks = [r for r in ranks.values() if r != float("inf")]
    first_hit_rank = min(finite_ranks) if finite_ranks else None

    # Recall@k: 至少有一个 expected URI 出现在前 k 个召回结果中
    recall_at_1 = 1.0 if first_hit_rank and first_hit_rank <= 1 else 0.0
    recall_at_3 = 1.0 if first_hit_rank and first_hit_rank <= 3 else 0.0
    recall_at_5 = 1.0 if first_hit_rank and first_hit_rank <= 5 else 0.0

    # MRR = 平均倒数排名（按 expected URI 数量归一化）
    if ranks and len(ranks) > 0:
        mrr = sum(1.0 / r for r in ranks.values() if r != float("inf")) / len(ranks)
    else:
        mrr = 0.0

    hits = [uri for uri in expected_recalls if uri in recalled_uris]
    misses = [uri for uri in expected_recalls if uri not in recalled_uris]

    return RecallMetrics(
        test_case_id=test_case_id,
        expected=expected_recalls,
        recalled=recalled_uris,
        recall_at_1=recall_at_1,
        recall_at_3=recall_at_3,
        recall_at_5=recall_at_5,
        mrr=mrr,
        hits=hits,
        misses=misses,
        first_hit_rank=first_hit_rank,
    )


def aggregate_metrics(metrics_list: List[RecallMetrics]) -> AggregatedMetrics:
    """聚合多个测试用例的指标。"""
    if not metrics_list:
        return AggregatedMetrics(
            total_cases=0,
            avg_recall_at_1=0.0, avg_recall_at_3=0.0, avg_recall_at_5=0.0, avg_mrr=0.0,
            detail=[],
        )

    n = len(metrics_list)
    return AggregatedMetrics(
        total_cases=n,
        avg_recall_at_1=sum(m.recall_at_1 for m in metrics_list) / n,
        avg_recall_at_3=sum(m.recall_at_3 for m in metrics_list) / n,
        avg_recall_at_5=sum(m.recall_at_5 for m in metrics_list) / n,
        avg_mrr=sum(m.mrr for m in metrics_list) / n,
        detail=metrics_list,
    )


def compute_evaluation(result) -> RecallMetrics:
    """从 TestCaseResult 计算 recall 指标（便捷函数）。"""
    return evaluate_run_result(
        test_case_id=result.test_case_id,
        expected_recalls=result.expected_recalls,
        tool_calls=result.tool_calls,
    )
