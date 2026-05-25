#!/usr/bin/env python3
"""
Memory Recall Benchmark — 主入口脚本。

配置来源：config.json（SSOT），首次运行时自动生成。
API Key 在 config.json 中留空或设为 "ENV" 则从环境变量回退。

用法:
    python run_benchmark.py                          # 运行全部测试
    python run_benchmark.py --scenario roleplay      # 只测试指定场景
    python run_benchmark.py --llm openai,gpt-4o      # 只测试指定 LLM
    python run_benchmark.py --context standard       # 只测试指定上下文
    python run_benchmark.py --dry-run                # 不调用 LLM，验证框架
    python run_benchmark.py --init                   # 仅生成 config.json 并退出

环境变量 (仅在 config.json 中 api_key 为空或 "ENV" 时生效):
    OPENAI_API_KEY
    ANTHROPIC_API_KEY
    DEEPSEEK_API_KEY
    GEMINI_API_KEY
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import BenchmarkConfig, ensure_config_exists
from benchmark.runner import run_benchmark, RunResult
from benchmark.evaluator import compute_evaluation, aggregate_metrics, AggregatedMetrics
from benchmark.reporter import print_comparison_table, generate_markdown_report, export_json


def main():
    parser = argparse.ArgumentParser(
        description="[*] LLM Memory Recall Benchmark — config.json 驱动",
    )

    parser.add_argument("--scenario", type=str, default=None,
                        help="只测试指定场景（逗号分隔多个）")
    parser.add_argument("--llm", type=str, default=None,
                        help="只测试指定 LLM（格式: provider,model）")
    parser.add_argument("--context", type=str, default=None,
                        help="只测试指定上下文变体（standard/minimal/verbose/no_tool_hint）")
    parser.add_argument("--dry-run", action="store_true",
                        help="验证框架不实际调用 LLM")
    parser.add_argument("--init", action="store_true",
                        help="仅生成 config.json 并退出")
    parser.add_argument("--output", type=str, default="results",
                        help="结果输出目录（默认: results）")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="打印详细的工具调用追踪")

    args = parser.parse_args()

    # --init: 仅创建 config.json
    if args.init:
        ensure_config_exists()
        print("[完成] config.json 已生成。编辑它来配置 LLM 和 API 密钥。")
        return

    # 加载配置
    print("[加载] 读取 config.json...")
    config = BenchmarkConfig.from_config_json()

    if not config.llms:
        print("[错误] config.json 中没有启用的 LLM。请添加 LLM 或将 'enabled' 设为 true。")
        return

    # 过滤 LLM（命令行覆盖）
    if args.llm:
        provider, model = args.llm.split(",", 1)
        config.llms = [l for l in config.llms if l.provider == provider and l.model == model]
        if not config.llms:
            print(f"[错误] LLM 未找到: {args.llm}")
            return

    # 过滤上下文（命令行覆盖）
    if args.context:
        if args.context not in config.context_variations:
            print(f"[错误] 未知的上下文变体: {args.context}")
            return
        config.context_variations = [args.context]

    # 跳过 api_key 为空的 LLM
    active_llms = [l for l in config.llms if l.api_key]
    skipped = len(config.llms) - len(active_llms)
    if skipped:
        print(f"[跳过] 忽略 {skipped} 个无 API 密钥的 LLM")
        config.llms = active_llms
    if not config.llms:
        print("[错误] 没有可用的 LLM（缺少 API 密钥）。检查 config.json 或环境变量。")
        return

    # 显示配置
    print("=" * 60)
    print("LLM 记忆召回基准测试")
    print("=" * 60)
    print(f"LLM: {', '.join(f'{l.provider}/{l.model}' for l in config.llms)}")
    print(f"上下文变体: {', '.join(config.context_variations)}")
    print(f"场景目录: data/scenarios/")
    print(f"结果目录: {config.results_dir}")

    if args.dry_run:
        print("\n[干跑] 验证框架...")
        _dry_run()
        return

    # 运行测试
    scenario_names = args.scenario.split(",") if args.scenario else None
    results = run_benchmark(config, scenario_names=scenario_names, verbose=args.verbose)

    if not results:
        print("\n[FAIL] No results. Check API keys and scenario files.")
        return

    # 聚合：按 (场景 × LLM × 上下文) 分组
    aggregates: List[AggregatedMetrics] = []
    for run_result in results:
        metrics_list = [compute_evaluation(tc) for tc in run_result.test_case_results if tc.success]
        if not metrics_list:
            continue
        agg = aggregate_metrics(metrics_list)
        # 附加元数据（临时做法：通过动态属性）
        agg.llm_provider = run_result.llm_provider
        agg.llm_model = run_result.llm_model
        agg.scenario_name = run_result.scenario_name
        agg.context_variation = run_result.context_variation
        agg.total_tokens = run_result.total_tokens
        agg.total_time_seconds = run_result.total_time_seconds
        aggregates.append(agg)

    # 报告
    print_comparison_table(aggregates)

    out_dir = args.output
    generate_markdown_report(aggregates, out_dir)
    export_json(aggregates, out_dir)

    print(f"\n[通过] 完成！报告已保存到 {out_dir}/")


def _dry_run():
    """验证框架完整性（不调用 LLM）。"""
    from benchmark.scenarios import load_scenario
    from memory import reset_memory, get_graph_service

    scenario = load_scenario("data/scenarios/roleplay_identity.json")
    print(f"  [通过] 场景已加载: {scenario.name}")
    print(f"    记忆数: {len(scenario.memories)}")
    print(f"    测试用例: {len(scenario.test_cases)}")

    # 测试记忆系统基础操作
    reset_memory()
    from memory import get_search_indexer
    try:
        results = get_search_indexer().search("test", limit=1)
        print(f"    搜索测试: {len(results)} 条结果")
    except Exception as e:
        print(f"    [警告] 搜索测试失败: {e}")

    print(f"  [通过] 记忆系统就绪")

    from memory.tools import TOOLS
    print(f"  [通过] 工具已注册: {len(TOOLS)} ({', '.join(TOOLS.keys())})")

    print(f"\n[通过] 框架验证通过！")


if __name__ == "__main__":
    main()
