# memory-recall-bench

🧠 LLM 记忆召回率基准测试框架 — 复刻 Nocturne Memory 架构，测试不同 LLM 在不同上下文中通过 MCP 工具接口召回记忆的能力。

## 快速开始

```bash
cd memory-recall-bench
pip install -r requirements.txt

# 首次运行：自动生成 config.json
python run_benchmark.py --init

# 编辑 config.json 配置 LLM 和 API key
# api_key 设为 "ENV" 则从环境变量读取
# api_key 直接填写则优先使用 config.json 中的值

# 设置 API Keys（如果 config.json 中 api_key 为 "ENV"）
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...

# 运行基准测试
python run_benchmark.py
```

## 配置（config.json）

config.json 是 LLM 配置的**唯一来源（SSOT）**。首次运行 `--init` 自动生成。

```json
{
  "llms": [
    {
      "provider": "openai",
      "model": "gpt-4o",
      "api_key": "ENV",
      "api_url": null,
      "enabled": true
    }
  ],
  "context_variations": ["standard", "minimal", "verbose", "no_tool_hint"],
  "max_tool_calls_per_case": 20
}
```

**api_key 解析规则：**
| 值 | 行为 |
|----|------|
| `"sk-abc123..."` | 直接使用 config.json 中的值 |
| `"ENV"` 或 `""` | 从环境变量回退（OPENAI_API_KEY 等） |

**添加自定义 LLM：** 在 `llms` 数组中追加条目，设置 `provider`、`model`、`api_key`、`api_url`（可选）。

## 项目结构

```
memory-recall-bench/
├── memory/                # 复刻的记忆核心（四实体图拓扑 + MCP 工具）
│   ├── models.py          # Node/Memory/Edge/Path ORM 模型
│   ├── graph.py           # 图引擎 + URI 路由 CRUD
│   ├── search.py          # 全文搜索索引器（FTS5）
│   ├── glossary.py        # 豆辞典（关键词↔跨节点链接）
│   └── tools.py           # 7 个 MCP 工具接口（纯函数）
├── llm/                   # LLM 适配层（统一接口 + 多模型）
│   ├── base.py            # 基类 + 工厂函数
│   ├── openai.py          # OpenAI / DeepSeek
│   ├── anthropic.py       # Anthropic Claude
│   └── gemini.py          # Google Gemini
├── benchmark/             # 测试框架
│   ├── agent.py           # 模拟 AI Agent（LLM + 工具调用循环）
│   ├── scenarios.py       # 场景定义 + JSON 加载器
│   ├── runner.py          # 测试编排器
│   ├── evaluator.py       # Recall@k + MRR 计算
│   └── reporter.py        # 报告生成（Markdown + JSON）
├── data/scenarios/          # 真实场景数据集
│   ├── roleplay_identity.json  # 角色扮演身份记忆
│   └── project_knowledge.json  # 项目管理知识
├── run_benchmark.py         # 主入口脚本
├── config.py                # 配置管理（config.json SSOT）
├── config.json              # LLM + 测试参数配置
└── requirements.txt         # 依赖
```

## 核心指标

| 指标 | 含义 |
|------|------|
| **Recall@k** | 前 k 次工具调用中至少召回一条预期记忆的概率 |
| **MRR** | 第一条预期记忆的倒数排名均值（1.0 = 首次调用命中） |
| **Precision@k** | 前 k 次调用中正确记忆的比例 |
| **Avg Tool Calls** | 每查询平均工具调用次数 |
| **Avg Tokens** | 每查询平均 token 用量 |

## 上下文变体

| 变体 | 说明 |
|------|------|
| `minimal` | 最小系统提示，仅告知工具存在 |
| `standard` | 标准提示，含工具清单 + 使用提示 |
| `verbose` | 详细提示，含最佳实践 + 示例 |
| `no_tool_hint` | 无工具使用提示（测试 LLM 主动性） |

## 测试场景

### 1. 角色扮演身份记忆 (`roleplay.json`)
模拟 AI 角色（Salem）与用户的长期互动历史。测试 LLM 是否能根据对话上下文召回身份、性格、偏好、关系等记忆。

### 2. 项目管理知识 (`project.json`)
模拟开发者使用 AI 助手管理项目上下文。测试 LLM 是否能召回技术架构、决策记录、功能列表等结构化知识。

## License

MIT — 与 Nocturne Memory 相同
