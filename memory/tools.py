"""
复刻 Nocturne Memory 的 7 个 MCP 工具接口。

这些是 LLM 在对话中可调用的工具（纯 Python 函数版本）。
函数签名和返回值格式与原始 MCP Server 保持一致，
使得 benchmark 能准确模拟 AI 使用 nocturne_memory 的真实场景。

注意：
  - 返回值是格式化文本字符串（AI 可直接阅读），而非 JSON
  - system:// 系统 URI 单独处理
"""

from typing import Optional, List

from . import get_graph_service, get_search_indexer, get_glossary_service
from .graph import parse_uri, make_uri, DEFAULT_DOMAIN


# ── 系统视图生成器 ──

def _boot_view() -> str:
    """生成 system://boot 视图。"""
    graph = get_graph_service()
    indexer = get_search_indexer()

    # 列出已知的核心记忆
    known_uris = [
        "core://agent/identity",
        "core://my_user/identity",
        "core://agent/identity/personality",
        "core://agent/identity/purpose",
    ]

    lines = ["═══ System Boot ═══", ""]
    found = False
    for uri in known_uris:
        memory = graph.read_memory(uri)
        if memory:
            found = True
            lines.append(f"[{memory['name']}]")
            lines.append(f"   URI: {uri}")
            content_preview = memory['content'][:100].replace('\n', ' ')
            lines.append(f"   Content: {content_preview}...")
            if memory.get('disclosure'):
                lines.append(f"   (Disclosure: {memory['disclosure']})")
            lines.append("")

    if not found:
        # fallback: search for common terms
        results = indexer.search("identity", domain="core", limit=20)
        if results:
            for r in results:
                lines.append(f"[{r['name']}]")
                lines.append(f"   URI: {r['uri']}")
                content_preview = r.get('content', '')[:100].replace('\n', ' ')
                lines.append(f"   Content: {content_preview}...")
                lines.append("")

    if not lines or len(lines) <= 2:
        return "No memories found."

    return "\n".join(lines)


def _index_view(domain: str) -> str:
    """生成 system://index/<domain> 视图。"""
    indexer = get_search_indexer()
    results = indexer.search("", domain=domain, limit=50)

    if not results:
        return f"No memories in domain '{domain}'."

    lines = [f"═══ Index: {domain} ═══", ""]
    for r in results:
        lines.append(f"📌 [{r['name']}] ({r['uri']})")
        if r['disclosure']:
            lines.append(f"   Disclosure: {r['disclosure']}")
        lines.append("")
    return "\n".join(lines)


# ── Tool 1: read_memory ──

def read_memory(uri: str) -> str:
    """Read a memory by its URI.

    Usage:
        read_memory("core://agent/identity")
        read_memory("system://boot")          # auto-loads core memories
        read_memory("system://index/core")    # list all memories in core domain

    Parameters:
        uri: The URI of the memory to read. Supports system:// URIs.
    """
    # 处理系统 URI
    if uri.startswith("system://"):
        system_path = uri[len("system://"):]
        if system_path == "boot":
            return _boot_view()
        if system_path.startswith("index/"):
            domain = system_path[len("index/"):]
            return _index_view(domain)
        return f"Unknown system view: {uri}"

    graph = get_graph_service()
    try:
        memory = graph.read_memory(uri)
    except ValueError as e:
        return f"Error: {e}"

    if not memory:
        # 尝试搜索（simulate 原始行为：失败时给相近 URI）
        domain, path = parse_uri(uri)
        results = get_search_indexer().search(path.split("/")[-1], domain=domain, limit=3)
        if results:
            lines = [f"URI not found: {uri}", "", "Similar memories:"]
            for r in results:
                lines.append(f"  - {r['uri']}")
            return "\n".join(lines)
        return f"URI not found: {uri}"

    # 格式化输出
    lines = [
        f"═══ {memory['uri']} ═══",
        f"Label: {memory['node_label']}",
        f"Priority: {memory['priority']}",
        f"Created: {memory['created_at']}",
    ]
    if memory['disclosure']:
        lines.append(f"Disclosure: {memory['disclosure']}")
    lines.extend(["", memory['content'], ""])

    return "\n".join(lines)


# ── Tool 2: create_memory ──

def create_memory(
    parent_uri: str,
    content: str,
    name: str = "",
    priority: float = 1.0,
    disclosure: str = "",
) -> str:
    """Create a new memory under a parent node.

    Priority: 1.0 = normal, >1.0 = high priority, <1.0 = low priority.
    Disclosure: natural language trigger condition (e.g., "When user asks about identity").

    Parameters:
        parent_uri: URI of the parent memory
        content: The memory content to store
        name: Short name for this memory (appended to parent URI path)
        priority: Priority weight (1.0 = normal)
        disclosure: When this memory should be recalled
    """
    graph = get_graph_service()
    try:
        result = graph.create_memory(
            parent_uri=parent_uri,
            content=content,
            name=name,
            priority=priority,
            disclosure=disclosure,
        )
        return f"✅ Memory created: {result['uri']}"
    except ValueError as e:
        return f"❌ Error: {e}"


# ── Tool 3: update_memory ──

def update_memory(uri: str, content: str, mode: str = "patch") -> str:
    """Update an existing memory.

    Modes:
        'patch'  — append to existing content (safe, prevents overwrite)
        'replace' — replace entire content

    Parameters:
        uri: URI of the memory to update
        content: New content (appended or replaced based on mode)
        mode: 'patch' or 'replace'
    """
    graph = get_graph_service()
    try:
        result = graph.update_memory(uri=uri, new_content=content, mode=mode)
        return f"✅ Memory updated: {result['uri']}"
    except ValueError as e:
        return f"❌ Error: {e}"


# ── Tool 4: delete_memory ──

def delete_memory(uri: str) -> str:
    """Delete a memory access path (does not remove the memory data).

    Parameters:
        uri: URI of the memory to delete access to
    """
    graph = get_graph_service()
    try:
        result = graph.delete_memory(uri=uri)
        return f"✅ Memory path deleted: {result['uri']}"
    except ValueError as e:
        return f"❌ Error: {e}"


# ── Tool 5: add_alias ──

def add_alias(
    target_uri: str,
    alias_uri: str,
    priority: float = 1.0,
    disclosure: str = "",
) -> str:
    """Add an alias entry for an existing memory.

    An alias creates a second access path to the same memory.
    Different aliases can have different priorities and disclosure triggers.

    Parameters:
        target_uri: URI of the existing memory
        alias_uri: New URI to access the same memory
        priority: Priority for this alias
        disclosure: Disclosure condition for this alias
    """
    graph = get_graph_service()
    try:
        result = graph.add_alias(
            target_uri=target_uri,
            alias_uri=alias_uri,
            priority=priority,
            disclosure=disclosure,
        )
        return f"✅ Alias created: {result['uri']} → {result['aliased_to']}"
    except ValueError as e:
        return f"❌ Error: {e}"


# ── Tool 6: manage_triggers ──

def manage_triggers(action: str, keyword: str, target_uri: str = "", description: str = "") -> str:
    """Manage trigger keywords that create cross-node links.

    When a keyword appears in any memory content, it automatically
    becomes a hyperlink to the target memory.

    Actions:
        'add'    — register a new keyword
        'remove' — unregister a keyword
        'list'   — show all registered keywords

    Parameters:
        action: 'add', 'remove', or 'list'
        keyword: The keyword to manage
        target_uri: The URI to link to (for 'add' action)
        description: Brief description of the keyword (for 'add' action)
    """
    glossary = get_glossary_service()

    if action == "add":
        if not target_uri:
            return "❌ Error: target_uri is required for 'add' action"
        result = glossary.add_keyword(keyword=keyword, target_uri=target_uri, description=description)
        return f"✅ Keyword registered: '{result['keyword']}' → {result['target_uri']}"

    elif action == "remove":
        result = glossary.remove_keyword(keyword=keyword)
        if result["removed"]:
            return f"✅ Keyword removed: '{keyword}'"
        return f"⚠️ Keyword not found: '{keyword}'"

    elif action == "list":
        entries = glossary.get_all_keywords()
        if not entries:
            return "No keywords registered."
        lines = ["═══ Registered Keywords ═══", ""]
        for e in entries:
            lines.append(f"📌 '{e['keyword']}' → {e['target_uri']}")
            if e['description']:
                lines.append(f"   {e['description']}")
        return "\n".join(lines)

    return f"❌ Unknown action: {action}"


# ── Tool 7: search_memory ──

def search_memory(query: str, domain: Optional[str] = None, limit: int = 10) -> str:
    """Search for memories by keyword.

    This is a FULL-TEXT search, not a semantic/vector search.
    Searches memory content, URI paths, and disclosure triggers.

    Parameters:
        query: Keywords to search for
        domain: Optional domain filter (e.g., 'core', 'writer')
        limit: Maximum number of results (default 10)
    """
    indexer = get_search_indexer()
    results = indexer.search(query=query, domain=domain, limit=limit)

    if not results:
        return f"No results for '{query}'" + (f" in domain '{domain}'" if domain else "")

    lines = [f"═══ Search: '{query}' ═══", f"Results: {len(results)}", ""]
    for r in results:
        lines.append(f"📌 [{r['name']}] ({r['uri']})")
        lines.append(f"   Priority: {r['priority']}")
        if r['snippet']:
            lines.append(f"   Snippet: ...{r['snippet']}...")
        if r['disclosure']:
            lines.append(f"   Disclosure: {r['disclosure']}")
        lines.append("")
    return "\n".join(lines)


# ── 工具函数映射（供 LLM function calling 使用） ──

TOOL_DEFINITIONS = [
    {
        "name": "read_memory",
        "description": "Read a memory by its URI. Supports system://boot, system://index/<domain>.",
        "parameters": {
            "type": "object",
            "properties": {
                "uri": {"type": "string", "description": "Memory URI to read"}
            },
            "required": ["uri"]
        }
    },
    {
        "name": "create_memory",
        "description": "Create a new memory under a parent node.",
        "parameters": {
            "type": "object",
            "properties": {
                "parent_uri": {"type": "string", "description": "URI of parent memory"},
                "content": {"type": "string", "description": "Memory content"},
                "name": {"type": "string", "description": "Short name (optional)"},
                "priority": {"type": "number", "description": "Priority weight (1.0 = normal)"},
                "disclosure": {"type": "string", "description": "When this memory should be recalled"}
            },
            "required": ["parent_uri", "content"]
        }
    },
    {
        "name": "update_memory",
        "description": "Update existing memory content (patch or replace mode).",
        "parameters": {
            "type": "object",
            "properties": {
                "uri": {"type": "string", "description": "Memory URI"},
                "content": {"type": "string", "description": "New content"},
                "mode": {"type": "string", "description": "'patch' or 'replace'", "enum": ["patch", "replace"]}
            },
            "required": ["uri", "content"]
        }
    },
    {
        "name": "delete_memory",
        "description": "Delete access to a memory (keeps the data).",
        "parameters": {
            "type": "object",
            "properties": {
                "uri": {"type": "string", "description": "Memory URI"}
            },
            "required": ["uri"]
        }
    },
    {
        "name": "add_alias",
        "description": "Create an alias entry for an existing memory.",
        "parameters": {
            "type": "object",
            "properties": {
                "target_uri": {"type": "string", "description": "Existing memory URI"},
                "alias_uri": {"type": "string", "description": "New URI for this memory"},
                "priority": {"type": "number", "description": "Priority weight"},
                "disclosure": {"type": "string", "description": "Disclosure condition"}
            },
            "required": ["target_uri", "alias_uri"]
        }
    },
    {
        "name": "manage_triggers",
        "description": "Manage trigger keywords for cross-node linking.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "description": "'add', 'remove', or 'list'"},
                "keyword": {"type": "string", "description": "Keyword to manage"},
                "target_uri": {"type": "string", "description": "URI to link (for add)"},
                "description": {"type": "string", "description": "Keyword description (for add)"}
            },
            "required": ["action", "keyword"]
        }
    },
    {
        "name": "search_memory",
        "description": "Full-text search for memories by keyword.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search keywords"},
                "domain": {"type": "string", "description": "Optional domain filter"},
                "limit": {"type": "integer", "description": "Max results (default 10)"}
            },
            "required": ["query"]
        }
    },
]

# 工具函数执行映射
TOOL_EXECUTORS = {
    "read_memory": read_memory,
    "create_memory": create_memory,
    "update_memory": update_memory,
    "delete_memory": delete_memory,
    "add_alias": add_alias,
    "manage_triggers": manage_triggers,
    "search_memory": search_memory,
}

# Agent 使用的 TOOLS 字典（格式：{"name": {"function": callable}}）
TOOLS = {name: {"function": func} for name, func in TOOL_EXECUTORS.items()}
