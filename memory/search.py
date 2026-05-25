"""
搜索索引 — 简化版 FTS 实现。

benchmark 版本简化：
  - 仅 SQLite FTS5 后端
  - 去掉 CJK 分词（benchmark 场景可控，不需要 jieba）
  - 同步 API
"""

from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy import text

from .models import new_uuid


class SearchIndexer:
    """全文搜索索引器（FTS5 虚拟表）。"""

    def __init__(self, session_factory):
        self._Session = session_factory
        self._ensure_fts_table()

    def _session(self) -> Session:
        return self._Session()

    def _ensure_fts_table(self):
        """确保 FTS5 虚拟表存在。"""
        session = self._session()
        try:
            session.execute(text("""
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                    node_id,
                    uri,
                    content,
                    disclosure,
                    tokenize='unicode61'
                )
            """))
            session.commit()
        finally:
            session.close()

    def index_memory(
        self, node_id: str, uri: str, content: str, disclosure: str = ""
    ):
        """索引/更新一条记忆的搜索条目。"""
        session = self._session()
        try:
            # 删除旧的
            session.execute(
                text("DELETE FROM memory_fts WHERE node_id = :node_id"),
                {"node_id": node_id}
            )
            # 插入新的
            session.execute(
                text("""
                    INSERT INTO memory_fts (node_id, uri, content, disclosure)
                    VALUES (:node_id, :uri, :content, :disclosure)
                """),
                {"node_id": node_id, "uri": uri, "content": content, "disclosure": disclosure}
            )
            session.commit()
        finally:
            session.close()

    def search(
        self, query: str, domain: Optional[str] = None, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """全文搜索记忆。

        返回格式:
        [
            {
                "uri": "core://agent/identity",
                "domain": "core",
                "path": "agent/identity",
                "name": "identity",
                "snippet": "...匹配上下文...",
                "priority": 1.0,
                "disclosure": "...",
                "content": "完整内容"
            }
        ]
        """
        if not query or not query.strip():
            return []

        session = self._session()
        try:
            # 构建 FTS5 查询
            fts_query = _build_fts_query(query)

            # FTS5 搜索
            fts_results = session.execute(
                text(f"""
                    SELECT node_id, uri, content, disclosure,
                           snippet(memory_fts, 2, '<mark>', '</mark>', '...', 30) as snippet
                    FROM memory_fts
                    WHERE memory_fts MATCH :query
                    ORDER BY rank
                    LIMIT :limit
                """),
                {"query": fts_query, "limit": limit * 2}  # 多取一些用于 domain 过滤
            ).fetchall()

            # 收集结果
            results = []
            seen_nodes = set()

            from .graph import parse_uri
            from .models import Edge, Path

            for row in fts_results:
                if row.node_id in seen_nodes:
                    continue
                seen_nodes.add(row.node_id)

                uri = row.uri
                try:
                    uri_domain, uri_path = parse_uri(uri)
                except ValueError:
                    continue

                # domain 过滤
                if domain and uri_domain != domain:
                    continue

                # 获取 priority（从 Edge 表）
                path_record = session.query(Path).filter(
                    Path.domain == uri_domain,
                    Path.path_string == uri_path,
                ).first()

                priority = 1.0
                disclosure = row.disclosure or ""
                if path_record:
                    edge = session.query(Edge).filter(
                        Edge.id == path_record.edge_id
                    ).first()
                    if edge:
                        priority = edge.priority
                        disclosure = edge.disclosure or disclosure

                results.append({
                    "uri": uri,
                    "domain": uri_domain,
                    "path": uri_path,
                    "name": uri_path.rsplit("/", 1)[-1] if "/" in uri_path else uri_path,
                    "snippet": _clean_snippet(row.snippet),
                    "priority": priority,
                    "disclosure": disclosure,
                    "content": row.content,
                })

                if len(results) >= limit:
                    break

            return results
        finally:
            session.close()

    def clear(self):
        """清空所有搜索索引。"""
        session = self._session()
        try:
            session.execute(text("DELETE FROM memory_fts"))
            session.commit()
        finally:
            session.close()


def _build_fts_query(query: str) -> str:
    """构建 FTS5 MATCH 查询。

    "hello world" → '"hello" AND "world"'
    """
    tokens = query.strip().split()
    if not tokens:
        return '""'
    # 每个 token 加引号后 AND 连接
    return " AND ".join(f'"{t}"' for t in tokens)


def _clean_snippet(snippet: str) -> str:
    """清理 FTS5 snippet 输出。"""
    if not snippet:
        return ""
    # 去掉多余的空白
    return " ".join(snippet.split())
