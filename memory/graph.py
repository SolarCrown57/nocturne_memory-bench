"""
复刻 Nocturne Memory 的核心图引擎。

提供 URI 路由 + 记忆 CRUD + 版本链管理。

与原始实现的差异：
  - 同步 API（benchmark 不需要异步）
  - SQLite only（benchmark 不需要 PostgreSQL 双后端）
  - 去掉 namespace 支持
"""

import re
from datetime import datetime
from typing import Optional, List, Dict, Any

from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy import and_

from .models import Node, Memory, Edge, Path, ROOT_NODE_UUID, DEFAULT_DOMAIN, new_uuid, utcnow

# ── URI 工具函数 ──

def parse_uri(uri: str) -> tuple[str, str]:
    """解析 URI: 'core://agent/identity' → ('core', 'agent/identity')"""
    match = re.match(r"^(system|[a-zA-Z][a-zA-Z0-9_-]*)://(.+)$", uri)
    if not match:
        raise ValueError(f"Invalid URI format: {uri}")
    return match.group(1), match.group(2)


def make_uri(domain: str, path: str) -> str:
    """构建 URI: ('core', 'agent/identity') → 'core://agent/identity'"""
    return f"{domain}://{path}"


# ── 图引擎服务 ──

class GraphService:
    """
    四实体图拓扑 CRUD 操作。

    设计原则：
      - Node UUID 是永久锚点（内容变化不影响身份）
      - Memory 是 Node 的内容版本（deprecated 实现版本链）
      - Edge 是有向关系（携带 priority + disclosure）
      - Path 是 human-readable 的 URI 路由缓存
    """

    def __init__(self, session_factory, search_indexer=None, glossary_service=None):
        self._Session = session_factory
        self._search = search_indexer
        self._glossary = glossary_service

    def _session(self) -> Session:
        return self._Session()

    # ── 读取 ──

    def read_memory(self, uri: str) -> Optional[Dict[str, Any]]:
        """通过 URI 读取一条记忆。"""
        domain, path_str = parse_uri(uri)

        # 先查 Path → 找 Edge → 找 child_node → 找 Memory
        session = self._session()
        try:
            path_record = session.query(Path).filter(
                and_(Path.domain == domain, Path.path_string == path_str)
            ).first()

            if not path_record:
                return None

            edge = session.query(Edge).filter(Edge.id == path_record.edge_id).first()
            if not edge:
                return None

            memory = session.query(Memory).filter(
                and_(Memory.node_id == edge.child_node_id, Memory.deprecated == False)
            ).order_by(Memory.created_at.desc()).first()

            if not memory:
                return None

            node = session.query(Node).filter(Node.id == memory.node_id).first()

            return {
                "uri": uri,
                "domain": domain,
                "path": path_str,
                "name": path_str.rsplit("/", 1)[-1] if "/" in path_str else path_str,
                "content": memory.content,
                "node_id": memory.node_id,
                "memory_id": memory.id,
                "priority": edge.priority,
                "disclosure": edge.disclosure or "",
                "created_at": memory.created_at.isoformat() if memory.created_at else "",
                "node_label": node.label if node else path_str.rsplit("/", 1)[-1],
            }
        finally:
            session.close()

    def list_children(self, parent_uri: str) -> List[Dict[str, Any]]:
        """列出父节点下所有子记忆（实际返回 children）。"""
        parent = self.read_memory(parent_uri)
        if not parent:
            return []

        session = self._session()
        try:
            edges = session.query(Edge).filter(
                Edge.parent_node_id == parent["node_id"]
            ).order_by(Edge.priority).all()

            results = []
            for edge in edges:
                memory = session.query(Memory).filter(
                    and_(Memory.node_id == edge.child_node_id, Memory.deprecated == False)
                ).order_by(Memory.created_at.desc()).first()

                if not memory:
                    continue

                # 找到指向这个子节点的 Path
                path_record = session.query(Path).filter(Path.edge_id == edge.id).first()
                child_uri = make_uri(
                    path_record.domain, path_record.path_string
                ) if path_record else f"node://{memory.node_id}"

                results.append({
                    "uri": child_uri,
                    "content": memory.content,
                    "priority": edge.priority,
                    "disclosure": edge.disclosure or "",
                })

            return results
        finally:
            session.close()

    # ── 创建 ──

    def create_memory(
        self,
        parent_uri: str,
        content: str,
        name: str = "",
        priority: float = 1.0,
        disclosure: str = "",
    ) -> Dict[str, Any]:
        """在指定父节点下创建新记忆。"""
        parent = self.read_memory(parent_uri)
        if not parent:
            raise ValueError(f"Parent not found: {parent_uri}")

        domain, parent_path = parse_uri(parent_uri)
        child_path = f"{parent_path}/{name}" if name else f"{parent_path}/memory_{new_uuid()[:8]}"

        session = self._session()
        try:
            # 创建子节点
            child_node = Node(id=new_uuid(), label=name or child_path.rsplit("/", 1)[-1])
            session.add(child_node)

            # 创建记忆内容
            child_memory = Memory(
                id=new_uuid(),
                node_id=child_node.id,
                content=content,
            )
            session.add(child_memory)

            # 创建边
            edge = Edge(
                id=new_uuid(),
                parent_node_id=parent["node_id"],
                child_node_id=child_node.id,
                priority=priority,
                disclosure=disclosure or None,
            )
            session.add(edge)

            # 创建路径缓存
            path_record = Path(
                id=new_uuid(),
                domain=domain,
                path_string=child_path,
                edge_id=edge.id,
            )
            session.add(path_record)

            session.commit()

            result_uri = make_uri(domain, child_path)

            # 更新搜索索引
            if self._search:
                self._search.index_memory(
                    node_id=child_node.id,
                    uri=result_uri,
                    content=content,
                    disclosure=disclosure,
                )

            return {"uri": result_uri, "content": content, "node_id": child_node.id}
        finally:
            session.close()

    # ── 更新 ──

    def update_memory(
        self, uri: str, new_content: str, mode: str = "patch"
    ) -> Dict[str, Any]:
        """更新记忆内容。patch 模式追加，replace 模式替换。"""
        memory_data = self.read_memory(uri)
        if not memory_data:
            raise ValueError(f"Memory not found: {uri}")

        session = self._session()
        try:
            old_memory = session.query(Memory).filter(
                Memory.id == memory_data["memory_id"]
            ).first()

            if mode == "patch":
                # 追加模式
                new_memory = Memory(
                    id=new_uuid(),
                    node_id=old_memory.node_id,
                    content=new_content,
                )
                session.add(new_memory)
                session.flush()

                # 标记旧版本
                old_memory.deprecated = True
                old_memory.migrated_to = new_memory.id

            elif mode == "replace":
                # 替换当前版本
                old_memory.content = new_content
                old_memory.created_at = utcnow()
                new_memory = old_memory

            else:
                raise ValueError(f"Unknown mode: {mode}")

            session.commit()

            # 更新搜索索引
            if self._search:
                self._search.index_memory(
                    node_id=new_memory.node_id,
                    uri=uri,
                    content=new_memory.content,
                    disclosure=memory_data.get("disclosure", ""),
                )

            return {"uri": uri, "content": new_memory.content}
        finally:
            session.close()

    # ── 删除 ──

    def delete_memory(self, uri: str) -> Dict[str, Any]:
        """删除访问路径（不删除记忆本体，保留在图中）。"""
        domain, path_str = parse_uri(uri)

        session = self._session()
        try:
            path_record = session.query(Path).filter(
                and_(Path.domain == domain, Path.path_string == path_str)
            ).first()

            if not path_record:
                raise ValueError(f"Path not found: {uri}")

            session.delete(path_record)
            session.commit()

            return {"uri": uri, "deleted": True}
        finally:
            session.close()

    # ── 别名 ──

    def add_alias(
        self,
        target_uri: str,
        alias_uri: str,
        priority: float = 1.0,
        disclosure: str = "",
    ) -> Dict[str, Any]:
        """为记忆创建别名入口（同一个 node 通过不同 path 访问）。"""
        target = self.read_memory(target_uri)
        if not target:
            raise ValueError(f"Target not found: {target_uri}")

        alias_domain, alias_path = parse_uri(alias_uri)
        parent_path = "/".join(alias_path.split("/")[:-1])

        session = self._session()
        try:
            # 找到父节点
            parent_path_record = session.query(Path).filter(
                and_(Path.domain == alias_domain, Path.path_string == parent_path)
            ).first()

            if not parent_path_record:
                raise ValueError(f"Parent path not found: {make_uri(alias_domain, parent_path)}")

            parent_edge = session.query(Edge).filter(
                Edge.id == parent_path_record.edge_id
            ).first()

            # 创建新边
            edge = Edge(
                id=new_uuid(),
                parent_node_id=parent_edge.child_node_id,
                child_node_id=target["node_id"],
                priority=priority,
                disclosure=disclosure or None,
            )
            session.add(edge)

            # 创建新路径
            path_record = Path(
                id=new_uuid(),
                domain=alias_domain,
                path_string=alias_path,
                edge_id=edge.id,
            )
            session.add(path_record)

            session.commit()

            return {"uri": alias_uri, "aliased_to": target_uri}
        finally:
            session.close()

    # ── 批量加载场景 ──

    def load_scenario(self, memories: List[Dict[str, Any]], domain: str = "core"):
        """批量加载记忆（用于 benchmark 初始化测试场景）。

        memories 格式:
        [
            {
                "path": "agent/identity",
                "content": "我是 Salem...",
                "priority": 10.0,
                "disclosure": "当用户询问我的身份时",
                "children": [...],  # 递归子节点
            }
        ]
        """
        uris = []
        for mem in memories:
            uri = self._create_recursive(None, mem, domain, "")[0]
            uris.append(uri)
        return uris

    def _create_recursive(
        self, parent_node_id: Optional[str], memory_def: Dict[str, Any],
        domain: str, parent_path: str
    ) -> tuple[str, Optional[str]]:
        """递归创建记忆树。"""
        path = memory_def["path"]  # "agent/identity" 或 "identity"（相对路径）
        full_path = f"{parent_path}/{path}" if parent_path else path
        name = path.rsplit("/", 1)[-1]

        session = self._session()
        try:
            # 创建节点
            node = Node(id=new_uuid(), label=name)
            session.add(node)

            # 创建记忆
            memory = Memory(
                id=new_uuid(),
                node_id=node.id,
                content=memory_def["content"],
            )
            session.add(memory)

            if parent_node_id:
                # 创建边
                edge = Edge(
                    id=new_uuid(),
                    parent_node_id=parent_node_id,
                    child_node_id=node.id,
                    priority=memory_def.get("priority", 1.0),
                    disclosure=memory_def.get("disclosure") or None,
                )
                session.add(edge)

                # 创建路径
                path_record = Path(
                    id=new_uuid(),
                    domain=domain,
                    path_string=full_path,
                    edge_id=edge.id,
                )
                session.add(path_record)

            session.commit()

            uri = make_uri(domain, full_path)

            # 更新搜索索引
            if self._search:
                self._search.index_memory(
                    node_id=node.id, uri=uri,
                    content=memory_def["content"],
                    disclosure=memory_def.get("disclosure", ""),
                )

            # 递归创建子节点
            for child in memory_def.get("children", []):
                self._create_recursive(node.id, child, domain, full_path)

            return uri, node.id
        finally:
            session.close()
