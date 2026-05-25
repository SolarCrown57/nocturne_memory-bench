"""
复刻 Nocturne Memory 的 SQLAlchemy ORM 模型。

四实体图拓扑：
  Node (身份层) ──► Memory (内容层)
  Edge (关系层) ──► Path (路由层)

简化说明：
  - 去掉 namespace 支持（benchmark 用单数据库实例）
  - 去掉 Changeset/Snapshot（benchmark 不需要版本回滚）
  - 去掉 ChangeCollector（不需要变更追踪）
"""

from datetime import datetime
from typing import Optional
from uuid import uuid4

from sqlalchemy import (
    Column, String, Text, Integer, Float, DateTime, ForeignKey,
    Index, Boolean, UniqueConstraint
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

# ── 常量 ──
ROOT_NODE_UUID = "00000000-0000-0000-0000-000000000000"
DEFAULT_DOMAIN = "core"


def new_uuid() -> str:
    return uuid4().hex


def utcnow() -> datetime:
    return datetime.utcnow()


# ── 记忆节点 (身份层) ──
class Node(Base):
    """概念的永久锚点。内容更新时 UUID 不变。"""
    __tablename__ = "nodes"

    id = Column(String(32), primary_key=True, default=new_uuid)
    label = Column(String(255), nullable=True, comment="节点标签（可选，人类可读）")
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)

    # 关系
    memories = relationship("Memory", back_populates="node", lazy="dynamic")


# ── 记忆内容 (内容层) ──
class Memory(Base):
    """节点的内容版本快照。deprecated + migrated_to 实现版本链。"""
    __tablename__ = "memories"

    id = Column(String(32), primary_key=True, default=new_uuid)
    node_id = Column(String(32), ForeignKey("nodes.id"), nullable=False)
    content = Column(Text, nullable=False, comment="记忆正文")

    # 版本链
    deprecated = Column(Boolean, default=False, comment="是否已被新版本替代")
    migrated_to = Column(String(32), nullable=True, comment="迁移到的新版本 memory_id")

    created_at = Column(DateTime, default=utcnow)

    node = relationship("Node", back_populates="memories")


# ── 有向关系 (关系层) ──
class Edge(Base):
    """Node 间的有向关系，携带 priority 和 disclosure。"""
    __tablename__ = "edges"

    id = Column(String(32), primary_key=True, default=new_uuid)
    parent_node_id = Column(String(32), ForeignKey("nodes.id"), nullable=False)
    child_node_id = Column(String(32), ForeignKey("nodes.id"), nullable=False)

    # AI 行为控制
    priority = Column(Float, default=1.0, comment="权重（越高越重要）")
    disclosure = Column(Text, nullable=True, comment="触发条件（人类可读的回忆条件）")

    created_at = Column(DateTime, default=utcnow)

    # 唯一约束：同一父子对只能有一条边
    __table_args__ = (
        UniqueConstraint("parent_node_id", "child_node_id", name="uq_edge_parent_child"),
    )


# ── URI 路由缓存 (路由层) ──
class Path(Base):
    """将 human-readable 的 URI 路径映射到对应的 Edge。"""
    __tablename__ = "paths"

    id = Column(String(32), primary_key=True, default=new_uuid)
    domain = Column(String(64), nullable=False, comment="域名（core/writer/game 等）")
    path_string = Column(String(512), nullable=False, comment="路径部分（不含域名）")
    edge_id = Column(String(32), ForeignKey("edges.id"), nullable=False)

    created_at = Column(DateTime, default=utcnow)

    __table_args__ = (
        # 同一 domain+path 只能指向一条边
        UniqueConstraint("domain", "path_string", name="uq_path"),
        Index("idx_path_domain", "domain"),
    )


# ── 豆辞典 (跨节点链接) ──
class GlossaryEntry(Base):
    """关键词到记忆节点的映射。支持 Aho-Corasick 多模式匹配。"""
    __tablename__ = "glossary"

    id = Column(String(32), primary_key=True, default=new_uuid)
    keyword = Column(String(255), nullable=False, comment="关键词")
    target_uri = Column(String(512), nullable=False, comment="目标记忆的 URI")
    description = Column(Text, nullable=True, comment="关键词说明")

    __table_args__ = (
        UniqueConstraint("keyword", name="uq_glossary_keyword"),
    )
