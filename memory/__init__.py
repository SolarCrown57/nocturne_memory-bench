"""
复刻 Nocturne Memory 的记忆核心层公共 API。

提供惰性初始化 + 服务 getter 模式。
"""

from typing import Optional, TYPE_CHECKING

from .models import Base, Node, Memory, Edge, Path, GlossaryEntry, ROOT_NODE_UUID

if TYPE_CHECKING:
    from .graph import GraphService
    from .search import SearchIndexer
    from .glossary import GlossaryService

_db_engine = None
_graph_service: Optional["GraphService"] = None
_search_indexer: Optional["SearchIndexer"] = None
_glossary_service: Optional["GlossaryService"] = None


def get_graph_service() -> "GraphService":
    """获取图引擎服务"""
    _ensure_initialized()
    return _graph_service  # type: ignore


def get_search_indexer() -> "SearchIndexer":
    """获取搜索索引服务"""
    _ensure_initialized()
    return _search_indexer  # type: ignore


def get_glossary_service() -> "GlossaryService":
    """获取豆辞典服务"""
    _ensure_initialized()
    return _glossary_service  # type: ignore


def _ensure_initialized():
    """惰性初始化所有服务"""
    global _graph_service, _search_indexer, _glossary_service, _db_engine
    if _graph_service is not None:
        return

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from .search import SearchIndexer
    from .glossary import GlossaryService
    from .graph import GraphService

    import config as cfg
    bench_cfg = cfg.BenchmarkConfig()
    _db_engine = create_engine(bench_cfg.database_url, echo=False)
    Base.metadata.create_all(_db_engine)
    _Session = sessionmaker(bind=_db_engine)

    _search_indexer = SearchIndexer(_Session)
    _glossary_service = GlossaryService(_Session, _search_indexer)
    _graph_service = GraphService(_Session, _search_indexer, _glossary_service)


def reset_memory():
    """重置所有记忆（用于每次测试运行前清理）"""
    global _graph_service, _search_indexer, _glossary_service, _db_engine
    if _db_engine:
        Base.metadata.drop_all(_db_engine)
        Base.metadata.create_all(_db_engine)
    _graph_service = None
    _search_indexer = None
    _glossary_service = None


__all__ = [
    "Base", "Node", "Memory", "Edge", "Path", "GlossaryEntry", "ROOT_NODE_UUID",
    "get_graph_service", "get_search_indexer", "get_glossary_service",
    "reset_memory",
]
