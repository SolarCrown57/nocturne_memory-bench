"""
豆辞典 — 简化版关键词↔跨节点链接引擎。

benchmark 版本简化：
  - 不依赖 pyahocorasick（用简单的字符串匹配代替，benchmark 场景可控）
  - 同步 API
"""

from typing import Optional, List, Dict

from sqlalchemy.orm import Session, sessionmaker
from .models import GlossaryEntry, new_uuid


class GlossaryService:
    """关键词 ↔ 记忆节点映射服务。"""

    def __init__(self, session_factory, search_indexer=None):
        self._Session = session_factory
        self._search = search_indexer

    def _session(self) -> Session:
        return self._Session()

    def add_keyword(self, keyword: str, target_uri: str, description: str = "") -> Dict:
        """注册关键词。"""
        session = self._session()
        try:
            entry = GlossaryEntry(
                id=new_uuid(),
                keyword=keyword,
                target_uri=target_uri,
                description=description,
            )
            session.add(entry)
            session.commit()
            return {"keyword": keyword, "target_uri": target_uri}
        finally:
            session.close()

    def remove_keyword(self, keyword: str) -> Dict:
        """移除关键词。"""
        session = self._session()
        try:
            entry = session.query(GlossaryEntry).filter(
                GlossaryEntry.keyword == keyword
            ).first()
            if not entry:
                return {"keyword": keyword, "removed": False}
            session.delete(entry)
            session.commit()
            return {"keyword": keyword, "removed": True}
        finally:
            session.close()

    def find_references(self, text: str) -> List[Dict[str, str]]:
        """在文本中找出所有匹配的关键词引用。

        简化为线性扫描（benchmark 场景可控），原始实现用 Aho-Corasick。
        """
        session = self._session()
        try:
            all_keywords = session.query(GlossaryEntry).all()
            references = []
            for entry in all_keywords:
                if entry.keyword.lower() in text.lower():
                    references.append({
                        "keyword": entry.keyword,
                        "target_uri": entry.target_uri,
                        "description": entry.description or "",
                    })
            return references
        finally:
            session.close()

    def get_all_keywords(self) -> List[Dict]:
        """获取所有已注册关键词。"""
        session = self._session()
        try:
            entries = session.query(GlossaryEntry).all()
            return [
                {"keyword": e.keyword, "target_uri": e.target_uri, "description": e.description or ""}
                for e in entries
            ]
        finally:
            session.close()

    def clear(self):
        """清空所有关键词。"""
        session = self._session()
        try:
            session.query(GlossaryEntry).delete()
            session.commit()
        finally:
            session.close()
