"""
检索Agent：负责找资料并生成回答
"""
import logging

from app.core.log import get_logger

logger = get_logger(__name__)


class RetrievalAgent:
    def __init__(self, rag_service):
        self.rag = rag_service

    def handle(self, query, session_id="default", top_k=5):
        """处理查询"""
        logger.info("[检索Agent] 正在检索: %s", query)

        result = self.rag.query(session_id, query, top_k=top_k)

        return {
            "answer": result["answer"],
            "sources": result["references"],
            "images": result.get("images", []),
            "agent": "retrieval"
        }

    def rewrite_query(self, query, history):
        """
 查询改写：结合历史对话把模糊的问题改清楚
 比如用户问"那它有什么特点？" → 改成"XX产品有什么特点？"
 """
        if not history:
            return query
        last_q = history[-1]["content"] if history else ""
        return f"{last_q} {query}"
