"""
检索链路端到端测试（离线）

做法：
- 只用「确定性哈希词袋 Embedding」打桩在线 Embedding 这一外部依赖（monkeypatch
  app.services.embeddings.embed_texts）。其余全部真实：构建 FAISS 索引、BM25、
  向量 + BM25 加权融合的混合检索。
- 进一步端到端跑通 RAGService.query（在线 LLM 同样打桩），验证检索结果能进入生成、
  且多轮对话记忆正确落库到 Redis。

这比在路由层整体 mock 更有说服力：它真实验证了「建库 → 混合检索 → 答案 → 记忆」
这条主链路在离线条件下也能闭环。
"""
import os
import tempfile

import numpy as np
import faiss
import pytest

from app.services.vector_store import FAISSVectorStore
from app.services.bm25_retriever import BM25Retriever
from app.services.hybrid_search import HybridSearcher
from app.services.rag_service import RAGService
from app.services.conversation_memory import ConversationMemory


def _hash_embed(texts, normalize=True):
    """
    确定性离线 Embedding 桩：基于字符哈希的「词袋」向量，
    共享字符越多的文本向量越接近（余弦相似度越高）。
    """
    dim = 1024
    out = []
    for t in texts:
        v = np.zeros(dim, dtype=np.float32)
        for ch in t:
            h = (ord(ch) * 2654435761) % dim
            v[h] += 1.0
        n = np.linalg.norm(v)
        if n > 0:
            v /= n
        out.append(v)
    arr = np.array(out, dtype=np.float32)
    if normalize and len(arr):
        faiss.normalize_L2(arr)
    return arr, list(texts)


@pytest.fixture
def fake_embed(monkeypatch):
    monkeypatch.setattr("app.services.embeddings.embed_texts", _hash_embed)


DOCS = [
    "混合检索结合了向量检索和BM25关键词检索，并用重排提升精度。",
    "LoRA微调可以用很小的代价把大模型适配到下游任务。",
    "Celery负责把文档入库做成异步流水线任务，避免阻塞主线程。",
]


def _build_searcher(tmp):
    vs = FAISSVectorStore(index_path=os.path.join(tmp, "faiss.pkl"))
    bm = BM25Retriever(index_path=os.path.join(tmp, "bm25.pkl"))
    vs.add_texts(DOCS)
    bm.add_texts(DOCS)
    return HybridSearcher(vs, bm, reranker=None)


def test_hybrid_retrieval_ranks_relevant_doc_first(fake_embed):
    with tempfile.TemporaryDirectory() as tmp:
        searcher = _build_searcher(tmp)
        res = searcher.search("BM25关键词检索怎么用？", top_k=3)
        assert len(res) >= 1
        # 含 BM25 关键词的文档应排在最前（BM25 召回主导相关文档）
        assert "BM25" in res[0]["text"]


def test_rag_service_query_end_to_end(fake_embed, fake_redis, monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        searcher = _build_searcher(tmp)
        memory = ConversationMemory(fake_redis)
        rag = RAGService(searcher, memory)

        # 打桩在线 LLM 生成（monkeypatch 结束后自动还原）
        def _fake_llm(self, messages, temperature=0.7, max_tokens=1024):
            return "这是一段基于检索结果的模拟回答。"

        monkeypatch.setattr(RAGService, "_llm_chat", _fake_llm)

        out = rag.query("sess1", "混合检索是什么？", top_k=3)

        assert "模拟回答" in out["answer"]
        assert len(out["references"]) >= 1

        # 会话记忆已落库（用户 + 助手两条）
        hist = memory.get_history_as_list("sess1")
        assert len(hist) == 2
        assert hist[0]["role"] == "user"
        assert hist[1]["role"] == "assistant"
