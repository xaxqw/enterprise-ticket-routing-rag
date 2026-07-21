"""
检索层单元测试（离线）
覆盖：BM25 关键词检索、混合检索的归一化与融合逻辑、多租户路径隔离
"""
import os
import sys
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.bm25_retriever import BM25Retriever
from app.services.hybrid_search import HybridSearcher
from app.core.tenancy import safe_tenant_id, faiss_path, bm25_path


def test_bm25_add_and_search():
    with tempfile.TemporaryDirectory() as d:
        bm25 = BM25Retriever(index_path=os.path.join(d, "bm25.pkl"))
        bm25.add_texts([
            "混合检索结合了向量检索和BM25关键词检索",
            "LoRA微调可以用很小的代价适配下游任务",
            "Celery负责异步处理文档入库任务",
        ])
        res = bm25.search("BM25关键词检索", top_k=2)
        assert len(res) >= 1
        assert "BM25" in res[0]["text"]


def test_hybrid_normalize_and_merge():
    searcher = HybridSearcher(vector_store=None, bm25_retriever=None, reranker=None)
    vec = [{"text": "文档A很相关", "score": 0.9, "metadata": {"source": "a"}},
           {"text": "文档B一般", "score": 0.3}]
    bm = [{"text": "文档A很相关", "score": 8.0},
          {"text": "文档C关键词命中", "score": 2.0}]
    vec = searcher._normalize_scores(vec, "score")
    bm = searcher._normalize_scores(bm, "score")
    merged = searcher._merge_results(vec, bm)
    # 文档A 同时被两路命中，combined_score 应最高
    merged.sort(key=lambda x: x["combined_score"], reverse=True)
    assert merged[0]["text"] == "文档A很相关"
    assert 0.0 <= merged[0]["combined_score"] <= 1.0


def test_tenant_isolation_paths():
    assert safe_tenant_id("../evil") == "___evil"
    assert safe_tenant_id("") == "default"
    a = faiss_path("tenantA")
    b = faiss_path("tenantB")
    assert a != b
    assert "tenantA" in a and "tenantB" in b
    assert bm25_path("tenantA") != bm25_path("tenantB")
