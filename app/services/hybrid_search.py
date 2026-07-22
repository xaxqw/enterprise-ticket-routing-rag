"""
混合检索：向量检索 + BM25关键词检索 + RRF重排（本地、免费）
三路融合，召回率通常优于单一路径
"""
import os
from dotenv import load_dotenv

load_dotenv()


class HybridSearcher:
    def __init__(self, vector_store, bm25_retriever, reranker=None):
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever
        self.reranker = reranker

        # 两路融合权重（向量 0.6 / BM25 0.4）：经验起步值。
        # 向量占主导保证语义召回，BM25 占四成补关键词召回；可用评测集反推调优。
        self.vector_weight = 0.6 # 向量检索权重
        self.bm25_weight = 0.4 # BM25权重

    def search(self, query, top_k=10, rerank_top_k=5):
        """混合检索主函数"""
        # 第1步：两路分别召回（各取top_k，扩大召回）
        vector_results = self.vector_store.search(query, top_k=top_k)
        bm25_results = self.bm25_retriever.search(query, top_k=top_k)

        # 第2步：归一化分数（不同检索方式的分数范围不一样，要统一到0-1）
        vector_results = self._normalize_scores(vector_results, "score")
        bm25_results = self._normalize_scores(bm25_results, "score")

        # 第3步：加权融合（相同的文本块分数加起来）
        merged = self._merge_results(vector_results, bm25_results)

        # 第4步：RRF 重排（精细排序）
        if self.reranker and len(merged) > 0:
            merged = self.reranker.rerank(query, merged, top_k=rerank_top_k)
            # 重排后用重排分数作为最终分数
            for r in merged:
                r["final_score"] = r["rerank_score"]
        else:
            for r in merged:
                r["final_score"] = r["combined_score"]
        # 第5步：排序返回
        merged.sort(key=lambda x: x["final_score"], reverse=True)
        return merged[:rerank_top_k if self.reranker else top_k]

    def _normalize_scores(self, results, score_key):
        """Min-Max归一化到[0, 1]区间"""
        if not results:
            return results
        scores = [r[score_key] for r in results]
        min_s, max_s = min(scores), max(scores)
        if max_s - min_s < 1e-6:
            for r in results:
                r[f"norm_{score_key}"] = 1.0
        else:
            for r in results:
                r[f"norm_{score_key}"] = (r[score_key] - min_s) / (max_s - min_s)
        return results

    def _merge_results(self, vector_results, bm25_results):
        """合并两路结果，相同文本加权求和"""
        text_map = {}

        # 加入向量检索的结果（保留原始余弦相似度 raw_vector_score，供相关性闸门判断“是否真有匹配”）
        for r in vector_results:
            key = r["text"][:100] # 用前100字当key，近似去重
            if key not in text_map:
                text_map[key] = {
                    "text": r["text"],
                    "metadata": r.get("metadata", {}),
                    "vector_score": r["norm_score"],
                    "bm25_score": 0.0,
                    "raw_vector_score": r.get("score", 0.0),
                    "raw_bm25_score": 0.0,
                }
        # 加入BM25的结果（保留原始 BM25 分数 raw_bm25_score）
        for r in bm25_results:
            key = r["text"][:100]
            if key not in text_map:
                text_map[key] = {
                    "text": r["text"],
                    "metadata": {},
                    "vector_score": 0.0,
                    "bm25_score": r["norm_score"],
                    "raw_vector_score": 0.0,
                    "raw_bm25_score": r.get("score", 0.0),
                }
            else:
                text_map[key]["bm25_score"] = r["norm_score"]
                text_map[key]["raw_bm25_score"] = r.get("score", 0.0)
        # 计算加权总分
        merged = list(text_map.values())
        for r in merged:
            r["combined_score"] = (
                self.vector_weight * r["vector_score"] +
                self.bm25_weight * r["bm25_score"]
            )
        return merged
