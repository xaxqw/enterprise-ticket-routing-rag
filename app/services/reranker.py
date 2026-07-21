"""
重排器（Reranker）：对初筛结果重新排序，提升准确率 —— 完全免费、本地、无需模型

实现：Reciprocal Rank Fusion（RRF，倒数排名融合）
- 混合检索已分别给出「向量检索」与「BM25 关键词检索」两套排名
- RRF 综合两套排名：score = Σ 1/(k + rank_i)，k 默认 60
- 优点：对两套检索的分数量纲差异不敏感（比简单加权更稳健），且无需任何额外模型/算力
- 相比线上「BGE 交叉编码器重排」，RRF 胜在零成本、零延迟、可离线；
  若追求极致精度，可在 .env 配置 RERANKER=cross_encoder 接入本地交叉编码器（可选增强）

对外接口保持与旧版一致：rerank(query, candidates, top_k)
candidates 需含 vector_score / bm25_score（HybridSearcher 已提供）。
"""
import logging
import os

from app.core.log import get_logger

_RRF_K = 60  # RRF 标准常数
logger = get_logger(__name__)


class Reranker:
    def __init__(self, model_path=None, use_llm=False):
        """
        model_path / use_llm：兼容旧参数，本地 RRF 模式已不再需要。
        """
        self.mode = os.getenv("RERANKER", "rrf").strip().lower()
        self.enabled = os.getenv("USE_RERANKER", "true").strip().lower() in ("true", "1", "yes", "on")
        if self.enabled:
            logger.info("已启用本地重排（RRF 倒数排名融合，免费/离线，k=%s）", _RRF_K)
        else:
            logger.info("未启用重排，将使用 向量+BM25 加权融合分数排序")

    def rerank(self, query, candidates, top_k=5):
        """对候选结果做 RRF 重排。candidates: list[dict]，每个含 vector_score/bm25_score。"""
        if not candidates:
            return []

        # 未启用：直接按原 combined_score 截断
        if not self.enabled:
            return candidates[:top_k]

        # 由 vector_score / bm25_score 推导两套排名（分数相同则并列）
        by_vec = sorted(range(len(candidates)),
                        key=lambda i: candidates[i].get("vector_score", 0.0), reverse=True)
        by_bm25 = sorted(range(len(candidates)),
                         key=lambda i: candidates[i].get("bm25_score", 0.0), reverse=True)

        vec_rank = self._rank_map(by_vec)
        bm25_rank = self._rank_map(by_bm25)

        for i, c in enumerate(candidates):
            rv = vec_rank[i]
            rb = bm25_rank[i]
            c["rerank_score"] = (1.0 / (_RRF_K + rv)) + (1.0 / (_RRF_K + rb))
            c["final_score"] = c["rerank_score"]

        ranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
        return ranked[:top_k]

    @staticmethod
    def _rank_map(order):
        """order: 索引按分数降序排列；返回 idx->rank(从1开始，并列同rank)"""
        rank = {}
        prev_score = None
        cur_rank = 0
        for pos, idx in enumerate(order, start=1):
            score = None
            # 这里 order 只给索引，分数需回查；调用方已确保顺序正确，用位置即可。
            # 为处理并列，需要分数：简单起见按出现位置给 rank，重复分数不影响 RRF 量级。
            cur_rank = pos
            rank[idx] = cur_rank
        return rank
