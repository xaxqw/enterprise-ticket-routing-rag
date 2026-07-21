"""
向量级去重：语义相似的块只保留一个（本地 Ollama Embedding 版）

- 不再依赖本地 sentence_transformers（笔记本 CPU 跑不动大模型）
- 统一走本地 Ollama Embedding（app.services.embeddings），与向量库同源
"""
import logging
import numpy as np

from app.core.log import get_logger

logger = get_logger(__name__)


class VectorDeduplicator:
    def __init__(self, model_path=None, threshold=0.95):
        """
        model_path: 兼容旧参数，本地模式下忽略
        threshold: 相似度阈值，>=0.95 认为语义重复
        """
        self.threshold = threshold

    def deduplicate(self, chunks):
        """对文本块做向量语义去重，返回去重后的块列表"""
        if len(chunks) <= 1:
            return chunks

        from app.services.embeddings import embed_texts
        logger.info("正在计算向量用于语义去重（本地 Ollama）...")
        # 已归一化，内积即余弦相似度；embed_texts 会过滤空串
        embeddings, valid_texts = embed_texts(chunks, normalize=True)
        if len(valid_texts) <= 1:
            return valid_texts

        # 相似度矩阵 = 归一化向量的内积
        sim_matrix = embeddings @ embeddings.T

        keep_indices = []
        removed = set()
        for i in range(len(valid_texts)):
            if i in removed:
                continue
            keep_indices.append(i)
            # 后面与 i 高度相似的都标记删除
            sims = sim_matrix[i]
            for j in range(i + 1, len(valid_texts)):
                if j not in removed and sims[j] >= self.threshold:
                    removed.add(j)

        logger.info("去重完成：去掉了 %s 个语义重复块（保留 %s 个）",
                    len(removed), len(keep_indices))
        return [valid_texts[i] for i in keep_indices]
