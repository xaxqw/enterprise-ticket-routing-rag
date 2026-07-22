"""
统一 Embedding 入口 —— 本地 Ollama 离线向量化（完全免费，无需任何 API Key）

- 默认模型 nomic-embed-text（768 维，英文/通用检索效果优秀）
- 调用本机 Ollama /api/embed，离线运行，不消耗任何额度
- 分批 + 重试，返回 L2 归一化后的 numpy 向量（内积=余弦相似度）
- 与线上 SiliconFlow 版本的对外接口完全一致（embed_texts / get_embedding_dim），
  上层（vector_store / build_vector_db / 评测）无需改动即可切换到本地。
"""
import os
import numpy as np
from app.services.ollama_client import ollama_embed

# 兼容旧常量
_MAX_CHARS = 500  # 仅作语义占位；Ollama 端模型自带上下文长度处理


def get_embedding_dim():
    return int(os.getenv("EMBEDDING_DIM", "768"))


def embed_texts(texts, normalize=True):
    """
    批量向量化。返回 (vectors, valid_texts)
     - vectors: np.float32, shape=(N, dim)，仅包含有效（非空）文本对应的向量
     - valid_texts: 与 vectors 一一对应的原始文本（已过滤空串）
    """
    return ollama_embed(texts, normalize=normalize)
