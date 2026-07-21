"""
FAISS 向量存储：本地向量库 + 本地 Ollama 向量化（完全免费/离线）

- 向量生成走本地 Ollama bge-m3（1024 维多语言/中文向量，无需 API Key）
- 索引构建与持久化走本地 FAISS（IndexFlatIP，离线、免费）
- 检索阶段对归一化向量矩阵做精确内积（= 余弦相似度），与 FAISS IndexFlatIP
  数学等价；FAISS 索引负责落盘，检索逻辑与重排/融合共享同一 numpy 矩阵。
"""
import os
import pickle
import time
import logging
import numpy as np
import faiss
from dotenv import load_dotenv

from app.core.log import get_logger

load_dotenv()
logger = get_logger(__name__)


class FAISSVectorStore:
    def __init__(self, model_path=None, index_path="./data/vector_db/faiss_index.pkl"):
        """
        model_path: 兼容旧参数，现由本地 Ollama 提供向量化，此参数忽略
        index_path: 向量索引保存路径
        """
        self.index_path = index_path
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "bge-m3")
        self.dimension = int(os.getenv("EMBEDDING_DIM", 1024))
        self.texts = []  # 存原始文本
        self.metadata = []  # 存元数据（来源、页码等）
        self.index = None  # FAISS IndexFlatIP 索引（负责构建与持久化）
        self.matrix = None  # 检索用的 numpy 向量矩阵（归一化向量精确内积 = 余弦相似度）
        self._load_or_create()

    def _get_client(self):
        """（已废弃）在线 SiliconFlow 后端已移至可选路径，本地默认模式不依赖付费 API。"""
        raise RuntimeError(
            "本地默认模式不依赖 SiliconFlow；如需在线后端请在 rag_service 中按 "
            "LLM_BACKEND=siliconflow 分派，并确保配置 SILICONFLOW_API_KEY。"
        )

    def _embed(self, texts):
        """
        调用本地 Ollama Embedding 入口把文本批量转成向量
        返回 numpy 数组，shape = (有效文本数, dimension)，已 L2 归一化
        注意：入参最好已过滤空串（add_texts 已做），空串会被忽略
        """
        from app.services.embeddings import embed_texts
        vecs, _ = embed_texts(texts, normalize=True)
        return vecs

    def _load_or_create(self):
        """加载已有索引，没有就创建新的"""
        if os.path.exists(self.index_path):
            try:
                with open(self.index_path, "rb") as f:
                    data = pickle.load(f)
                    self.index = data["index"]
                    self.texts = data["texts"]
                    self.metadata = data.get("metadata", [])
                    # 检索矩阵优先用持久化的 numpy 矩阵；缺失时在 search() 里用
                    # _ensure_matrix() 重新向量化恢复（FAISS 仅负责索引持久化）。
                    self.matrix = data.get("matrix")
                    # 校验维度
                    if self.index.d != self.dimension:
                        logger.warning("已有索引维度(%s)与配置(%s)不符，重建索引",
                                       self.index.d, self.dimension)
                        self.index = faiss.IndexFlatIP(self.dimension)
                        self.texts = []
                        self.metadata = []
                        self.matrix = None
                    else:
                        logger.info("加载向量索引成功，共 %s 条", len(self.texts))
            except Exception as e:
                logger.warning("加载索引失败(%s)，重建索引", e)
                self.index = faiss.IndexFlatIP(self.dimension)
        else:
            # IndexFlatIP：内积索引，适合归一化后的向量（等价于余弦相似度）
            self.index = faiss.IndexFlatIP(self.dimension)
            logger.info("创建新的向量索引")

    def _ensure_matrix(self):
        """
        确保检索用 numpy 矩阵存在；缺失时（旧索引/重建后）通过重新向量化语料恢复。
        检索阶段直接对归一化向量矩阵做精确内积，与 FAISS IndexFlatIP 数学等价，
        且能复用同一矩阵做重排/融合，避免对 FAISS 索引的重复访问。
        """
        if self.matrix is not None and self.matrix.shape[0] == len(self.texts):
            return
        if len(self.texts) == 0:
            self.matrix = np.empty((0, self.dimension), dtype="float32")
            return
        logger.info("重建检索矩阵（重新向量化 %s 条语料，仅此一次）...", len(self.texts))
        vecs = self._embed(self.texts)
        self.matrix = vecs
        # 立即落盘，下次启动直接加载矩阵，无需再向量化
        try:
            self._save()
        except Exception:
            pass

    def add_texts(self, texts, metadata_list=None):
        """添加文本到向量库"""
        if not texts:
            return

        # 过滤空文本，保证向量和文本一一对应
        valid_indices = [i for i, t in enumerate(texts) if t and t.strip()]
        if not valid_indices:
            logger.info("所有文本块均为空，跳过")
            return

        valid_texts = [texts[i] for i in valid_indices]
        valid_metadata = (
            [metadata_list[i] for i in valid_indices] if metadata_list else None
        )

        n_filtered = len(texts) - len(valid_texts)
        if n_filtered > 0:
            logger.info("过滤 %s 个空文本块", n_filtered)

        logger.info("正在向量化 %s 条文本（本地 Ollama）...", len(valid_texts))
        embeddings = self._embed(valid_texts)
        self.index.add(embeddings)
        # 同步维护检索用 numpy 矩阵（归一化向量精确内积 = 余弦相似度）
        if self.matrix is not None and self.matrix.shape[1] == embeddings.shape[1]:
            self.matrix = np.vstack([self.matrix, embeddings])
        else:
            self.matrix = embeddings.copy()
        self.texts.extend(valid_texts)

        if valid_metadata:
            self.metadata.extend(valid_metadata)
        else:
            self.metadata.extend([{}] * len(valid_texts))

        self._save()
        logger.info("已添加 %s 条，总计 %s 条", len(valid_texts), len(self.texts))

    def search(self, query, top_k=10):
        """
        向量检索：返回最相似的 top_k 个文本块
        query: 用户的问题
        top_k: 返回前几个结果

        实现说明：向量已 L2 归一化，检索用 numpy 精确内积（与 FAISS IndexFlatIP
        的余弦相似度数学等价）；FAISS 索引负责构建与持久化，二者共享同一组向量。
        """
        if len(self.texts) == 0:
            return []

        query_vec = self._embed([query])  # shape (1, dim)，已 L2 归一化

        # 确保检索矩阵可用（缺失时重新向量化语料，绝不调用 faiss.search / get_xb）
        self._ensure_matrix()

        if self.matrix is not None and self.matrix.shape[0] == len(self.texts):
            sims = self.matrix @ query_vec[0]  # (N,) 精确内积 = 余弦相似度
            k = min(top_k, len(sims))
            top_idx = np.argsort(-sims)[:k]
            results = []
            for idx in top_idx:
                results.append({
                    "text": self.texts[idx],
                    "score": float(sims[idx]),
                    "metadata": self.metadata[idx] if idx < len(self.metadata) else {}
                })
            return results

        # 极端兜底：矩阵仍不可用（如向量化失败且无备选），返回空，避免进程崩溃
        return []

    def _save(self):
        """保存索引到磁盘（含检索用 numpy 矩阵，避免每次重启重新从 FAISS 取回）"""
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump({
                "index": self.index,
                "texts": self.texts,
                "metadata": self.metadata,
                "matrix": self.matrix
            }, f)
