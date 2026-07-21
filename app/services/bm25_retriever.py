"""
BM25关键词检索：和向量检索互补，关键词精确匹配效果好
"""

import logging
from app.core.log import get_logger
logger = get_logger(__name__)

import os
import pickle
import jieba
from rank_bm25 import BM25Okapi


class BM25Retriever:
    def __init__(self, index_path="./data/vector_db/bm25_index.pkl"):
        self.index_path = index_path
        self.corpus = [] # 所有文本
        self.bm25 = None # BM25索引
        self._load_or_create()

    def _load_or_create(self):
        if os.path.exists(self.index_path):
            with open(self.index_path, "rb") as f:
                data = pickle.load(f)
                self.corpus = data["corpus"]
                self.bm25 = data["bm25"]
                logger.info(f" 加载BM25索引成功，共 {len(self.corpus)} 条")
        else:
            logger.info(" 创建新的BM25索引")

    def add_texts(self, texts):
        """添加文本到BM25索引"""
        self.corpus.extend(texts)
        # 用jieba分词，BM25需要分词后的结果
        tokenized_corpus = [list(jieba.cut(doc)) for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self._save()
        logger.info(f" BM25索引更新完成，共 {len(self.corpus)} 条")

    def search(self, query, top_k=10):
        """BM25检索"""
        if not self.bm25 or len(self.corpus) == 0:
            return []
        tokenized_query = list(jieba.cut(query)) # 问题也分词
        scores = self.bm25.get_scores(tokenized_query)

        # 按分数排序，取前top_k
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "text": self.corpus[idx],
                "score": float(scores[idx])
            })
        return results

    def _save(self):
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump({
 "corpus": self.corpus,
 "bm25": self.bm25
 }, f)
