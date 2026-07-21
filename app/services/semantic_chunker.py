"""
语义分块器：基于「句向量断点检测」的真实语义分块（参考 Greg Kamradt 的
Semantic Chunking 思路）

做法：
1. 先把文档切成句子；
2. 用同一套 embedding 模型给每个句子编码（L2 归一化，余弦相似度 = 向量内积）；
3. 计算相邻句子的相似度，相似度低于阈值的缝隙视为「话题切换」，作为断点；
4. 按断点切出语义完整的片段，再做一次贪心合并，使每个块尽量接近 chunk_size，
   相邻块保留 chunk_overlap 的句子级重叠，避免边界信息丢失。

相比「按固定字数滑动窗口」，语义分块能保证一个知识点不被从中间切断，
检索召回更准。embedding 不可用时自动退回「段落 + 滑动窗口」兜底。
"""

import logging
from app.core.log import get_logger
logger = get_logger(__name__)

import re


class SemanticChunker:
    def __init__(self, chunk_size=500, chunk_overlap=50, breakpoint_threshold=0.55):
        """
        chunk_size: 每个块目标字数（默认 500）
        chunk_overlap: 相邻块重叠句子数折合的字数上限（默认 50）
        breakpoint_threshold: 相邻句余弦相似度低于该值即断句（默认 0.55，可调）
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.breakpoint_threshold = breakpoint_threshold

    def chunk(self, text):
        """主分块函数：返回语义完整的文本块列表"""
        sentences = self._split_sentences(text)
        if not sentences:
            return []
        if len(sentences) == 1:
            return [sentences[0]]

        emb = self._embed_sentences(sentences)
        if emb is None:
            # embedding 不可用 → 退回段落 + 滑动窗口兜底
            return self._window_chunk(text)

        # 相邻句余弦相似度（已归一化，内积即余弦）
        sims = [float(emb[i] @ emb[i + 1]) for i in range(len(emb) - 1)]
        # 断点：相似度低于阈值处断开
        breakpoints = [i for i, s in enumerate(sims) if s < self.breakpoint_threshold]

        return self._merge(sentences, breakpoints)

    # ---------------------- 内部方法 ----------------------
    def _split_sentences(self, text):
        """中英文混合句子切分（按句末标点切，保留标点）"""
        parts = re.split(r"(?<=[。！？!?；;])\s*", text)
        return [p.strip() for p in parts if p and p.strip()]

    def _embed_sentences(self, sentences):
        """批量句向量；失败返回 None（触发兜底）"""
        try:
            from app.services.embeddings import embed_texts
            vecs, _ = embed_texts(sentences, normalize=True)
            return vecs
        except Exception as e:
            logger.info(f" 语义分块 embedding 失败，退回窗口切分：{e}")
            return None

    def _merge(self, sentences, breakpoints):
        """按断点切段，再贪心合并成接近 chunk_size 的块（句级重叠）"""
        # 1) 按断点切出初始语义段
        segments, start = [], 0
        for bp in breakpoints:
            segments.append(sentences[start:bp + 1])
            start = bp + 1
        segments.append(sentences[start:])
        segments = [s for s in segments if s]

        # 2) 贪心合并
        chunks, cur, cur_len = [], [], 0
        for seg in segments:
            seg_text = "".join(seg)
            seg_len = len(seg_text)
            if cur_len + seg_len <= self.chunk_size or not cur:
                cur.extend(seg)
                cur_len += seg_len
            else:
                chunks.append("".join(cur))
                # 新块带重叠：取上一块末尾 1~2 句
                overlap = cur[-2:] if self.chunk_overlap >= 50 and len(cur) >= 2 else cur[-1:]
                cur = list(overlap) + seg
                cur_len = sum(len(s) for s in cur)
        if cur:
            chunks.append("".join(cur))
        return [c for c in chunks if c.strip()]

    def _window_chunk(self, text):
        """兜底：段落 + 字符滑动窗口（原实现）"""
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        chunks, current = [], ""
        for para in paragraphs:
            if len(current) + len(para) < self.chunk_size:
                current += para + "\n"
            else:
                if current:
                    chunks.append(current.strip())
                current = self._get_overlap(current) + para + "\n"
        if current.strip():
            chunks.append(current.strip())
        return chunks

    def _get_overlap(self, text):
        if len(text) <= self.chunk_overlap:
            return text
        return text[-self.chunk_overlap:]
