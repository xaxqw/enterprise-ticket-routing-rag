"""
数据流水线单元测试（离线，不依赖网络/Redis）
覆盖：文本清洗、语义分块、低质量过滤、去重
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.text_cleaner import TextCleaner
from app.services.semantic_chunker import SemanticChunker


def test_clean_removes_extra_whitespace_and_dup_punct():
    cleaner = TextCleaner()
    dirty = "你好    世界！！！\n\n\n\n这是   测试。。。"
    clean = cleaner.clean(dirty)
    assert "    " not in clean
    assert "！！！" not in clean
    assert "。。。" not in clean


def test_filter_low_quality_drops_short_and_symbol_chunks():
    cleaner = TextCleaner()
    chunks = [
        "这是一段足够长的正常中文内容，应该被保留下来用于检索。",
        "abc",                       # 太短
        "###@@@!!!***---+++===///",  # 几乎无文字
    ]
    kept = cleaner.filter_low_quality(chunks)
    assert len(kept) == 1
    assert kept[0].startswith("这是一段")


def test_remove_duplicate_chunks():
    cleaner = TextCleaner()
    chunks = ["重复内容重复内容重复内容", "重复内容重复内容重复内容", "唯一内容唯一内容唯一内容"]
    unique = cleaner.remove_duplicate_chunks(chunks)
    assert len(unique) == 2


def test_semantic_chunker_respects_size():
    chunker = SemanticChunker(chunk_size=50, chunk_overlap=10)
    text = "\n\n".join([f"这是第{i}段测试文本，用来验证分块逻辑是否按大小切分。" for i in range(6)])
    chunks = chunker.chunk(text)
    assert len(chunks) >= 2
    assert all(isinstance(c, str) and c for c in chunks)
