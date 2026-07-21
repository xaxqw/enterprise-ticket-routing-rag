"""
    Celery 异步任务：真实的文档入库流水线

    流水线：
    parse(多格式/URL) -> clean(清洗) -> chunk(语义分块) -> filter/去重(文本+向量) ->
    embed(在线Embedding) -> FAISS 向量索引 + BM25 关键词索引 -> 失效该租户缓存
    每个租户的向量库相互隔离（多租户）。
    """

import logging
from app.core.log import get_logger
logger = get_logger(__name__)

import os
import traceback
from dotenv import load_dotenv

from app.core.celery_app import celery_app
from app.core.tenancy import faiss_path, bm25_path, tenant_doc_dir, safe_tenant_id

load_dotenv()


def _run_pipeline(self, text, source_name, tenant_id, extra_meta=None):
    """把一段已解析的文本走完清洗->分块->去重->入库，返回入库块数"""
    from app.services.text_cleaner import TextCleaner
    from app.services.semantic_chunker import SemanticChunker
    from app.services.vector_dedup import VectorDeduplicator
    from app.services.vector_store import FAISSVectorStore
    from app.services.bm25_retriever import BM25Retriever

    tenant_id = safe_tenant_id(tenant_id)
    cleaner = TextCleaner()
    chunker = SemanticChunker(chunk_size=500, chunk_overlap=50)

    # 1. 清洗
    self.update_state(state="PROGRESS", meta={"progress": 30, "status": "清洗文本"})
    text = cleaner.clean(text)

    # 2. 语义分块
    self.update_state(state="PROGRESS", meta={"progress": 45, "status": "语义分块"})
    chunks = chunker.chunk(text)
    chunks = cleaner.filter_low_quality(chunks)
    chunks = cleaner.remove_duplicate_chunks(chunks)  # 文本级精确去重
    if not chunks:
        return 0

    # 3. 向量级语义去重（本地 Ollama Embedding，去掉"意思重复"的块）
    self.update_state(state="PROGRESS", meta={"progress": 60, "status": "语义去重"})
    try:
        dedup = VectorDeduplicator(threshold=0.95)
        chunks = dedup.deduplicate(chunks)
    except Exception as e:
        logger.info(f" 向量去重跳过（{e}），继续用文本去重结果")

    # 4. 向量化 + 建索引（租户隔离路径）
    self.update_state(state="PROGRESS", meta={"progress": 80, "status": "向量化入库"})
    metadata = [{"source": source_name, "tenant_id": tenant_id, **(extra_meta or {})} for _ in chunks]

    vector_store = FAISSVectorStore(index_path=faiss_path(tenant_id))
    vector_store.add_texts(chunks, metadata)

    bm25 = BM25Retriever(index_path=bm25_path(tenant_id))
    bm25.add_texts(chunks)

    # 5. 该租户查询缓存失效（避免旧缓存盖住新知识）
    _invalidate_tenant_cache(tenant_id)
    return len(chunks)


def _invalidate_tenant_cache(tenant_id):
    try:
        import redis as redis_lib
        client = redis_lib.Redis(
            host=os.getenv("REDIS_HOST", "127.0.0.1"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            db=int(os.getenv("REDIS_DB", 0)),
            decode_responses=True,
        )
        keys = client.keys(f"cache:rag:{tenant_id}:*")
        if keys:
            client.delete(*keys)
            logger.info(f" 已清理租户[{tenant_id}]的 {len(keys)} 条查询缓存")
    except Exception as e:
        logger.info(f" 缓存清理失败（忽略）：{e}")


@celery_app.task(bind=True, name="process_document")
def process_document_task(self, file_path, tenant_id="default"):
    """
        异步处理单个上传文档：解析->清洗->分块->去重->向量化->建索引
        """
    try:
        from app.services.file_parser import FileParser
        self.update_state(state="PROGRESS", meta={"progress": 10, "status": "解析文档"})

        parser = FileParser()
        text = parser.auto_parse(file_path)
        if not text or len(text) < 20:
            return {"status": "failed", "error": "文档内容为空或过短", "chunks_added": 0}

        n = _run_pipeline(self, text, os.path.basename(file_path), tenant_id)
        if n == 0:
            return {"status": "failed", "error": "未提取到有效文本块", "chunks_added": 0}

        return {"status": "success", "file_path": file_path,
                "tenant_id": safe_tenant_id(tenant_id), "chunks_added": n}
    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "error": str(e), "chunks_added": 0}


@celery_app.task(bind=True, name="process_url")
def process_url_task(self, url, tenant_id="default"):
    """异步抓取网页 URL 并入库（多源数据流水线）"""
    try:
        from app.services.file_parser import FileParser
        self.update_state(state="PROGRESS", meta={"progress": 10, "status": "抓取网页"})

        parser = FileParser()
        text = parser.parse_url(url)
        if not text or len(text) < 20:
            return {"status": "failed", "error": "网页正文为空", "chunks_added": 0}

        n = _run_pipeline(self, text, url, tenant_id, extra_meta={"type": "url"})
        return {"status": "success", "url": url,
                "tenant_id": safe_tenant_id(tenant_id), "chunks_added": n}
    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "error": str(e), "chunks_added": 0}


@celery_app.task(bind=True, name="rebuild_tenant_index")
def rebuild_tenant_index_task(self, tenant_id="default"):
    """
        重建某租户的整个索引：清空后重新扫描该租户 data/raw 目录全量入库
        用于文档删除后保持索引一致。
        """
    try:
        tenant_id = safe_tenant_id(tenant_id)
        # 清空旧索引
        for p in (faiss_path(tenant_id), bm25_path(tenant_id)):
            if os.path.exists(p):
                os.remove(p)

        doc_dir = tenant_doc_dir(tenant_id)
        from app.services.file_parser import FileParser
        parser = FileParser()
        supported = {".pdf", ".txt", ".md", ".markdown", ".docx", ".doc", ".xlsx", ".xls"}

        total = 0
        for fname in os.listdir(doc_dir):
            fpath = os.path.join(doc_dir, fname)
            if not os.path.isfile(fpath):
                continue
            if os.path.splitext(fname)[1].lower() not in supported:
                continue
            try:
                text = parser.auto_parse(fpath)
                if text and len(text) >= 20:
                    total += _run_pipeline(self, text, fname, tenant_id)
            except Exception as e:
                logger.info(f" 重建时跳过 {fname}：{e}")

        return {"status": "success", "tenant_id": tenant_id, "chunks_added": total}
    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "error": str(e)}
