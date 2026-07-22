"""
图片入库：把文档里的图片转成可检索的多模态描述，并落盘供前端渲染。

- PDF：每页渲染成图片（pdf2image），图片落盘到 data/image_store/{tenant}
- 单图文件：直接落盘到 data/image_store/{tenant}
- 每张图用 image_caption 生成描述文字，作为「文本块」进入现有向量/BM25 检索空间
  （图片描述与文档文字天然处于同一向量空间，因此问文字问题即可召回对应图片）
- 文本块的 metadata 带 image_path（落盘后的绝对路径），问答命中即返回该图
"""
import os
import shutil
import tempfile
import logging

from app.core.log import get_logger
logger = get_logger(__name__)

from dotenv import load_dotenv
load_dotenv()

from app.core.tenancy import (
    image_store_dir, safe_tenant_id, faiss_path, bm25_path,
)
from app.services.image_caption import generate_image_caption
from app.services.file_parser import FileParser


def _store_image(src_path, tenant_id):
    """把图片压缩落盘到 image_store/{tenant}，返回落盘后的绝对路径"""
    store = image_store_dir(tenant_id)
    base = f"{safe_tenant_id(tenant_id)}_{os.path.splitext(os.path.basename(src_path))[0]}"
    dst = os.path.join(store, base + ".jpg")
    try:
        from PIL import Image
        im = Image.open(src_path).convert("RGB")
        max_side = 1280
        w, h = im.size
        if max(w, h) > max_side:
            scale = max_side / max(w, h)
            im = im.resize((int(w * scale), int(h * scale)))
        im.save(dst, "JPEG", quality=82)
    except Exception as e:
        logger.info(" 图片压缩失败（%s），直接复制原图", e)
        shutil.copy2(src_path, dst)
    return os.path.abspath(dst)


def _add_caption_block(caption, image_path, source_name, page, tenant_id,
                       vector_store=None, bm25=None):
    """把图片描述作为文本块加入检索索引（带 image_path 元数据）"""
    from app.services.vector_store import FAISSVectorStore
    from app.services.bm25_retriever import BM25Retriever
    if vector_store is None:
        vector_store = FAISSVectorStore(index_path=faiss_path(tenant_id))
    if bm25 is None:
        bm25 = BM25Retriever(index_path=bm25_path(tenant_id))
    meta = {
        "source": source_name or os.path.basename(image_path),
        "image_path": image_path,
        "type": "image",
    }
    if page is not None:
        meta["page"] = page
    vector_store.add_texts([caption], [meta])
    bm25.add_texts([caption])
    return 1


def ingest_image(image_path, source_name, tenant_id, page=None,
                 vector_store=None, bm25=None):
    """入库单张图片：落盘 + 生成描述 + 加入检索空间。返回 1（成功）或 0。"""
    tenant_id = safe_tenant_id(tenant_id)
    stored = _store_image(image_path, tenant_id)
    caption = generate_image_caption(stored, source_name, page)
    if not caption:
        return 0
    return _add_caption_block(caption, stored, source_name, page, tenant_id,
                              vector_store, bm25)


def ingest_document_images(file_path, tenant_id, vector_store=None, bm25=None):
    """
    入库一个文档里的全部图片（PDF 按页渲染；图片文件直接入库）。
    返回成功入库的图片数。任何异常都不阻塞主文本建库流程。
    """
    tenant_id = safe_tenant_id(tenant_id)
    ext = os.path.splitext(file_path)[1].lower()
    source_name = os.path.basename(file_path)
    count = 0
    try:
        if ext == ".pdf":
            parser = FileParser()
            tmp = tempfile.mkdtemp(prefix="rag_img_")
            try:
                pages = parser.extract_page_images(file_path, tmp)
                for img_path, page_no in pages:
                    count += ingest_image(img_path, source_name, tenant_id,
                                          page=page_no, vector_store=vector_store, bm25=bm25)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
        elif ext in (".png", ".jpg", ".jpeg"):
            count += ingest_image(file_path, source_name, tenant_id,
                                  vector_store=vector_store, bm25=bm25)
    except Exception as e:
        logger.info(" 图片入库失败（%s），跳过", e)
    return count
