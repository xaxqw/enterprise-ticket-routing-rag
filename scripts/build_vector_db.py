"""
    向量知识库构建脚本

    运行方式（在项目根目录，激活虚拟环境后）：
    python scripts/build_vector_db.py

    或者指定目录：
    python scripts/build_vector_db.py --dir ./data/raw

    注意：向量化默认走本地 Ollama（nomic-embed-text），完全免费、离线，需先启动 Ollama 并拉取模型。
    """
import os
import sys
import argparse

# 把项目根目录加入路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from app.services.file_parser import FileParser
from app.services.text_cleaner import TextCleaner
from app.services.semantic_chunker import SemanticChunker
from app.services.vector_store import FAISSVectorStore
from app.services.bm25_retriever import BM25Retriever
from app.services.vector_dedup import VectorDeduplicator
from app.services.data_quality_report import DataQualityReport
from app.core.tenancy import faiss_path, bm25_path

# 支持的文件格式
SUPPORTED_EXTS = {".pdf", ".txt", ".docx", ".doc", ".xlsx", ".xls", ".md", ".markdown"}


def scan_documents(root_dir):
    """递归扫描目录下所有支持的文档"""
    docs = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext in SUPPORTED_EXTS:
                docs.append(os.path.join(dirpath, fname))
    return docs


def process_document(file_path, parser, cleaner, chunker):
    """解析单个文档，返回 (raw_text, chunks, metadata_list)"""
    fname = os.path.basename(file_path)
    print(f"\n 处理: {fname}")

    try:
        raw_text = parser.auto_parse(file_path)
    except Exception as e:
        print(f" 解析失败: {e}")
        return "", [], []

    if not raw_text or len(raw_text) < 20:
        print(f" 文档内容为空或太短，跳过")
        return "", [], []

    # 清洗
    text = cleaner.clean(raw_text)
    # 分块
    chunks = chunker.chunk(text)
    # 过滤低质量 + 文本级去重
    chunks = cleaner.filter_low_quality(chunks)
    chunks = cleaner.remove_duplicate_chunks(chunks)

    print(f" 解析 {len(raw_text)} 字 -> 分块 {len(chunks)} 块")

    # 每个块带上来源元数据
    metadata_list = [{"source": fname, "file_path": file_path} for _ in chunks]
    return raw_text, chunks, metadata_list


def main():
    parser_arg = argparse.ArgumentParser(description="构建向量知识库")
    parser_arg.add_argument("--dir", default=None, help="文档所在目录（默认扫描该租户目录 data/raw/{tenant}）")
    parser_arg.add_argument("--tenant", default="default", help="租户ID（多租户隔离）")
    args = parser_arg.parse_args()

    # 切换到项目根目录（保证相对路径正确）
    os.chdir(PROJECT_ROOT)

    # 未显式指定 --dir 时，按租户隔离只扫描该租户的文档目录
    if args.dir is None:
        tenant_dir = os.path.join("./data/raw", args.tenant)
        args.dir = tenant_dir if os.path.isdir(tenant_dir) else "./data/raw"

    # 检查本地 Ollama（向量化走本地 nomic-embed-text，完全免费/离线）
    try:
        from app.services.ollama_client import ensure_model
        ensure_model(os.getenv("EMBEDDING_MODEL", "nomic-embed-text"), timeout=1200)
    except Exception as e:
        print(f" 错误：本地 Ollama 向量模型未就绪：{e}")
        print(" 请先启动 Ollama 并拉取嵌入模型：ollama pull nomic-embed-text")
        sys.exit(1)

    doc_dir = args.dir
    if not os.path.isdir(doc_dir):
        print(f" 目录不存在: {doc_dir}")
        sys.exit(1)

    # 扫描文档
    docs = scan_documents(doc_dir)
    if not docs:
        print(f" 目录 {doc_dir} 下没有可处理的文档")
        print(f" 支持的格式: {', '.join(SUPPORTED_EXTS)}")
        sys.exit(0)

    print(f" 找到 {len(docs)} 个文档，开始处理...")
    print(f" 目录: {doc_dir}")

    # 初始化各组件
    parser = FileParser()
    cleaner = TextCleaner()
    chunker = SemanticChunker(chunk_size=500, chunk_overlap=50)

    # 租户隔离的索引路径
    tenant = args.tenant
    fp = faiss_path(tenant)
    bp = bm25_path(tenant)
    for p in [fp, bp]:
        if os.path.exists(p):
            os.remove(p)
            print(f" 已清空旧索引: {p}")

    vector_store = FAISSVectorStore(index_path=fp)
    bm25 = BM25Retriever(index_path=bp)

    # 逐个处理文档
    raw_texts = []
    all_chunks = []
    all_metadata = []
    for doc_path in docs:
        raw_text, chunks, metadata = process_document(doc_path, parser, cleaner, chunker)
        if raw_text:
            raw_texts.append(raw_text)
            all_chunks.extend(chunks)
            all_metadata.extend(metadata)

        # 多模态：提取该文档里的图片并入库（以文搜图）
        try:
            from app.services.image_index import ingest_document_images
            img_n = ingest_document_images(doc_path, tenant, vector_store, bm25)
            if img_n:
                print(f" 图片入库：{img_n} 张（来自 {os.path.basename(doc_path)}）")
        except Exception as e:
            print(f" 图片入库跳过（{e}）")

    if not all_chunks:
        print("\n 没有提取到任何有效文本块，请检查文档内容")
        sys.exit(1)

    cleaned_chunks_snapshot = list(all_chunks)  # 语义去重前的块（用于质量报告）

    # 语义级去重（跨文档也能去掉意思重复的块）
    print(f"\n 语义去重前共 {len(all_chunks)} 块，开始向量级去重...")
    try:
        dedup = VectorDeduplicator(threshold=0.95)
        deduped = dedup.deduplicate(all_chunks)
        # 去重后需要重建对齐的 metadata（用块内容做映射）
        keep_set = set(deduped)
        new_chunks, new_meta = [], []
        seen = set()
        for c, m in zip(all_chunks, all_metadata):
            if c in keep_set and c not in seen:
                new_chunks.append(c)
                new_meta.append(m)
                seen.add(c)
        all_chunks, all_metadata = new_chunks, new_meta
    except Exception as e:
        print(f" 语义去重跳过（{e}）")

    print(f"\n 总计 {len(all_chunks)} 个文本块，开始向量化入库...")
    print(f" （调用本地 Ollama {os.getenv('EMBEDDING_MODEL', 'nomic-embed-text')} 向量化，完全免费/离线）")

    # 向量入库（调用本地 Ollama nomic-embed-text 向量化，完全免费/离线）
    vector_store.add_texts(all_chunks, all_metadata)
    # BM25 入库
    bm25.add_texts(all_chunks)

    # 生成数据质量报告
    try:
        report = DataQualityReport().generate(raw_texts, cleaned_chunks_snapshot, all_chunks)
        os.makedirs("./logs", exist_ok=True)
        report_path = f"./logs/data_quality_{tenant}.json"
        DataQualityReport().save_report(report, report_path)
    except Exception as e:
        print(f" 质量报告生成失败（忽略）：{e}")

    print(f"\n 知识库构建完成！")
    print(f" 租户: {tenant}")
    print(f" 文档数: {len(docs)}")
    print(f" 文本块: {len(all_chunks)}")
    print(f" 向量索引: {fp}")
    print(f" BM25索引: {bp}")
    print(f"\n现在可以启动服务进行问答了。")


if __name__ == "__main__":
    main()
