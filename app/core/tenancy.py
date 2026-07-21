"""
多租户路径与工具：每个租户的向量库/文档目录相互隔离
"""
import os
import re
from dotenv import load_dotenv

load_dotenv()

VECTOR_DB_ROOT = os.getenv("VECTOR_DB_PATH", "./data/vector_db")
DOC_ROOT = os.getenv("DOCUMENT_STORAGE_PATH", "./data/raw")


def safe_tenant_id(tenant_id: str) -> str:
    """清洗 tenant_id，避免路径穿越（../）等注入"""
    tenant_id = (tenant_id or "default").strip()
    tenant_id = re.sub(r"[^A-Za-z0-9_\-]", "_", tenant_id)
    return tenant_id or "default"


def tenant_vector_dir(tenant_id: str) -> str:
    """某租户的向量库目录"""
    d = os.path.join(VECTOR_DB_ROOT, safe_tenant_id(tenant_id))
    os.makedirs(d, exist_ok=True)
    return d


def faiss_path(tenant_id: str) -> str:
    return os.path.join(tenant_vector_dir(tenant_id), "faiss_index.pkl")


def bm25_path(tenant_id: str) -> str:
    return os.path.join(tenant_vector_dir(tenant_id), "bm25_index.pkl")


def tenant_doc_dir(tenant_id: str) -> str:
    """某租户的上传文档目录"""
    d = os.path.join(DOC_ROOT, safe_tenant_id(tenant_id))
    os.makedirs(d, exist_ok=True)
    return d
