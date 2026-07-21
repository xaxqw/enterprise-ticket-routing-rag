"""
    文件管理路由：上传/列表/删除 + URL 数据源导入（多源数据流水线入口）
    所有操作按 tenant_id 隔离，A 租户看不到 B 租户的文档。
    """
import os
import shutil
from datetime import datetime
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from pydantic import BaseModel

from app.api.auth_router import get_current_user
from app.core.tenancy import tenant_doc_dir, safe_tenant_id
from app.services.tasks import process_document_task, process_url_task, rebuild_tenant_index_task

router = APIRouter()


@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
    ):
    tenant_id = safe_tenant_id(current_user["tenant_id"])
    tenant_dir = tenant_doc_dir(tenant_id)

    file_path = os.path.join(tenant_dir, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

        # 丢进 Celery 异步流水线（真实的 解析→清洗→分块→去重→向量化→建索引）
        task = process_document_task.delay(file_path, tenant_id)

        return {
            "filename": file.filename,
            "size": os.path.getsize(file_path),
            "task_id": task.id,
            "status": "processing",
            "message": "已提交后台处理，稍后即可检索到该文档内容",
            }


class URLIngest(BaseModel):
    url: str


@router.post("/ingest_url")
async def ingest_url(body: URLIngest, current_user: dict = Depends(get_current_user)):
    """多源数据流水线：把一个网页 URL 抓取入库"""
    tenant_id = safe_tenant_id(current_user["tenant_id"])
    task = process_url_task.delay(body.url, tenant_id)
    return {"url": body.url, "task_id": task.id, "status": "processing"}


@router.get("/task/{task_id}")
async def get_task_status(task_id: str, current_user: dict = Depends(get_current_user)):
    """查询异步入库任务进度"""
    from app.core.celery_app import celery_app
    res = celery_app.AsyncResult(task_id)
    return {
        "task_id": task_id,
        "state": res.state,
        "info": res.info if isinstance(res.info, dict) else str(res.info),
        }


@router.get("/list")
async def list_files(current_user: dict = Depends(get_current_user)):
    tenant_id = safe_tenant_id(current_user["tenant_id"])
    tenant_dir = tenant_doc_dir(tenant_id)

    files = []
    for f in os.listdir(tenant_dir):
        fpath = os.path.join(tenant_dir, f)
        if os.path.isfile(fpath):
            stat = os.stat(fpath)
            files.append({
                "filename": f,
                "size": stat.st_size,
                "upload_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            })
    return files


@router.delete("/{filename}")
async def delete_file(filename: str, current_user: dict = Depends(get_current_user)):
    tenant_id = safe_tenant_id(current_user["tenant_id"])
    file_path = os.path.join(tenant_doc_dir(tenant_id), filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="文件不存在")

    os.remove(file_path)
    # 删除文档后重建该租户索引，保证检索结果与文档一致
    task = rebuild_tenant_index_task.delay(tenant_id)
    return {
        "status": "success",
        "message": f"已删除 {filename}，正在后台重建索引",
        "rebuild_task_id": task.id,
        }
