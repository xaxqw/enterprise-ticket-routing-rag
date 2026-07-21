"""
    RAG 智能问答平台 —— FastAPI 后端入口
    启动：uvicorn main:app --host 0.0.0.0 --port 8000
"""
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.auth_router import router as auth_router
from app.api.rag_router import router as rag_router
from app.api.file_router import router as file_router
from app.core.logging_middleware import LoggingMiddleware

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="RAG 智能问答平台 API",
    description="基于混合检索与多智能体的文档问答后端",
    version="1.0.0",
)

# 允许跨域（前端 Streamlit / 调试页面调用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 请求日志中间件
app.add_middleware(LoggingMiddleware)

# 路由挂载
app.include_router(auth_router, prefix="/api/auth")
app.include_router(rag_router, prefix="/api/rag")
app.include_router(file_router, prefix="/api/files")


@app.get("/health")
def health():
    redis_status = "ok"
    try:
        from app.api.auth_router import _redis_client
        _redis_client.ping()
    except Exception:
        redis_status = "down"
    return {"status": "healthy", "redis": redis_status}


@app.on_event("startup")
def _create_default_admin():
    from app.api.auth_router import auth_service
    auth_service.ensure_default_user("xuanxu", "xuanxu123")
