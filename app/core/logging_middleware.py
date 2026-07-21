"""
日志中间件：记录每个请求的详细信息
"""
import os
import time
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

# 确保日志目录存在（避免某些环境 / CI 全新 checkout 时 ./logs 不存在，
# 导致 FileHandler 初始化失败使整个服务起不来）
os.makedirs("./logs", exist_ok=True)

# 配置日志
logging.basicConfig(
 level=logging.INFO,
 format="%(asctime)s - %(levelname)s - %(message)s",
 handlers=[
 logging.FileHandler("./logs/app.log"),
 logging.StreamHandler()
 ]
)
logger = logging.getLogger("rag_api")


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()

        # 请求信息
        client_ip = request.client.host if request.client else "unknown"
        method = request.method
        path = request.url.path

        # 处理请求
        response = await call_next(request)

        # 耗时
        duration = round((time.time() - start_time) * 1000, 2)

        # 记录日志
        logger.info(
 f"{client_ip} - {method} {path} - "
 f"状态码: {response.status_code} - 耗时: {duration}ms"
 )

        response.headers["X-Response-Time"] = str(duration)
        return response
