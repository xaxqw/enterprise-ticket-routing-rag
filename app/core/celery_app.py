"""
Celery配置：异步任务队列，处理耗时操作
"""
import os
from celery import Celery
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# Redis地址，从环境变量读取，方便Docker部署（用 db 1 做队列，与缓存/会话的 db 0 分开）
REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT = os.getenv("REDIS_PORT", 6379)
redis_url = f"redis://{REDIS_HOST}:{REDIS_PORT}/1"

celery_app = Celery(
 "rag_tasks",
 broker=redis_url,
 backend=redis_url,
)

# Celery配置优化
celery_app.conf.update(
 task_serializer="json", # 任务序列化方式
 result_serializer="json", # 结果序列化方式
 accept_content=["json"], # 支持的内容格式
 timezone="Asia/Shanghai", # 时区（重要，避免时间混乱）
 enable_utc=True,
 task_track_started=True, # 记录任务开始时间
 result_expires=3600, # 结果1小时后过期，释放内存
 worker_prefetch_multiplier=1, # 每次取1个任务，避免任务堆积
)

# 自动发现任务（让Celery去app.services包里找tasks.py）
celery_app.autodiscover_tasks(["app.services"])