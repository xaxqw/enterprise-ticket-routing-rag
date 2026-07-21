# ============================================================
# RAG 平台运行镜像（后端 API / Celery Worker / 前端 共用）
# 基础镜像：官方 Python 3.10 slim，体积小
# ============================================================
FROM python:3.10-slim

# 环境变量：不写 .pyc、日志实时刷出、pip 不缓存
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 系统依赖：faiss / numpy 等偶尔需要编译工具；curl 用于健康检查
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 先装依赖，充分利用 Docker 层缓存（改代码不必重装依赖）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# 复制项目代码
COPY . .

# 后端 API 端口 + Streamlit 前端端口
EXPOSE 8000 8501

# 默认启动后端（compose 中 celery-worker / frontend 会覆盖 command）
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
