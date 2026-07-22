# 多智能体混合检索 RAG 问答平台

一个面向企业知识库场景的 **RAG（检索增强生成）问答平台**，覆盖从多源数据流水线、混合检索、多智能体调度、异步任务、缓存、多租户权限，到自动化评测的完整链路。

> **默认完全本地化、免费、离线运行**：向量化用本机 Ollama 的 `nomic-embed-text`、重排用本地 RRF（Reciprocal Rank Fusion）、生成用本机 Ollama 的 `deepseek-r1`（自动调用 RTX 4050 GPU）。整套系统**不依赖任何付费 API、不需要外网、零 token 成本**。在线 SiliconFlow API 作为**可选**后端保留，按需切换。

---

## 一、核心特性

| 能力 | 说明 |
| --- | --- |
| **多源数据流水线** | 支持 PDF / Word / Excel / Markdown / TXT / 网页 URL 多种数据源，统一走「清洗 → 分块 → 质量过滤 → 文本去重 → 语义去重 → 建索引」流水线 |
| **混合检索** | BM25 关键词检索 + FAISS 向量检索（IndexFlatIP 索引 + 精确内积，与余弦相似度等价） + 本地 RRF 重排，三路融合（向量 0.6 / BM25 0.4，再重排） |
| **多智能体调度** | 意图识别 → 路由（检索 / 工具 / 闲聊）→ 幻觉检测 → 反思重写的 Agent 编排 |
| **Celery 异步队列** | 文档入库、URL 抓取、索引重建均通过 Celery + Redis 异步执行，接口秒回任务 ID |
| **Redis 缓存** | 查询结果按「租户 + 问题」维度缓存，命中直接返回，显著降低延迟 |
| **多租户权限** | JWT（sub + tenant_id + role）鉴权；每个租户独立的 FAISS / BM25 索引目录、文档目录与缓存命名空间，物理隔离 |
| **双/三 LLM 后端** | 本地 Ollama（默认，免费离线/GPU）+ 在线 SiliconFlow（可选）+ 本地 LoRA（可选私有化） |
| **Docker 部署** | docker-compose 一键编排 redis / rag-api / celery-worker / frontend 四服务，带健康检查 |
| **自动化评测** | 检索层 ablation（确定性）+ 生成层 RAGAS 风格 LLM-as-Judge 评测，输出 JSON + HTML 报告 |

---

## 二、系统架构

```
                        ┌─────────────────────────┐
                        │   Streamlit 前端 (8501)  │
                        │   登录 / 问答 / 上传 / 监控 │
                        └────────────┬────────────┘
                                     │ HTTP + JWT
                        ┌────────────▼────────────┐
                        │    FastAPI 后端 (8000)   │
                        │  auth / file / rag / agent │
                        └──┬──────────┬─────────┬──┘
                           │          │         │
              ┌────────────▼──┐  ┌────▼─────┐  ┌▼──────────────┐
              │ 多智能体编排    │  │ 混合检索  │  │ Celery 异步任务 │
              │ 意图/路由/幻觉  │  │BM25+向量  │  │ 入库/抓取/重建  │
              │ /反思          │  │ +RRF重排 │  └───┬───────────┘
              └───────┬───────┘  └────┬─────┘      │
                      │               │            │
                ┌─────▼───────────────▼────────────▼──────┐
                │        Redis (缓存 db0 / Celery db1)      │
                └──────────────────────────────────────────┘
                      │
        ┌─────────────▼──────────────┐   ┌──────────────────────────────┐
        │  租户隔离索引               │   │  本地 Ollama（免费/离线/GPU）   │
        │  data/vector_db/{tenant}/  │   │  nomic-embed-text 向量化 + deepseek-r1 生成 │
        │  faiss + bm25              │   │  （可选 SiliconFlow 在线兜底）   │
        └────────────────────────────┘   └──────────────────────────────┘
```

---

## 三、技术栈

- **后端**：FastAPI + Uvicorn
- **前端**：Streamlit + Plotly
- **检索**：FAISS 向量索引（IndexFlatIP，检索用归一化向量精确内积，与 FAISS 余弦检索数学等价）+ rank-bm25 + jieba（中文分词）+ 本地 RRF 重排
- **模型（本地，默认）**：Ollama 运行 `deepseek-r1`（LLM，推理模型，GPU 加速）+ `nomic-embed-text`（Embedding，768 维）
- **模型（在线，可选）**：硅基流动 Qwen2.5 系列（需 API Key）
- **微调**：PEFT + Transformers（Qwen2.5-0.5B，CPU LoRA）
- **异步**：Celery + Redis
- **鉴权**：python-jose（JWT HS256）+ SHA256 口令散列
- **评测**：检索层 ablation（确定性）+ LLM-as-Judge（RAGAS 风格）
- **部署**：Docker + docker-compose

---

## 四、快速开始（本地，完全免费/离线）

### 1. 准备环境 + 安装 Ollama

```bash
# 1) 安装 Ollama（本地大模型运行时，免费）：https://ollama.com 下载 Windows 版并安装
# 2) 拉取本地模型（只需一次，后续永久离线可用）
ollama pull nomic-embed-text   # 向量化（Embedding，768 维）
ollama pull deepseek-r1        # 问答生成（LLM，推理模型，自动用 GPU）

# 3) 项目依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> 不想手动拉模型也可以：启动脚本 `start.py` 会在模型缺失时自动尝试拉取。

### 2. 配置 `.env`（无需任何 API Key）

默认即本地模式，`.env` 关键项：

```dotenv
LLM_BACKEND=ollama          # ollama(默认,免费) / siliconflow(可选) / local_lora
LLM_MODEL=deepseek-r1        # Ollama 本地模型名（推理模型）
EMBEDDING_MODEL=nomic-embed-text   # Ollama 本地嵌入模型（768 维）
OLLAMA_HOST=http://localhost:11434
# 可选：仅在 LLM_BACKEND=siliconflow 时需要
# SILICONFLOW_API_KEY=
```

### 3. 启动 Redis

```bash
# 需本地已安装 Redis，或用 Docker：
docker run -d -p 6379:6379 redis:7-alpine
```

### 4. 建立知识库索引

```bash
# 把文档放入 data/raw/{租户ID}/ 后建库（默认租户 default）
python scripts/build_vector_db.py --tenant default
```

### 5. 启动服务

```bash
# 一键启动（后端 + 前端 + Celery），自动拉起 Ollama 缺失模型
python start.py
# 或分别启动
uvicorn main:app --host 0.0.0.0 --port 8000
celery -A app.core.celery_app.celery_app worker --loglevel=info --pool=solo
streamlit run dashboard/chat.py --server.port 8501
```

> **文档上传窗口**：`python start.py` 已自动拉起后端 / 前端 / Celery worker 三个服务。
> 打开 `http://localhost:8501` 后，左侧导航会出现 **「上传文档」** 页面：
> 选文件（PDF/Word/Excel/TXT/Markdown/CSV/HTML，可多选）→ 上传 → 后台自动解析/清洗/分块/向量化/建索引 →
> 进度实时可见，完成后即可在「聊天」页直接问到这份文档。删除文档会触发该租户索引自动重建。

访问：前端 http://localhost:8501 ，后端文档 http://localhost:8000/docs
默认账号：**xuanxu / xuanxu123**

---

## 五、Docker 一键部署

```bash
docker-compose up -d --build
```

启动后包含 4 个服务：

| 服务 | 端口 | 说明 |
| --- | --- | --- |
| redis | 6379 | 缓存 + Celery broker，appendonly 持久化 |
| rag-api | 8000 | FastAPI 后端，带 /health 健康检查 |
| celery-worker | - | 异步任务消费者（--pool=solo 适配 Windows/CPU）|
| frontend | 8501 | Streamlit 前端，API_BASE 指向 rag-api |

> 注：Docker 部署默认仍走本地 Ollama（需在宿主机或容器内可访问 11434）。纯离线内网可改用 `LLM_BACKEND=local_lora`。

停止：`docker-compose down`（数据卷 redis_data 保留）

---

## 六、LLM 后端切换（本地 / 在线 / 本地 LoRA）

统一生成入口按 `.env` 的 `LLM_BACKEND` 切换，**共用同一套 messages 格式**：

- `LLM_BACKEND=ollama`（**默认**）：问答/多智能体/记忆摘要/评测全部走本机 Ollama 开源模型（免费、离线、调用 GPU），不依赖任何付费 API 或外网。
- `LLM_BACKEND=siliconflow`（可选）：改走硅基流动在线 API（OpenAI 兼容），需配置 `SILICONFLOW_API_KEY`，适合无 GPU 环境。
- `LLM_BACKEND=local_lora`（可选私有化）：走本地 `Qwen2.5-0.5B + LoRA` 适配器（CPU），适合内网/离线闭环。

```bash
# 1. 训练 LoRA（见下）
# 2. 在 .env 中切换
LLM_BACKEND=local_lora
LLM_LOCAL_MODEL=./models/base/Qwen/Qwen2.5-0.5B-Instruct
LLM_LORA_PATH=./models/lora/qwen2.5-0.5b-lora-v1
# 3. 重启后端
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## 七、LoRA 微调（CPU 可跑）

```bash
pip install -r requirements-finetune.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
python scripts/download_model.py      # 下载基座 Qwen2.5-0.5B-Instruct
python scripts/run_finetune.py         # 运行 LoRA 微调（含微调前/后对比评估）
```

产物：适配器权重 `models/lora/qwen2.5-0.5b-lora-v1/`，对比 `logs/finetune_compare.json`。

---

## 八、自动化评测（全本地）

```bash
python scripts/auto_evaluation.py
```

- **检索层 ablation（确定性，无需 LLM）**：每题跑 向量-only / BM25-only / 混合(无重排) / 混合+RRF重排 四档，算 recall@1/3/5 与 MRR；并对融合权重 α ∈ [0,1] 做扫描找最优值。
- **生成层（RAGAS 风格 LLM 裁判，走本地 Ollama）**：事实召回率 `fact_recall`（确定性：gold_facts 命中比例）+ 忠实度 / 答案相关性 / 上下文精度（LLM 打分）+ 平均延迟。
- 报告产物：`logs/evaluation_report.json` 与 `logs/evaluation_report.html`（含指标卡片 + 明细表 + ablation 表）。

---

## 九、主要 API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/auth/register` | 注册（可指定 role） |
| POST | `/api/auth/login` | 登录，返回 JWT（含 tenant_id / role） |
| GET | `/api/auth/me` | 当前用户信息 |
| GET | `/api/auth/users` | 用户列表（仅管理员） |
| POST | `/api/files/upload` | 上传文档，异步入库，返回 task_id |
| POST | `/api/files/ingest_url` | 抓取网页 URL 入库 |
| GET | `/api/files/task/{task_id}` | 查询异步任务状态 |
| GET | `/api/files/list` | 当前租户文档列表 |
| DELETE | `/api/files/{filename}` | 删除文档并触发索引重建 |
| POST | `/api/rag/query` | 混合检索问答（带缓存） |
| POST | `/api/rag/agent` | 多智能体问答（意图路由 + 幻觉检测） |
| GET | `/api/rag/chat/history` | 拉取当前用户的历史问答（按「租户:用户名」隔离） |
| DELETE | `/api/rag/chat/message/{msg_id}` | 删除一条历史（删提问会连带其后的回答） |
| DELETE | `/api/rag/chat/history` | 清空当前用户全部历史问答 |
| GET | `/health` | 健康检查 |

> 交互式接口文档：启动后访问 http://localhost:8000/docs

### 会话历史：持久化 + 单条删除

问答记录默认写入 Redis（key 前缀 `chat:history:{租户}:{用户名}`），**TTL 30 天**——你隔几天再登录仍能看到自己问过的问题；不同用户按「租户:用户名」物理隔离，互不可见。每条消息带服务端生成的稳定 `id`，用于精准单条删除。

---

## 十、测试与持续集成

测试覆盖数据流水线、检索、多智能体、缓存与鉴权、API 与端到端链路。

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pytest -q   # 39 项，全部离线、无需真实 Redis / 在线 API
```

**测试分层**（详见 `tests/`）：

| 层 | 文件 | 覆盖 |
|----|------|------|
| 单元 · 数据流水线 | `test_pipeline.py` | 文本清洗 / 低质量过滤 / 去重 / 语义分块 |
| 单元 · 检索 | `test_retrieval.py` | BM25、向量+BM25 加权融合归一化、租户路径隔离 |
| 单元 · 多智能体 | `test_agents.py` | 工具 Agent、幻觉检测、意图识别与路由、反思 |
| 单元 · 缓存与鉴权 | `test_cache.py` | Redis 缓存读写、租户隔离、失效；注册/登录/JWT/角色/多租户 |
| 单元 · Redis 抽象 | `test_redis.py` | set/get/setex/keys/delete（连接抽象正确性） |
| 集成 · API（真实 HTTP） | `test_api.py` | FastAPI TestClient 打真实接口：鉴权 401/200、注册、`/me`、管理员 403/200、`/query` 缓存命中、`/agent` |
| 端到端 · 检索链路 | `test_retrieval_e2e.py` | mock 仅「Embedding / LLM」两项外部依赖，**真实**构建 FAISS+BM25 并跑通混合检索；端到端跑通 `RAGService.query` 且会话记忆落库 |
| 集成 · 会话历史 | `test_chat_history.py` | 历史按用户隔离、带稳定 id 可单条删除、删提问连带回答、不同用户互不可见、清空 |

**关键工程做法**：
- **`fakeredis` 替代真实 Redis**：`tests/conftest.py` 把全局 `redis.Redis` 替换为内存版且多服务共享同一实例，CI 无需起 Redis。
- **外部依赖隔离**：在线 LLM / Embedding 通过 `monkeypatch` 在边界打桩，只验证系统自身逻辑，测试可重复、可离线、不依赖密钥。
- **确定性环境**：conftest 固定默认账号 / `SECRET_KEY` 并清空 `SILICONFLOW_API_KEY` 与 `LLM_BACKEND`（默认回退 ollama）。

**持续集成**：`.github/workflows/ci.yml` 在 `push` / `pull_request` 时自动 `pytest -q`（GitHub Actions，Python 3.11）。全程离线、无需任何密钥，保证每次提交都不破坏核心链路。

---

## 十一、目录结构

```
RAG/
├── main.py                     # FastAPI 入口
├── app/
│   ├── api/                    # 路由：auth / file / rag
│   ├── agents/                 # 多智能体：编排/检索/工具/幻觉
│   ├── core/                   # celery_app / tenancy / tasks
│   └── services/               # 检索/嵌入/缓存/鉴权/流水线等
├── dashboard/                  # Streamlit 前端 chat/monitor
├── scripts/                    # 建库/微调/评测脚本
├── tests/                      # pytest 测试
├── data/
│   ├── raw/{tenant}/           # 各租户原始文档
│   └── vector_db/{tenant}/      # 各租户 FAISS + BM25 索引
├── requirements.txt            # 运行依赖
├── requirements-finetune.txt   # 微调依赖
├── Dockerfile / docker-compose.yml
└── README.md
```
