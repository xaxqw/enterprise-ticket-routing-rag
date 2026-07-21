"""
    RAG 问答路由
    - 多租户：每个租户独立的向量库 + 会话，互不可见
    - Redis 缓存：相同问题命中缓存直接返回（含租户维度）
    - 多智能体：/agent 走「意图识别→路由→幻觉校验」的编排流程
    - 索引热更新：Celery worker 写盘后，API 依据索引文件 mtime 自动重载
    """
import os
import redis as redis_lib
from dotenv import load_dotenv
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Optional, List

from app.services.rag_service import RAGService
from app.services.hybrid_search import HybridSearcher
from app.services.vector_store import FAISSVectorStore
from app.services.bm25_retriever import BM25Retriever
from app.services.reranker import Reranker
from app.services.conversation_memory import ConversationMemory
from app.services.cache_service import CacheService
from app.agents.retrieval_agent import RetrievalAgent
from app.agents.tool_agent import ToolAgent
from app.agents.hallucination_agent import HallucinationAgent
from app.agents.agent_orchestrator import AgentOrchestrator
from app.core.tenancy import faiss_path, bm25_path, safe_tenant_id
from app.api.auth_router import get_current_user

load_dotenv()
router = APIRouter()

# 会话/缓存共用 db 0
_redis_client = redis_lib.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    db=int(os.getenv("REDIS_DB", 0)),
    decode_responses=True,
    )
_cache = CacheService(_redis_client, ttl=3600)
# 会话历史共用同一个 Redis 客户端（与 RAGService 内的 memory 共享存储）
_memory = ConversationMemory(_redis_client)


# 每租户服务缓存：{tenant: {"rag":, "orch":, "mtime":}}
_services = {}

def _session_id(current_user):
    """按「租户:用户名」派生稳定会话ID，保证用户再次进入能读到自己的历史"""
    return f"{current_user['tenant_id']}:{current_user['username']}"


def _index_mtime(tenant_id):
    """索引文件最后修改时间，用于判断是否需要重载"""
    m = 0.0
    for p in (faiss_path(tenant_id), bm25_path(tenant_id)):
        if os.path.exists(p):
            m = max(m, os.path.getmtime(p))
    return m


def get_services(tenant_id):
    """按租户构建（或复用）RAG 服务与多智能体编排器；索引变更则自动重载"""
    tenant_id = safe_tenant_id(tenant_id)
    mt = _index_mtime(tenant_id)
    cached = _services.get(tenant_id)
    if cached and cached["mtime"] == mt:
        return cached
    vector_store = FAISSVectorStore(index_path=faiss_path(tenant_id))
    bm25 = BM25Retriever(index_path=bm25_path(tenant_id))
    reranker = Reranker()
    searcher = HybridSearcher(vector_store, bm25, reranker)
    memory = ConversationMemory(_redis_client)
    rag = RAGService(searcher, memory)

    # 多智能体：检索/工具/幻觉 + 编排器（LLM 用 rag 做意图判断与闲聊）
    retrieval_agent = RetrievalAgent(rag)
    tool_agent = ToolAgent(rag)
    hallucination_agent = HallucinationAgent(rag)
    orchestrator = AgentOrchestrator(retrieval_agent, tool_agent, hallucination_agent, rag)

    bundle = {"rag": rag, "orch": orchestrator, "mtime": mt}
    _services[tenant_id] = bundle
    return bundle


class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = "default"
    top_k: Optional[int] = 5


class Reference(BaseModel):
    text: str
    score: float
    metadata: dict = {}


class QueryResponse(BaseModel):
    answer: str
    query_type: str
    references: List[Reference]
    hallucination_level: str
    cache_hit: bool = False


@router.post("/query", response_model=QueryResponse)
async def rag_query(request: QueryRequest, current_user=Depends(get_current_user)):
    """
    标准 RAG 问答：经多智能体编排（意图识别 → 路由 → 幻觉校验）后返回。
    带租户级 Redis 缓存；hallucination_level 由真实幻觉检测推导（不再硬编码）。
    """
    tenant_id = safe_tenant_id(current_user["tenant_id"])

    # 1. 查缓存（命中直接返回）
    cached = _cache.get(request.query, request.top_k, tenant_id)
    if cached:
        cached["cache_hit"] = True
        return QueryResponse(**cached)

    # 2. 多智能体编排：意图路由（检索/工具/闲聊）+ 幻觉校验
    orch = get_services(tenant_id)["orch"]
    res = orch.process(request.query, request.session_id, top_k=request.top_k)

    # 3. 由真实幻觉检测推导风险等级（规则法：事实覆盖率越高越可信）
    hc = res.get("hallucination_check") or {}
    fact_coverage = hc.get("fact_coverage", 1.0) if isinstance(hc, dict) else 1.0
    hallucination_level = "low" if fact_coverage >= 0.6 else ("medium" if fact_coverage >= 0.3 else "high")

    intent = res.get("intent", "knowledge_qa")
    query_type = {"knowledge_qa": "factual", "tool_call": "tool", "general_chat": "chat"}.get(intent, "factual")

    payload = {
        "answer": res.get("answer", ""),
        "query_type": query_type,
        "references": [
            {"text": r.get("text", ""), "score": r.get("score", 0.0), "metadata": r.get("metadata", {})}
            for r in res.get("sources", [])
        ],
        "hallucination_level": hallucination_level,
        "cache_hit": False,
    }
    # 4. 写缓存
    _cache.set(request.query, payload, request.top_k, tenant_id)
    return QueryResponse(**payload)


class AgentResponse(BaseModel):
    answer: str
    agent: str
    intent: Optional[str] = None
    sources: List[dict] = []
    hallucination_check: dict = {}


@router.post("/agent", response_model=AgentResponse)
async def agent_query(request: QueryRequest, current_user=Depends(get_current_user)):
    """
        多智能体编排问答：意图识别 → 路由（检索/工具/闲聊）→ 幻觉校验
        与 /query 的区别：会自动判断问题类型，计算/时间等走工具 Agent
        """
    tenant_id = safe_tenant_id(current_user["tenant_id"])
    orch = get_services(tenant_id)["orch"]
    result = orch.process(request.query, request.session_id)
    return AgentResponse(
        answer=result.get("answer", ""),
        agent=result.get("agent", "unknown"),
        intent=result.get("intent"),
        sources=result.get("sources", []) or [],
        hallucination_check=result.get("hallucination_check", {}) or {},
        )


class ChatHistoryResponse(BaseModel):
    messages: List[dict] = []


@router.get("/chat/history", response_model=ChatHistoryResponse)
async def chat_history(current_user=Depends(get_current_user)):
    """
        拉取当前用户的历史问答（按 tenant:username 隔离）。
        用户再次进入系统时，前端调用此接口即可恢复自己问过的问题。
        """
    messages = _memory.get_history(_session_id(current_user), limit=200)
    return ChatHistoryResponse(messages=messages)


@router.delete("/chat/message/{msg_id}")
async def delete_chat_message(msg_id: str, current_user=Depends(get_current_user)):
    """
        删除一条历史记录。
        - 删除「用户提问」时，连同其后的「助手回答」一起删（保持问答配对）。
        - 删除「助手回答」时，只删该条。
        """
    removed = _memory.delete_turn(_session_id(current_user), msg_id)
    return {"ok": True, "removed": removed}


@router.delete("/chat/history")
async def clear_chat_history(current_user=Depends(get_current_user)):
    """清空当前用户的全部历史问答"""
    _memory.clear(_session_id(current_user))
    return {"ok": True}
