"""
API 层集成测试（真实 HTTP，FastAPI TestClient）

覆盖：
- 健康检查
- 认证：未带 token → 401；错误密码 → 401；注册 / 登录 → 200；/me 鉴权；管理员接口 403/200
- RAG：/query、/agent 必须鉴权；相同问题二次请求命中 Redis 缓存（cache_hit=True）

说明：
- redis 已由 conftest 替换为 fakeredis，全程离线、无需真实 Redis。
- 外部 LLM / 在线 Embedding 通过 monkeypatch 在「路由边界」隔离，
  只验证 API 契约（鉴权、请求/响应结构、缓存与租户隔离），不验证模型效果。
"""
import pytest

from app.api import rag_router


def _fake_services(tenant_id=None):
    """打桩的 RAG 服务与多智能体编排器，避免联网调用 LLM / 在线 Embedding。"""

    class _FakeRAG:
        def query(self, session_id, user_query, top_k=5):
            return {
                "answer": f"[mock] 针对「{user_query}」的回答",
                "references": [
                    {"text": "混合检索结合了向量检索和BM25关键词检索", "score": 0.9, "metadata": {"source": "doc1"}},
                    {"text": "LoRA微调可以用很小的代价适配下游任务", "score": 0.7, "metadata": {"source": "doc2"}},
                ],
            }

    class _FakeOrch:
        def process(self, query, session_id="default", top_k=5):
            return {
                "answer": f"[mock-agent] {query}",
                "agent": "retrieval",
                "intent": "knowledge_qa",
                "sources": [{"text": "参考资料"}],
                "hallucination_check": {"keyword_coverage": 0.8},
            }

    return {"rag": _FakeRAG(), "orch": _FakeOrch(), "mtime": 1.0}


@pytest.fixture
def mock_services(monkeypatch):
    monkeypatch.setattr(rag_router, "get_services", _fake_services)


def _admin_token(client):
    r = client.post("/api/auth/login", data={"username": "xuanxu", "password": "xuanxu123"})
    return r.json()["access_token"]


# ---------------- 健康检查 ----------------
def test_health_endpoint(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "healthy"
    assert body["redis"] == "ok"


# ---------------- 认证 ----------------
def test_login_wrong_password_401(client):
    r = client.post("/api/auth/login", data={"username": "xuanxu", "password": "wrong"})
    assert r.status_code == 401


def test_register_and_me(client):
    r = client.post("/api/auth/register", json={
        "username": "alice", "password": "pwd123", "tenant_id": "teamX", "role": "user"
    })
    assert r.status_code == 200
    token = r.json()["access_token"]

    me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200
    body = me.json()
    assert body["username"] == "alice"
    assert body["tenant_id"] == "teamX"
    assert body["role"] == "user"


def test_me_requires_token(client):
    assert client.get("/api/auth/me").status_code == 401


def test_admin_only_403_for_normal_user(client):
    client.post("/api/auth/register", json={"username": "bob", "password": "pwd", "role": "user"})
    tok = client.post("/api/auth/login", data={"username": "bob", "password": "pwd"}).json()["access_token"]
    r = client.get("/api/auth/users", headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 403


def test_admin_list_users_200(client):
    tok = _admin_token(client)
    r = client.get("/api/auth/users", headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 200
    assert any(u["username"] == "xuanxu" for u in r.json())


# ---------------- RAG 问答（鉴权 + 缓存）----------------
def test_rag_query_requires_auth(client):
    r = client.post("/api/rag/query", json={"query": "混合检索是什么？"})
    assert r.status_code == 401


def test_rag_query_cache_hit(client, mock_services):
    tok = _admin_token(client)
    headers = {"Authorization": f"Bearer {tok}"}
    q = {"query": "混合检索是什么？", "top_k": 3}

    r1 = client.post("/api/rag/query", json=q, headers=headers)
    assert r1.status_code == 200
    b1 = r1.json()
    # /query 现已统一走多智能体编排器（orchestrator），答案由编排器产出
    assert b1["answer"].startswith("[mock")
    assert b1["cache_hit"] is False
    assert len(b1["references"]) >= 1

    # 相同问题二次请求应命中 Redis 缓存
    r2 = client.post("/api/rag/query", json=q, headers=headers)
    assert r2.status_code == 200
    assert r2.json()["cache_hit"] is True


def test_agent_query(client, mock_services):
    tok = _admin_token(client)
    r = client.post("/api/rag/agent", json={"query": "介绍一下混合检索"},
                    headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 200
    assert r.json()["agent"] == "retrieval"
