"""
会话历史持久化与单条删除（真实 HTTP 接口，fakeredis 离线）
覆盖：
- 历史按「租户:用户名」隔离，登录后可拉取
- 每条消息带稳定 id，便于单条删除
- 删除「用户提问」时连同其「助手回答」一起删
- 删除「助手回答」只删该条
- 不同用户互不可见
"""
import pytest


def _login(client):
    r = client.post("/api/auth/login", data={"username": "xuanxu", "password": "xuanxu123"})
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


def _seed(redis_client, session_key, q, a):
    """直接往 Redis 写入一对问答（绕过在线检索，纯测历史链路）"""
    import json, time, uuid
    uid = uuid.uuid4().hex
    aid = uuid.uuid4().hex
    redis_client.rpush(
        f"chat:history:{session_key}",
        json.dumps({"id": uid, "role": "user", "content": q, "timestamp": time.time()}),
        json.dumps({"id": aid, "role": "assistant", "content": a,
                    "timestamp": time.time(),
                    "references": [{"text": "src", "score": 0.9, "metadata": {"source": "x.md"}}]}),
    )
    return uid, aid


def test_history_requires_auth(client):
    r = client.get("/api/rag/chat/history")
    assert r.status_code == 401


def test_history_visible_after_login(client):
    tok = _login(client)
    uid, aid = _seed(_redis(client), "default:xuanxu", "什么是混合检索？", "混合检索是...")
    h = client.get("/api/rag/chat/history", headers={"Authorization": f"Bearer {tok}"})
    assert h.status_code == 200
    msgs = h.json()["messages"]
    assert len(msgs) == 2
    assert msgs[0]["id"] == uid
    assert msgs[1]["id"] == aid
    assert msgs[1]["references"][0]["metadata"]["source"] == "x.md"


def test_delete_question_removes_answer(client):
    tok = _login(client)
    uid, aid = _seed(_redis(client), "default:xuanxu", "问题A", "回答A")
    # 删「用户提问」应连带删「回答」
    d = client.delete(f"/api/rag/chat/message/{uid}", headers={"Authorization": f"Bearer {tok}"})
    assert d.status_code == 200
    assert d.json()["removed"] == 2
    h = client.get("/api/rag/chat/history", headers={"Authorization": f"Bearer {tok}"})
    assert h.json()["messages"] == []


def test_delete_answer_only(client):
    tok = _login(client)
    uid, aid = _seed(_redis(client), "default:xuanxu", "问题B", "回答B")
    # 只删「回答」，应只删 1 条，提问保留
    d = client.delete(f"/api/rag/chat/message/{aid}", headers={"Authorization": f"Bearer {tok}"})
    assert d.json()["removed"] == 1
    h = client.get("/api/rag/chat/history", headers={"Authorization": f"Bearer {tok}"})
    assert len(h.json()["messages"]) == 1
    assert h.json()["messages"][0]["role"] == "user"


def test_history_isolated_by_user(client):
    tok = _login(client)
    # 给另一个用户塞数据，当前用户不应看到
    _seed(_redis(client), "default:other", "别人的问题", "别人的回答")
    h = client.get("/api/rag/chat/history", headers={"Authorization": f"Bearer {tok}"})
    assert h.json()["messages"] == []


def test_clear_history(client):
    tok = _login(client)
    _seed(_redis(client), "default:xuanxu", "Q", "A")
    d = client.delete("/api/rag/chat/history", headers={"Authorization": f"Bearer {tok}"})
    assert d.status_code == 200
    h = client.get("/api/rag/chat/history", headers={"Authorization": f"Bearer {tok}"})
    assert h.json()["messages"] == []


def _redis(client):
    """取 fakeredis 客户端（与后端共享同一份内存数据）"""
    from app.api import rag_router as rr
    return rr._redis_client
