"""
共享测试夹具（CI / 本地通用）

核心做法：
1. 用 fakeredis 把全局 `redis.Redis` 替换成内存版，并让所有模块级 Redis 客户端
   共用同一份数据（模拟一个真实 Redis 实例）。这样测试全程不需要真实 Redis，
   CI 里也能稳定跑通鉴权、缓存、多租户隔离等依赖 Redis 的逻辑。
2. 设定确定性环境变量：默认账号、SECRET_KEY，并将 LLM_BACKEND 固定为本地 Ollama，
   再用全局桩把 Ollama 的 Embedding / 生成接口替换为「确定性离线桩」，
   确保测试不触发任何联网的大模型 / Embedding 调用，可重复、可离线。
"""
import os

os.environ.setdefault("DEFAULT_USERNAME", "xuanxu")
os.environ.setdefault("DEFAULT_PASSWORD", "xuanxu123")
os.environ.setdefault("SECRET_KEY", "test-secret-key-not-for-production")
os.environ.setdefault("REDIS_HOST", "localhost")
os.environ.setdefault("LLM_BACKEND", "ollama")  # 固定本地后端，配合下方离线桩
os.environ["SILICONFLOW_API_KEY"] = ""  # 测试中禁止联网调用大模型

import fakeredis
import redis as _redis_lib

# 全局共享一个内存 Redis 实例：所有模块级 redis.Redis(...) 拿到同一个对象，
# 既模拟「一个真实 Redis 被多服务共享」，又让 decode_responses 正常生效（返回 str）。
_SHARED_REDIS = fakeredis.FakeStrictRedis(decode_responses=True)


def _fake_redis(*args, **kwargs):
    return _SHARED_REDIS


_redis_lib.Redis = _fake_redis

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def fake_redis():
    """独立的 fakeredis 实例，用于需要 Redis 的单元 / 集成测试"""
    return fakeredis.FakeStrictRedis(decode_responses=True)


@pytest.fixture(autouse=True)
def stub_ollama_offline(monkeypatch):
    """
    全局把 Ollama 的 Embedding / 生成接口替换为离线桩，保证测试不联网。

    - Embedding：模拟「不可用」，触发下游语义分块的窗口兜底（与无模型环境一致），
      与离线/CI 行为对齐；需要真实向量的测试（test_retrieval_e2e）自行打桩 embed_texts。
    - 生成：返回确定性模拟回答，任何 stray 生成都不会联网、不依赖 GPU。
    """
    def _embed_unavailable(texts, normalize=True, model=None, timeout=120):
        raise RuntimeError("offline test: Ollama embedding 不可用（模拟无模型环境）")

    def _chat(messages, model=None, temperature=0.7, max_tokens=1024,
              timeout=180, keep_alive="5m"):
        return "（测试桩）这是基于检索结果的模拟回答。"

    # 最底层：直接桩 Ollama 客户端，覆盖所有间接调用路径
    monkeypatch.setattr("app.services.ollama_client.ollama_embed", _embed_unavailable)
    monkeypatch.setattr("app.services.ollama_client.ollama_chat", _chat)
    # 上一层：桩统一 Embedding 入口与 RAG 生成，双重保险
    monkeypatch.setattr("app.services.embeddings.embed_texts", _embed_unavailable)
    monkeypatch.setattr(
        "app.services.rag_service.RAGService._llm_chat",
        lambda self, messages, temperature=0.7, max_tokens=1024: _chat(messages),
    )


@pytest.fixture
def client():
    """
    FastAPI 测试客户端（真实 HTTP）。
    每次测试前清空共享 Redis，并触发 startup 重建默认管理员账号，
    保证测试之间互不污染、默认账号一定存在。
    """
    _SHARED_REDIS.flushall()
    from main import app
    with TestClient(app) as c:
        yield c
