"""
缓存与认证单元测试（离线，使用内存版 Fake Redis）
覆盖：CacheService 读写/租户隔离/失效；AuthService 注册登录/角色/多租户
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.cache_service import CacheService
from app.services.auth_service import AuthService


class FakeRedis:
    """最小可用的内存版 Redis，满足 CacheService / AuthService 所需接口"""
    def __init__(self):
        self.store = {}

    def get(self, k):
        return self.store.get(k)

    def set(self, k, v):
        self.store[k] = v

    def setex(self, k, ttl, v):
        self.store[k] = v

    def keys(self, pattern="*"):
        import fnmatch
        return [k for k in self.store if fnmatch.fnmatch(k, pattern)]

    def delete(self, *keys):
        n = 0
        for k in keys:
            if k in self.store:
                del self.store[k]
                n += 1
        return n


def test_cache_set_get_and_tenant_isolation():
    cache = CacheService(FakeRedis())
    cache.set("同一个问题", {"answer": "A租户答案"}, top_k=5, tenant_id="A")
    # 同问题不同租户，互不命中
    assert cache.get("同一个问题", 5, "A")["answer"] == "A租户答案"
    assert cache.get("同一个问题", 5, "B") is None


def test_cache_invalidate_by_tenant():
    r = FakeRedis()
    cache = CacheService(r)
    cache.set("q1", {"a": 1}, tenant_id="A")
    cache.set("q2", {"a": 2}, tenant_id="A")
    cache.set("q3", {"a": 3}, tenant_id="B")
    deleted = cache.invalidate(tenant_id="A")
    assert deleted == 2
    assert cache.get("q3", 5, "B") is not None


def test_auth_register_login_role_and_tenant():
    auth = AuthService(FakeRedis())
    auth.ensure_default_user("admin", "admin123")
    # 默认账号是管理员
    tok = auth.authenticate("admin", "admin123")
    assert tok is not None and tok["role"] == "admin"

    # 新用户默认普通角色 + 指定租户
    auth.create_user("alice", "pwd", tenant_id="teamX", role="user")
    tok2 = auth.authenticate("alice", "pwd")
    assert tok2["tenant_id"] == "teamX"
    assert tok2["role"] == "user"

    # 密码错误
    assert auth.authenticate("alice", "wrong") is None

    # JWT 解析回租户与角色
    decoded = auth.get_current_user(tok2["access_token"])
    assert decoded["username"] == "alice"
    assert decoded["tenant_id"] == "teamX"
    assert decoded["role"] == "user"


def test_auth_duplicate_user_rejected():
    auth = AuthService(FakeRedis())
    auth.create_user("bob", "p")
    try:
        auth.create_user("bob", "p2")
        assert False, "应抛出重复用户异常"
    except ValueError:
        pass
