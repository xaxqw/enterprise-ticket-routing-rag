"""
Redis 连接抽象测试。
说明：在 CI / 本地测试中，conftest 已把 redis.Redis 替换为内存版 fakeredis，
因此这里无需真实 Redis 即可验证「连接 → 写入 → 读取 → 按模式枚举 → 过期删除」的正确性。
"""
import redis


def test_redis_set_get_and_keys():
    r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
    r.flushall()
    r.set("user:alice", '{"role":"user"}')
    r.set("user:bob", '{"role":"user"}')
    assert r.get("user:alice") == '{"role":"user"}'
    # 多租户 / 多用户按前缀枚举
    assert set(r.keys("user:*")) == {"user:alice", "user:bob"}


def test_redis_setex_and_delete():
    r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
    r.flushall()
    r.setex("k", 100, "v")
    assert r.get("k") == "v"
    assert r.delete("k") == 1
    assert r.get("k") is None
