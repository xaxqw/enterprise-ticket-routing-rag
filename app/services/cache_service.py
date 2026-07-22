"""
缓存服务：相同问题直接返回缓存，提升响应速度。

容错设计：若 Redis 不可用，自动降级为进程内内存缓存，
保证请求不会因为缓存层抛异常而 500（仅重启后缓存失效，功能不受影响）。
"""

import logging
from app.core.log import get_logger
logger = get_logger(__name__)

import hashlib
import json
import fnmatch
import redis
import os
from dotenv import load_dotenv

load_dotenv()


class CacheService:
    def __init__(self, redis_client=None, ttl=3600):
        """
        ttl: 缓存过期时间，默认1小时（3600秒），过期后自动删除
        redis_client: Redis客户端实例，不传则自动创建
        """
        self.ttl = ttl
        if redis_client is None:
            # 从环境变量读取Redis配置，兼容Docker部署
            redis_host = os.getenv("REDIS_HOST", "localhost")
            redis_port = int(os.getenv("REDIS_PORT", 6379))
            redis_client = redis.Redis(host=redis_host, port=redis_port, db=0, decode_responses=True)
        # Redis 可用性探测：不可用则降级为内存缓存
        try:
            redis_client.ping()
            self.redis = redis_client
            self._degraded = False
        except Exception:
            self._degraded = True
            self._cache = {}
            logger.warning(
                "Redis 不可用，CacheService 降级为内存缓存"
                "（重启服务后缓存失效，问答功能不受影响）"
            )

    def _key(self, query, top_k=5, tenant_id="default"):
        """生成唯一缓存key（含租户，避免跨租户缓存串味）"""
        raw = f"{query}:{top_k}"
        digest = hashlib.md5(raw.encode()).hexdigest()
        return f"cache:rag:{tenant_id}:{digest}"

    def get(self, query, top_k=5, tenant_id="default"):
        """读取缓存：如果有缓存，返回结果；没有返回None"""
        key = self._key(query, top_k, tenant_id)
        if self._degraded:
            data = self._cache.get(key)
        else:
            data = self.redis.get(key)
        if data:
            logger.info(f" 命中缓存，key：{key}")
            return json.loads(data)
        logger.info(f" 未命中缓存，key：{key}")
        return None

    def set(self, query, result, top_k=5, tenant_id="default"):
        """写入缓存：将查询结果存入Redis（或内存），设置过期时间"""
        key = self._key(query, top_k, tenant_id)
        # ensure_ascii=False：支持中文存储
        payload = json.dumps(result, ensure_ascii=False)
        if self._degraded:
            self._cache[key] = payload
            logger.info(f" 内存缓存写入成功，key：{key}")
            return
        self.redis.setex(key, self.ttl, payload)
        logger.info(f" 缓存写入成功，key：{key}，过期时间：{self.ttl}秒")

    def invalidate(self, tenant_id=None):
        """
        清除RAG缓存（文档更新/删除后调用，避免缓存与实际数据不一致）
        tenant_id 为空则清全部；否则只清该租户
        """
        pattern = f"cache:rag:{tenant_id}:*" if tenant_id else "cache:rag:*"
        if self._degraded:
            removed = 0
            for k in list(self._cache.keys()):
                if fnmatch.fnmatch(k, pattern):
                    del self._cache[k]
                    removed += 1
            logger.info(f" 清除内存缓存成功，共删除{removed}个缓存条目")
            return removed
        keys = self.redis.keys(pattern)
        if keys:
            deleted_count = self.redis.delete(*keys)
            logger.info(f" 清除缓存成功，共删除{deleted_count}个缓存条目")
            return deleted_count
        logger.info(" 没有需要清除的缓存条目")
        return 0
