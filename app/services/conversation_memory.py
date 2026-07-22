"""
会话记忆：用Redis存储多轮对话历史，支持多用户隔离。

容错设计：若 Redis 不可用（如未启动 / 崩溃），自动降级为进程内内存存储，
保证问答功能不中断（仅进程重启后历史丢失，对本地 demo 可接受）。
这样即使 Redis 没运行，也不会让整个问答请求 500。
"""
import redis
import json
import time
import uuid
import logging

logger = logging.getLogger(__name__)


class ConversationMemory:
    def __init__(self, redis_client, ttl=2592000):
        """
        ttl: 过期时间，默认 2592000秒 = 30天
        说明：拉长到 30 天，用户隔几天再进入系统仍能看到自己问过的问题。
        """
        self.ttl = ttl
        # Redis 可用性探测：本地未启动会立即抛 ConnectionRefusedError，快速降级。
        try:
            redis_client.ping()
            self.redis = redis_client
            self._degraded = False
        except Exception:
            self._degraded = True
            self._mem = {}  # 内存兜底：key -> list[json_str]
            logger.warning(
                "Redis 不可用，ConversationMemory 降级为内存模式"
                "（重启服务后历史丢失，问答功能不受影响）"
            )

    def _key(self, session_id):
        return f"chat:history:{session_id}"

    def add_message(self, session_id, role, content, extra=None):
        """添加一条消息到历史（自动带唯一 id，便于后续单条删除）"""
        message = {
            "id": uuid.uuid4().hex,
            "role": role,
            "content": content,
            "timestamp": time.time(),
        }
        # 额外字段（如助手回答的参考资料溯源）一并持久化
        if extra:
            message.update(extra)
        payload = json.dumps(message, ensure_ascii=False)
        if self._degraded:
            self._mem.setdefault(self._key(session_id), []).append(payload)
            return
        self.redis.rpush(self._key(session_id), payload)
        self.redis.expire(self._key(session_id), self.ttl)  # 刷新过期时间

    def get_history(self, session_id, limit=200):
        """获取最近 N 条对话（默认 200，覆盖完整历史）"""
        if self._degraded:
            msgs = self._mem.get(self._key(session_id), [])
            if limit and limit > 0:
                msgs = msgs[-limit:]
            return [json.loads(m) for m in msgs]
        messages = self.redis.lrange(self._key(session_id), -limit, -1)
        # 兼容 decode_responses=True（str）和 False（bytes）两种情况
        return [json.loads(m if isinstance(m, str) else m.decode("utf-8")) for m in messages]

    def get_history_as_list(self, session_id, limit=10):
        """转成标准格式的消息列表（仅 role/content，给大模型当上下文）"""
        history = self.get_history(session_id, limit)
        return [{"role": h["role"], "content": h["content"]} for h in history]

    def delete_turn(self, session_id, msg_id):
        """
        删除一条历史记录。
        - 若删除的是「用户提问」，则连同其后的「助手回答」一起删，
          保持问答配对，避免出现孤立的回答。
        - 返回实际删除的条数。
        """
        key = self._key(session_id)
        if self._degraded:
            msgs = [json.loads(m) for m in self._mem.get(key, [])]
        else:
            raw = self.redis.lrange(key, 0, -1)
            msgs = [json.loads(m if isinstance(m, str) else m.decode("utf-8")) for m in raw]
        idx = next((i for i, m in enumerate(msgs) if m.get("id") == msg_id), None)
        if idx is None:
            return 0
        remove_idx = {idx}
        if (msgs[idx].get("role") == "user"
                and idx + 1 < len(msgs)
                and msgs[idx + 1].get("role") == "assistant"):
            remove_idx.add(idx + 1)
        kept = [m for i, m in enumerate(msgs) if i not in remove_idx]
        # 重写列表（低并发 demo 场景足够，保证删除精准）
        if self._degraded:
            self._mem[key] = [json.dumps(m, ensure_ascii=False) for m in kept]
        else:
            self.redis.delete(key)
            if kept:
                self.redis.rpush(key, *[json.dumps(m, ensure_ascii=False) for m in kept])
                self.redis.expire(key, self.ttl)
        return len(remove_idx)

    def clear(self, session_id):
        """清空某个会话的历史"""
        key = self._key(session_id)
        if self._degraded:
            self._mem.pop(key, None)
            return
        self.redis.delete(key)
