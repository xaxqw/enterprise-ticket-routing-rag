"""
认证服务：用户注册、登录、JWT令牌

设计说明：
- 用户信息持久化到 Redis（重启不丢失）
- 启动时自动创建默认账号 admin/admin123
"""

import logging
from app.core.log import get_logger
logger = get_logger(__name__)

import hashlib
import os
import json
from datetime import datetime, timedelta
from jose import jwt
from dotenv import load_dotenv

load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440 # token有效期24小时

# Redis 用户存储的 key 前缀
_USER_KEY_PREFIX = "user:"


class AuthService:
    def __init__(self, redis_client=None):
        """
 redis_client: 可选，传入已连接的 redis 客户端用于持久化用户
 不传则回退到内存模式（重启丢失）
 """
        self.redis = redis_client
        self._mem_users = {} # 内存兜底

    def _get_user(self, username):
        """读取用户信息"""
        if self.redis is not None:
            data = self.redis.get(_USER_KEY_PREFIX + username)
            if data:
                return json.loads(data)
            return None
        return self._mem_users.get(username)

    def _save_user(self, username, user_data):
        """保存用户信息"""
        if self.redis is not None:
            self.redis.set(_USER_KEY_PREFIX + username, json.dumps(user_data, ensure_ascii=False))
        else:
            self._mem_users[username] = user_data

    def hash_password(self, password):
        """使用 SHA256 哈希密码"""
        salt = "rag_salt_2024"
        return hashlib.sha256((salt + password).encode()).hexdigest()

    def verify_password(self, plain_password, hashed_password):
        return self.hash_password(plain_password) == hashed_password

    def ensure_default_user(self, username, password):
        """确保默认管理员账号存在（不存在则创建，已存在则跳过）"""
        if self._get_user(username) is None:
            self._save_user(username, {
 "username": username,
 "password_hash": self.hash_password(password),
 "tenant_id": "default",
 "role": "admin", # 默认账号是管理员
 "created_at": datetime.now().isoformat()
 })
            logger.info(f" 已创建默认管理员账号：{username}")
        else:
            logger.info(f"ℹ 默认账号已存在：{username}")

    def create_user(self, username, password, tenant_id="default", role="user"):
        if self._get_user(username) is not None:
            raise ValueError("用户名已存在")
        if role not in ("admin", "user"):
            role = "user"
        user_data = {
            "username": username,
            "password_hash": self.hash_password(password),
            "tenant_id": tenant_id,
            "role": role,
            "created_at": datetime.now().isoformat()
        }
        self._save_user(username, user_data)
        return {"username": username, "tenant_id": tenant_id, "role": role}

    def authenticate(self, username, password):
        user = self._get_user(username)
        if not user or not self.verify_password(password, user["password_hash"]):
            return None
        role = user.get("role", "user")
        access_token = self._create_access_token(
            data={"sub": username, "tenant_id": user["tenant_id"], "role": role}
        )
        return {
            "access_token": access_token, "token_type": "bearer",
            "tenant_id": user["tenant_id"], "role": role,
        }

    def _create_access_token(self, data: dict):
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        to_encode.update({"exp": expire})
        return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

    def get_current_user(self, token: str):
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username = payload.get("sub")
            tenant_id = payload.get("tenant_id", "default")
            role = payload.get("role", "user")
            if username is None:
                return None
            return {"username": username, "tenant_id": tenant_id, "role": role}
        except Exception:
            return None

    def list_users(self):
        """列出所有用户（管理员用）"""
        users = []
        if self.redis is not None:
            for key in self.redis.keys(_USER_KEY_PREFIX + "*"):
                data = self.redis.get(key)
                if data:
                    u = json.loads(data)
                    users.append({
                        "username": u.get("username"),
                        "tenant_id": u.get("tenant_id"),
                        "role": u.get("role", "user"),
                        "created_at": u.get("created_at"),
                    })
        else:
            for u in self._mem_users.values():
                users.append({
                    "username": u.get("username"),
                    "tenant_id": u.get("tenant_id"),
                    "role": u.get("role", "user"),
                    "created_at": u.get("created_at"),
                })
        return users
