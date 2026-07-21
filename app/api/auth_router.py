"""
    认证路由：注册、登录、获取当前用户
    """
import os
import redis as redis_lib
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from app.services.auth_service import AuthService

load_dotenv()

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

# 创建 Redis 客户端（懒连接，import 时不会报错）
_redis_client = redis_lib.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    db=int(os.getenv("REDIS_DB", 0)),
    decode_responses=True
    )
auth_service = AuthService(_redis_client)

# 兼容外部 from app.api.auth_router import auth_router 的写法
auth_router = router


class UserCreate(BaseModel):
    username: str
    password: str
    tenant_id: str = "default"
    role: str = "user"


class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    tenant_id: str
    role: str = "user"


@router.post("/register", response_model=TokenResponse)
async def register(user: UserCreate):
    try:
        auth_service.create_user(user.username, user.password, user.tenant_id, user.role)
        token = auth_service.authenticate(user.username, user.password)
        return token
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/login", response_model=TokenResponse)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    token = auth_service.authenticate(form_data.username, form_data.password)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token


async def get_current_user(token: str = Depends(oauth2_scheme)):
    user = auth_service.get_current_user(token)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token 无效或已过期",
        )
    return user


async def require_admin(current_user: dict = Depends(get_current_user)):
    """管理员权限依赖：非 admin 角色一律 403"""
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="需要管理员权限")
    return current_user


@router.get("/me")
async def me(current_user: dict = Depends(get_current_user)):
    """返回当前登录用户信息（用户名 / 租户 / 角色）"""
    return current_user


@router.get("/users")
async def list_users(admin: dict = Depends(require_admin)):
    """列出所有用户（仅管理员）"""
    return auth_service.list_users()
