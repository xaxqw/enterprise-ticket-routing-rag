"""
    RAG 账号注册窗口（Streamlit 多页面）
    之后就能在「聊天」「上传文档」页面使用。支持指定租户（多租户隔离）和角色。

    运行方式：保持 streamlit run dashboard/chat.py 不变，左上角导航会出现「注册账号」页面。
    """
import os
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# 后端 API 地址（Docker 内由环境变量覆盖为 http://rag-api:8000）
API_BASE = os.getenv("API_BASE", "http://localhost:8000")

st.set_page_config(
    page_title="注册账号",
    layout="centered",
)

# ===== 会话状态初始化（与其它页面共享登录态）=====
if "token" not in st.session_state:
    st.session_state.token = None
if "username" not in st.session_state:
    st.session_state.username = None


def call_register(username, password, tenant_id, role):
    """调用注册接口，成功返回后端 JSON（含 access_token），失败返回 (None, 错误信息)"""
    try:
        resp = requests.post(
            f"{API_BASE}/api/auth/register",
            json={
                "username": username,
                "password": password,
                "tenant_id": tenant_id,
                "role": role,
            },
            timeout=15,
        )
        if resp.status_code == 200:
            return resp.json(), None
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        return None, f"（{resp.status_code}）{detail}"
    except requests.exceptions.ConnectionError:
        return None, "无法连接后端服务，请确认后端已启动（端口 8000）"
    except Exception as e:
        return None, str(e)


# ===== 页面 =====
st.title(" 注册新账号")
st.caption("创建一个新账号即可使用问答、文档上传等全部功能")

# 后端状态提示
try:
    hb = requests.get(f"{API_BASE}/health", timeout=3)
    if hb.status_code == 200:
        st.success(" 后端服务在线")
    else:
        st.warning(" 后端服务异常")
except Exception:
    st.error(" 后端服务未启动，请先启动后端（端口 8000）")

# 已登录提示（仍允许注册其它账号）
if st.session_state.token:
    st.info(f"你当前已登录为 **{st.session_state.username}**。在此注册的是新账号，不影响当前登录状态。")

st.divider()

with st.form("register_form", clear_on_submit=False):
    st.subheader("填写账号信息")

    username = st.text_input("用户名 *", placeholder="4-20 位，字母/数字/下划线")
    col1, col2 = st.columns(2)
    with col1:
        password = st.text_input("密码 *", type="password", placeholder="至少 6 位")
    with col2:
        password2 = st.text_input("确认密码 *", type="password", placeholder="再输入一次")

    col3, col4 = st.columns(2)
    with col3:
        tenant_id = st.text_input(
            "租户 ID", value="default",
            help="多租户隔离：不同租户的知识库互不可见。个人使用保持 default 即可。",
        )
    with col4:
        role = st.selectbox(
            "角色", options=["user", "admin"], index=0,
            help="user=普通用户；admin=管理员（可查看用户列表）。可选 admin。",
        )

    submitted = st.form_submit_button(" 注册", type="primary", use_container_width=True)

    if submitted:
        # 前端校验
        errors = []
        if not username or len(username.strip()) < 2:
            errors.append("用户名至少 2 个字符")
        if not password or len(password) < 6:
            errors.append("密码至少 6 位")
        if password != password2:
            errors.append("两次输入的密码不一致")

        if errors:
            for e in errors:
                st.error(f" {e}")
        else:
            with st.spinner("正在创建账号..."):
                data, err = call_register(
                    username.strip(), password,
                    (tenant_id or "default").strip(), role,
                )
                if data:
                    st.success(f" 账号 `{username.strip()}` 注册成功！")
                    st.balloons()
                    # 注册接口已直接返回 token，自动登录，省去再去登录页
                    token = data.get("access_token")
                    if token:
                        st.session_state.token = token
                        st.session_state.username = username.strip()
                        st.info("已自动登录，现在可切换到「聊天」或「上传文档」页面直接使用 ")
                    else:
                        st.info("请切换到「聊天」页面用刚注册的账号登录。")
                else:
                    st.error(f" 注册失败：{err}")

st.divider()
st.caption("已有账号？切换到左侧「聊天」页面登录即可。")
