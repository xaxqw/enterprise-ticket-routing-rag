"""
RAG 智能问答界面（Streamlit）

启动命令：streamlit run dashboard/chat.py
"""
import os
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# 后端API地址
API_BASE = os.getenv("API_BASE", "http://localhost:8000")

# 页面配置
st.set_page_config(
    page_title="RAG 智能问答",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== 初始化会话状态 =====
if "token" not in st.session_state:
    st.session_state.token = None
if "username" not in st.session_state:
    st.session_state.username = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    # 默认占位；登录后会改成稳定的「租户:用户名」，保证再次进入能读到自己的历史
    st.session_state.session_id = "guest"
if "tenant_id" not in st.session_state:
    st.session_state.tenant_id = "default"
if "history_loaded" not in st.session_state:
    # 是否已经从后端拉过历史（避免每次 rerun 重复拉取）
    st.session_state.history_loaded = False


def render_images(images):
    """渲染「以文搜图」返回的图片列表（图片已落盘为绝对路径，Streamlit 直接读取）"""
    if not images:
        return
    st.subheader(" 相关图片")
    for im in images:
        p = im.get("path")
        if p and os.path.exists(p):
            st.image(p, caption=im.get("caption", ""), use_column_width=True)
        else:
            st.caption(f"（图片缺失：{p}）")


def render_answer(content, no_match=False):
    """渲染助手回答；未检索到相关内容时用醒目告警样式，明确告知已拒答。"""
    if no_match:
        st.warning(content)
    else:
        st.markdown(content)


def call_chat_history():
    """从后端拉取当前用户的历史问答（按用户隔离）"""
    headers = {"Authorization": f"Bearer {st.session_state.token}"}
    try:
        resp = requests.get(f"{API_BASE}/api/rag/chat/history", headers=headers, timeout=10)
        if resp.status_code == 200:
            return resp.json().get("messages", [])
    except Exception:
        return []
    return []


def load_history():
    """拉取历史并写入 session_state.messages（每条带服务端 id，用于删除）"""
    msgs = call_chat_history()
    st.session_state.messages = [
        {
            "role": m.get("role"),
            "content": m.get("content", ""),
            "id": m.get("id"),
            "references": m.get("references", []),
            "images": m.get("images", []),
            "no_match": m.get("no_match", False),
        }
        for m in msgs
    ]
    st.session_state.history_loaded = True


def call_delete_message(msg_id):
    """删除一条历史记录（删提问会连带其回答）"""
    headers = {"Authorization": f"Bearer {st.session_state.token}"}
    try:
        resp = requests.delete(f"{API_BASE}/api/rag/chat/message/{msg_id}", headers=headers, timeout=10)
        return resp.status_code == 200
    except Exception:
        return False


def call_login(username, password):
    """调用登录接口"""
    try:
        resp = requests.post(
            f"{API_BASE}/api/auth/login",
            data={"username": username, "password": password},
            timeout=10
        )
        if resp.status_code == 200:
            return resp.json()
        return None
    except requests.exceptions.ConnectionError:
        st.error(" 无法连接后端服务，请确认后端已启动（端口8000）")
        return None
    except Exception as e:
        st.error(f" 登录出错：{e}")
        return None


def call_rag_query(query, top_k=5):
    """调用RAG问答接口"""
    headers = {"Authorization": f"Bearer {st.session_state.token}"}
    try:
        resp = requests.post(
            f"{API_BASE}/api/rag/query",
            json={
                "query": query,
                "session_id": st.session_state.session_id,
                "top_k": top_k
            },
            headers=headers,
            timeout=60
        )
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 401:
            st.error(" 登录已过期，请重新登录")
            st.session_state.token = None
            st.rerun()
        else:
            st.error(f" 请求失败（{resp.status_code}）：{resp.text}")
            return None
    except Exception as e:
        st.error(f" 请求出错：{e}")
        return None


def check_backend():
    """检查后端是否在线"""
    try:
        resp = requests.get(f"{API_BASE}/health", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


# ===== 侧边栏 =====
with st.sidebar:
    st.header(" 控制面板")

    # 后端状态
    if check_backend():
        st.success(" 后端服务在线")
    else:
        st.error(" 后端服务未启动")

    st.divider()

    # 登录区
    if st.session_state.token:
        st.success(f"已登录：{st.session_state.username}")
        if st.button(" 退出登录", use_container_width=True):
            st.session_state.token = None
            st.session_state.username = None
            st.session_state.messages = []
            st.rerun()
    else:
        st.subheader(" 登录")
        username = st.text_input("用户名", value="xuanxu", key="login_user")
        password = st.text_input("密码", value="xuanxu123", type="password", key="login_pwd")
        if st.button("登录", use_container_width=True, type="primary"):
            token = call_login(username, password)
            if token:
                st.session_state.token = token["access_token"]
                st.session_state.username = token.get("username", username)
                tenant = token.get("tenant_id", "default")
                st.session_state.tenant_id = tenant
                # 稳定会话ID：租户:用户名，保证再次进入能读到自己的历史
                st.session_state.session_id = f"{tenant}:{st.session_state.username}"
                load_history()
                st.success("登录成功！")
                st.rerun()
            else:
                st.error("用户名或密码错误")

    st.divider()

    # 会话控制
    st.subheader(" 会话")
    if st.session_state.token:
        st.caption(f"历史会话：`{st.session_state.session_id}`（跨次进入可见，可单条删除）")

        top_k = st.slider("参考资料数量", min_value=1, max_value=10, value=5,
            help="每次问答检索的参考资料条数")

        if st.button(" 清空对话", use_container_width=True):
            # 同时清空后端持久化历史，保证再次进入也是干净的
            if st.session_state.token:
                try:
                    requests.delete(
                        f"{API_BASE}/api/rag/chat/history",
                        headers={"Authorization": f"Bearer {st.session_state.token}"},
                        timeout=10,
                    )
                except Exception:
                    pass
                st.session_state.messages = []
                st.session_state.history_loaded = False
                st.rerun()


# ===== 主界面 =====
st.title(" RAG 智能问答助手")
st.caption("基于你的文档知识库回答问题，回答附带参考资料溯源")

if not st.session_state.token:
    st.info(" 请在左侧登录后开始提问（默认账号 admin / admin123）")
    st.stop()

# 进入即恢复历史（首次登录已加载；整页刷新/重进时重新拉取）
if not st.session_state.history_loaded:
    load_history()

# 欢迎提示
if not st.session_state.messages:
    st.markdown("""
    你好！我是基于知识库的问答助手。

    我会从你上传的文档中检索相关内容，然后生成回答。每个回答都会附带**参考资料溯源**，你可以查看来源。

    **试试问我：**
    - 这个项目用了哪些技术？
    - RAG平台的核心架构是什么？
    - 混合检索是怎么工作的？
    """)

# 显示历史对话
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        render_answer(msg["content"], msg.get("no_match", False))
        # 显示参考资料
        if msg.get("references"):
            with st.expander(f" 参考资料（{len(msg['references'])} 条）"):
                for i, ref in enumerate(msg["references"], 1):
                    source = ref.get("metadata", {}).get("source", "未知来源")
                    score = ref.get("score", 0)
                    # 分数条
                    score_pct = max(0, min(100, int(score * 100)))
                    st.markdown(f"**参考 {i}** — 来源：`{source}`")
                    st.progress(score_pct / 100, text=f"相关度 {score_pct}%")
                    st.caption(ref.get("text", ""))
                    st.divider()
        # 以文搜图：渲染命中的图片
        render_images(msg.get("images", []))
        # 单条删除按钮（删提问会连带其回答），放在引用区外部避免重复 key
        if msg.get("id"):
            label = " 删除此问题" if msg["role"] == "user" else " 删除此回答"
            if st.button(label, key=f"del_{msg['id']}", use_container_width=True):
                if call_delete_message(msg["id"]):
                    # 重新拉取后端历史，本地与服务端保持一致
                    load_history()
                    st.rerun()

# 输入框
if query := st.chat_input("请输入你的问题..."):
    # 显示用户消息
    with st.chat_message("user"):
        st.markdown(query)

    # 调用RAG接口
    with st.chat_message("assistant"):
        with st.spinner(" 正在检索知识库并生成回答..."):
            result = call_rag_query(query, top_k=top_k)

            if result:
                answer = result.get("answer", "（无回答）")
                refs = result.get("references", [])

                render_answer(answer, result.get("no_match", False))

                if refs:
                    with st.expander(f" 参考资料（{len(refs)} 条）"):
                        for i, ref in enumerate(refs, 1):
                            source = ref.get("metadata", {}).get("source", "未知来源")
                            score = ref.get("score", 0)
                            score_pct = max(0, min(100, int(score * 100)))
                            st.markdown(f"**参考 {i}** — 来源：`{source}`")
                            st.progress(score_pct / 100, text=f"相关度 {score_pct}%")
                            st.caption(ref.get("text", ""))
                            st.divider()

                # 以文搜图：渲染命中的图片
                render_images(result.get("images", []))

                # 同步到后端历史并获取服务端消息 id（保证可删除 / 可溯源）
                if result:
                    load_history()
                    st.rerun()
