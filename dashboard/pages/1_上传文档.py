"""
    RAG 文档上传窗口（Streamlit 多页面）
    分块/向量化/建索引 -> 立刻就能在聊天页问到这份文档里的内容。支持 PDF / Word / Excel /
    TXT / Markdown / CSV / HTML。

    运行方式：保持 streamlit run dashboard/chat.py 不变，左上角导航会出现「上传文档」页面。
    （本页面与聊天页共享登录状态）
    """
import os
import time
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# 后端 API 地址（Docker 内由环境变量覆盖为 http://rag-api:8000）
API_BASE = os.getenv("API_BASE", "http://localhost:8000")

# 页面配置（多页面模式下每个页面独立设置，必须放在最前面）
st.set_page_config(
    page_title="文档上传",
    layout="wide",
)

# 支持的文件类型（不含点号）
SUPPORTED_TYPES = [
    "pdf", "docx", "doc", "txt", "md", "markdown",
    "xlsx", "xls", "csv", "html", "htm",
]

# ===== 会话状态初始化 =====
if "token" not in st.session_state:
    st.session_state.token = None
if "username" not in st.session_state:
    st.session_state.username = None
if "upload_tasks" not in st.session_state:
    st.session_state.upload_tasks = []


# ===== 后端调用 =====
def call_login(username, password):
    """登录获取 JWT"""
    try:
        resp = requests.post(
            f"{API_BASE}/api/auth/login",
            data={"username": username, "password": password},
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.json().get("access_token")
        return None
    except requests.exceptions.ConnectionError:
        st.error(" 无法连接后端服务，请确认后端已启动（端口 8000）")
        return None
    except Exception as e:
        st.error(f" 登录出错：{e}")
        return None


def _headers():
    return {"Authorization": f"Bearer {st.session_state.token}"}


def call_upload(uploaded_file):
    """上传单个文件，返回后端 JSON 或 None"""
    try:
        data = uploaded_file.getvalue()
        files = {
            "file": (
                uploaded_file.name,
                data,
                uploaded_file.type or "application/octet-stream",
            )
        }
        resp = requests.post(
            f"{API_BASE}/api/files/upload",
            files=files,
            headers=_headers(),
            timeout=60,
        )
        if resp.status_code == 200:
            return resp.json()
        st.error(f" 上传失败（{resp.status_code}）：{resp.text}")
        return None
    except Exception as e:
        st.error(f" 上传出错：{e}")
        return None


def call_task_status(task_id):
    try:
        resp = requests.get(
            f"{API_BASE}/api/files/task/{task_id}",
            headers=_headers(),
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.json()
        return {"state": "UNKNOWN", "info": resp.text}
    except Exception as e:
        return {"state": "UNKNOWN", "info": str(e)}


def call_list_files():
    try:
        resp = requests.get(
            f"{API_BASE}/api/files/list",
            headers=_headers(),
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.json()
        st.error(f" 获取文档列表失败（{resp.status_code}）：{resp.text}")
        return []
    except Exception as e:
        st.error(f" 获取文档列表出错：{e}")
        return []


def call_delete(filename):
    import urllib.parse
    try:
        encoded = urllib.parse.quote(filename, safe="")
        resp = requests.delete(
            f"{API_BASE}/api/files/{encoded}",
            headers=_headers(),
            timeout=10,
        )
        return resp.status_code == 200, resp.json() if resp.status_code == 200 else resp.text
    except Exception as e:
        return False, str(e)


# ===== 登录拦截 =====
if not st.session_state.token:
    st.title(" 请先登录")
    st.info("上传文档需要登录（当前租户的文档只会出现在当前租户知识库里）")
    with st.form("login_form"):
        username = st.text_input("用户名", value="xuanxu")
        password = st.text_input("密码", value="xuanxu123", type="password")
        submitted = st.form_submit_button("登录", type="primary")
        if submitted:
            token = call_login(username, password)
            if token:
                st.session_state.token = token
                st.session_state.username = username
                st.rerun()
            else:
                st.error("用户名或密码错误")
                st.stop()

# ===== 主界面 =====
st.title(" 文档上传窗口")
st.caption("选择文件 → 上传 → 后台自动解析/清洗/分块/向量化/建索引，完成后即可在聊天页提问")

# 顶部状态条
col_status, col_user = st.columns([4, 1])
with col_status:
    try:
        hb = requests.get(f"{API_BASE}/health", timeout=3)
        if hb.status_code == 200:
            st.success(" 后端服务在线")
        else:
            st.error(" 后端服务异常")
    except Exception:
        st.error(" 后端服务未启动，请先启动后端（端口 8000）")
with col_user:
    st.write(f" {st.session_state.username}")
    if st.button(" 退出", use_container_width=True):
        st.session_state.token = None
        st.session_state.username = None
        st.session_state.upload_tasks = []
        st.rerun()

st.divider()

# ===== 1) 上传区 =====
st.subheader("① 选择并上传文档")
uploaded_files = st.file_uploader(
    "支持 PDF / Word / Excel / TXT / Markdown / CSV / HTML（可多选）",
    accept_multiple_files=True,
    type=SUPPORTED_TYPES,
)

if uploaded_files:
    st.caption(f"已选择 {len(uploaded_files)} 个文件：")
    for f in uploaded_files:
        st.write(f"• `{f.name}` （{len(f.getvalue())} 字节，类型：{f.type or '未知'}）")

    if st.button(" 上传并入知识库", type="primary", use_container_width=True):
        with st.spinner("正在上传文件到后端..."):
            ok_count = 0
            for f in uploaded_files:
                result = call_upload(f)
                if result and result.get("task_id"):
                    st.session_state.upload_tasks.append({
                        "task_id": result["task_id"],
                        "filename": result.get("filename", f.name),
                        "submitted_at": time.strftime("%H:%M:%S"),
                    })
                    ok_count += 1
            if ok_count:
                st.success(
                    f" 已提交 {ok_count} 个文件进入后台处理"
                    f"（task_id 见下方进度，完成后即可在聊天页提问）"
                )
                st.rerun()
            else:
                st.warning(" 没有文件上传成功，请检查文件类型或后端状态")

st.divider()

# ===== 2) 入库进度（任务状态轮询） =====
st.subheader("② 后台入库进度")
auto_refresh = st.checkbox("自动刷新（每 3 秒）", value=False)

STATE_LABEL = {
    "PENDING": ("⏳ 排队中", "blue"),
    "RECEIVED": (" 已接收", "blue"),
    "STARTED": (" 处理中", "blue"),
    "PROGRESS": (" 处理中", "blue"),
    "SUCCESS": (" 完成", "green"),
    "FAILURE": (" 失败", "red"),
    "RETRY": (" 重试中", "orange"),
    "UNKNOWN": (" 未知", "gray"),
}

if not st.session_state.upload_tasks:
    st.info("还没有提交过上传任务。在上方选择文件并上传后，这里会显示处理进度。")
else:
    for task in reversed(st.session_state.upload_tasks):
        status = call_task_status(task["task_id"])
        state = status.get("state", "UNKNOWN")
        label, color = STATE_LABEL.get(state, (" 未知", "gray"))
        info = status.get("info", "")
        if isinstance(info, dict):
            info_text = info.get("message") or info.get("stage") or ""
        else:
            info_text = str(info)

        with st.container(border=True):
            c1, c2 = st.columns([3, 1])
            with c1:
                st.markdown(f"**`{task['filename']}`** — {label}")
                if info_text:
                    st.caption(f"详情：{info_text}")
                    st.caption(f"提交时间 {task['submitted_at']} · task_id：{task['task_id']}")
            with c2:
                if st.button(" 删除", key=f"del_task_{task['task_id']}", use_container_width=True):
                    st.session_state.upload_tasks = [
                        t for t in st.session_state.upload_tasks
                        if t["task_id"] != task["task_id"]
                    ]
                    st.rerun()
                if st.button(" 刷新状态", key=f"refresh_{task['task_id']}", use_container_width=True):
                    st.rerun()

st.divider()

# ===== 3) 当前租户文档库 =====
st.subheader("③ 当前租户文档库")
if st.button(" 刷新列表", use_container_width=True):
    st.rerun()

files = call_list_files()
if not files:
    st.info("当前租户还没有任何文档。上传文件后会出现在这里；删除文件会触发索引重建。")
else:
    for f in files:
        c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
        with c1:
            st.write(f" `{f['filename']}`")
        with c2:
            size_kb = f.get("size", 0) / 1024
            st.caption(f"{size_kb:.1f} KB")
        with c3:
            st.caption(f.get("upload_time", "")[:19].replace("T", " "))
        with c4:
            if st.button(" 删除", key=f"del_{f['filename']}"):
                ok, msg = call_delete(f["filename"])
                if ok:
                    st.success(f"已删除 `{f['filename']}`，正在后台重建索引")
                else:
                    st.error(f"删除失败：{msg}")
                st.rerun()

# ===== 自动刷新（放在脚本末尾，避免遮挡上方③区文档库） =====
if auto_refresh:
    time.sleep(3)
    st.rerun()
