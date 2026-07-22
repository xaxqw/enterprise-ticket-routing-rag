"""
RAG 平台一键启动脚本

这个脚本会依次：
1. 检查 Redis 是否运行
2. 检查本地 Ollama 服务与所需模型（完全免费、离线，无需 API Key）
3. 如果向量库为空，自动建库（本地向量化）
4. 启动 FastAPI 后端（端口8000）
5. 启动 Streamlit 问答界面（端口8501）
6. 自动打开浏览器
"""
import os
import sys

# 解决 Windows 控制台中文编码问题
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import time
import subprocess
import webbrowser
import requests

# 项目根目录（本文件所在目录）
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_DIR)

# 端口配置
API_PORT = 8000
WEB_PORT = 8501
API_BASE = f"http://localhost:{API_PORT}"

# 颜色输出（Windows Terminal / cmd 都支持 ANSI）
def info(msg):
    print(f"\033[36m[INFO]\033[0m {msg}")

def ok(msg):
    print(f"\033[32m[OK]\033[0m {msg}")

def warn(msg):
    print(f"\033[33m[WARN]\033[0m {msg}")

def err(msg):
    print(f"\033[31m[ERROR]\033[0m {msg}")

def banner():
    print("=" * 50)
    print(" RAG 智能问答平台 - 一键启动")
    print(" （本地 Ollama · 完全免费 · 离线 · 调用本机 GPU）")
    print("=" * 50)
    print()


def check_redis():
    """检查 Redis 是否运行"""
    info("检查 Redis 服务...")
    try:
        import redis
        r = redis.Redis(host="localhost", port=6379, db=0)
        r.ping()
        ok("Redis 运行中")
        return True
    except ImportError:
        warn("redis-py 未安装，跳过检查")
        return True
    except Exception:
        err("Redis 未运行！请先启动 Redis：")
        print(" 方式1：打开 cmd 执行 net start Redis")
        print(" 方式2：双击 Redis 安装目录下的 redis-server.exe")
        print(" 方式3：如果用 Docker，执行 docker run -d -p 6379:6379 redis")
        return False


def check_ollama():
    """检查本地 Ollama 是否运行、所需模型是否已拉取（完全免费、离线，无需 API Key）"""
    import json as _json
    import urllib.request, urllib.error
    from dotenv import load_dotenv
    load_dotenv()

    host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    embed_model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    llm_model = os.getenv("LLM_MODEL", "deepseek-r1")

    info("检查本地 Ollama 服务...")
    try:
        req = urllib.request.Request(f"{host}/api/tags",
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            tags = _json.loads(resp.read().decode("utf-8")).get("models", [])
        names = {m.get("name") for m in tags}
        ok(f"Ollama 服务已运行（{host}）")
    except Exception as e:
        err(f"无法连接 Ollama 服务：{e}")
        print(" 请先启动 Ollama（二选一）：")
        print("  方式1：双击桌面 / 开始菜单的 Ollama 程序")
        print("  方式2：命令行执行  ollama serve")
        print(" 启动后再双击「运行RAG平台」即可。")
        return False

    # 检查所需模型是否已拉取，缺失则尝试自动拉取（仅首次需联网）
    missing = []
    for mdl in (embed_model, llm_model):
        if mdl not in names and not any(
            n.startswith(mdl + ":") or n == mdl for n in names
        ):
            missing.append(mdl)

    if missing:
        warn(f"以下模型未拉取：{', '.join(missing)}")
        print(" 首次运行需联网拉取（之后永久离线可用），正在尝试自动拉取...")
        try:
            from app.services.ollama_client import ensure_model
            for mdl in missing:
                ensure_model(mdl, timeout=1200)
            ok("模型已就绪")
        except Exception as e:
            err(f"模型自动拉取失败：{e}")
            print(f" 请手动执行：ollama pull {embed_model}  和  ollama pull {llm_model}")
            return False
    else:
        ok(f"所需模型已就绪（向量：{embed_model}，对话：{llm_model}）")
    return True


def check_vector_db():
    """检查向量库是否已建立，没有就自动建库"""
    faiss_path = "./data/vector_db/faiss_index.pkl"
    if os.path.exists(faiss_path):
        ok("向量知识库已存在")
        return True

    warn("向量知识库为空，开始自动建库...")
    print(" （会用本地 Ollama 向量化 data/raw 下的文档，可能需要几十秒）")
    print()
    build_script = os.path.join(PROJECT_DIR, "scripts", "build_vector_db.py")
    result = subprocess.call([sys.executable, build_script, "--dir", "./data/raw"])
    if result != 0:
        err("建库失败，请检查文档和 API Key")
        return False
    ok("知识库建库完成")
    return True


def start_backend():
    """启动 FastAPI 后端"""
    info(f"启动后端 API 服务（端口 {API_PORT}）...")
    # 用新控制台窗口启动，方便看日志
    creationflags = subprocess.CREATE_NEW_CONSOLE if os.name == "nt" else 0
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app",
        "--host", "0.0.0.0", "--port", str(API_PORT)],
        cwd=PROJECT_DIR,
        creationflags=creationflags
    )
    # 等待后端就绪
    info("等待后端启动...")
    for i in range(30):
        try:
            resp = requests.get(f"{API_BASE}/health", timeout=2)
            if resp.status_code == 200:
                ok(f"后端已就绪（{API_BASE}）")
                return proc
        except Exception:
            time.sleep(1)
    err("后端启动超时")
    return proc


def start_frontend():
    """启动 Streamlit 问答界面"""
    info(f"启动问答界面（端口 {WEB_PORT}）...")
    creationflags = subprocess.CREATE_NEW_CONSOLE if os.name == "nt" else 0
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "dashboard/chat.py",
        "--server.port", str(WEB_PORT),
        "--server.headless", "true"],
        cwd=PROJECT_DIR,
        creationflags=creationflags
    )
    # 等待前端就绪
    info("等待问答界面启动...")
    for i in range(20):
        try:
            resp = requests.get(f"http://localhost:{WEB_PORT}", timeout=2)
            if resp.status_code == 200:
                ok(f"问答界面已就绪（http://localhost:{WEB_PORT}）")
                return proc
        except Exception:
            time.sleep(1)
    ok("问答界面已启动（正在初始化）")
    return proc


def start_worker():
    """启动 Celery 异步任务 worker（Windows 用 --pool solo）"""
    info("启动 Celery 异步队列（端口 6379 / db1）...")
    creationflags = subprocess.CREATE_NEW_CONSOLE if os.name == "nt" else 0
    proc = subprocess.Popen(
        [sys.executable, "-m", "celery",
        "-A", "app.core.celery_app.celery_app",
        "worker", "--pool", "solo", "--loglevel", "info",
        "--without-gossip", "--without-mingle", "--without-heartbeat"],
        cwd=PROJECT_DIR,
        creationflags=creationflags
    )
    time.sleep(3)
    ok("Celery worker 已启动（处理上传/建库等异步任务）")
    return proc


def main():
    banner()

    # 1. 检查 Redis
    if not check_redis():
        print()
        input("解决 Redis 问题后按回车继续，或按 Ctrl+C 退出...")
        if not check_redis():
            return

    # 2. 检查本地 Ollama（服务 + 模型，完全免费/离线）
    if not check_ollama():
        err("Ollama 未就绪，无法启动。请按上方提示启动 Ollama 并拉取模型。")
        input("按回车键退出...")
        return

    # 3. 检查/建立向量库
    if not check_vector_db():
        print()
        input("建库遇到问题，按回车继续启动（或 Ctrl+C 退出）...")

    # 4. 启动 Celery worker（必须先于后端，上传窗口的任务才会被消费）
    worker_proc = start_worker()

    # 5. 启动后端
    backend_proc = start_backend()

    # 6. 启动前端
    frontend_proc = start_frontend()

    # 7. 打开浏览器
    print()
    ok(" 全部启动完成！正在打开浏览器...")
    web_url = f"http://localhost:{WEB_PORT}"
    webbrowser.open(web_url)

    print()
    print("=" * 50)
    print(" 服务地址：")
    print(f" 问答界面：{web_url}")
    print(f" API文档： http://localhost:{API_PORT}/docs")
    print(f" 健康检查：http://localhost:{API_PORT}/health")
    print()
    print(" 默认登录账号：xuanxu / xuanxu123")
    print()
    print(" 关闭此窗口会停止服务。")
    print(" 如需停止，直接关闭弹出的两个黑色窗口即可。")
    print("=" * 50)
    print()

    # 保持主进程不退出（子进程在独立窗口运行）
    try:
        input("按回车键停止所有服务...")
    except KeyboardInterrupt:
        pass
    finally:
        info("正在停止服务...")
        for proc in [frontend_proc, backend_proc, worker_proc]:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
        ok("服务已停止")


if __name__ == "__main__":
    main()
