"""
本地 Ollama 客户端（完全免费、离线、无需任何 API Key）

- 提供两类能力：embed（向量化，供检索/建库用）与 chat（对话生成，供问答用）
- 底层调用本机 Ollama 服务（默认 http://localhost:11434）
- 模型由 .env 指定：EMBEDDING_MODEL（如 nomic-embed-text）、LLM_MODEL（如 deepseek-r1）
- 全程不依赖外网 / 付费服务；Ollama 未启动或模型缺失时抛出清晰错误
"""

import logging
import re
from app.core.log import get_logger
logger = get_logger(__name__)

import os
import time
import json
from dotenv import load_dotenv

load_dotenv()

_OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")


def _post(path, payload, timeout=120):
    """对 Ollama 原生 API 发 POST 请求（同步，用标准库，零额外依赖）"""
    import urllib.request
    url = f"{_OLLAMA_HOST}{path}"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8", "ignore")[:300]
        except Exception:
            pass
        raise RuntimeError(f"Ollama 接口错误 {e.code}：{detail}")
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"无法连接 Ollama 服务（{_OLLAMA_HOST}）。请先启动 Ollama："
            "运行 ollama serve，或双击 Ollama 桌面程序。"
        )


def ensure_model(model_name, timeout=600):
    """确认模型已拉取；缺失则尝试自动拉取（需联网一次，之后永久离线可用）。"""
    try:
        from urllib.request import Request, urlopen
        url = f"{_OLLAMA_HOST}/api/tags"
        req = Request(url, headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=10) as resp:
            tags = json.loads(resp.read().decode("utf-8")).get("models", [])
        names = {m.get("name") for m in tags}
        if model_name in names or any(m.get("name", "").startswith(model_name + ":") for m in tags):
            return True
    except Exception:
        # 连不上就不阻塞，交给真正调用时再报错
        return False

    logger.info(f" 模型 {model_name} 未找到，尝试自动拉取（仅此一次，需联网）...")
    import subprocess
    ollama_bin = os.getenv("OLLAMA_BIN", "ollama")
    try:
        subprocess.run([ollama_bin, "pull", model_name], check=True,
                       timeout=timeout, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception as e:
        raise RuntimeError(
            f"模型 {model_name} 未安装且自动拉取失败：{e}。\n"
            f"请手动执行：ollama pull {model_name}"
        )


def _strip_think_tags(text):
    """过滤 deepseek-r1 等推理模型自带的 <think>...</think> 思考链，只保留正式回答。"""
    if not text:
        return text
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def ollama_embed(texts, normalize=True, model=None, timeout=120):
    """
    批量向量化（调用 Ollama /api/embed）。
    texts: list[str]
    返回 (np.ndarray shape=(N,dim) 已 L2 归一化, list[str] 对应的有效文本)
    """
    import numpy as np
    import faiss

    if model is None:
        model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

    valid_texts = [t for t in texts if t and t.strip()]
    if not valid_texts:
        dim = int(os.getenv("EMBEDDING_DIM", "768"))
        return np.zeros((0, dim), dtype=np.float32), []

    # 分批（Ollama 单次输入过长易超时，按批调用）
    batch_size = int(os.getenv("EMBED_BATCH_SIZE", 32))
    all_vecs = []
    for i in range(0, len(valid_texts), batch_size):
        batch = valid_texts[i:i + batch_size]
        last_err = None
        for attempt in range(3):
            try:
                resp = _post("/api/embed",
                             {"model": model, "input": batch, "keep_alive": 0},
                             timeout=timeout)
                embs = resp.get("embeddings")
                if not embs or len(embs) != len(batch):
                    raise RuntimeError("Ollama 返回向量数量与输入不一致")
                all_vecs.extend(embs)
                last_err = None
                break
            except RuntimeError as e:
                last_err = e
                if attempt < 2:
                    time.sleep(1.5)
        if last_err is not None:
            raise last_err

    arr = np.array(all_vecs, dtype=np.float32)
    if normalize and len(arr) > 0:
        faiss.normalize_L2(arr)
    return arr, valid_texts


def ollama_vision_caption(images, prompt, model=None, timeout=180, keep_alive="5m"):
    """
    多模态图像描述（调用 Ollama /api/generate，给图片生成文字描述）。
    images: list[str] 本地图片路径；prompt: 中文指令（如「请描述这张图片的内容」）
    返回描述文本。底层走本机 Ollama 视觉模型（如 llava / minicpm-v），免费/离线。
    """
    if model is None:
        model = os.getenv("VISION_MODEL", "llava")
    if not images:
        return ""
    import base64
    b64_imgs = []
    for img_path in images:
        try:
            with open(img_path, "rb") as f:
                b64_imgs.append(base64.b64encode(f.read()).decode("utf-8"))
        except Exception as e:
            logger.warning("读取图片失败 %s：%s", img_path, e)
    if not b64_imgs:
        return ""
    try:
        resp = _post("/api/generate", {
            "model": model,
            "prompt": prompt,
            "images": b64_imgs,
            "stream": False,
            "keep_alive": keep_alive,
        }, timeout=timeout)
        return (resp.get("response") or "").strip()
    except RuntimeError as e:
        # 视觉模型缺失时不自动拉取（避免 4.5GB 下载卡住建库/入库流程），
        # 由上层 image_caption 降级到 OCR 或文件名兜底。
        if "not found" in str(e).lower() or "pull" in str(e).lower():
            logger.info("视觉模型 %s 未安装，跳过自动拉取，交由 OCR/文件名兜底", model)
            return ""
        raise


def ollama_chat(messages, model=None, temperature=0.7, max_tokens=1024,
                timeout=180, keep_alive="5m"):
    """
    对话生成（调用 Ollama /api/chat，OpenAI 风格 messages）。
    返回模型回复文本；对 deepseek-r1 等推理模型会自动过滤 <think> 思考链。
    """
    if model is None:
        model = os.getenv("LLM_MODEL", "deepseek-r1")
    # 兜底：模型未就绪时尝试自动拉取
    try:
        resp = _post("/api/chat", {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
            "keep_alive": keep_alive,
        }, timeout=timeout)
        return _strip_think_tags((resp.get("message", {}).get("content") or "").strip())
    except RuntimeError as e:
        if "not found" in str(e).lower() or "pull" in str(e).lower():
            ensure_model(model)
            resp = _post("/api/chat", {
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens},
                "keep_alive": keep_alive,
            }, timeout=timeout)
            return _strip_think_tags((resp.get("message", {}).get("content") or "").strip())
        raise
