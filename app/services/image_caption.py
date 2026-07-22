"""
图片描述生成：把图片转成可用于文本检索的描述文字（「以文搜图」的关键一步）

策略（按可用性与质量排序）：
1. 优先用本地 Ollama 多模态模型（llava / minicpm-v）生成中文描述 —— 语义最丰富；
2. 多模态模型不可用（未安装/拉取失败）时，用 OCR（PaddleOCR）提取图中文字；
3. 都不可用则用「来源文档 + 页码 + 文件名」作为最弱但可用的描述。

最终描述会与上下文一起作为「文本块」进入现有向量/BM25 检索空间，
因此图片与文档文字天然处于同一向量空间，问文字问题即可召回对应图片。
"""
import os
import logging

from app.core.log import get_logger
logger = get_logger(__name__)

from dotenv import load_dotenv
load_dotenv()

_VISION_PROMPT = (
    "请用简洁的中文描述这张图片的关键内容，尤其是其中的文字、图表、结构、"
    "流程图、界面或重要信息。如果图片是一页文档或截图，请概括这一页在讲什么。"
    "只输出描述本身，不要加开场白。"
)


def _ocr_text(image_path):
    """用 PaddleOCR 提取图片中的文字（懒加载重依赖，失败返回空串）"""
    try:
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(use_angle_cls=True, lang="ch")
        import numpy as np
        from PIL import Image
        img = np.array(Image.open(image_path).convert("RGB"))
        result = ocr.ocr(img, cls=True)
        lines = []
        for line in result:
            for word_info in line:
                lines.append(word_info[1][0])
        return "\n".join(lines).strip()
    except Exception as e:
        logger.info(" OCR 提取失败（%s），跳过", e)
        return ""


def generate_image_caption(image_path, source_name=None, page=None):
    """
    生成一张图片的检索描述文字。返回字符串（可能为空，调用方需判断）。
    image_path: 已落盘的图片绝对路径
    source_name: 来源文档名（如 PDF 文件名），用于上下文
    page: 页码（PDF 页面渲染图时填入）
    """
    parts = []
    used_vision = False

    # 1) 多模态模型描述（语义最丰富）
    try:
        from app.services.ollama_client import ollama_vision_caption
        cap = ollama_vision_caption([image_path], _VISION_PROMPT)
        if cap:
            parts.append(cap)
            used_vision = True
    except Exception as e:
        logger.info(" 多模态描述生成失败（%s），尝试 OCR 兜底", e)

    # 2) OCR 文字兜底（多模态未成功时）
    if not used_vision:
        ocr = _ocr_text(image_path)
        if ocr:
            parts.append(f"图片中的文字：{ocr}")

    # 3) 来源与页码上下文（始终附加，帮助定位图片出处）
    ctx = []
    if source_name:
        ctx.append(f"来源文档：{source_name}")
    if page is not None:
        ctx.append(f"第 {page} 页")
    if ctx:
        parts.append("；".join(ctx))

    # 4) 文件名兜底（前面都失败时用）
    if not parts:
        parts.append(f"图片文件：{os.path.basename(image_path)}")

    return "\n".join(parts).strip()
