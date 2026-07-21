"""
多格式文档解析器：支持 PDF、Word、Excel、TXT、Markdown、扫描件PDF（OCR）

- 常用格式（PDF/Word/Excel/TXT/MD）依赖轻量库，随运行镜像安装
- 扫描件 OCR（paddleocr + pdf2image/poppler）体积大且需系统依赖，改为"用到再导入"，
  运行镜像不装也能正常跑普通文档，避免拖垮 Docker 镜像体积
"""

import logging
from app.core.log import get_logger
logger = get_logger(__name__)

import os


class FileParser:
    def __init__(self):
        self.ocr = None  # OCR懒加载，用到再初始化，省内存

    def parse_pdf(self, file_path):
        """解析普通PDF（可以复制文字的那种）"""
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += (page.extract_text() or "") + "\n"
        return text.strip()

    def parse_scanned_pdf(self, file_path):
        """解析扫描件PDF（图片型，用OCR识别文字）"""
        # 懒加载OCR模型，第一次用才加载（paddleocr + poppler 属重依赖，按需安装）
        from pdf2image import convert_from_path
        if self.ocr is None:
            from paddleocr import PaddleOCR
            logger.info("正在加载OCR模型...")
            self.ocr = PaddleOCR(use_angle_cls=True, lang="ch")

        # PDF每页转成图片
        images = convert_from_path(file_path)
        full_text = ""

        import numpy as np
        for i, img in enumerate(images):
            logger.info(f"OCR识别第 {i+1}/{len(images)} 页...")
            result = self.ocr.ocr(np.array(img), cls=True)
            for line in result:
                for word_info in line:
                    full_text += word_info[1][0] + "\n"

        return full_text.strip()

    def parse_txt(self, file_path):
        """解析TXT / Markdown 纯文本文件"""
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()

    def parse_word(self, file_path):
        """解析Word文档（.docx）"""
        from docx import Document
        doc = Document(file_path)
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text.strip()

    def parse_excel(self, file_path):
        """解析Excel表格"""
        import openpyxl
        wb = openpyxl.load_workbook(file_path)
        text = ""
        for sheet_name in wb.sheetnames:
            sheet = wb[sheet_name]
            text += f"=== 工作表: {sheet_name} ===\n"
            for row in sheet.iter_rows(values_only=True):
                row_text = " | ".join([str(cell) if cell else "" for cell in row])
                text += row_text + "\n"
        return text.strip()

    def auto_parse(self, file_path):
        """
        自动识别文件格式并解析
        这是对外的主接口，调用这个就行
        """
        ext = os.path.splitext(file_path)[1].lower()  # 取文件后缀

        if ext == ".pdf":
            # 先尝试普通解析，如果提取的文字太少，说明是扫描件，用OCR
            text = self.parse_pdf(file_path)
            if len(text) < 100:  # 文字太少，大概率是扫描件
                logger.info("检测到扫描件PDF，启用OCR识别...")
                text = self.parse_scanned_pdf(file_path)
            return text
        elif ext in [".txt", ".md", ".markdown"]:
            return self.parse_txt(file_path)
        elif ext in [".docx", ".doc"]:
            return self.parse_word(file_path)
        elif ext in [".xlsx", ".xls"]:
            return self.parse_excel(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {ext}")

    def parse_url(self, url, timeout=15):
        """
        网页数据源：抓取 URL 正文并清洗为纯文本
        多源数据流水线的一环——支持把在线文档/文章直接纳入知识库
        """
        import requests
        from bs4 import BeautifulSoup

        headers = {"User-Agent": "Mozilla/5.0 (compatible; RAG-Pipeline/1.0)"}
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding or "utf-8"

        soup = BeautifulSoup(resp.text, "lxml")
        # 去掉脚本/样式/导航等噪声
        for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "form"]):
            tag.decompose()

        # 优先取 <article> / <main>，否则取 body
        main = soup.find("article") or soup.find("main") or soup.body or soup
        text = main.get_text(separator="\n")
        # 压缩多余空行
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        return "\n".join(lines)
