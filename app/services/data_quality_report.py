"""
生成数据质量报告
"""

import logging
from app.core.log import get_logger
logger = get_logger(__name__)

import json
from datetime import datetime


class DataQualityReport:
    def generate(self, raw_texts, cleaned_chunks, final_chunks):
        """生成质量报告"""
        report = {
 "report_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
 "原始文档数": len(raw_texts),
 "原始总字数": sum(len(t) for t in raw_texts),
 "清洗后块数": len(cleaned_chunks),
 "清洗后总字数": sum(len(c) for c in cleaned_chunks),
 "最终块数": len(final_chunks),
 "最终总字数": sum(len(c) for c in final_chunks),
 "平均块大小": sum(len(c) for c in final_chunks) / max(len(final_chunks), 1),
 "有效率": len(final_chunks) / max(len(cleaned_chunks), 1)
 }
        return report

    def save_report(self, report, output_path):
        """保存为JSON文件"""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            logger.info(f"质量报告已保存到 {output_path}")
