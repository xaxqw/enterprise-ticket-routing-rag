"""
自动生成SFT微调数据集：从文档块生成问答对
"""

import logging
from app.core.log import get_logger
logger = get_logger(__name__)

import json
import random


class SFTDatasetGenerator:
    def __init__(self, llm_pipeline=None):
        self.llm = llm_pipeline  # 可以用LLM生成高质量问题，也可以用规则

    def generate_qa_pairs(self, chunks, num_questions_per_chunk=2):
        """从每个文本块生成问答对"""
        dataset = []

        for chunk in chunks:
            # 简单规则生成（也可以用LLM生成质量更高的）
            questions = self._generate_questions(chunk, num_questions_per_chunk)

            for q in questions:
                dataset.append({
                    "instruction": "请根据文档内容回答问题",
                    "input": q,
                    "output": chunk
                })

        return dataset

    def _generate_questions(self, text, num):
        """简单的问题生成（基于关键词，简化版）"""
        sentences = text.split("。")
        questions = []

        for i, sent in enumerate(sentences[:num]):
            if len(sent) > 10:
                questions.append(f"关于{sent[:15]}...的内容是什么？")

        if not questions:
            questions.append("这段内容讲了什么？")

        return questions[:num]

    def split_dataset(self, dataset, train_ratio=0.8, val_ratio=0.1):
        """按 8:1:1 划分训练集/验证集/测试集"""
        random.shuffle(dataset)
        n = len(dataset)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        return {
            "train": dataset[:train_end],
            "validation": dataset[train_end:val_end],
            "test": dataset[val_end:]
        }

    def save_to_jsonl(self, dataset, output_path):
        """保存为JSONL格式（标准训练格式）"""
        with open(output_path, "w", encoding="utf-8") as f:
            for item in dataset:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info(f"已保存 {len(dataset)} 条数据到 {output_path}")
