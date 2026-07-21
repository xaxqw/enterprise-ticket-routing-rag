"""
4bit量化加载：用bitsandbytes把模型量化加载，省内存
"""

import logging
from app.core.log import get_logger
logger = get_logger(__name__)

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from dotenv import load_dotenv

load_dotenv()


def load_quantized_model(model_path=None, load_in_4bit=True):
    """
    加载量化后的模型
    load_in_4bit=True: 4bit量化，最省内存
    load_in_8bit=True: 8bit量化，效果稍好但占内存多一倍
    """
    if model_path is None:
        model_path = os.getenv("MODEL_PATH")

    logger.info(f"正在以4bit量化方式加载模型: {model_path}")

    # 量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_use_double_quant=True,  # 双量化，更省内存
        bnb_4bit_quant_type="nf4",  # NF4类型，效果最好
        bnb_4bit_compute_dtype=torch.bfloat16  # 计算时用bf16
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",  # 自动分配设备
        trust_remote_code=True
    )

    logger.info(" 4bit量化模型加载完成！")
    logger.info(f" 显存占用约: {model.get_memory_footprint() / 1024**3:.2f} GB")

    return model, tokenizer


if __name__ == "__main__":
    model, tokenizer = load_quantized_model()

    # 简单测试
    prompt = "你好，请介绍一下你自己。"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    outputs = model.generate(**inputs, max_new_tokens=100)
    logger.info(" ".join(str(x) for x in ["回复:", tokenizer.decode(outputs[0], skip_special_tokens=True)]))
