"""
AWQ量化：极致压缩，推理速度更快
"""
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

MODEL_PATH = "../models/base/qwen/Qwen2-7B-Instruct"
OUTPUT_PATH = "../models/base/qwen/Qwen2-7B-Instruct-AWQ"

# 量化配置
quant_config = {
 "w_bit": 4,
 "q_group_size": 128,
 "zero_point": True,
 "q_version": "GEMM",
}

print("加载模型...")
model = AutoAWQForCausalLM.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

print("开始量化...")
model.quantize(tokenizer, quant_config=quant_config)

print("保存量化模型...")
model.save_quantized(OUTPUT_PATH)
tokenizer.save_pretrained(OUTPUT_PATH)

print(f" AWQ量化完成，保存在: {OUTPUT_PATH}")
