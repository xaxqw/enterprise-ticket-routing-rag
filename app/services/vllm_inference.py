"""
vLLM 本地推理
"""

import logging
from app.core.log import get_logger
logger = get_logger(__name__)

from vllm import LLM, SamplingParams


class VLLMInference:
    def __init__(self, model_path):
        logger.info("正在用vLLM加载模型...")
        self.llm = LLM(
 model=model_path,
 trust_remote_code=True,
 gpu_memory_utilization=0.8, # 用80%显存
 )
        self.sampling_params = SamplingParams(
 temperature=0.7,
 top_p=0.9,
 max_tokens=512,
 )
        logger.info(" vLLM模型加载完成")

    def generate(self, prompt):
        """生成回答"""
        outputs = self.llm.generate([prompt], self.sampling_params)
        return outputs[0].outputs[0].text

    def generate_batch(self, prompts):
        """批量生成，vLLM擅长批量，速度优势明显"""
        outputs = self.llm.generate(prompts, self.sampling_params)
        return [o.outputs[0].text for o in outputs]
