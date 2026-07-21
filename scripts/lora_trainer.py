"""
LoRA微调配置：用PEFT库配置LoRA参数
"""
from peft import LoraConfig, get_peft_model, TaskType


def get_lora_config():
    """获取LoRA配置"""
    config = LoraConfig(
 task_type=TaskType.CAUSAL_LM, # 因果语言模型（生成式）
 r=8, # LoRA秩，越大效果越好但参数越多，8是常用值
 lora_alpha=32, # 缩放系数，一般是r的4倍
 lora_dropout=0.05, # dropout防止过拟合
 target_modules=[ # 要训练哪些层，Qwen一般是q_proj和v_proj
 "q_proj",
 "v_proj",
 "k_proj",
 "o_proj",
 ],
 bias="none",
 )
    return config


def wrap_with_lora(model):
    """给模型套上LoRA"""
    lora_config = get_lora_config()
    lora_model = get_peft_model(model, lora_config)

    # 打印可训练参数量
    trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in lora_model.parameters())
    print(f"总参数量: {total_params / 1e6:.2f} M")
    print(f"可训练参数量: {trainable_params / 1e6:.2f} M")
    print(f"可训练占比: {100 * trainable_params / total_params:.2f}%")

    return lora_model
