"""
知识蒸馏：大模型教小模型
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class DistillationTrainer:
    def __init__(self, teacher_model, student_model, tokenizer, temperature=2.0):
        """
 teacher: 大模型（老师）
 student: 小模型（学生）
 temperature: 温度，越高软化越明显
 """
        self.teacher = teacher_model
        self.student = student_model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def train_step(self, input_ids, attention_mask, labels):
        """单步蒸馏训练"""
        # 老师模型生成soft label（不更新梯度）
        with torch.no_grad():
            teacher_outputs = self.teacher(
 input_ids=input_ids,
 attention_mask=attention_mask
 )
            teacher_logits = teacher_outputs.logits / self.temperature

            # 学生模型前向传播
            student_outputs = self.student(
 input_ids=input_ids,
 attention_mask=attention_mask,
 labels=labels
 )
            student_logits = student_outputs.logits / self.temperature

            # 蒸馏损失 + 原始损失
            distill_loss = self.kl_loss(
 torch.log_softmax(student_logits, dim=-1),
 torch.softmax(teacher_logits, dim=-1)
 )
            hard_loss = student_outputs.loss

            total_loss = 0.7 * hard_loss + 0.3 * distill_loss * (self.temperature ** 2)
            return total_loss
