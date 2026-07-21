"""
微调效果评测：对比微调前后的回答质量
"""
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


import torch


def load_base_model(model_path):
    """加载基座模型（CPU, float32）"""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
 model_path, torch_dtype=torch.float32, trust_remote_code=True
 ).to("cpu")
    return model, tokenizer


def load_lora_model(base_path, lora_path):
    """加载基座+LoRA（CPU）"""
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
 base_path, torch_dtype=torch.float32, trust_remote_code=True
 ).to("cpu")
    model = PeftModel.from_pretrained(base_model, lora_path)
    return model, tokenizer


def ask(model, tokenizer, question):
    """问一个问题（Qwen chat template）"""
    messages = [
 {"role": "system", "content": "你是本项目的技术助手，请简洁准确地回答。"},
 {"role": "user", "content": question},
 ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to("cpu")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=128, do_sample=False,
 pad_token_id=tokenizer.eos_token_id)
        answer = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return answer.strip()


def evaluate(model, tokenizer, test_questions):
    """批量评测"""
    results = []
    for q in test_questions:
        a = ask(model, tokenizer, q)
        results.append({"question": q, "answer": a})
        return results


        if __name__ == "__main__":
            import os
            ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            BASE_MODEL = os.path.join(ROOT, "models", "base", "Qwen", "Qwen2.5-0.5B-Instruct")
            LORA_PATH = os.path.join(ROOT, "models", "lora", "qwen2.5-0.5b-lora-v1")

            # 测试问题（换成你自己领域的）
            test_questions = [
 "这个 RAG 平台用了哪种混合检索？",
 "系统怎么做多租户隔离？",
 "LoRA 微调在这个项目里怎么做的？",
 ]

            print("=== 基座模型回答 ===")
            base_model, tokenizer = load_base_model(BASE_MODEL)
            base_results = evaluate(base_model, tokenizer, test_questions)
            for r in base_results:
                print(f"\nQ: {r['question']}")
                print(f"A: {r['answer']}")

                del base_model # 释放显存

                print("\n=== LoRA微调后回答 ===")
                lora_model, tokenizer = load_lora_model(BASE_MODEL, LORA_PATH)
                lora_results = evaluate(lora_model, tokenizer, test_questions)
                for r in lora_results:
                    print(f"\nQ: {r['question']}")
                    print(f"A: {r['answer']}")

                    # 保存对比结果
                    out_path = os.path.join(ROOT, "data", "processed", "eval_comparison.json")
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump({
 "base": base_results,
 "lora": lora_results
 }, f, ensure_ascii=False, indent=2)

                        print("\n 评测完成，对比结果已保存")
