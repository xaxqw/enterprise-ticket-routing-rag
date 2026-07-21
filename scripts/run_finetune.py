"""
CPU 可跑的 LoRA 微调完整流水线（Qwen2.5-0.5B-Instruct）

流程：
 1) 准备 SFT 数据集（无则自动生成一份关于本 RAG 项目的样例问答）
 2) 加载 Qwen2.5-0.5B（float32, CPU）
 3) 套 LoRA（peft），只训练极少参数
 4) Trainer 训练（CPU，小步数）
 5) 保存 LoRA 适配器
 6) 微调前 vs 微调后 同题对比，结果存 JSON

运行：
 pip install -r requirements-finetune.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
 python scripts/download_model.py # 先下载基座
 python scripts/run_finetune.py

注意：0.5B 模型 + CPU，训练几十条数据约几分钟；这是为了在无 GPU 环境“跑通全流程”，
 真正提升效果需要更多数据/更大模型/GPU。
"""
import os
import sys
import json

# Windows 上 os.putenv 单个环境变量上限 32767 字符；若环境中存在超长变量
# （如某些桌面 / 加速类产品注入的 ACC_PRODUCT_CONFIG_V3），accelerate / datasets
# 在保存或恢复环境时会触发 ValueError: the environment variable is longer than
# 32767 characters。这里在训练前清理掉超长变量，避免崩溃（不影响训练本身）。
for _k, _v in list(os.environ.items()):
    if len(_v) > 32767:
        del os.environ[_k]

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ---------------- 配置 ----------------
BASE_MODEL = os.path.join(PROJECT_ROOT, "models", "base", "Qwen", "Qwen2.5-0.5B-Instruct")
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "sft", "train.jsonl")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models", "lora", "qwen2.5-0.5b-lora-v1")
EVAL_OUT = os.path.join(PROJECT_ROOT, "logs", "finetune_compare.json")

NUM_EPOCHS = 6
BATCH_SIZE = 1
GRAD_ACCUM = 4
LR = 2e-4
MAX_LEN = 512


SAMPLE_DATA = [
 {"instruction": "你是本项目的技术助手，请简洁准确地回答。",
  "input": "这个 RAG 平台用了哪种混合检索？",
  "output": "采用 BM25 关键词检索 + FAISS 向量检索 + 本地 RRF（倒数排名融合）重排三路融合的混合检索。向量与 BM25 加权融合（默认向量 0.6、BM25 0.4）后再用 RRF 重排，召回更全、排序更准，且全程免费离线。"},
 {"instruction": "你是本项目的技术助手，请简洁准确地回答。",
  "input": "文档上传后是怎么入库的？",
  "output": "上传后由 Celery 异步流水线处理：解析→清洗→语义分块→向量去重→本地 Ollama bge-m3 向量化→写入 FAISS 索引与 BM25 索引。"},
 {"instruction": "你是本项目的技术助手，请简洁准确地回答。",
  "input": "系统怎么做多租户隔离？",
  "output": "每个租户有独立的向量库目录与文档目录，JWT 里带 tenant_id 与 role，查询与缓存都按租户维度隔离。"},
 {"instruction": "你是本项目的技术助手，请简洁准确地回答。",
  "input": "为什么要接 Redis 缓存？",
  "output": "相同问题命中缓存可跳过检索与大模型生成，显著降低延迟和算力消耗；文档更新时按租户失效缓存保证一致性。"},
 {"instruction": "你是本项目的技术助手，请简洁准确地回答。",
  "input": "多智能体是怎么调度的？",
  "output": "编排器先做意图识别，再路由到检索 Agent / 工具 Agent（计算器）/ 闲聊，知识问答结果还会经幻觉校验 Agent 复核，必要时触发反思重检索。"},
 {"instruction": "你是本项目的技术助手，请简洁准确地回答。",
  "input": "大模型和向量用的是本地还是在线？",
  "output": "全部本地、免费、可离线：用 Ollama 在本机 GPU 跑 qwen2.5:7b 做生成、bge-m3 做向量化，无需任何 API Key，断网也能演示。"},
 {"instruction": "你是本项目的技术助手，请简洁准确地回答。",
  "input": "怎么部署这个系统？",
  "output": "用 docker compose 一键编排 redis、后端 API、Celery worker、Streamlit 前端四个服务，本地启动即可。"},
 {"instruction": "你是本项目的技术助手，请简洁准确地回答。",
  "input": "LoRA 微调在这个项目里怎么做的？",
  "output": "基于 Qwen2.5-0.5B 用 PEFT 做 LoRA 微调，CPU 即可跑通，只训练少量低秩参数并保存适配器，支持微调前后效果对比。"},
 {"instruction": "你是本项目的技术助手，请简洁准确地回答。",
  "input": "怎么防止大模型胡说（幻觉）？",
  "output": "两层防护：一是生成时数字落地校验，把答案里的年份/百分比/人数纠回参考资料原文；二是幻觉检测 Agent 判定高风险时触发反思重检索，用更准的上下文重答。"},
 {"instruction": "你是本项目的技术助手，请简洁准确地回答。",
  "input": "为什么不用在线大模型 API？",
  "output": "主打零成本、可离线、可复现的演示效果，普通笔记本也能跑；同时保留 siliconflow 在线后端作为可选升级路径。"},
]


TEST_QUESTIONS = [
 "这个 RAG 平台用了哪种混合检索？",
 "系统怎么做多租户隔离？",
 "大模型和向量用的是本地还是在线？",
 "怎么防止大模型胡说（幻觉）？",
]


def ensure_dataset():
    if os.path.exists(DATA_PATH):
        print(f" 使用已有数据集：{DATA_PATH}")
        return
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        for item in SAMPLE_DATA:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f" 已生成样例 SFT 数据集（{len(SAMPLE_DATA)} 条）：{DATA_PATH}")


def build_example(tokenizer, item):
    """用 Qwen chat template 构造样本，并把 prompt 部分的 label 屏蔽为 -100"""
    system = item.get("instruction") or "你是一个有帮助的助手。"
    user = item.get("input", "")
    answer = item.get("output", "")

    prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True,
    )
    full = prompt + answer + tokenizer.eos_token

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full, add_special_tokens=False, truncation=True, max_length=MAX_LEN)["input_ids"]

    labels = list(full_ids)
    for i in range(min(len(prompt_ids), len(labels))):
        labels[i] = -100  # 只对答案计算 loss
    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}


def ask(model, tokenizer, question, device):
    messages = [
        {"role": "system", "content": "你是本项目的技术助手，请简洁准确地回答。"},
        {"role": "user", "content": question},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    import torch
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=128, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
        gen = out[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(gen, skip_special_tokens=True).strip()


def main():
    if not os.path.isdir(BASE_MODEL):
        print(f" 未找到基座模型：{BASE_MODEL}")
        print(" 请先运行： python scripts/download_model.py")
        sys.exit(1)

    try:
        import torch
        from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                  TrainingArguments, Trainer, DataCollatorForSeq2Seq)
        from peft import LoraConfig, get_peft_model, TaskType, PeftModel
    except ImportError as e:
        print(f" 依赖缺失：{e}")
        print(" 请先： pip install -r requirements-finetune.txt -i https://pypi.tuna.tsinghua.edu.cn/simple")
        sys.exit(1)

    torch.manual_seed(42)
    device = "cpu"
    ensure_dataset()

    print("\n[1/6] 加载 tokenizer 与基座模型（CPU, float32）...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float32, trust_remote_code=True).to(device)

    print("[2/6] 微调前基线回答...")
    before = [{"question": q, "answer": ask(model, tokenizer, q, device)} for q in TEST_QUESTIONS]

    print("[3/6] 套 LoRA（只训练低秩参数）...")
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    print("[4/6] 处理数据集...")
    import json as _json

    class _SFTDataset(torch.utils.data.Dataset):
        def __init__(self, items, tok):
            self.examples = [build_example(tok, it) for it in items]

        def __len__(self):
            return len(self.examples)

        def __getitem__(self, idx):
            return self.examples[idx]

    _raw = []
    with open(DATA_PATH, "r", encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line:
                _raw.append(_json.loads(_line))
    ds = _SFTDataset(_raw, tokenizer)
    print(f" 样本数：{len(ds)}")

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        logging_steps=1,
        save_strategy="no",
        fp16=False, bf16=False,  # CPU 用 float32
        optim="adamw_torch",
        report_to=[],
        use_cpu=True,
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
    )

    print("[5/6] 开始训练（CPU，稍等几分钟）...")
    trainer.train()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f" LoRA 适配器已保存：{OUTPUT_DIR}")

    print("[6/6] 微调后回答 & 对比...")
    after = [{"question": q, "answer": ask(model, tokenizer, q, device)} for q in TEST_QUESTIONS]

    os.makedirs(os.path.dirname(EVAL_OUT), exist_ok=True)
    with open(EVAL_OUT, "w", encoding="utf-8") as f:
        json.dump({"base_model": BASE_MODEL, "before": before, "after": after},
                  f, ensure_ascii=False, indent=2)

    print("\n===== 微调前后对比 =====")
    for b, a in zip(before, after):
        print(f"\nQ: {b['question']}")
        print(f" [微调前] {b['answer'][:120]}")
        print(f" [微调后] {a['answer'][:120]}")
    print(f"\n 完成！对比结果已保存：{EVAL_OUT}")


if __name__ == "__main__":
    main()
