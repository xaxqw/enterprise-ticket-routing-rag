"""
    完整RAG服务：检索 + 生成 + 记忆，一站式调用

    设计说明（生成后端，由 .env 的 LLM_BACKEND 切换）：
    * ollama（默认）：本地 Ollama 运行开源模型（如 deepseek-r1），完全免费、离线、
                      调用本机 GPU（RTX 4050），无需任何 API Key 或外网
    * siliconflow   ：硅基流动 SiliconFlow 在线API（OpenAI兼容），需 API Key
    * local_lora    ：本地 Qwen2.5-0.5B + LoRA 适配器（CPU 推理，需先训练）
    - ollama 默认走本机 GPU，回答质量与在线大模型相当，且零成本、可离线演示
    - 检索/向量化同样本地化（Ollama nomic-embed-text 嵌入），整套系统 100% 免费离线
    """

import logging
from app.core.log import get_logger
logger = get_logger(__name__)

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# 硅基流动 SiliconFlow API 基址（OpenAI 兼容）—— 仅 LLM_BACKEND=siliconflow 时使用
_SILICONFLOW_BASE = "https://api.siliconflow.cn/v1"


# 本地 LoRA 推理（可切换，懒加载）
# 仅当 LLM_MODE=local_lora 时按需加载 torch/transformers/peft；
# 默认 online 模式完全不依赖这些重型依赖，保证轻量启动。
_LOCAL_LLM = None # 单例：{"model": PeftModel, "tokenizer": tokenizer}


def _load_local_llm():
    """懒加载本地基座 + LoRA 适配器（CPU）。失败抛清晰错误。"""
    global _LOCAL_LLM
    if _LOCAL_LLM is not None:
        return _LOCAL_LLM
    try:
        import torch # noqa
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except ImportError as e:
        raise RuntimeError(
            f"本地 LoRA 模式需要 torch/transformers/peft，但未安装：{e}。"
            "请执行：pip install -r requirements-finetune.txt"
        )
    base = os.getenv("LLM_LOCAL_MODEL", "./models/base/Qwen/Qwen2.5-0.5B-Instruct")
    lora = os.getenv("LLM_LORA_PATH", "./models/lora/qwen2.5-0.5b-lora-v1")
    if not os.path.isdir(base):
        raise RuntimeError(f"本地基座模型不存在：{base}。请先运行 python scripts/download_model.py")
    if not os.path.isdir(lora):
        raise RuntimeError(f"LoRA 适配器不存在：{lora}。请先运行 python scripts/run_finetune.py")
    logger.info(f"⏳ 正在加载本地 LoRA 模型（CPU）：{lora} ...")
    tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.float32, device_map="cpu", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(model, lora)
    model.eval()
    _LOCAL_LLM = {"model": model, "tokenizer": tokenizer}
    logger.info(" 本地 LoRA 模型加载完成")
    return _LOCAL_LLM


def _local_chat(messages, temperature=0.7, max_tokens=1024):
    """用本地 Qwen2.5-0.5B + LoRA 生成回答。messages 为 OpenAI 风格列表。"""
    import torch
    eng = _load_local_llm()
    model, tokenizer = eng["model"], eng["tokenizer"]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    gen_kwargs = dict(
        max_new_tokens=min(int(max_tokens), 512),
        do_sample=temperature > 0,
        temperature=max(float(temperature), 1e-5),
        top_p=0.9,
        pad_token_id=pad_id,
        )
    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
        new_ids = out[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


class RAGService:
    def __init__(self, hybrid_searcher, conversation_memory):
        self.searcher = hybrid_searcher
        self.memory = conversation_memory
        self._client = None # 懒加载，避免没配API Key时初始化就报错
        self.backend = os.getenv("LLM_BACKEND", "ollama").lower()
        self.llm_model = os.getenv("LLM_MODEL", "deepseek-r1")
        if self.backend == "ollama":
            logger.info(f" RAG服务初始化完成（本地 Ollama LLM: {self.llm_model}）")
        elif self.backend == "siliconflow":
            logger.info(f" RAG服务初始化完成（在线 SiliconFlow LLM: {self.llm_model}）")
        else:
            logger.info(f" RAG服务初始化完成（本地 LoRA 模式）")

    def _get_client(self):
        """懒加载 OpenAI 兼容客户端（仅 LLM_BACKEND=siliconflow 时使用）"""
        if self._client is None:
            api_key = os.getenv("SILICONFLOW_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "SILICONFLOW_API_KEY 未配置。请打开项目目录下的 .env 文件，"
                    "在 SILICONFLOW_API_KEY= 后面填入你的硅基流动 API Key。"
                    "免费获取：https://cloud.siliconflow.cn/"
                )
            self._client = OpenAI(api_key=api_key, base_url=_SILICONFLOW_BASE)
        return self._client

    def _llm_chat(self, messages, temperature=0.7, max_tokens=1024):
        """
            统一生成入口：按 LLM_BACKEND 切换
            - ollama（默认）：本地 Ollama 开源模型（免费/离线/GPU）
            - siliconflow    ：硅基流动在线API
            - local_lora     ：本地 Qwen2.5-0.5B + LoRA 适配器（CPU）
            """
        backend = os.getenv("LLM_BACKEND", "ollama").lower()
        if backend == "local_lora":
            try:
                return _local_chat(messages, temperature=temperature, max_tokens=max_tokens)
            except Exception as e:
                raise RuntimeError(f"本地 LoRA 推理失败：{e}")
        if backend == "siliconflow":
            client = self._get_client()
            resp = client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                temperature=temperature,
                top_p=0.9,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip()
        # 默认：本地 Ollama
        try:
            from app.services.ollama_client import ollama_chat
            return ollama_chat(messages, model=self.llm_model,
                               temperature=temperature, max_tokens=max_tokens)
        except RuntimeError as e:
            raise RuntimeError(f"本地 Ollama 生成失败：{e}")

    def generate(self, prompt, system=None, temperature=0.7, max_tokens=1024):
        """
            纯 LLM 生成（不做检索）。供多智能体闲聊、记忆摘要、幻觉复核等复用。
            自动按 LLM_MODE 选择在线API或本地 LoRA。
            """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            return self._llm_chat(messages, temperature=temperature, max_tokens=max_tokens)
        except Exception as e:
            return f" 生成失败：{e}"

    def _build_messages(self, query, contexts, history):
        """构建对话的 messages（含系统提示 + 历史 + 当前问题 + 参考资料）"""
        context_str = "\n\n".join([
            f"【参考资料{i+1}】{ctx['text']}"
            for i, ctx in enumerate(contexts)
            ]) if contexts else "（无相关参考资料）"

        system_prompt = (
            "你是一个专业的知识问答助手。请根据用户提供的参考资料回答问题。\n"
            "要求：\n"
            "1. 优先使用参考资料中的内容回答；\n"
            "2. 如果参考资料中没有答案，请如实说明，不要编造；\n"
            "3. 回答要简洁、准确、有条理；\n"
            "4. 年份、金额、人数、占比等数字必须严格照抄参考资料原文对应字段，"
            "不得改写、推算，也不得挪用其他字段的数字（例如不得把成立年份当作员工人数）；\n"
            "5. 数字只能由阿拉伯数字 0-9 组成，严禁把英文字母（如 Z）混入数字中。"
            )

        # 当前问题附带参考资料
        user_content = f"{query}\n\n【参考资料】\n{context_str}"

        messages = [{"role": "system", "content": system_prompt}]
        # 加入历史对话（多轮上下文）
        messages.extend(history)
        messages.append({"role": "user", "content": user_content})
        return messages

    @staticmethod
    def _ground_numbers(answer, contexts):
        """
        数字落地校验：把答案里与参考资料相差「单个数字」（错字或相邻易位）的
        年份 / 百分比 / 整数 纠回原文，避免小模型偶发数字幻觉（2018->2218、62%->61%）。
        仅在来源中存在唯一近邻时才修正，避免误改。
        """
        import re
        src_text = " ".join(contexts)
        # 按类别收集来源中的数字
        src_years = set(re.findall(r"\b(?:19|20)\d{2}\b", src_text))
        src_pcts = set(re.findall(r"\d+(?:\.\d+)?%", src_text))
        src_ints = set(re.findall(r"(?<![\d.])(\d{2,})(?![\d%])", src_text))

        def edit1(a, b):
            """编辑距离<=1（含相邻易位 / 单字符插入 / 删除），用于纠正
            小模型把 2018 写成 22018、把 2400 写成 24000 这类错字"""
            if a == b:
                return False
            if abs(len(a) - len(b)) > 1:
                return False
            # 同长度：逐字符比较，允许 1 处不同（相邻易位也算 1 处）
            if len(a) == len(b):
                diff = sum(1 for x, y in zip(a, b) if x != y)
                if diff == 1:
                    return True
                if diff == 2:  # 相邻易位
                    for i in range(len(a) - 1):
                        if a[:i] + a[i + 1] + a[i] + a[i + 2:] == b:
                            return True
                return False
            # 长度差 1：较长串删掉一个字符应能变成较短串
            longer, shorter = (a, b) if len(a) > len(b) else (b, a)
            for i in range(len(longer)):
                if longer[:i] + longer[i + 1:] == shorter:
                    return True
            return False

        def digit_overlap(a, b):
            """两串共同出现的数字个数（顺序无关）"""
            return len(set(a) & set(b))

        def fix(num_str, src_set):
            if num_str in src_set:
                return num_str
            # 1) 单字符编辑距离（含易位/插入删除）：如 2218<->2018、22018<->2018
            close = [s for s in src_set if edit1(num_str, s)]
            if len(close) == 1:
                return close[0]
            # 2) 数字重叠 >= len-1 的兜底：如 2208<->2018、61%<->62%
            ov = [
                s for s in src_set
                if len(s) == len(num_str) and digit_overlap(num_str, s) >= len(num_str) - 1
            ]
            if len(ov) == 1:
                return ov[0]
            return num_str

        answer = re.sub(r"\b(?:19|20)\d{2}\b", lambda m: fix(m.group(0), src_years), answer)
        answer = re.sub(r"\d+(?:\.\d+)?%", lambda m: fix(m.group(0), src_pcts), answer)
        answer = re.sub(r"(?<![\d.])(\d{2,})(?![\d%])", lambda m: fix(m.group(0), src_ints), answer)

        # 字母污染数字恢复：6Z% / 2Z8 / Z20Z8 这类被模型混入字母的数字，
        # 提取其数字子序列与同类别来源数字做子序列匹配后纠回（如 6Z%->62%、Z20Z8->2018）
        def _is_subseq(a, b):
            it = iter(b)
            return all(c in it for c in a)

        contam_pat = re.compile(r"(?<![0-9A-Za-z])([0-9A-Za-z]*[0-9][0-9A-Za-z]*)(%?)")

        def _recover(m):
            core, pct = m.group(1), m.group(2)
            if not re.search(r"[A-Za-z]", core):
                return m.group(0)
            digits = re.sub(r"\D", "", core)
            if len(digits) < 1:
                return m.group(0)
            pool = src_pcts if pct else src_ints
            def match_src(s):
                sd = re.sub(r"\D", "", s)
                # 单数字用前缀匹配防误改；多位数字用子序列匹配
                if len(digits) == 1:
                    return sd.startswith(digits)
                return _is_subseq(digits, sd) and abs(len(sd) - len(digits)) <= 2
            cands = [s for s in pool if match_src(s)]
            return (cands[0] + pct) if len(cands) == 1 else m.group(0)

        answer = contam_pat.sub(_recover, answer)
        return answer

    def query(self, session_id, user_query, top_k=5):
        """
            主问答函数
            session_id: 会话ID，区分不同用户
            user_query: 用户的问题
            """
        # 1. 混合检索，找相关资料（余额不足 / API 异常时返回友好提示，而非 500 崩溃）
        try:
            contexts = self.searcher.search(user_query, top_k=top_k, rerank_top_k=top_k)
        except RuntimeError as e:
            return {"answer": f" 无法生成回答：{e}", "references": []}
        except Exception as e:
            return {"answer": f" 检索环节出错：{e}", "references": []}

        # 2. 获取历史对话
        history = self.memory.get_history_as_list(session_id, limit=6)

        # 3. 构建消息
        messages = self._build_messages(user_query, contexts, history)

        # 4. 调用大模型生成回答（自动按 LLM_MODE 切换在线/本地 LoRA）
        try:
            answer = self._llm_chat(messages, temperature=0.7, max_tokens=1024)
        except RuntimeError as e:
            # API Key 未配置 / 本地 LoRA 加载失败 的友好提示
            answer = f" 无法生成回答：{e}"
        except Exception as e:
            answer = (
                f" 生成回答时出错：{e}\n"
                "请检查：本地 Ollama 模式需先启动 Ollama 并拉取模型"
                f"（ollama pull {self.llm_model}）；"
                "若切换为 siliconflow / local_lora，请确认对应配置已就绪。"
            )
        # 4.5 数字落地校验：纠正与参考资料相差单个数字（错字/易位）的年份、百分比、
        # 人数等，避免小模型偶发的数字幻觉（如 2018->2218、62%->61%）。仅在来源中
        # 存在唯一近邻时修正，保证不引入新的错误。
        answer = self._ground_numbers(answer, [c["text"] for c in contexts])
        # 5. 保存对话历史（问题 + 回答，回答附带参考资料溯源，供下次进入查看）
        self.memory.add_message(session_id, "user", user_query)
        references = [
            {"text": c["text"][:200], "score": c.get("final_score", 0.0), "metadata": c.get("metadata", {})}
            for c in contexts
        ]
        self.memory.add_message(session_id, "assistant", answer, extra={"references": references})
        return {
            "answer": answer,
            "references": references,
        }
