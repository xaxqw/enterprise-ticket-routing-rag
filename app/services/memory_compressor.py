"""
记忆压缩：把长对话历史总结成摘要，节省token
"""


class MemoryCompressor:
    def __init__(self, llm, max_history=10, compress_threshold=8):
        self.llm = llm
        self.max_history = max_history
        self.compress_threshold = compress_threshold

    def compress_if_needed(self, history):
        """超过阈值就压缩"""
        if len(history) <= self.compress_threshold:
            return history, False

        # 把前面一半压缩成摘要
        split_idx = len(history) // 2
        old_messages = history[:split_idx]
        recent_messages = history[split_idx:]

        summary = self._summarize(old_messages)

        compressed = [
            {"role": "system", "content": f"【历史对话摘要】{summary}"}
        ] + recent_messages

        return compressed, True

    def _summarize(self, messages):
        """用LLM总结对话"""
        conversation = "\n".join([
            f"{m['role']}: {m['content']}" for m in messages
        ])

        prompt = f"""请用简短的话总结下面这段对话的核心内容，不超过200字。

对话：
{conversation}

摘要："""

        # 调用LLM生成摘要
        summary = self.llm.generate(prompt) if hasattr(self.llm, 'generate') else "历史对话摘要"
        return summary
