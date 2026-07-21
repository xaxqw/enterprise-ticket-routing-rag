"""
幻觉校验Agent：检查回答是否基于原文，有没有编造

校验思路（可解释、可讲清楚）：
1. 句子级切分答案，过滤掉「根据参考资料…」「注意：…」这类元信息套话/免责声明，
   只对实质性陈述句做原文支撑度检查（避免把提示语误判为幻觉）。
2. 关键词覆盖：用 jieba 分词后做词级包含度（比逐字硬匹配更稳健，能容忍
   原文中穿插的英文/括号/空格）。
3. 事实陈述校验：对每条实质性陈述句，检查其核心词是否能在参考资料中找到依据。
综合「关键词覆盖率」与「事实覆盖率」给出可信度判定。
（说明：默认走规则法，足够快且可解释；如需更强校验可接入 LLM-NLI，但默认不开。）
"""
import re
import jieba


# 元信息/套话标记：命中则整句不参与事实校验
_SOURCING_START = ("根据", "参考", "综上", "如下", "以下是", "依据")
_DISCLAIMER = ("注意", "请核对", "可能与原文", "出入", "提示：", "说明：", "建议", "免责")


class HallucinationAgent:
    def __init__(self, llm=None):
        self.llm = llm  # 可以用 LLM 做更精细的校验（NLI），默认不启用

    def check(self, answer, sources):
        """
        校验回答是否有事实依据
        返回: (是否可信, 校验详情)
        """
        if not sources:
            return False, "没有参考资料，无法验证"

        source_text = " ".join(
            [s.get("text", "") if isinstance(s, dict) else str(s) for s in sources]
        )

        # 1) 句子切分，过滤套话/免责声明，只对实质性陈述做校验
        sentences = [s.strip() for s in re.split(r"[。！？\n]", answer) if s.strip()]
        substantive = [s for s in sentences if not self._is_meta(s)]
        if not substantive:
            substantive = sentences  # 全是套话时退回用原文，避免误判
        checked_text = "。".join(substantive)

        # 2) 关键词覆盖（jieba 分词，词级包含度）
        answer_keywords = self._extract_keywords(checked_text)
        matched = sum(1 for kw in answer_keywords if kw in source_text)
        coverage = matched / max(len(answer_keywords), 1)

        # 3) 事实陈述校验（实体陈述句，至少 6 字）
        facts = [s for s in substantive if len(s) > 6]
        verified_facts = sum(1 for f in facts if self._fact_in_sources(f, source_text))
        fact_coverage = verified_facts / max(len(facts), 1)

        # 综合判断：关键词覆盖与事实覆盖都足够才认为可信
        is_factual = coverage >= 0.5 and fact_coverage >= 0.5

        details = {
            "keyword_coverage": round(coverage, 2),
            "fact_coverage": round(fact_coverage, 2),
            "unmatched_keywords": [kw for kw in answer_keywords if kw not in source_text][:10],
            "total_facts": len(facts),
            "verified_facts": verified_facts,
        }

        return is_factual, details

    def _is_meta(self, sentence):
        """判断是否为元信息/套话/免责声明句（不参与事实校验）"""
        s = sentence.strip()
        if any(s.startswith(p) for p in _SOURCING_START):
            return True
        if any(d in s for d in _DISCLAIMER):
            return True
        return False

    def _extract_keywords(self, text):
        """用 jieba 分词提取中文词（长度>=2），去重"""
        words = [
            w.strip()
            for w in jieba.cut(text)
            if len(w.strip()) >= 2 and re.search(r"[\u4e00-\u9fa5]", w)
        ]
        return list(set(words))

    def _extract_facts(self, text):
        """（保留接口）提取事实陈述句：按句切分后的实质性句子"""
        sentences = [s.strip() for s in re.split(r"[。！？\n]", text) if s.strip()]
        return [s for s in sentences if not self._is_meta(s) and len(s) > 6]

    def _fact_in_sources(self, fact, source_text):
        """判断一个事实是否在资料中有依据：核心词命中比例 >= 0.7 即认为有支撑"""
        words = self._extract_keywords(fact)
        if not words:
            return True
        hit = sum(1 for w in words if w in source_text)
        return hit / len(words) >= 0.7
