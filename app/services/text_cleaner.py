"""
文本清洗器：去空白、去乱码、去重复、过滤低质量文本
"""
import re
import hashlib


class TextCleaner:
    def clean(self, text):
        """完整清洗流程，一条龙服务"""
        text = self.remove_extra_whitespace(text)
        text = self.normalize_brand_names(text)
        text = self.remove_latin_parentheticals(text)
        text = self.fix_garbled(text)
        text = self.remove_duplicate_punctuation(text)
        return text.strip()

    def normalize_brand_names(self, text):
        """把英文品牌名归一为中文名（如「ZhiTu Technology」→「智图科技」）。
        中文问答语料里混入外文品牌名会诱发模型把字母混进数字（如把 2018 写成 2Z8），
        统一成中文名可去掉这一噪声源。"""
        return text.replace("ZhiTu Technology", "智图科技").replace("ZhiTu", "智图科技")

    def remove_latin_parentheticals(self, text):
        """去掉不含中文的括号注释（如「（ZhiTu Technology）」这类英文/品牌注记）。
        中文问答里这些纯外文括号是噪声，且易诱发模型把字母混进数字；保留含中文的括号。"""
        return re.sub(r"[（(][^（）()\u4e00-\u9fa5]*[）)]", "", text)

    def remove_extra_whitespace(self, text):
        """去掉多余的空格和换行"""
        text = re.sub(r" +", " ", text) # 多个空格变一个
        text = re.sub(r"\n{3,}", "\n\n", text) # 三个以上换行变两个
        return text

    def fix_garbled(self, text):
        """去掉不可见字符、乱码"""
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        return text

    def remove_duplicate_punctuation(self, text):
        """去掉重复标点，比如。。。变成。"""
        text = re.sub(r"([。！？，、；：])\1+", r"\1", text)
        return text

    def filter_low_quality(self, chunks):
        """过滤低质量文本块：太短的、纯数字的、全是符号的"""
        filtered = []
        for chunk in chunks:
            if len(chunk) < 20: # 太短了，没信息量
                continue
            # 计算非文字字符比例
            non_alpha = len(re.findall(r"[^\u4e00-\u9fa5a-zA-Z]", chunk))
            if non_alpha / len(chunk) > 0.8: # 80%以上都不是文字
                continue
            filtered.append(chunk)
        return filtered

    def remove_duplicate_chunks(self, chunks):
        """完全重复的块去重（用MD5判断）"""
        seen = set()
        unique = []
        for chunk in chunks:
            h = hashlib.md5(chunk.encode()).hexdigest()
            if h not in seen:
                seen.add(h)
                unique.append(chunk)
        return unique
