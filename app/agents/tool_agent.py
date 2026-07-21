"""
工具Agent：调用各种工具（计算器、时间查询等）
"""
import re
import datetime
import math


class ToolAgent:
    def __init__(self, llm=None):
        self.llm = llm
        # 注册可用的工具
        self.tools = {
            "calculator": self._calculator,
            "get_time": self._get_current_time,
            }

    def handle(self, query):
        """判断该用哪个工具，调用后返回结果"""
        calc_hit = ["计算", "算一下", "算算", "等于", "多少", "乘以", "除以", "加上",
            "减去", "乘", "除", "+", "-", "*", "/", "×", "÷"]
        time_hit = ["几点", "时间", "日期", "今天", "报时"]
        if any(k in query for k in calc_hit):
            tool_name = "calculator"
        elif any(k in query for k in time_hit):
            tool_name = "get_time"
        else:
            return {"answer": "抱歉，我暂时不会处理这个请求", "agent": "tool"}

        # 解析参数（简化版）
        params = self._extract_params(query, tool_name)

        # 调用工具
        result = self.tools[tool_name](params)

        return {
            "answer": result,
            "tool_used": tool_name,
            "agent": "tool"
            }

    def _calculator(self, expression):
        """计算器工具"""
        try:
            # 安全计算，只允许数学表达式
            allowed = set("0123456789+-*/(). ")
            if all(c in allowed for c in expression):
                result = eval(expression)
                return f"计算结果：{expression} = {result}"
            else:
                return "表达式不合法"
        except Exception as e:
            return f"计算出错: {e}"

    def _get_current_time(self, _):
        """获取当前时间"""
        now = datetime.datetime.now()
        return f"当前时间：{now.strftime('%Y年%m月%d日 %H:%M:%S')}"

    def _extract_params(self, query, tool_name):
        """从问题里提取参数（简化版）"""
        if tool_name == "calculator":
            # 先把中文运算符/全角符号归一化为标准算术符号
            expr = query
            replaces = {
                "乘以": "*", "乘": "*", "×": "*",
                "除以": "/", "除": "/", "÷": "/",
                "加上": "+", "加": "+",
                "减去": "-", "减": "-",
                }
            for cn, op in replaces.items():
                expr = expr.replace(cn, op)
            # 提取形如 128*6 / 12 + 3 的算术表达式
            match = re.search(r"(\d+\s*[\+\-\*/]\s*\d+(?:\s*[\+\-\*/]\s*\d+)*)", expr)
            if match:
                return match.group(1).replace(" ", "")
            return expr
        return query
