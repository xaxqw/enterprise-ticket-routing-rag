"""
多智能体单元测试（离线，不依赖网络/Redis）
覆盖：工具Agent（计算/时间）、幻觉Agent、编排器意图识别与路由
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.agents.tool_agent import ToolAgent
from app.agents.hallucination_agent import HallucinationAgent
from app.agents.agent_orchestrator import AgentOrchestrator


def test_tool_agent_calculator():
    agent = ToolAgent()
    res = agent.handle("帮我计算 12+8 等于多少")
    assert res["agent"] == "tool"
    assert "20" in res["answer"]


def test_tool_agent_time():
    agent = ToolAgent()
    res = agent.handle("现在几点了")
    assert res["tool_used"] == "get_time"
    assert "当前时间" in res["answer"]


def test_hallucination_check_grounded():
    agent = HallucinationAgent()
    sources = [{"text": "混合检索由向量检索和BM25关键词检索组成，并用重排提升精度。"}]
    ok, detail = agent.check("混合检索由向量检索和BM25组成。", sources)
    assert isinstance(detail, dict)
    assert detail["keyword_coverage"] >= 0.5


def test_hallucination_check_no_sources():
    agent = HallucinationAgent()
    ok, detail = agent.check("随便编的答案", [])
    assert ok is False


class _FakeRetrieval:
    def handle(self, query, session_id="default", top_k=5):
        return {"answer": "检索型回答", "sources": [{"text": "资料"}], "agent": "retrieval"}


class _FakeTool:
    def handle(self, query):
        return {"answer": "工具型回答", "tool_used": "calculator", "agent": "tool"}


class _FakeLLM:
    def generate(self, q, **kw):
        return "闲聊型回答"


def _make_orch():
    return AgentOrchestrator(_FakeRetrieval(), _FakeTool(), HallucinationAgent(), _FakeLLM())


def test_orchestrator_routes_tool():
    orch = _make_orch()
    res = orch.process("请计算 3*3")
    assert res["intent"] == "tool_call"
    assert res["agent"] == "tool"


def test_orchestrator_routes_chat():
    orch = _make_orch()
    res = orch.process("你好")
    assert res["intent"] == "general_chat"
    assert res["agent"] == "direct"


def test_orchestrator_routes_knowledge():
    orch = _make_orch()
    res = orch.process("这个项目的架构是怎样的？")
    assert res["intent"] == "knowledge_qa"
    assert res["agent"] == "retrieval"


def test_orchestrator_reflection_runs():
    orch = _make_orch()
    res = orch.process_with_reflection("介绍一下混合检索", max_rounds=2)
    assert "answer" in res
