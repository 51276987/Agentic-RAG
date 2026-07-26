"""Tests for the bounded Agentic RAG loop."""

import asyncio
from typing import Any

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
)
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph
from langgraph.types import Command

from app.core.langgraph.agent_loop import AgentLoop
from app.schemas import (
    EvidenceAssessment,
    GraphState,
    GroundednessAssessment,
    IntentAnalysis,
    QueryRewriteResult,
    RewrittenQuery,
    RetrievalPlan,
    RetrievalTask,
)


class FakeLLMService:
    """Return a predefined response for each loop LLM call."""

    def __init__(self, responses: list[Any]):
        """Initialize the fake with ordered responses."""
        self.responses = responses

    async def call(self, messages: Any, model_name: str | None = None, response_format: Any = None, **kwargs: Any) -> Any:
        """Return the next predefined response."""
        del messages, model_name, response_format, kwargs
        if not self.responses:
            raise AssertionError("unexpected LLM call")
        return self.responses.pop(0)


class FakeOpenVikingAPI:
    """Minimal read-only OpenViking API fake."""

    def __init__(self, find_results: list[Any] | None = None):
        """Initialize an empty call log."""
        self.calls: list[tuple[Any, ...]] = []
        self.find_results = list(find_results or [])

    async def find(self, query: str, target_uri: str, limit: int) -> Any:
        """Record a semantic retrieval call."""
        self.calls.append(("find", query, target_uri, limit))
        if self.find_results:
            return self.find_results.pop(0)
        return []

    async def list_resources(self, uri: str, recursive: bool, node_limit: int) -> Any:
        """Return one deterministic resource."""
        self.calls.append(("list_resources", uri, recursive, node_limit))
        return [
            {
                "uri": "viking://resources/guide.md",
                "isDir": False,
                "abstract": "OpenViking 使用指南。",
            }
        ]

    async def read(self, uri: str, level: str, offset: int = 0, limit: int = 200) -> Any:
        """Return deterministic knowledge content."""
        self.calls.append(("read", uri, level, offset, limit))
        return {"content": "OpenViking 使用指南正文。"}

    async def stat(self, uri: str) -> Any:
        """Return a ready resource state."""
        self.calls.append(("stat", uri))
        return {"uri": uri, "status": "ready"}


def _build_graph(llm: FakeLLMService, api: FakeOpenVikingAPI, *, checkpointer: Any = None) -> Any:
    builder = StateGraph(GraphState)
    AgentLoop(llm, api).configure(builder)  # pyright: ignore[reportArgumentType]
    return builder.compile(checkpointer=checkpointer)


def test_agent_loop_runs_read_only_retrieval_to_verified_answer() -> None:
    """An explicit role should proceed through retrieval without HITL."""
    llm = FakeLLMService(
        [
            IntentAnalysis(
                intent="fact_lookup",
                needs_retrieval=True,
                answer_requirements=["列出知识库文件"],
            ),
            RetrievalPlan(
                tasks=[
                    RetrievalTask(
                        task_id="r1",
                        purpose="列出文件",
                        operation="list_resources",
                        information_need="知识库文件列表",
                        hydration_level="abstract",
                    )
                ]
            ),
            EvidenceAssessment(
                sufficient=True,
                covered_requirements=["列出知识库文件"],
                missing_requirements=[],
                reason="目录结果和摘要足够",
            ),
            AIMessage(content="知识库包含 guide.md。[来源: viking://resources/guide.md]"),
            GroundednessAssessment(
                passed=True,
                action="pass",
                unsupported_claims=[],
                missing_requirements=[],
            ),
        ]
    )
    api = FakeOpenVikingAPI()
    graph = _build_graph(llm, api)

    result = asyncio.run(
        graph.ainvoke(
            {
                "messages": [HumanMessage(content="我是开发，知识库有什么文件？")],
                "long_term_memory": "",
            }
        )
    )

    assert result["user_role"] == "developer"
    assert result["retrieval_round"] == 1
    assert result["route"] == "completed"
    assert "viking://resources/guide.md" in result["final_answer"]
    assert any(call[0] == "list_resources" for call in api.calls)
    assert any(call[0] == "read" for call in api.calls)
    assert not any(call[0] in {"add_url", "write", "delete"} for call in api.calls)
    assert not llm.responses


def test_agent_loop_interrupts_and_resumes_for_unknown_role() -> None:
    """An unknown role should pause and resume the original request."""
    llm = FakeLLMService(
        [
            IntentAnalysis(
                intent="conversational",
                needs_retrieval=False,
                answer_requirements=["友好回应用户"],
            ),
            AIMessage(content="你好，很高兴为你服务。"),
        ]
    )
    api = FakeOpenVikingAPI()
    graph = _build_graph(llm, api, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "role-test:agentic-rag-v1"}}

    first_result = asyncio.run(
        graph.ainvoke(
            {
                "messages": [HumanMessage(content="你好")],
                "long_term_memory": "",
            },
            config=config,
        )
    )
    state = asyncio.run(graph.aget_state(config))

    assert "__interrupt__" in first_result
    assert state.next == ("role_clarification",)

    final_result = asyncio.run(graph.ainvoke(Command(resume="开发"), config=config))

    assert final_result["user_role"] == "developer"
    assert final_result["role_source"] == "hitl"
    assert final_result["route"] == "completed"
    assert final_result["final_answer"] == "你好，很高兴为你服务。"
    assert not api.calls


def test_agent_loop_repairs_missing_evidence_within_two_rounds() -> None:
    """Evidence gaps should trigger one bounded retrieval repair round."""
    llm = FakeLLMService(
        [
            IntentAnalysis(
                intent="procedure",
                needs_retrieval=True,
                answer_requirements=["说明认证配置", "给出验证方法"],
            ),
            RetrievalPlan(
                tasks=[
                    RetrievalTask(
                        task_id="r1",
                        purpose="查找认证配置",
                        operation="find",
                        information_need="认证配置",
                    )
                ]
            ),
            QueryRewriteResult(queries=[RewrittenQuery(task_id="r1", query="认证配置参数")]),
            EvidenceAssessment(
                sufficient=False,
                covered_requirements=["说明认证配置"],
                missing_requirements=["给出验证方法"],
                reason="缺少验证步骤",
            ),
            RetrievalPlan(
                tasks=[
                    RetrievalTask(
                        task_id="r2",
                        purpose="查找验证方法",
                        operation="find",
                        information_need="认证配置验证步骤",
                    )
                ]
            ),
            QueryRewriteResult(
                queries=[RewrittenQuery(task_id="r2", query="认证配置验证命令和预期结果")]
            ),
            EvidenceAssessment(
                sufficient=True,
                covered_requirements=["说明认证配置", "给出验证方法"],
                missing_requirements=[],
                reason="配置和验证证据完整",
            ),
            AIMessage(content="按照文档配置并执行验证命令。[来源: viking://resources/auth.md]"),
            GroundednessAssessment(
                passed=True,
                action="pass",
                unsupported_claims=[],
                missing_requirements=[],
            ),
        ]
    )
    api = FakeOpenVikingAPI(
        find_results=[
            [{"uri": "viking://resources/auth.md", "score": 0.8}],
            [{"uri": "viking://resources/validation.md", "score": 0.9}],
        ]
    )
    graph = _build_graph(llm, api)

    result = asyncio.run(
        graph.ainvoke(
            {
                "messages": [HumanMessage(content="我是产品经理，请说明认证配置和验证方法")],
                "long_term_memory": "",
            }
        )
    )

    assert result["retrieval_round"] == 2
    assert result["executed_queries"] == ["认证配置参数", "认证配置验证命令和预期结果"]
    assert result["route"] == "completed"
    assert len([call for call in api.calls if call[0] == "find"]) == 2
    assert {item["uri"] for item in result["selected_evidence"]} == {
        "viking://resources/auth.md",
        "viking://resources/validation.md",
    }
    assert not llm.responses
