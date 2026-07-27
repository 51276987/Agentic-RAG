"""Tests for the bounded Agentic RAG loop."""

import asyncio
import json
from typing import Any

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
)
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.core.langgraph.agent_loop import AgentLoop
from app.schemas import (
    AnswerRequirement,
    EvidenceAssessment,
    GrepKeywordResult,
    GraphState,
    GroundednessAssessment,
    IntentAnalysis,
    QueryRewriteResult,
    RewrittenQuery,
    RetrievalTask,
    SystemScopeResult,
)


class FakeLLMService:
    """Return a predefined response for each loop LLM call."""

    def __init__(self, responses: list[Any]):
        """Initialize the fake with ordered responses."""
        self.responses = responses

    async def call(self, messages: Any, model_name: str | None = None, response_format: Any = None, **kwargs: Any) -> Any:
        """Return the next predefined response."""
        del model_name, kwargs
        if response_format is SystemScopeResult and (
            not self.responses or not isinstance(self.responses[0], SystemScopeResult)
        ):
            payload = json.loads(messages[-1].content)
            return SystemScopeResult(
                system_explicit=False,
                scoped_query=payload["query"],
            )
        if not self.responses:
            raise AssertionError("unexpected LLM call")
        return self.responses.pop(0)


class FakeOpenVikingAPI:
    """Minimal read-only OpenViking API fake."""

    def __init__(
        self,
        find_results: list[Any] | None = None,
        grep_results: list[Any] | None = None,
    ):
        """Initialize an empty call log."""
        self.calls: list[tuple[Any, ...]] = []
        self.find_results = list(find_results or [])
        self.grep_results = list(grep_results or [])

    async def find(self, query: str, target_uri: str, limit: int) -> Any:
        """Record a semantic retrieval call."""
        self.calls.append(("find", query, target_uri, limit))
        if self.find_results:
            return self.find_results.pop(0)
        return []

    async def grep(
        self,
        pattern: str,
        target_uri: str,
        *,
        case_insensitive: bool,
        node_limit: int,
        level_limit: int,
    ) -> Any:
        """Record one exact keyword retrieval call."""
        self.calls.append(
            (
                "grep",
                pattern,
                target_uri,
                case_insensitive,
                node_limit,
                level_limit,
            )
        )
        if self.grep_results:
            return self.grep_results.pop(0)
        return {"matches": [], "count": 0}

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


def test_result_fusion_sorts_find_scores_and_preserves_unscored_grep_matches() -> None:
    """Find hits should be score-sorted while grep matches retain source order."""
    loop = AgentLoop(FakeLLMService([]), FakeOpenVikingAPI())  # pyright: ignore[reportArgumentType]
    find_items = [
        {
            "uri": f"viking://resources/find_{index}.md",
            "level": 2,
            "score": score,
        }
        for index, score in enumerate((0.1, 0.8, 0.3, 0.7, 0.2))
    ]
    grep_items = [
        {
            "uri": f"viking://resources/grep_{index}.md",
            "line": index + 1,
            "content": f"keyword match {index}",
        }
        for index in range(3)
    ]
    state = GraphState(
        raw_results=[
            {
                "ok": True,
                "task_id": "find_docs",
                "operation": "find",
                "target_uri": "viking://resources",
                "result": find_items,
            },
            {
                "ok": True,
                "task_id": "grep_keywords",
                "operation": "grep",
                "target_uri": "viking://resources",
                "result": {"matches": grep_items, "count": len(grep_items)},
            },
        ]
    )

    result = asyncio.run(loop.result_fusion(state))
    candidates = result["candidate_items"]
    scored = [item for item in candidates if item["score"] is not None]
    unscored = [item for item in candidates if item["score"] is None]

    assert len(candidates) == 8
    assert [item["score"] for item in scored] == sorted(
        [item["score"] for item in scored],
        reverse=True,
    )
    assert len(unscored) == 3
    assert all(item["operation"] == "grep" for item in unscored)
    assert all(item["source_level"] == "full" for item in scored)


def test_evidence_hydration_reads_find_level_two_as_full_content() -> None:
    """A find L2 hit must use the full endpoint even without an isDir field."""
    api = FakeOpenVikingAPI()
    loop = AgentLoop(FakeLLMService([]), api)  # pyright: ignore[reportArgumentType]
    state = GraphState(
        retrieval_tasks=[
            RetrievalTask(
                task_id="find_timemoe",
                purpose="查找 TimeMoe 正文",
                operation="find",
                information_need="TimeMoe",
                hydration_level="abstract",
            ).model_dump()
        ],
        candidate_items=[
            {
                "uri": "viking://resources/paper/References_1.md",
                "score": 0.9,
                "task_id": "find_timemoe",
                "task_ids": ["find_timemoe"],
                "operation": "find",
                "operations": ["find"],
                "source_level": "full",
                "is_directory": False,
                "task_order": 0,
                "source_rank": 0,
                "metadata": {
                    "uri": "viking://resources/paper/References_1.md",
                    "level": 2,
                    "abstract": "Reference summary",
                },
            }
        ],
    )

    result = asyncio.run(loop.evidence_hydration(state))

    assert (
        "read",
        "viking://resources/paper/References_1.md",
        "full",
        0,
        200,
    ) in api.calls
    assert not any(call[0] == "read" and call[2] == "abstract" for call in api.calls)
    assert any(item["level"] == "full" for item in result["selected_evidence"])


def test_agent_loop_runs_read_only_retrieval_to_verified_answer() -> None:
    """An explicit role should proceed through retrieval without HITL."""
    llm = FakeLLMService(
        [
            IntentAnalysis(
                intent="fact_lookup",
                needs_retrieval=True,
                answer_requirements=[
                    AnswerRequirement(
                        requirement_id="req_1",
                        description="列出知识库文件",
                        priority="required",
                        evidence_source="knowledge_base",
                    )
                ],
            ),
            EvidenceAssessment(
                required_sufficient=True,
                covered_required_ids=["req_1"],
                missing_required_ids=[],
                covered_optional_ids=[],
                missing_optional_ids=[],
                reason="目录结果和摘要足够",
            ),
            AIMessage(content="知识库包含 guide.md。[来源: viking://resources/guide.md]"),
            GroundednessAssessment(
                passed=True,
                action="pass",
                unsupported_claims=[],
                missing_required_ids=[],
                missing_optional_ids=[],
            ),
        ]
    )
    api = FakeOpenVikingAPI(
        find_results=[
            [
                {
                    "uri": "viking://resources/guide.md",
                    "level": 2,
                    "score": 0.9,
                }
            ]
        ]
    )
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
    assert len([call for call in api.calls if call[0] == "find"]) == 1
    assert not any(call[0] == "grep" for call in api.calls)
    assert any(call[0] == "read" for call in api.calls)
    assert not any(call[0] in {"add_url", "write", "delete"} for call in api.calls)
    assert not llm.responses


def test_system_scope_limits_all_retrieval_to_explicit_system_folder() -> None:
    """An explicit system name should deterministically narrow the OpenViking root URI."""
    llm = FakeLLMService(
        [
            IntentAnalysis(
                intent="fact_lookup",
                needs_retrieval=True,
                answer_requirements=[
                    AnswerRequirement(
                        requirement_id="req_1",
                        description="说明支付系统认证配置",
                        priority="required",
                        evidence_source="knowledge_base",
                    )
                ],
            ),
            SystemScopeResult(
                system_explicit=True,
                system_name="支付系统",
                scoped_query="仅查询支付系统知识库：认证配置是什么？",
            ),
            EvidenceAssessment(
                required_sufficient=True,
                covered_required_ids=["req_1"],
                missing_required_ids=[],
                covered_optional_ids=[],
                missing_optional_ids=[],
                reason="已找到支付系统认证配置",
            ),
            AIMessage(
                content="支付系统使用令牌认证。[来源: viking://resources/支付系统/auth.md]"
            ),
            GroundednessAssessment(
                passed=True,
                action="pass",
                unsupported_claims=[],
                missing_required_ids=[],
                missing_optional_ids=[],
            ),
        ]
    )
    api = FakeOpenVikingAPI(
        find_results=[
            [
                {
                    "uri": "viking://resources/支付系统/auth.md",
                    "level": 2,
                    "score": 0.9,
                }
            ]
        ]
    )
    graph = _build_graph(llm, api)

    result = asyncio.run(
        graph.ainvoke(
            {
                "messages": [HumanMessage(content="我是开发，查询支付系统的认证配置")],
                "long_term_memory": "",
            }
        )
    )

    assert result["system_scope_explicit"] is True
    assert result["system_name"] == "支付系统"
    assert result["allowed_target_uris"] == ["viking://resources/支付系统"]
    assert result["normalized_query"] == "仅查询支付系统知识库：认证配置是什么？"
    find_call = next(call for call in api.calls if call[0] == "find")
    assert find_call[2] == "viking://resources/支付系统"
    assert result["route"] == "completed"
    assert not llm.responses


def test_agent_loop_answers_greeting_without_role_clarification() -> None:
    """A conversational request should not ask for a role or retrieve knowledge."""
    llm = FakeLLMService(
        [
            IntentAnalysis(
                intent="conversational",
                needs_retrieval=False,
                answer_requirements=[
                    AnswerRequirement(
                        requirement_id="req_1",
                        description="友好回应用户",
                        priority="required",
                        evidence_source="user_context",
                    )
                ],
            ),
            AIMessage(content="你好，很高兴为你服务。"),
        ]
    )
    api = FakeOpenVikingAPI()
    graph = _build_graph(llm, api)

    result = asyncio.run(
        graph.ainvoke(
            {
                "messages": [HumanMessage(content="你好")],
                "long_term_memory": "",
            }
        )
    )

    assert result["user_role"] is None
    assert result["needs_role_clarification"] is False
    assert result["route"] == "completed"
    assert result["final_answer"] == "你好，很高兴为你服务。"
    assert not api.calls
    assert not llm.responses


def test_agent_loop_repairs_missing_evidence_within_two_rounds() -> None:
    """Evidence gaps should trigger one bounded retrieval repair round."""
    llm = FakeLLMService(
        [
            IntentAnalysis(
                intent="procedure",
                needs_retrieval=True,
                answer_requirements=[
                    AnswerRequirement(
                        requirement_id="req_1",
                        description="说明认证配置",
                        priority="required",
                        evidence_source="knowledge_base",
                    ),
                    AnswerRequirement(
                        requirement_id="req_2",
                        description="给出验证方法",
                        priority="required",
                        evidence_source="knowledge_base",
                    ),
                ],
            ),
            EvidenceAssessment(
                required_sufficient=False,
                covered_required_ids=["req_1"],
                missing_required_ids=["req_2"],
                covered_optional_ids=[],
                missing_optional_ids=[],
                reason="缺少验证步骤",
            ),
            QueryRewriteResult(
                queries=[
                    RewrittenQuery(
                        task_id="rewritten_find",
                        query="认证配置验证命令和预期结果",
                    )
                ]
            ),
            EvidenceAssessment(
                required_sufficient=True,
                covered_required_ids=["req_1", "req_2"],
                missing_required_ids=[],
                covered_optional_ids=[],
                missing_optional_ids=[],
                reason="配置和验证证据完整",
            ),
            AIMessage(content="按照文档配置并执行验证命令。[来源: viking://resources/auth.md]"),
            GroundednessAssessment(
                passed=True,
                action="pass",
                unsupported_claims=[],
                missing_required_ids=[],
                missing_optional_ids=[],
            ),
        ]
    )
    api = FakeOpenVikingAPI(
        find_results=[
            [{"uri": "viking://resources/auth.md", "level": 2, "score": 0.8}],
            [{"uri": "viking://resources/validation.md", "level": 2, "score": 0.9}],
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
    assert result["executed_queries"] == [
        "我是产品经理，请说明认证配置和验证方法",
        "认证配置验证命令和预期结果",
    ]
    assert result["route"] == "completed"
    assert len([call for call in api.calls if call[0] == "find"]) == 2
    assert {item["uri"] for item in result["selected_evidence"]} == {
        "viking://resources/auth.md",
        "viking://resources/validation.md",
    }
    assert not llm.responses


def test_agent_loop_uses_one_grep_round_then_requests_question_clarification() -> None:
    """Empty initial, rewritten, and grep retrievals must end in a resumable HITL prompt."""
    llm = FakeLLMService(
        [
            IntentAnalysis(
                intent="procedure",
                needs_retrieval=True,
                answer_requirements=[
                    AnswerRequirement(
                        requirement_id="req_1",
                        description="说明认证配置和验证方法",
                        priority="required",
                        evidence_source="knowledge_base",
                    )
                ],
            ),
            QueryRewriteResult(
                queries=[
                    RewrittenQuery(
                        task_id="rewritten_find",
                        query="认证配置参数 验证命令",
                    )
                ]
            ),
            GrepKeywordResult(keywords=["认证配置", "验证命令"]),
            IntentAnalysis(
                intent="procedure",
                needs_retrieval=True,
                answer_requirements=[
                    AnswerRequirement(
                        requirement_id="req_1",
                        description="说明认证配置和验证方法",
                        priority="required",
                        evidence_source="knowledge_base",
                    )
                ],
            ),
            EvidenceAssessment(
                required_sufficient=True,
                covered_required_ids=["req_1"],
                missing_required_ids=[],
                covered_optional_ids=[],
                missing_optional_ids=[],
                reason="用户补充版本后检索到完整证据",
            ),
            AIMessage(content="请按 v2 配置并验证。[来源: viking://resources/auth_v2.md]"),
            GroundednessAssessment(
                passed=True,
                action="pass",
                unsupported_claims=[],
                missing_required_ids=[],
                missing_optional_ids=[],
            ),
        ]
    )
    api = FakeOpenVikingAPI(
        find_results=[
            [],
            [],
            [
                {
                    "uri": "viking://resources/auth_v2.md",
                    "level": 2,
                    "score": 0.95,
                }
            ],
        ],
        grep_results=[{"matches": [], "count": 0}],
    )
    graph = _build_graph(llm, api, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "grep-hitl-test"}}

    result = asyncio.run(
        graph.ainvoke(
            {
                "messages": [HumanMessage(content="我是开发，请说明认证方案")],
                "long_term_memory": "",
            },
            config=config,
        )
    )
    snapshot = asyncio.run(graph.aget_state(config))

    assert "__interrupt__" in result
    assert snapshot.next == ("question_clarification",)
    assert snapshot.values["retrieval_stage"] == "grep"
    assert snapshot.values["retrieval_round"] == 3
    assert [call[0] for call in api.calls if call[0] in {"find", "grep"}] == [
        "find",
        "find",
        "grep",
    ]
    grep_call = next(call for call in api.calls if call[0] == "grep")
    assert grep_call[1] == "认证配置|验证命令"
    interrupt_value = snapshot.tasks[0].interrupts[0].value
    assert interrupt_value["type"] == "question_clarification"
    assert interrupt_value["missing_information"] == ["说明认证配置和验证方法"]
    assert interrupt_value["requires_system_name"] is True
    assert "声明需要查询哪个系统的知识库" in interrupt_value["question"]

    resumed = asyncio.run(
        graph.ainvoke(
            Command(resume="目标版本是 v2，需要部署后的验证命令"),
            config=config,
        )
    )

    assert resumed["route"] == "completed"
    assert "用户补充：目标版本是 v2" in resumed["normalized_query"]
    assert resumed["retrieval_stage"] == "initial_find"
    assert resumed["retrieval_round"] == 1
    assert len([call for call in api.calls if call[0] == "find"]) == 3
    assert len([call for call in api.calls if call[0] == "grep"]) == 1
    assert not llm.responses


def test_second_exhausted_cycle_after_hitl_returns_knowledge_not_found() -> None:
    """Only one HITL supplement is allowed before a deterministic not-found answer."""
    requirement = AnswerRequirement(
        requirement_id="req_1",
        description="说明目标系统的认证配置",
        priority="required",
        evidence_source="knowledge_base",
    )
    llm = FakeLLMService(
        [
            IntentAnalysis(
                intent="procedure",
                needs_retrieval=True,
                answer_requirements=[requirement],
            ),
            QueryRewriteResult(
                queries=[
                    RewrittenQuery(
                        task_id="rewritten_find",
                        query="认证配置 参数",
                    )
                ]
            ),
            GrepKeywordResult(keywords=["认证配置"]),
            IntentAnalysis(
                intent="procedure",
                needs_retrieval=True,
                answer_requirements=[requirement],
            ),
            SystemScopeResult(
                system_explicit=True,
                system_name="支付系统",
                scoped_query="仅查询支付系统知识库：认证配置和验证方法",
            ),
            QueryRewriteResult(
                queries=[
                    RewrittenQuery(
                        task_id="rewritten_find",
                        query="支付系统 认证配置 验证方法",
                    )
                ]
            ),
            GrepKeywordResult(keywords=["认证配置", "验证方法"]),
        ]
    )
    api = FakeOpenVikingAPI(
        find_results=[[], [], [], []],
        grep_results=[
            {"matches": [], "count": 0},
            {"matches": [], "count": 0},
        ],
    )
    graph = _build_graph(llm, api, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "single-hitl-budget-test"}}

    interrupted = asyncio.run(
        graph.ainvoke(
            {
                "messages": [HumanMessage(content="我是开发，认证怎么配置？")],
                "long_term_memory": "",
            },
            config=config,
        )
    )
    assert "__interrupt__" in interrupted

    resumed = asyncio.run(
        graph.ainvoke(
            Command(resume="查询支付系统，问题是认证配置和部署后验证方法"),
            config=config,
        )
    )

    assert resumed["hitl_retry_used"] is True
    assert resumed["system_name"] == "支付系统"
    assert resumed["route"] == "knowledge_not_found"
    assert resumed["final_answer"] == "在“支付系统”系统知识库中检索不到足以回答该问题的信息。"
    assert len([call for call in api.calls if call[0] == "find"]) == 4
    assert len([call for call in api.calls if call[0] == "grep"]) == 2
    assert not llm.responses


def test_agent_loop_uses_grep_match_as_evidence() -> None:
    """The final grep round should hydrate its matched file and complete when sufficient."""
    llm = FakeLLMService(
        [
            IntentAnalysis(
                intent="troubleshooting",
                needs_retrieval=True,
                answer_requirements=[
                    AnswerRequirement(
                        requirement_id="req_1",
                        description="找到错误码对应的处理方法",
                        priority="required",
                        evidence_source="knowledge_base",
                    )
                ],
            ),
            QueryRewriteResult(
                queries=[
                    RewrittenQuery(
                        task_id="rewritten_find",
                        query="E401 认证失败 排障",
                    )
                ]
            ),
            GrepKeywordResult(keywords=["E401"]),
            EvidenceAssessment(
                required_sufficient=True,
                covered_required_ids=["req_1"],
                missing_required_ids=[],
                covered_optional_ids=[],
                missing_optional_ids=[],
                reason="grep 命中错误码及处理方法",
            ),
            AIMessage(content="按认证文档更新令牌。[来源: viking://resources/errors.md]"),
            GroundednessAssessment(
                passed=True,
                action="pass",
                unsupported_claims=[],
                missing_required_ids=[],
                missing_optional_ids=[],
            ),
        ]
    )
    api = FakeOpenVikingAPI(
        find_results=[[], []],
        grep_results=[
            {
                "matches": [
                    {
                        "uri": "viking://resources/errors.md",
                        "line": 15,
                        "content": "E401 表示认证令牌无效，请更新令牌后重试。",
                    }
                ],
                "count": 1,
            }
        ],
    )
    graph = _build_graph(llm, api)

    result = asyncio.run(
        graph.ainvoke(
            {
                "messages": [HumanMessage(content="我是开发，E401 怎么处理？")],
                "long_term_memory": "",
            }
        )
    )

    assert result["route"] == "completed"
    assert result["retrieval_stage"] == "grep"
    assert result["retrieval_round"] == 3
    assert [call[0] for call in api.calls if call[0] in {"find", "grep"}] == [
        "find",
        "find",
        "grep",
    ]
    assert any(item["level"] == "grep_match" for item in result["selected_evidence"])
    assert any(item["level"] == "full" for item in result["selected_evidence"])
    assert not llm.responses


def test_optional_requirement_does_not_trigger_retrieval_fallback() -> None:
    """An uncovered optional personalization item must not cause another retrieval round."""
    llm = FakeLLMService(
        [
            IntentAnalysis(
                intent="fact_lookup",
                needs_retrieval=True,
                answer_requirements=[
                    AnswerRequirement(
                        requirement_id="req_1",
                        description="列出知识库资源",
                        priority="required",
                        evidence_source="knowledge_base",
                    ),
                    AnswerRequirement(
                        requirement_id="req_2",
                        description="提供资源路径",
                        priority="required",
                        evidence_source="knowledge_base",
                    ),
                    AnswerRequirement(
                        requirement_id="req_3",
                        description="根据新员工角色推荐优先学习资源",
                        priority="optional",
                        evidence_source="knowledge_and_context",
                    ),
                ],
            ),
            EvidenceAssessment(
                required_sufficient=True,
                covered_required_ids=["req_1", "req_2"],
                missing_required_ids=[],
                covered_optional_ids=[],
                missing_optional_ids=["req_3"],
                reason="必须项证据充分，可选推荐缺少依据",
            ),
            AIMessage(content="知识库包含 guide.md。[来源: viking://resources/guide.md]"),
            GroundednessAssessment(
                passed=True,
                action="pass",
                unsupported_claims=[],
                missing_required_ids=[],
                missing_optional_ids=["req_3"],
            ),
        ]
    )
    api = FakeOpenVikingAPI(
        find_results=[
            [
                {
                    "uri": "viking://resources/guide.md",
                    "level": 2,
                    "score": 0.9,
                }
            ]
        ]
    )
    graph = _build_graph(llm, api)

    result = asyncio.run(
        graph.ainvoke(
            {
                "messages": [HumanMessage(content="我是新员工，知识库有哪些资源？")],
                "long_term_memory": "",
            }
        )
    )

    assert result["retrieval_round"] == 1
    assert result["missing_required_ids"] == []
    assert result["missing_optional_ids"] == ["req_3"]
    assert result["route"] == "completed"
    assert not llm.responses
