"""First-phase Agentic RAG loop backed by deterministic OpenViking API calls."""

import asyncio
import json
import re
from collections.abc import Awaitable, Callable, Mapping
from typing import (
    Any,
    Literal,
)

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langgraph.graph import (
    END,
    START,
    StateGraph,
)
from langgraph.config import get_stream_writer
from langgraph.errors import GraphInterrupt
from langgraph.types import interrupt

from app.core.logging import logger
from app.core.prompts.agentic_rag import get_agentic_rag_prompt
from app.schemas import (
    AnswerRequirement,
    EvidenceAssessment,
    GrepKeywordResult,
    GraphState,
    GroundednessAssessment,
    IntentAnalysis,
    QueryRewriteResult,
    RetrievalTask,
    SystemScopeResult,
)
from app.services.llm import LLMService
from app.services.openviking import OpenVikingKnowledgeAPI
from app.utils import extract_text_content

MAX_PARALLEL_TASKS = 4
MAX_RESULTS_PER_TASK = 10
MAX_HYDRATED_RESOURCES = 8
MAX_FULL_CONTENT_RESOURCES = 4
MAX_EVIDENCE_CHARS = 24_000
MAX_TOOL_RESULT_DOCUMENTS = 10
ROOT_RESOURCES_URI = "viking://resources"
SourceLevel = Literal["abstract", "overview", "full"]
_SOURCE_MARKER_PATTERN = re.compile(r"【知识来源：(SRC-\d{3})】")
_SOURCE_MARKER_CANDIDATE_PATTERN = re.compile(r"【知识来源：([^】]*)】")
_VIKING_URI_PATTERN = re.compile(r"viking://[^\s\]）)】}>，。；;、]+", re.IGNORECASE)
_LEGACY_SOURCE_MARKER_PATTERN = re.compile(r"(?:\[来源\s*:|【来源\s*：)")

_CONTEXT_USAGE_INSTRUCTIONS = """
recent_messages 是系统构建的有界会话上下文：
- type=user/assistant 表示保留的原始消息；
- type=conversation_summary 且 compressed=true 表示对应历史轮次的可信压缩摘要。
可以使用它解析“它、这个、继续、上一个方案”等上下文指代，但不得把摘要内容当成知识库事实。
知识库系统范围只能依据当前 query 中明确出现的系统名称确定，不允许从 recent_messages 或长期记忆继承。
""".strip()

_ROLE_LABELS = {
    "product_manager": "产品经理",
    "developer": "开发",
    "new_employee": "新入职员工",
}
_ROLE_ALIASES = {
    "product_manager": "product_manager",
    "产品经理": "product_manager",
    "developer": "developer",
    "开发": "developer",
    "开发人员": "developer",
    "程序员": "developer",
    "new_employee": "new_employee",
    "新入职员工": "new_employee",
    "新员工": "new_employee",
}

_NODE_TITLES = {
    "prepare_request": "整理当前问题",
    "intent_analyzer": "分析问题意图",
    "role_clarification": "确认用户角色",
    "system_scope_determination": "确定知识库系统范围",
    "initial_find": "准备首次知识库检索",
    "query_rewrite": "改写检索问题",
    "grep_query_builder": "生成关键词检索条件",
    "retrieval_executor": "执行知识库检索",
    "result_fusion": "融合检索结果",
    "evidence_hydration": "补全证据内容",
    "evidence_grader": "评估证据覆盖情况",
    "question_clarification": "请求补充查询信息",
    "knowledge_not_found": "生成未检索到结果说明",
    "answer_generator": "生成答案草稿",
    "groundedness_verifier": "校验答案依据",
    "finalizer": "生成最终回答",
    "direct_answer": "生成直接回答",
}

NodeHandler = Callable[[GraphState], Awaitable[dict[str, Any]]]


def _json(value: Any, *, max_chars: int | None = None) -> str:
    """Serialize prompt data without leaking non-serializable objects."""
    text = json.dumps(value, ensure_ascii=False, default=str)
    return text if max_chars is None else text[:max_chars]


def _message_text(message: Any) -> str:
    """Extract plain text from LangChain or OpenAI-style messages."""
    if isinstance(message, BaseMessage):
        return extract_text_content(message.content)
    if isinstance(message, dict):
        content = message.get("content", "")
        return extract_text_content(content) if isinstance(content, (str, list)) else str(content)
    return str(message)


def _redact_viking_uris(value: Any) -> Any:
    """Remove backend-owned resource URIs from data sent to an LLM."""
    if isinstance(value, str):
        return _VIKING_URI_PATTERN.sub("[内部资源地址已隐藏]", value)
    if isinstance(value, list):
        return [_redact_viking_uris(item) for item in value]
    if isinstance(value, dict):
        return {
            key: _redact_viking_uris(item)
            for key, item in value.items()
            if key.lower() != "uri" and not key.lower().endswith("_uri")
        }
    return value


def _build_llm_evidence(
    evidence: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Replace real evidence URIs with stable request-local source IDs."""
    uri_to_source_id: dict[str, str] = {}
    source_uri_map: dict[str, str] = {}
    llm_evidence: list[dict[str, Any]] = []
    for item in evidence:
        uri = str(item.get("uri") or "").strip()
        if not uri:
            continue
        source_id = uri_to_source_id.get(uri)
        if source_id is None:
            source_id = f"SRC-{len(uri_to_source_id) + 1:03d}"
            uri_to_source_id[uri] = source_id
            source_uri_map[source_id] = uri
        safe_item = _redact_viking_uris(item)
        if not isinstance(safe_item, dict):
            continue
        llm_evidence.append({**safe_item, "source_id": source_id})
    return llm_evidence, source_uri_map


def _validate_source_markers(answer: str, source_uri_map: Mapping[str, str]) -> list[str]:
    """Validate that a model used only backend-issued source IDs."""
    if _VIKING_URI_PATTERN.search(answer):
        raise ValueError("模型输出中禁止包含 viking URI")
    if _LEGACY_SOURCE_MARKER_PATTERN.search(answer):
        raise ValueError("模型必须使用【知识来源：SRC-XXX】格式")

    marker_values = _SOURCE_MARKER_CANDIDATE_PATTERN.findall(answer)
    source_ids = _SOURCE_MARKER_PATTERN.findall(answer)
    if len(marker_values) != len(source_ids):
        raise ValueError("模型输出了格式错误的知识来源标记")
    if source_uri_map and not source_ids:
        raise ValueError("知识库回答缺少知识来源标记")

    unknown_ids = sorted({source_id for source_id in source_ids if source_id not in source_uri_map})
    if unknown_ids:
        raise ValueError(f"模型输出了未知知识来源 ID: {', '.join(unknown_ids)}")
    return source_ids


def _render_source_uris(answer: str, source_uri_map: Mapping[str, str]) -> str:
    """Render verified source IDs with backend-owned OpenViking URIs."""
    _validate_source_markers(answer, source_uri_map)
    return _SOURCE_MARKER_PATTERN.sub(
        lambda match: f"【知识来源：{source_uri_map[match.group(1)]}】",
        answer,
    )


def _is_user_message(message: Any) -> bool:
    """Return whether a state message is a user/human message."""
    if isinstance(message, HumanMessage):
        return True
    if isinstance(message, BaseMessage):
        return message.type == "human"
    return isinstance(message, dict) and message.get("role") == "user"


def _extract_explicit_role(text: str) -> str | None:
    """Extract only explicit user self-identification, not task keywords."""
    for alias, role in _ROLE_ALIASES.items():
        pattern = rf"(?:我是|我是一名|我的角色是|我的岗位是|作为一名?|我刚入职是)\s*{re.escape(alias)}"
        if re.search(pattern, text, re.IGNORECASE):
            return role
    return None


def _parse_role_answer(value: Any) -> str | None:
    """Normalize a HITL resume value to the supported role enum."""
    parsed_value = _parse_json_object(value)
    if parsed_value is not None:
        value = parsed_value
    if isinstance(value, dict):
        value = value.get("value") or value.get("role") or value.get("answer")
    text = str(value).strip()
    if text in _ROLE_ALIASES:
        return _ROLE_ALIASES[text]
    lowered = text.lower()
    return _ROLE_ALIASES.get(lowered)


def _parse_json_object(value: Any) -> dict[str, Any] | None:
    """Parse a frontend HITL payload without involving an LLM."""
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _is_allowed_uri(uri: str, allowed_roots: list[str]) -> bool:
    """Enforce that a planned URI stays under an authorized root."""
    value = uri.rstrip("/")
    for root in allowed_roots:
        normalized_root = root.rstrip("/")
        if value == normalized_root or value.startswith(f"{normalized_root}/"):
            return True
    return False


def _iter_uri_items(value: Any) -> list[dict[str, Any]]:
    """Recursively collect OpenViking result objects containing a URI."""
    items: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if isinstance(value.get("uri"), str):
            items.append(value)
        for nested in value.values():
            if isinstance(nested, (dict, list)):
                items.extend(_iter_uri_items(nested))
    elif isinstance(value, list):
        for nested in value:
            items.extend(_iter_uri_items(nested))
    return items


def _tool_result_documents(value: Any) -> list[dict[str, Any]]:
    """Return a compact, deduplicated document list safe for SSE metadata."""
    documents: list[dict[str, Any]] = []
    seen_uris: set[str] = set()
    for item in _iter_uri_items(value):
        uri = str(item["uri"])
        if not uri or uri in seen_uris:
            continue
        seen_uris.add(uri)
        document: dict[str, Any] = {"uri": uri}
        if "level" in item:
            document["level"] = item["level"]
        score = _numeric_score(item)
        if score is not None:
            document["score"] = score
        if isinstance(item.get("line"), int):
            document["line"] = item["line"]
        if item.get("isDir") is True:
            document["is_directory"] = True
        documents.append(document)
        if len(documents) >= MAX_TOOL_RESULT_DOCUMENTS:
            break
    return documents


def _root_system_candidates(value: Any) -> list[dict[str, str]]:
    """Extract only verified direct child directories from a root listing."""
    root_prefix = f"{ROOT_RESOURCES_URI}/"
    candidates_by_uri: dict[str, dict[str, str]] = {}
    for item in _iter_uri_items(value):
        if item.get("isDir") is not True:
            continue
        uri = str(item["uri"]).rstrip("/")
        if not uri.startswith(root_prefix):
            continue
        relative_path = uri.removeprefix(root_prefix)
        if not relative_path or "/" in relative_path:
            continue
        candidates_by_uri[uri] = {
            "name": relative_path,
            "uri": uri,
            "abstract": str(item.get("abstract") or "")[:500],
        }
    return list(candidates_by_uri.values())


def _normalize_source_level(value: Any) -> SourceLevel | None:
    """Normalize OpenViking find result levels to read API levels."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        numeric_levels: dict[int, SourceLevel] = {0: "abstract", 1: "overview", 2: "full"}
        return numeric_levels.get(value)
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    text_levels: dict[str, SourceLevel] = {
        "0": "abstract",
        "l0": "abstract",
        "abstract": "abstract",
        "1": "overview",
        "l1": "overview",
        "overview": "overview",
        "2": "full",
        "l2": "full",
        "full": "full",
    }
    return text_levels.get(normalized)


def _numeric_score(item: dict[str, Any]) -> float | None:
    """Return a real retrieval score without inventing one for list results."""
    value = item.get("score", item.get("similarity"))
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _source_level_rank(level: str | None) -> int:
    """Return the relative amount of content represented by a source level."""
    return {None: -1, "abstract": 0, "overview": 1, "full": 2}[level]


def _merge_candidates(current: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Merge duplicate URIs while retaining the strongest score and richest level."""
    current_score = current.get("score")
    candidate_score = candidate.get("score")
    prefer_candidate = candidate_score is not None and (current_score is None or candidate_score > current_score)
    merged = dict(candidate if prefer_candidate else current)
    levels = [current.get("source_level"), candidate.get("source_level")]
    merged["source_level"] = max(levels, key=_source_level_rank)
    merged["is_directory"] = bool(current.get("is_directory") or candidate.get("is_directory"))
    merged["task_ids"] = list(dict.fromkeys([*current["task_ids"], *candidate["task_ids"]]))
    merged["operations"] = list(dict.fromkeys([*current["operations"], *candidate["operations"]]))
    return merged


def _select_fused_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort semantic hits by score and keep unscored grep matches in source order."""
    return sorted(
        candidates,
        key=lambda item: (
            item["score"] is None,
            -(item["score"] or 0.0),
            item["task_order"],
            item["source_rank"],
        ),
    )[:MAX_HYDRATED_RESOURCES]


def _parse_answer_requirements(items: list[dict[str, Any] | str]) -> list[AnswerRequirement]:
    """Normalize current requirements and legacy string-only checkpoints."""
    requirements: list[AnswerRequirement] = []
    for index, item in enumerate(items, start=1):
        if isinstance(item, str):
            requirements.append(
                AnswerRequirement(
                    requirement_id=f"req_{index}",
                    description=item,
                    priority="required" if not requirements else "optional",
                    evidence_source="knowledge_base",
                )
            )
            continue
        requirements.append(AnswerRequirement.model_validate(item))

    # Old checkpoints can contain multiple string-only requirements.  Preserve
    # their content but do not let legacy detail become a multi-item retrieval
    # gate after upgrading the workflow.
    required_seen = 0
    for requirement in requirements:
        if requirement.priority != "required":
            continue
        required_seen += 1
        if required_seen > 1:
            requirement.priority = "optional"
    return requirements


def _next_retrieval_route(
    retrieval_stage: str,
    hitl_retry_used: bool,
) -> Literal[
    "query_rewrite",
    "grep_query_builder",
    "question_clarification",
    "knowledge_not_found",
]:
    """Return the single allowed next step in the fixed retrieval fallback chain."""
    if retrieval_stage == "initial_find":
        return "query_rewrite"
    if retrieval_stage == "rewritten_find":
        return "grep_query_builder"
    return "knowledge_not_found" if hitl_retry_used else "question_clarification"


class AgentLoop:
    """Build and execute the bounded Agentic RAG workflow."""

    def __init__(self, llm_service: LLMService, openviking_api: OpenVikingKnowledgeAPI):
        """Initialize node dependencies."""
        self.llm_service = llm_service
        self.openviking_api = openviking_api

    @staticmethod
    def _emit_tool_progress(
        tool_name: str,
        tool_kind: Literal["node", "openviking"],
        tool_status: Literal["started", "completed", "failed"],
        title: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Publish small, frontend-safe progress events in stream mode."""
        try:
            writer = get_stream_writer()
        except RuntimeError:
            # Nodes are also invoked directly by unit tests and maintenance
            # code, where no LangGraph stream context exists.
            return
        writer(
            {
                "type": "tool_progress",
                "tool_name": tool_name,
                "tool_kind": tool_kind,
                "tool_status": tool_status,
                "title": title,
                "metadata": metadata or {},
            }
        )

    def _with_node_progress(self, node_name: str, handler: NodeHandler) -> Any:
        """Wrap every graph node with start/completion progress notifications."""

        async def wrapped(state: GraphState) -> dict[str, Any]:
            title = _NODE_TITLES[node_name]
            self._emit_tool_progress(
                tool_name=f"node.{node_name}",
                tool_kind="node",
                tool_status="started",
                title=title,
                metadata={"node": node_name},
            )
            try:
                result = await handler(state)
            except GraphInterrupt:
                # Dynamic interrupts are an expected pause, not a failed node.
                # Let LangGraph persist and surface the pending interrupt.
                raise
            except Exception as exc:
                self._emit_tool_progress(
                    tool_name=f"node.{node_name}",
                    tool_kind="node",
                    tool_status="failed",
                    title=title,
                    metadata={"node": node_name, "error_type": type(exc).__name__},
                )
                raise
            self._emit_tool_progress(
                tool_name=f"node.{node_name}",
                tool_kind="node",
                tool_status="completed",
                title=title,
                metadata={"node": node_name},
            )
            return result

        return wrapped

    def configure(self, graph: StateGraph) -> None:
        """Register nodes and routes on a StateGraph."""
        graph.add_node("prepare_request", self._with_node_progress("prepare_request", self.prepare_request))
        graph.add_node("intent_analyzer", self._with_node_progress("intent_analyzer", self.intent_analyzer))
        graph.add_node("role_clarification", self._with_node_progress("role_clarification", self.role_clarification))
        graph.add_node(
            "system_scope_determination",
            self._with_node_progress("system_scope_determination", self.system_scope_determination),
        )
        graph.add_node("initial_find", self._with_node_progress("initial_find", self.initial_find))
        graph.add_node("query_rewrite", self._with_node_progress("query_rewrite", self.query_rewrite))
        graph.add_node("grep_query_builder", self._with_node_progress("grep_query_builder", self.grep_query_builder))
        graph.add_node("retrieval_executor", self._with_node_progress("retrieval_executor", self.retrieval_executor))
        graph.add_node("result_fusion", self._with_node_progress("result_fusion", self.result_fusion))
        graph.add_node("evidence_hydration", self._with_node_progress("evidence_hydration", self.evidence_hydration))
        graph.add_node("evidence_grader", self._with_node_progress("evidence_grader", self.evidence_grader))
        graph.add_node(
            "question_clarification",
            self._with_node_progress("question_clarification", self.question_clarification),
        )
        graph.add_node("knowledge_not_found", self._with_node_progress("knowledge_not_found", self.knowledge_not_found))
        graph.add_node("answer_generator", self._with_node_progress("answer_generator", self.answer_generator))
        graph.add_node(
            "groundedness_verifier",
            self._with_node_progress("groundedness_verifier", self.groundedness_verifier),
        )
        graph.add_node("finalizer", self._with_node_progress("finalizer", self.finalizer))
        graph.add_node("direct_answer", self._with_node_progress("direct_answer", self.direct_answer))

        graph.add_edge(START, "prepare_request")
        graph.add_edge("prepare_request", "intent_analyzer")
        graph.add_conditional_edges(
            "intent_analyzer",
            self.route_after_intent,
            {
                "role_clarification": "role_clarification",
                "system_scope_determination": "system_scope_determination",
                "initial_find": "initial_find",
                "direct_answer": "direct_answer",
            },
        )
        graph.add_conditional_edges(
            "role_clarification",
            self.route_after_role,
            {
                "system_scope_determination": "system_scope_determination",
                "initial_find": "initial_find",
                "direct_answer": "direct_answer",
            },
        )
        graph.add_edge("system_scope_determination", "initial_find")
        graph.add_edge("initial_find", "retrieval_executor")
        graph.add_edge("query_rewrite", "retrieval_executor")
        graph.add_edge("grep_query_builder", "retrieval_executor")
        graph.add_edge("retrieval_executor", "result_fusion")
        graph.add_edge("result_fusion", "evidence_hydration")
        graph.add_edge("evidence_hydration", "evidence_grader")
        graph.add_conditional_edges(
            "evidence_grader",
            self.route_after_evidence,
            {
                "answer_generator": "answer_generator",
                "query_rewrite": "query_rewrite",
                "grep_query_builder": "grep_query_builder",
                "question_clarification": "question_clarification",
                "knowledge_not_found": "knowledge_not_found",
            },
        )
        graph.add_edge("question_clarification", "intent_analyzer")
        graph.add_edge("answer_generator", "groundedness_verifier")
        graph.add_conditional_edges(
            "groundedness_verifier",
            self.route_after_verification,
            {
                "finalizer": "finalizer",
                "answer_generator": "answer_generator",
                "query_rewrite": "query_rewrite",
                "grep_query_builder": "grep_query_builder",
                "question_clarification": "question_clarification",
                "knowledge_not_found": "knowledge_not_found",
            },
        )
        graph.add_edge("finalizer", END)
        graph.add_edge("direct_answer", END)
        graph.add_edge("knowledge_not_found", END)

    async def prepare_request(self, state: GraphState) -> dict[str, Any]:
        """Extract the latest user query and reset per-request loop fields."""
        user_query = next(
            (_message_text(message) for message in reversed(state.messages) if _is_user_message(message)),
            "",
        ).strip()
        if not user_query:
            raise ValueError("未找到可处理的用户消息")

        return {
            "user_query": user_query,
            "normalized_query": user_query,
            "needs_role_clarification": False,
            "intent": "",
            "needs_retrieval": True,
            "entities": [],
            "constraints": [],
            "answer_requirements": [],
            "allowed_target_uris": [ROOT_RESOURCES_URI],
            "system_name": "",
            "system_scope_explicit": False,
            "system_options": [],
            "scope_determination_completed": False,
            "hitl_retry_used": False,
            "retrieval_tasks": [],
            "executed_queries": [],
            "executed_operations": [],
            "retrieval_round": 0,
            "retrieval_stage": "initial_find",
            "active_retrieval_query": user_query,
            "grep_pattern": "",
            "raw_results": [],
            "candidate_uris": [],
            "candidate_items": [],
            "hydrated_evidence": [],
            "retrieval_errors": [],
            "selected_evidence": [],
            "covered_required_ids": [],
            "missing_required_ids": [],
            "covered_optional_ids": [],
            "missing_optional_ids": [],
            "draft_answer": "",
            "revision_instructions": "",
            "revision_count": 0,
            "final_answer": "",
            "route": "intent_analyzer",
        }

    async def intent_analyzer(self, state: GraphState) -> dict[str, Any]:
        """Analyze intent and perform evidence-based role classification."""
        payload = {
            "query": state.normalized_query,
            "confirmed_role": state.user_role,
            "recent_messages": state.conversation_context,
            "long_term_memory": state.long_term_memory,
        }
        analysis = await self.llm_service.call(
            [
                SystemMessage(
                    content=f"{get_agentic_rag_prompt('intent_analyzer')}\n\n{_CONTEXT_USAGE_INSTRUCTIONS}"
                ),
                HumanMessage(content=_json(payload)),
            ],
            response_format=IntentAnalysis,
        )

        role = state.user_role
        role_source = state.role_source
        confidence = state.role_confidence
        evidence = list(state.role_evidence)

        explicit_role = _extract_explicit_role(state.normalized_query)
        if role is None and explicit_role:
            role = explicit_role
            role_source = "explicit"
            confidence = 1.0
            evidence = [f"用户明确自述角色为{_ROLE_LABELS[explicit_role]}"]
        elif role is None and analysis.user_role is not None:
            inferred_is_sufficient = (
                analysis.role_source == "inferred"
                and analysis.role_confidence >= 0.85
                and len(analysis.role_evidence) >= 2
            )
            if inferred_is_sufficient:
                role = analysis.user_role
                role_source = "inferred"
                confidence = analysis.role_confidence
                evidence = analysis.role_evidence

        needs_role_clarification = analysis.needs_retrieval and role is None
        logger.info(
            "agent_intent_analyzed",
            intent=analysis.intent,
            needs_retrieval=analysis.needs_retrieval,
            user_role=role,
            needs_role_clarification=needs_role_clarification,
        )
        return {
            "intent": analysis.intent,
            "needs_retrieval": analysis.needs_retrieval,
            "user_role": role,
            "role_source": role_source,
            "role_confidence": confidence,
            "role_evidence": evidence,
            "needs_role_clarification": needs_role_clarification,
            "entities": analysis.entities,
            "constraints": analysis.constraints,
            "answer_requirements": [item.model_dump() for item in analysis.answer_requirements],
            "route": (
                "role_clarification"
                if needs_role_clarification
                else "initial_find"
                if analysis.needs_retrieval
                else "direct_answer"
            ),
        }

    async def role_clarification(self, state: GraphState) -> dict[str, Any]:
        """Interrupt until the user selects one supported role."""
        prompt: dict[str, Any] = {
            "type": "role_clarification",
            "input_type": "single_choice",
            "field": "role",
            "required": True,
            "question": "为了按合适的视角改写问题并检索知识库，请问您当前的角色是：产品经理、开发，还是新入职员工？",
            "options": [{"label": label, "value": value} for value, label in _ROLE_LABELS.items()],
        }
        role: str | None = None
        raw_answer: Any = None
        while role is None:
            raw_answer = interrupt(prompt)
            role = _parse_role_answer(raw_answer)
            if role is None:
                prompt = {**prompt, "error": "请选择产品经理、开发或新入职员工之一。"}

        return {
            "messages": [HumanMessage(content=f"我的角色是{_ROLE_LABELS[role]}")],
            "user_role": role,
            "role_source": "hitl",
            "role_confidence": 1.0,
            "role_evidence": ["用户通过角色澄清确认"],
            "needs_role_clarification": False,
            "route": (
                "initial_find"
                if state.needs_retrieval and state.scope_determination_completed
                else "system_scope_determination"
                if state.needs_retrieval
                else "direct_answer"
            ),
        }

    async def system_scope_determination(self, state: GraphState) -> dict[str, Any]:
        """List real root systems and let the LLM select one verified candidate."""
        self._emit_tool_progress(
            tool_name="openviking.list_resources",
            tool_kind="openviking",
            tool_status="started",
            title="读取知识库系统目录",
            metadata={"target_uri": ROOT_RESOURCES_URI, "recursive": False},
        )
        try:
            root_listing = await self.openviking_api.list_resources(
                ROOT_RESOURCES_URI,
                recursive=False,
                node_limit=100,
            )
        except Exception as exc:
            self._emit_tool_progress(
                tool_name="openviking.list_resources",
                tool_kind="openviking",
                tool_status="failed",
                title="读取知识库系统目录失败",
                metadata={"target_uri": ROOT_RESOURCES_URI, "error_type": type(exc).__name__},
            )
            logger.exception("openviking_system_scope_listing_failed")
            return {
                "allowed_target_uris": [ROOT_RESOURCES_URI],
                "system_name": "",
                "system_scope_explicit": False,
                "system_options": [],
                "scope_determination_completed": True,
                "route": "initial_find",
            }

        candidates = _root_system_candidates(root_listing)
        self._emit_tool_progress(
            tool_name="openviking.list_resources",
            tool_kind="openviking",
            tool_status="completed",
            title="已读取知识库系统目录",
            metadata={
                "target_uri": ROOT_RESOURCES_URI,
                "result_count": len(candidates),
                "documents": [
                    {"uri": candidate["uri"], "name": candidate["name"], "is_directory": True}
                    for candidate in candidates[:MAX_TOOL_RESULT_DOCUMENTS]
                ],
            },
        )
        system_options = [{"label": candidate["name"], "value": candidate["uri"]} for candidate in candidates]
        if not candidates:
            logger.warning("openviking_system_scope_candidates_empty")
            return {
                "allowed_target_uris": [ROOT_RESOURCES_URI],
                "system_name": "",
                "system_scope_explicit": False,
                "system_options": [],
                "scope_determination_completed": True,
                "route": "initial_find",
            }

        scope = await self.llm_service.call(
            [
                SystemMessage(content=get_agentic_rag_prompt("system_scope_determination")),
                HumanMessage(
                    content=_json(
                        {
                            "query": state.user_query,
                            "system_candidates": candidates,
                        }
                    )
                ),
            ],
            response_format=SystemScopeResult,
        )
        candidates_by_uri = {item["uri"]: item for item in candidates}
        selected_uri = (scope.selected_uri or "").strip().rstrip("/")
        selected_candidate = candidates_by_uri.get(selected_uri) if scope.scope_confident else None
        if scope.scope_confident and selected_candidate is None:
            logger.warning(
                "invalid_system_scope_ignored",
                proposed_uri=scope.selected_uri,
                candidate_count=len(candidates),
            )
        return {
            "normalized_query": (
                scope.scoped_query.strip() if selected_candidate is not None else state.normalized_query
            ),
            "allowed_target_uris": [
                selected_candidate["uri"] if selected_candidate is not None else ROOT_RESOURCES_URI
            ],
            "system_name": (selected_candidate["name"] if selected_candidate is not None else ""),
            "system_scope_explicit": selected_candidate is not None,
            "system_options": system_options,
            "scope_determination_completed": True,
            "route": "initial_find",
        }

    async def initial_find(self, state: GraphState) -> dict[str, Any]:
        """Create the fixed first-round semantic find task."""
        query = state.normalized_query.strip()
        task = RetrievalTask(
            task_id="initial_find",
            purpose="使用用户原始问题进行首轮语义检索",
            operation="find",
            information_need=query,
            target_uri=state.allowed_target_uris[0],
            query=query,
            limit=MAX_RESULTS_PER_TASK,
            hydration_level="full",
        )
        return {
            "retrieval_tasks": [task.model_dump()],
            "retrieval_stage": "initial_find",
            "active_retrieval_query": query,
            "grep_pattern": "",
            "raw_results": [],
            "candidate_uris": [],
            "candidate_items": [],
            "route": "retrieval_executor",
        }

    async def query_rewrite(self, state: GraphState) -> dict[str, Any]:
        """Rewrite the failed initial query for exactly one semantic retry."""
        requirements = _parse_answer_requirements(state.answer_requirements)
        missing_required_ids = set(state.missing_required_ids)
        rewrite = await self.llm_service.call(
            [
                SystemMessage(content=get_agentic_rag_prompt("query_rewrite")),
                HumanMessage(
                    content=_json(
                        {
                            "query": state.normalized_query,
                            "role": state.user_role,
                            "tasks": [
                                {
                                    "task_id": "rewritten_find",
                                    "operation": "find",
                                    "information_need": state.normalized_query,
                                }
                            ],
                            "executed_queries": state.executed_queries,
                            "missing_requirements": [
                                item.model_dump()
                                for item in requirements
                                if item.requirement_id in missing_required_ids
                            ],
                        }
                    )
                ),
            ],
            response_format=QueryRewriteResult,
        )
        query = next(
            (item.query.strip() for item in rewrite.queries if item.query.strip()),
            state.normalized_query,
        )
        task = RetrievalTask(
            task_id="rewritten_find",
            purpose="使用改写后的问题进行一次语义检索重试",
            operation="find",
            information_need=state.normalized_query,
            target_uri=state.allowed_target_uris[0],
            query=query,
            limit=MAX_RESULTS_PER_TASK,
            hydration_level="full",
        )
        return {
            "retrieval_tasks": [task.model_dump()],
            "retrieval_stage": "rewritten_find",
            "active_retrieval_query": query,
            "raw_results": [],
            "candidate_uris": [],
            "candidate_items": [],
            "route": "retrieval_executor",
        }

    async def grep_query_builder(self, state: GraphState) -> dict[str, Any]:
        """Build one escaped keyword regex for the final retrieval round."""
        requirements = _parse_answer_requirements(state.answer_requirements)
        missing_required_ids = set(state.missing_required_ids)
        keyword_result = await self.llm_service.call(
            [
                SystemMessage(content=get_agentic_rag_prompt("grep_query_builder")),
                HumanMessage(
                    content=_json(
                        {
                            "query": state.normalized_query,
                            "rewritten_query": state.active_retrieval_query,
                            "entities": state.entities,
                            "executed_queries": state.executed_queries,
                            "missing_requirements": [
                                item.model_dump()
                                for item in requirements
                                if item.requirement_id in missing_required_ids
                            ],
                        }
                    )
                ),
            ],
            response_format=GrepKeywordResult,
        )
        keywords = list(dict.fromkeys(keyword.strip() for keyword in keyword_result.keywords if keyword.strip()))
        pattern = "|".join(re.escape(keyword) for keyword in keywords)
        task = RetrievalTask(
            task_id="grep_keywords",
            purpose="使用关键词精确匹配知识库正文",
            operation="grep",
            information_need="；".join(keywords),
            target_uri=state.allowed_target_uris[0],
            query=pattern,
            node_limit=MAX_HYDRATED_RESOURCES,
            hydration_level="full",
        )
        return {
            "retrieval_tasks": [task.model_dump()],
            "retrieval_stage": "grep",
            "grep_pattern": pattern,
            "raw_results": [],
            "candidate_uris": [],
            "candidate_items": [],
            "route": "retrieval_executor",
        }

    async def retrieval_executor(self, state: GraphState) -> dict[str, Any]:
        """Execute only the planned read-only OpenViking operations."""
        semaphore = asyncio.Semaphore(MAX_PARALLEL_TASKS)
        tasks = [RetrievalTask.model_validate(task) for task in state.retrieval_tasks]

        async def execute(task: RetrievalTask) -> dict[str, Any]:
            async with semaphore:
                operation_name = f"openviking.{task.operation}"
                self._emit_tool_progress(
                    tool_name=operation_name,
                    tool_kind="openviking",
                    tool_status="started",
                    title=f"正在执行知识库 {task.operation} 检索",
                    metadata={
                        "task_id": task.task_id,
                        "query": task.query[:500],
                        "target_uri": task.target_uri,
                    },
                )
                try:
                    if task.operation == "find":
                        result = await self.openviking_api.find(task.query, task.target_uri, task.limit)
                    else:
                        result = await self.openviking_api.grep(
                            task.query,
                            task.target_uri,
                            case_insensitive=True,
                            node_limit=task.node_limit,
                            level_limit=10,
                        )
                    self._emit_tool_progress(
                        tool_name=operation_name,
                        tool_kind="openviking",
                        tool_status="completed",
                        title=f"知识库 {task.operation} 检索完成",
                        metadata={
                            "task_id": task.task_id,
                            "query": task.query[:500],
                            "target_uri": task.target_uri,
                            "result_count": len(_iter_uri_items(result)),
                            "documents": _tool_result_documents(result),
                        },
                    )
                    return {
                        "ok": True,
                        "task_id": task.task_id,
                        "operation": task.operation,
                        "target_uri": task.target_uri,
                        "result": result,
                    }
                except Exception as exc:
                    self._emit_tool_progress(
                        tool_name=operation_name,
                        tool_kind="openviking",
                        tool_status="failed",
                        title=f"知识库 {task.operation} 检索失败",
                        metadata={
                            "task_id": task.task_id,
                            "query": task.query[:500],
                            "target_uri": task.target_uri,
                            "error_type": type(exc).__name__,
                        },
                    )
                    logger.warning(
                        "openviking_retrieval_operation_failed",
                        task_id=task.task_id,
                        operation=task.operation,
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )
                    return {
                        "ok": False,
                        "task_id": task.task_id,
                        "operation": task.operation,
                        "target_uri": task.target_uri,
                        "error": str(exc),
                    }

        raw_results = list(await asyncio.gather(*(execute(task) for task in tasks)))
        queries = [task.query for task in tasks if task.query and task.query not in state.executed_queries]
        operations = [
            {
                "round": state.retrieval_round + 1,
                "task_id": result["task_id"],
                "operation": result["operation"],
                "target_uri": result["target_uri"],
                "ok": result["ok"],
            }
            for result in raw_results
        ]
        return {
            "raw_results": raw_results,
            "retrieval_errors": [
                *state.retrieval_errors,
                *(result for result in raw_results if not result["ok"]),
            ],
            "executed_queries": [*state.executed_queries, *queries],
            "executed_operations": [*state.executed_operations, *operations],
            "retrieval_round": state.retrieval_round + 1,
            "route": "result_fusion",
        }

    async def result_fusion(self, state: GraphState) -> dict[str, Any]:
        """Normalize, deduplicate, and rank heterogeneous OpenViking results."""
        by_uri: dict[str, dict[str, Any]] = {}
        for task_order, raw in enumerate(state.raw_results):
            if not raw.get("ok"):
                continue
            operation = str(raw["operation"])
            for source_rank, item in enumerate(_iter_uri_items(raw.get("result"))):
                uri = str(item["uri"])
                if not _is_allowed_uri(uri, state.allowed_target_uris):
                    continue
                score = _numeric_score(item) if operation == "find" else None
                candidate = {
                    "uri": uri,
                    "score": score,
                    "task_id": raw["task_id"],
                    "task_ids": [raw["task_id"]],
                    "operation": operation,
                    "operations": [operation],
                    "source_level": (
                        _normalize_source_level(item.get("level"))
                        if operation == "find"
                        else "full"
                        if operation == "grep"
                        else None
                    ),
                    "is_directory": False,
                    "task_order": task_order,
                    "source_rank": source_rank,
                    "metadata": item,
                }
                current = by_uri.get(uri)
                by_uri[uri] = candidate if current is None else _merge_candidates(current, candidate)

        candidates = _select_fused_candidates(list(by_uri.values()))
        return {
            "candidate_uris": [candidate["uri"] for candidate in candidates],
            "candidate_items": candidates,
            "route": "evidence_hydration",
        }

    async def evidence_hydration(self, state: GraphState) -> dict[str, Any]:
        """Read L0/L1 first and a bounded number of L2 documents."""
        semaphore = asyncio.Semaphore(MAX_PARALLEL_TASKS)
        task_by_id = {
            task.task_id: task for task in (RetrievalTask.model_validate(item) for item in state.retrieval_tasks)
        }

        async def hydrate(index: int, candidate: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
            evidence: list[dict[str, Any]] = []
            errors: list[dict[str, Any]] = []
            uri = candidate["uri"]
            metadata = candidate.get("metadata", {})
            inline_abstract = metadata.get("abstract") if isinstance(metadata, dict) else None
            if inline_abstract:
                evidence.append(
                    {
                        "uri": uri,
                        "level": "abstract",
                        "content": str(inline_abstract),
                        "task_id": candidate["task_id"],
                    }
                )
            if candidate.get("operation") == "grep" and isinstance(metadata, dict):
                match_content = metadata.get("content")
                if match_content:
                    evidence.append(
                        {
                            "uri": uri,
                            "level": "grep_match",
                            "content": str(match_content),
                            "line": metadata.get("line"),
                            "task_id": candidate["task_id"],
                        }
                    )

            task = task_by_id.get(candidate["task_id"])
            operation = candidate.get("operation")
            source_level = _normalize_source_level(candidate.get("source_level"))
            if source_level is None and operation == "find" and isinstance(metadata, dict):
                source_level = _normalize_source_level(metadata.get("level"))
            first_level: SourceLevel = source_level or "abstract"
            async with semaphore:
                try:
                    first_content = await self.openviking_api.read(uri, first_level)
                    evidence.append(
                        {
                            "uri": uri,
                            "level": first_level,
                            "content": _json(first_content, max_chars=6000),
                            "task_id": candidate["task_id"],
                        }
                    )
                    wants_full = task is not None and task.hydration_level == "full"
                    if first_level != "full" and wants_full and index < MAX_FULL_CONTENT_RESOURCES:
                        full_content = await self.openviking_api.read(uri, "full")
                        evidence.append(
                            {
                                "uri": uri,
                                "level": "full",
                                "content": _json(full_content, max_chars=10_000),
                                "task_id": candidate["task_id"],
                            }
                        )
                except Exception as exc:
                    status: Any = None
                    try:
                        status = await self.openviking_api.stat(uri)
                    except Exception:
                        status = None
                    errors.append(
                        {
                            "operation": "read",
                            "uri": uri,
                            "error": str(exc),
                            "status": status,
                        }
                    )
            return evidence, errors

        hydrated = list(
            await asyncio.gather(*(hydrate(index, candidate) for index, candidate in enumerate(state.candidate_items)))
        )
        new_evidence = [item for group, _ in hydrated for item in group]
        errors = [item for _, group in hydrated for item in group]
        evidence_by_key = {
            (item.get("uri"), item.get("level")): item for item in [*state.selected_evidence, *new_evidence]
        }
        evidence = list(evidence_by_key.values())
        return {
            "hydrated_evidence": evidence,
            "selected_evidence": evidence,
            "retrieval_errors": [*state.retrieval_errors, *errors],
            "route": "evidence_grader",
        }

    async def evidence_grader(self, state: GraphState) -> dict[str, Any]:
        """Assess whether retrieved evidence covers every answer requirement."""
        requirements = _parse_answer_requirements(state.answer_requirements)
        required_ids = [item.requirement_id for item in requirements if item.priority == "required"]
        optional_ids = [item.requirement_id for item in requirements if item.priority == "optional"]
        if not state.selected_evidence:
            return {
                "covered_required_ids": [],
                "missing_required_ids": required_ids,
                "covered_optional_ids": [],
                "missing_optional_ids": optional_ids,
                "route": (
                    _next_retrieval_route(state.retrieval_stage, state.hitl_retry_used)
                    if required_ids
                    else "answer_generator"
                ),
            }

        llm_evidence, _ = _build_llm_evidence(state.selected_evidence)
        assessment = await self.llm_service.call(
            [
                SystemMessage(content=get_agentic_rag_prompt("evidence_grader")),
                HumanMessage(
                    content=_json(
                        {
                            "query": state.normalized_query,
                            "user_context": {
                                "role": state.user_role,
                                "role_source": state.role_source,
                                "role_confidence": state.role_confidence,
                                "role_evidence": state.role_evidence,
                            },
                            "answer_requirements": [item.model_dump() for item in requirements],
                            "evidence": llm_evidence,
                        },
                        max_chars=MAX_EVIDENCE_CHARS,
                    )
                ),
            ],
            response_format=EvidenceAssessment,
        )
        covered_required_id_set = set(assessment.covered_required_ids)
        covered_optional_id_set = set(assessment.covered_optional_ids)
        covered_required_ids = [item for item in required_ids if item in covered_required_id_set]
        missing_required_ids = [item for item in required_ids if item not in covered_required_id_set]
        covered_optional_ids = [item for item in optional_ids if item in covered_optional_id_set]
        missing_optional_ids = [item for item in optional_ids if item not in covered_optional_id_set]
        required_sufficient = not missing_required_ids
        if assessment.required_sufficient != required_sufficient:
            logger.warning(
                "evidence_assessment_consistency_corrected",
                model_required_sufficient=assessment.required_sufficient,
                computed_required_sufficient=required_sufficient,
                missing_required_ids=missing_required_ids,
            )

        if required_sufficient:
            route = "answer_generator"
        else:
            route = _next_retrieval_route(state.retrieval_stage, state.hitl_retry_used)
        return {
            "covered_required_ids": covered_required_ids,
            "missing_required_ids": missing_required_ids,
            "covered_optional_ids": covered_optional_ids,
            "missing_optional_ids": missing_optional_ids,
            "route": route,
        }

    async def question_clarification(self, state: GraphState) -> dict[str, Any]:
        """Interrupt after exhausted retrieval and collect missing user context."""
        requirements = _parse_answer_requirements(state.answer_requirements)
        missing_id_set = set(state.missing_required_ids)
        missing_descriptions = [
            item.description
            for item in requirements
            if item.priority == "required" and item.requirement_id in missing_id_set
        ]
        if state.system_scope_explicit:
            question = (
                f"在“{state.system_name}”系统知识库中，现有问题和证据仍不足。"
                "请进一步明确提问对象、版本、场景、现象和期望结果。"
            )
        else:
            question = (
                "现有问题和知识库证据仍不足。请声明需要查询哪个系统的知识库，"
                "并明确具体问题、对象、版本、场景或期望结果。"
            )
        requires_system = not state.system_scope_explicit and bool(state.system_options)
        prompt: dict[str, Any] = {
            "type": "question_clarification",
            "input_type": "system_and_question",
            "question": question,
            "original_query": state.user_query,
            "current_system": state.system_name or None,
            "requires_system": requires_system,
            "system_field": "system_uri",
            "question_field": "question",
            "system_options": state.system_options,
            "missing_information": missing_descriptions,
        }
        supplement = ""
        selected_uri = (
            state.allowed_target_uris[0] if state.system_scope_explicit and state.allowed_target_uris else ""
        )
        selected_system_name = state.system_name
        options_by_uri = {
            str(option.get("value", "")).rstrip("/"): option for option in state.system_options if option.get("value")
        }
        while not supplement:
            raw_answer = interrupt(prompt)
            payload = _parse_json_object(raw_answer)
            submitted_uri = ""
            if payload is not None:
                submitted_uri = str(payload.get("system_uri") or "").strip().rstrip("/")
                supplement = str(
                    payload.get("question") or payload.get("answer") or payload.get("content") or ""
                ).strip()
            elif not requires_system:
                supplement = str(raw_answer).strip()

            if submitted_uri:
                selected_option = options_by_uri.get(submitted_uri)
                if selected_option is None:
                    supplement = ""
                    prompt = {**prompt, "error": "请选择 system_options 中提供的系统。"}
                    continue
                selected_uri = submitted_uri
                selected_system_name = str(selected_option.get("label") or "")
            elif requires_system:
                supplement = ""
                prompt = {
                    **prompt,
                    "error": "请选择 system_options 中的系统，并补充具体问题。",
                }
                continue

            if not supplement:
                prompt = {**prompt, "error": "补充问题不能为空。"}

        system_context = f"用户选择系统：{selected_system_name}\n" if selected_system_name else ""
        combined_query = (f"{state.user_query}\n{system_context}用户补充：{supplement}").strip()
        selected_scope_explicit = bool(selected_uri and selected_system_name)
        return {
            "messages": [
                HumanMessage(
                    content=(f"选择系统：{selected_system_name}\n" if selected_system_name else "")
                    + f"补充信息：{supplement}"
                )
            ],
            "user_query": combined_query,
            "normalized_query": combined_query,
            "allowed_target_uris": [selected_uri if selected_scope_explicit else ROOT_RESOURCES_URI],
            "system_name": selected_system_name if selected_scope_explicit else "",
            "system_scope_explicit": selected_scope_explicit,
            "scope_determination_completed": True,
            "hitl_retry_used": True,
            "retrieval_tasks": [],
            "executed_queries": [],
            "executed_operations": [],
            "retrieval_round": 0,
            "retrieval_stage": "initial_find",
            "active_retrieval_query": combined_query,
            "grep_pattern": "",
            "raw_results": [],
            "candidate_uris": [],
            "candidate_items": [],
            "hydrated_evidence": [],
            "selected_evidence": [],
            "covered_required_ids": [],
            "missing_required_ids": [],
            "covered_optional_ids": [],
            "missing_optional_ids": [],
            "revision_instructions": "",
            "revision_count": 0,
            "route": "intent_analyzer",
        }

    async def knowledge_not_found(self, state: GraphState) -> dict[str, Any]:
        """Finish after the single post-HITL retrieval cycle is exhausted."""
        if state.system_scope_explicit:
            answer = f"在“{state.system_name}”系统知识库中检索不到足以回答该问题的信息。"
        else:
            answer = "在知识库中检索不到足以回答该问题的信息。"
        return {
            "draft_answer": answer,
            "final_answer": answer,
            "messages": [AIMessage(content=answer)],
            "route": "knowledge_not_found",
        }

    async def answer_generator(self, state: GraphState) -> dict[str, Any]:
        """Generate or revise a cited answer using selected evidence only."""
        requirements = _parse_answer_requirements(state.answer_requirements)
        llm_evidence, _ = _build_llm_evidence(state.selected_evidence)
        response = await self.llm_service.call(
            [
                SystemMessage(content=get_agentic_rag_prompt("answer_generator")),
                HumanMessage(
                    content=_json(
                        {
                            "query": state.normalized_query,
                            "role": state.user_role,
                            "answer_requirements": [item.model_dump() for item in requirements],
                            "missing_optional_ids": state.missing_optional_ids,
                            "evidence": llm_evidence,
                            "revision_instructions": state.revision_instructions,
                        },
                        max_chars=MAX_EVIDENCE_CHARS,
                    )
                ),
            ]
        )
        answer = _message_text(response).strip()
        return {"draft_answer": answer, "route": "groundedness_verifier"}

    async def groundedness_verifier(self, state: GraphState) -> dict[str, Any]:
        """Verify evidence support and required-answer coverage."""
        requirements = _parse_answer_requirements(state.answer_requirements)
        required_id_set = {item.requirement_id for item in requirements if item.priority == "required"}
        optional_id_set = {item.requirement_id for item in requirements if item.priority == "optional"}
        llm_evidence, source_uri_map = _build_llm_evidence(state.selected_evidence)
        try:
            _validate_source_markers(state.draft_answer, source_uri_map)
        except ValueError as exc:
            logger.warning(
                "answer_source_marker_validation_failed",
                error=str(exc),
                revision_count=state.revision_count,
            )
            retry_allowed = state.revision_count < 1
            return {
                "revision_instructions": str(exc),
                "revision_count": state.revision_count + (1 if retry_allowed else 0),
                "route": "answer_generator" if retry_allowed else "knowledge_not_found",
            }
        assessment = await self.llm_service.call(
            [
                SystemMessage(content=get_agentic_rag_prompt("groundedness_verifier")),
                HumanMessage(
                    content=_json(
                        {
                            "query": state.normalized_query,
                            "user_context": {
                                "role": state.user_role,
                                "role_source": state.role_source,
                            },
                            "answer_requirements": [item.model_dump() for item in requirements],
                            "draft_answer": state.draft_answer,
                            "evidence": llm_evidence,
                        },
                        max_chars=MAX_EVIDENCE_CHARS,
                    )
                ),
            ],
            response_format=GroundednessAssessment,
        )

        missing_required_ids = [item for item in assessment.missing_required_ids if item in required_id_set]
        missing_optional_ids = [item for item in assessment.missing_optional_ids if item in optional_id_set]
        has_required_gap = bool(missing_required_ids)
        if assessment.passed and assessment.action == "pass" and not has_required_gap:
            route = "finalizer"
        elif has_required_gap and assessment.action == "retrieve":
            route = _next_retrieval_route(state.retrieval_stage, state.hitl_retry_used)
        elif state.revision_count < 1:
            route = "answer_generator"
        elif not has_required_gap and not assessment.unsupported_claims:
            route = "finalizer"
        else:
            route = "knowledge_not_found" if state.hitl_retry_used else "question_clarification"

        instructions = assessment.revision_instructions
        return {
            "missing_required_ids": missing_required_ids or state.missing_required_ids,
            "missing_optional_ids": missing_optional_ids or state.missing_optional_ids,
            "revision_instructions": instructions,
            "revision_count": state.revision_count + (1 if route == "answer_generator" else 0),
            "route": route,
        }

    async def finalizer(self, state: GraphState) -> dict[str, Any]:
        """Buffer the model response, then publish backend-rendered source URIs."""
        writer = get_stream_writer()
        llm_evidence, source_uri_map = _build_llm_evidence(state.selected_evidence)
        model_chunks: list[str] = []
        async for chunk in self.llm_service.stream(
            [
                SystemMessage(content=get_agentic_rag_prompt("final_answer_generator")),
                HumanMessage(
                    content=_json(
                        {
                            "query": state.normalized_query,
                            "role": state.user_role,
                            "verified_draft": state.draft_answer,
                            "evidence": llm_evidence,
                        },
                        max_chars=MAX_EVIDENCE_CHARS,
                    )
                ),
            ]
        ):
            model_chunks.append(chunk)
        model_answer = "".join(model_chunks).strip() or state.draft_answer
        try:
            final_answer = _render_source_uris(model_answer, source_uri_map)
        except ValueError:
            logger.warning("final_answer_source_marker_invalid_using_verified_draft")
            final_answer = _render_source_uris(state.draft_answer, source_uri_map)
        for index in range(0, len(final_answer), 256):
            writer({"type": "final_answer_chunk", "content": final_answer[index : index + 256]})
        return {
            "final_answer": final_answer,
            "messages": [AIMessage(content=final_answer)],
            "route": "completed",
        }

    async def direct_answer(self, state: GraphState) -> dict[str, Any]:
        """Stream a conversational answer without pretending to retrieve."""
        writer = get_stream_writer()
        chunks: list[str] = []
        async for chunk in self.llm_service.stream(
            [
                SystemMessage(
                    content=f"{get_agentic_rag_prompt('direct_answer')}\n\n{_CONTEXT_USAGE_INSTRUCTIONS}"
                ),
                HumanMessage(
                    content=_json(
                        {
                            "query": state.normalized_query,
                            "role": state.user_role,
                            "recent_messages": state.conversation_context,
                            "long_term_memory": state.long_term_memory,
                        }
                    )
                ),
            ]
        ):
            chunks.append(chunk)
            writer({"type": "final_answer_chunk", "content": chunk})
        answer = "".join(chunks).strip()
        if not answer:
            raise RuntimeError("直接回答模型未返回内容")
        return {
            "draft_answer": answer,
            "final_answer": answer,
            "messages": [AIMessage(content=answer)],
            "route": "completed",
        }

    @staticmethod
    def route_after_intent(
        state: GraphState,
    ) -> Literal[
        "role_clarification",
        "system_scope_determination",
        "initial_find",
        "direct_answer",
    ]:
        """Route after intent and role analysis."""
        if state.needs_role_clarification:
            return "role_clarification"
        if not state.needs_retrieval:
            return "direct_answer"
        return "initial_find" if state.scope_determination_completed else "system_scope_determination"

    @staticmethod
    def route_after_role(
        state: GraphState,
    ) -> Literal["system_scope_determination", "initial_find", "direct_answer"]:
        """Continue the original request after HITL role confirmation."""
        if not state.needs_retrieval:
            return "direct_answer"
        return "initial_find" if state.scope_determination_completed else "system_scope_determination"

    @staticmethod
    def route_after_evidence(
        state: GraphState,
    ) -> Literal[
        "answer_generator",
        "query_rewrite",
        "grep_query_builder",
        "question_clarification",
        "knowledge_not_found",
    ]:
        """Route according to evidence coverage and retry budget."""
        return state.route  # type: ignore[return-value]

    @staticmethod
    def route_after_verification(
        state: GraphState,
    ) -> Literal[
        "finalizer",
        "answer_generator",
        "query_rewrite",
        "grep_query_builder",
        "question_clarification",
        "knowledge_not_found",
    ]:
        """Route according to groundedness and revision budgets."""
        return state.route  # type: ignore[return-value]
