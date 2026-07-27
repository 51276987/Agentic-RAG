"""First-phase Agentic RAG loop backed by deterministic OpenViking API calls."""

import asyncio
import json
import re
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
from langgraph.types import interrupt

from app.core.logging import logger
from app.core.prompts.agentic_rag import get_agentic_rag_prompt
from app.schemas import (
    AnswerRequirement,
    EvidenceAssessment,
    GraphState,
    GroundednessAssessment,
    IntentAnalysis,
    QueryRewriteResult,
    RetrievalPlan,
    RetrievalTask,
)
from app.services.llm import LLMService
from app.services.openviking import OpenVikingKnowledgeAPI
from app.utils import extract_text_content

MAX_RETRIEVAL_ROUNDS = 2
MAX_PARALLEL_TASKS = 4
MAX_RESULTS_PER_TASK = 10
MAX_LISTED_RESOURCES = 100
MAX_HYDRATED_RESOURCES = 8
MAX_FULL_CONTENT_RESOURCES = 4
MAX_EVIDENCE_CHARS = 24_000

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
_DIRECTORY_TERMS = ("有哪些文件", "有什么文件", "目录", "文件列表", "知识库结构", "列出文件")


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
    if isinstance(value, dict):
        value = value.get("value") or value.get("role") or value.get("answer")
    text = str(value).strip()
    if text in _ROLE_ALIASES:
        return _ROLE_ALIASES[text]
    lowered = text.lower()
    return _ROLE_ALIASES.get(lowered)


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


def _parse_answer_requirements(items: list[dict[str, Any] | str]) -> list[AnswerRequirement]:
    """Normalize current requirements and legacy string-only checkpoints."""
    requirements: list[AnswerRequirement] = []
    for index, item in enumerate(items, start=1):
        if isinstance(item, str):
            requirements.append(
                AnswerRequirement(
                    requirement_id=f"req_{index}",
                    description=item,
                    priority="required",
                    evidence_source="knowledge_base",
                )
            )
            continue
        requirements.append(AnswerRequirement.model_validate(item))
    return requirements


class AgentLoop:
    """Build and execute the bounded Agentic RAG workflow."""

    def __init__(self, llm_service: LLMService, openviking_api: OpenVikingKnowledgeAPI):
        """Initialize node dependencies."""
        self.llm_service = llm_service
        self.openviking_api = openviking_api

    def configure(self, graph: StateGraph) -> None:
        """Register nodes and routes on a StateGraph."""
        graph.add_node("prepare_request", self.prepare_request)
        graph.add_node("intent_analyzer", self.intent_analyzer)
        graph.add_node("role_clarification", self.role_clarification)
        graph.add_node("retrieval_planner", self.retrieval_planner)
        graph.add_node("query_rewrite", self.query_rewrite)
        graph.add_node("retrieval_executor", self.retrieval_executor)
        graph.add_node("result_fusion", self.result_fusion)
        graph.add_node("evidence_hydration", self.evidence_hydration)
        graph.add_node("evidence_grader", self.evidence_grader)
        graph.add_node("retrieval_repair", self.retrieval_repair)
        graph.add_node("answer_generator", self.answer_generator)
        graph.add_node("groundedness_verifier", self.groundedness_verifier)
        graph.add_node("finalizer", self.finalizer)
        graph.add_node("insufficient_evidence", self.insufficient_evidence)
        graph.add_node("direct_answer", self.direct_answer)

        graph.add_edge(START, "prepare_request")
        graph.add_edge("prepare_request", "intent_analyzer")
        graph.add_conditional_edges(
            "intent_analyzer",
            self.route_after_intent,
            {
                "role_clarification": "role_clarification",
                "retrieval_planner": "retrieval_planner",
                "direct_answer": "direct_answer",
            },
        )
        graph.add_conditional_edges(
            "role_clarification",
            self.route_after_role,
            {
                "retrieval_planner": "retrieval_planner",
                "direct_answer": "direct_answer",
            },
        )
        graph.add_edge("retrieval_planner", "query_rewrite")
        graph.add_edge("query_rewrite", "retrieval_executor")
        graph.add_edge("retrieval_executor", "result_fusion")
        graph.add_edge("result_fusion", "evidence_hydration")
        graph.add_edge("evidence_hydration", "evidence_grader")
        graph.add_conditional_edges(
            "evidence_grader",
            self.route_after_evidence,
            {
                "answer_generator": "answer_generator",
                "retrieval_repair": "retrieval_repair",
                "insufficient_evidence": "insufficient_evidence",
            },
        )
        graph.add_edge("retrieval_repair", "query_rewrite")
        graph.add_edge("answer_generator", "groundedness_verifier")
        graph.add_conditional_edges(
            "groundedness_verifier",
            self.route_after_verification,
            {
                "finalizer": "finalizer",
                "answer_generator": "answer_generator",
                "retrieval_repair": "retrieval_repair",
                "insufficient_evidence": "insufficient_evidence",
            },
        )
        graph.add_edge("finalizer", END)
        graph.add_edge("insufficient_evidence", END)
        graph.add_edge("direct_answer", END)

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
            "retrieval_tasks": [],
            "executed_queries": [],
            "executed_operations": [],
            "retrieval_round": 0,
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
        recent_messages = [
            {"type": getattr(message, "type", "unknown"), "content": _message_text(message)}
            for message in state.messages[-8:]
        ]
        payload = {
            "query": state.normalized_query,
            "confirmed_role": state.user_role,
            "recent_messages": recent_messages,
            "long_term_memory": state.long_term_memory,
        }
        analysis = await self.llm_service.call(
            [
                SystemMessage(content=get_agentic_rag_prompt("intent_analyzer")),
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
                else "retrieval_planner"
                if analysis.needs_retrieval
                else "direct_answer"
            ),
        }

    async def role_clarification(self, state: GraphState) -> dict[str, Any]:
        """Interrupt until the user selects one supported role."""
        prompt: dict[str, Any] = {
            "type": "role_clarification",
            "question": "为了按合适的视角改写问题并检索知识库，请问您当前的角色是：产品经理、开发，还是新入职员工？",
            "options": [
                {"label": label, "value": value}
                for value, label in _ROLE_LABELS.items()
            ],
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
            "route": "retrieval_planner" if state.needs_retrieval else "direct_answer",
        }

    async def _create_plan(self, state: GraphState, *, repair: bool) -> list[RetrievalTask]:
        """Create and normalize a bounded retrieval plan."""
        requirements = _parse_answer_requirements(state.answer_requirements)
        missing_required_ids = set(state.missing_required_ids) if repair else set()
        payload = {
            "query": state.normalized_query,
            "intent": state.intent,
            "role": state.user_role,
            "answer_requirements": [item.model_dump() for item in requirements],
            "missing_requirements": [
                item.model_dump() for item in requirements if item.requirement_id in missing_required_ids
            ],
            "allowed_target_uris": state.allowed_target_uris,
            "executed_queries": state.executed_queries,
            "retrieval_round": state.retrieval_round,
        }
        plan = await self.llm_service.call(
            [
                SystemMessage(content=get_agentic_rag_prompt("retrieval_planner")),
                HumanMessage(content=_json(payload)),
            ],
            response_format=RetrievalPlan,
        )

        tasks: list[RetrievalTask] = []
        for index, task in enumerate(plan.tasks[:MAX_PARALLEL_TASKS], start=1):
            target_uri = task.target_uri.rstrip("/") or state.allowed_target_uris[0]
            if not _is_allowed_uri(target_uri, state.allowed_target_uris):
                target_uri = state.allowed_target_uris[0]
            tasks.append(
                task.model_copy(
                    update={
                        "task_id": task.task_id or f"r{state.retrieval_round + 1}_{index}",
                        "target_uri": target_uri,
                        "limit": min(task.limit, MAX_RESULTS_PER_TASK),
                        "node_limit": min(task.node_limit, MAX_LISTED_RESOURCES),
                    }
                )
            )

        needs_directory_listing = any(term in state.normalized_query for term in _DIRECTORY_TERMS)
        if needs_directory_listing and not any(task.operation == "list_resources" for task in tasks):
            tasks.insert(
                0,
                RetrievalTask(
                    task_id=f"r{state.retrieval_round + 1}_list",
                    purpose="列出知识库授权范围内的目录和文件",
                    operation="list_resources",
                    information_need=state.normalized_query,
                    target_uri=state.allowed_target_uris[0],
                    recursive=False,
                    node_limit=MAX_LISTED_RESOURCES,
                    hydration_level="overview",
                ),
            )
            tasks = tasks[:MAX_PARALLEL_TASKS]
        return tasks

    async def retrieval_planner(self, state: GraphState) -> dict[str, Any]:
        """Plan first-round OpenViking operations."""
        tasks = await self._create_plan(state, repair=False)
        logger.info("retrieval_plan_created", task_count=len(tasks), retrieval_round=state.retrieval_round + 1)
        return {"retrieval_tasks": [task.model_dump() for task in tasks], "route": "query_rewrite"}

    async def query_rewrite(self, state: GraphState) -> dict[str, Any]:
        """Rewrite all find tasks using the confirmed role."""
        tasks = [RetrievalTask.model_validate(task) for task in state.retrieval_tasks]
        requirements = _parse_answer_requirements(state.answer_requirements)
        missing_required_ids = set(state.missing_required_ids)
        find_tasks = [task for task in tasks if task.operation == "find"]
        if not find_tasks:
            return {"route": "retrieval_executor"}

        rewrite = await self.llm_service.call(
            [
                SystemMessage(content=get_agentic_rag_prompt("query_rewrite")),
                HumanMessage(
                    content=_json(
                        {
                            "query": state.normalized_query,
                            "role": state.user_role,
                            "tasks": [task.model_dump() for task in find_tasks],
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
        query_by_task = {item.task_id: item.query.strip() for item in rewrite.queries}
        updated_tasks: list[dict[str, Any]] = []
        for task in tasks:
            if task.operation == "find":
                query = query_by_task.get(task.task_id) or task.information_need
                task = task.model_copy(update={"query": query})
            updated_tasks.append(task.model_dump())
        return {"retrieval_tasks": updated_tasks, "route": "retrieval_executor"}

    async def retrieval_executor(self, state: GraphState) -> dict[str, Any]:
        """Execute only the planned read-only OpenViking operations."""
        semaphore = asyncio.Semaphore(MAX_PARALLEL_TASKS)
        tasks = [RetrievalTask.model_validate(task) for task in state.retrieval_tasks]

        async def execute(task: RetrievalTask) -> dict[str, Any]:
            async with semaphore:
                try:
                    if task.operation == "find":
                        result = await self.openviking_api.find(task.query, task.target_uri, task.limit)
                    elif task.operation == "list_resources":
                        result = await self.openviking_api.list_resources(
                            task.target_uri,
                            task.recursive,
                            task.node_limit,
                        )
                    else:
                        result = await self.openviking_api.stat(task.target_uri)
                    return {
                        "ok": True,
                        "task_id": task.task_id,
                        "operation": task.operation,
                        "target_uri": task.target_uri,
                        "result": result,
                    }
                except Exception as exc:
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
        queries = [
            task.query
            for task in tasks
            if task.operation == "find" and task.query and task.query not in state.executed_queries
        ]
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
        """Fuse heterogeneous API results into a deduplicated URI set."""
        by_uri: dict[str, dict[str, Any]] = {}
        for raw in state.raw_results:
            if not raw.get("ok"):
                continue
            for item in _iter_uri_items(raw.get("result")):
                uri = str(item["uri"])
                if not _is_allowed_uri(uri, state.allowed_target_uris):
                    continue
                score = item.get("score", item.get("similarity", 0.0))
                candidate = {
                    "uri": uri,
                    "score": float(score) if isinstance(score, (int, float)) else 0.0,
                    "task_id": raw["task_id"],
                    "operation": raw["operation"],
                    "metadata": item,
                }
                current = by_uri.get(uri)
                if current is None or candidate["score"] > current["score"]:
                    by_uri[uri] = candidate

        candidates = sorted(by_uri.values(), key=lambda item: item["score"], reverse=True)
        candidates = candidates[:MAX_HYDRATED_RESOURCES]
        return {
            "candidate_uris": [candidate["uri"] for candidate in candidates],
            "candidate_items": candidates,
            "route": "evidence_hydration",
        }

    async def evidence_hydration(self, state: GraphState) -> dict[str, Any]:
        """Read L0/L1 first and a bounded number of L2 documents."""
        semaphore = asyncio.Semaphore(MAX_PARALLEL_TASKS)
        task_by_id = {
            task.task_id: task
            for task in (RetrievalTask.model_validate(item) for item in state.retrieval_tasks)
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

            task = task_by_id.get(candidate["task_id"])
            is_directory = bool(metadata.get("isDir") or metadata.get("is_dir")) if isinstance(metadata, dict) else False
            first_level: Literal["abstract", "overview", "full"] = "overview" if is_directory else "abstract"
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
                    if not is_directory and wants_full and index < MAX_FULL_CONTENT_RESOURCES:
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
            await asyncio.gather(
                *(hydrate(index, candidate) for index, candidate in enumerate(state.candidate_items))
            )
        )
        new_evidence = [item for group, _ in hydrated for item in group]
        errors = [item for _, group in hydrated for item in group]
        evidence_by_key = {
            (item.get("uri"), item.get("level")): item
            for item in [*state.selected_evidence, *new_evidence]
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
                    "retrieval_repair"
                    if required_ids and state.retrieval_round < MAX_RETRIEVAL_ROUNDS
                    else "insufficient_evidence"
                    if required_ids
                    else "answer_generator"
                ),
            }

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
                            "evidence": state.selected_evidence,
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
        elif state.retrieval_round < MAX_RETRIEVAL_ROUNDS:
            route = "retrieval_repair"
        else:
            route = "insufficient_evidence"
        return {
            "covered_required_ids": covered_required_ids,
            "missing_required_ids": missing_required_ids,
            "covered_optional_ids": covered_optional_ids,
            "missing_optional_ids": missing_optional_ids,
            "route": route,
        }

    async def retrieval_repair(self, state: GraphState) -> dict[str, Any]:
        """Plan the next bounded retrieval round from evidence gaps."""
        tasks = await self._create_plan(state, repair=True)
        return {
            "retrieval_tasks": [task.model_dump() for task in tasks],
            "raw_results": [],
            "candidate_uris": [],
            "candidate_items": [],
            "route": "query_rewrite",
        }

    async def answer_generator(self, state: GraphState) -> dict[str, Any]:
        """Generate or revise a cited answer using selected evidence only."""
        requirements = _parse_answer_requirements(state.answer_requirements)
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
                            "evidence": state.selected_evidence,
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
        """Verify support, coverage, and citation validity."""
        requirements = _parse_answer_requirements(state.answer_requirements)
        required_id_set = {item.requirement_id for item in requirements if item.priority == "required"}
        optional_id_set = {item.requirement_id for item in requirements if item.priority == "optional"}
        available_uris = sorted({item["uri"] for item in state.selected_evidence if item.get("uri")})
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
                            "available_uris": available_uris,
                            "evidence": state.selected_evidence,
                        },
                        max_chars=MAX_EVIDENCE_CHARS,
                    )
                ),
            ],
            response_format=GroundednessAssessment,
        )

        has_valid_citation = not available_uris or any(uri in state.draft_answer for uri in available_uris)
        missing_required_ids = [
            item for item in assessment.missing_required_ids if item in required_id_set
        ]
        missing_optional_ids = [
            item for item in assessment.missing_optional_ids if item in optional_id_set
        ]
        has_required_gap = bool(missing_required_ids)
        if assessment.passed and assessment.action == "pass" and has_valid_citation and not has_required_gap:
            route = "finalizer"
        elif (
            has_required_gap
            and assessment.action == "retrieve"
            and state.retrieval_round < MAX_RETRIEVAL_ROUNDS
        ):
            route = "retrieval_repair"
        elif state.revision_count < 1:
            route = "answer_generator"
        elif not has_required_gap and not assessment.unsupported_claims:
            route = "finalizer"
        else:
            route = "insufficient_evidence"

        instructions = assessment.revision_instructions
        if not has_valid_citation:
            instructions = f"{instructions}\n为关键结论补充可用 URI 引用。".strip()
        return {
            "missing_required_ids": missing_required_ids or state.missing_required_ids,
            "missing_optional_ids": missing_optional_ids or state.missing_optional_ids,
            "revision_instructions": instructions,
            "revision_count": state.revision_count + (1 if route == "answer_generator" else 0),
            "route": route,
        }

    async def finalizer(self, state: GraphState) -> dict[str, Any]:
        """Publish the verified answer."""
        return {
            "final_answer": state.draft_answer,
            "messages": [AIMessage(content=state.draft_answer)],
            "route": "completed",
        }

    async def insufficient_evidence(self, state: GraphState) -> dict[str, Any]:
        """Return an explicit no-guess response after bounded retries."""
        requirements = _parse_answer_requirements(state.answer_requirements)
        missing_id_set = set(state.missing_required_ids)
        missing = [
            item.description
            for item in requirements
            if item.priority == "required" and item.requirement_id in missing_id_set
        ]
        if not missing:
            missing = [item.description for item in requirements if item.priority == "required"]
        sources = sorted({item["uri"] for item in state.selected_evidence if item.get("uri")})
        parts = [
            "当前知识库证据不足，我不会继续推测。",
            f"尚缺少的信息：{'；'.join(missing)}" if missing else "未检索到可直接支持结论的内容。",
        ]
        if sources:
            parts.append(f"已核查来源：{'、'.join(sources)}")
        answer = "\n\n".join(parts)
        return {
            "final_answer": answer,
            "messages": [AIMessage(content=answer)],
            "route": "insufficient_evidence",
        }

    async def direct_answer(self, state: GraphState) -> dict[str, Any]:
        """Answer conversational requests without pretending to retrieve."""
        response = await self.llm_service.call(
            [
                SystemMessage(content=get_agentic_rag_prompt("direct_answer")),
                HumanMessage(
                    content=_json(
                        {
                            "query": state.normalized_query,
                            "role": state.user_role,
                            "long_term_memory": state.long_term_memory,
                        }
                    )
                ),
            ]
        )
        answer = _message_text(response).strip()
        return {
            "draft_answer": answer,
            "final_answer": answer,
            "messages": [AIMessage(content=answer)],
            "route": "completed",
        }

    @staticmethod
    def route_after_intent(
        state: GraphState,
    ) -> Literal["role_clarification", "retrieval_planner", "direct_answer"]:
        """Route after intent and role analysis."""
        if state.needs_role_clarification:
            return "role_clarification"
        return "retrieval_planner" if state.needs_retrieval else "direct_answer"

    @staticmethod
    def route_after_role(state: GraphState) -> Literal["retrieval_planner", "direct_answer"]:
        """Continue the original request after HITL role confirmation."""
        return "retrieval_planner" if state.needs_retrieval else "direct_answer"

    @staticmethod
    def route_after_evidence(
        state: GraphState,
    ) -> Literal["answer_generator", "retrieval_repair", "insufficient_evidence"]:
        """Route according to evidence coverage and retry budget."""
        return state.route  # type: ignore[return-value]

    @staticmethod
    def route_after_verification(
        state: GraphState,
    ) -> Literal["finalizer", "answer_generator", "retrieval_repair", "insufficient_evidence"]:
        """Route according to groundedness and revision budgets."""
        return state.route  # type: ignore[return-value]
