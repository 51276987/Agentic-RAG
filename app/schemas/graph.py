"""State schema for the Agentic RAG graph."""

from typing import (
    Annotated,
    Any,
)

from langgraph.graph.message import add_messages
from pydantic import (
    BaseModel,
    Field,
)

from app.schemas.agentic_rag import (
    RoleSource,
    UserRole,
)


class GraphState(BaseModel):
    """Persistent and per-request state for the Agentic RAG loop."""

    messages: Annotated[list, add_messages] = Field(
        default_factory=list, description="The messages in the conversation"
    )
    long_term_memory: str = Field(default="", description="The long term memory of the conversation")
    user_query: str = ""
    normalized_query: str = ""

    user_role: UserRole | None = None
    role_source: RoleSource | None = None
    role_confidence: float = 0.0
    role_evidence: list[str] = Field(default_factory=list)
    needs_role_clarification: bool = False

    intent: str = ""
    needs_retrieval: bool = True
    entities: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    answer_requirements: list[str] = Field(default_factory=list)
    allowed_target_uris: list[str] = Field(default_factory=lambda: ["viking://resources"])

    retrieval_tasks: list[dict[str, Any]] = Field(default_factory=list)
    executed_queries: list[str] = Field(default_factory=list)
    executed_operations: list[dict[str, Any]] = Field(default_factory=list)
    retrieval_round: int = 0

    raw_results: list[dict[str, Any]] = Field(default_factory=list)
    candidate_uris: list[str] = Field(default_factory=list)
    candidate_items: list[dict[str, Any]] = Field(default_factory=list)
    hydrated_evidence: list[dict[str, Any]] = Field(default_factory=list)
    retrieval_errors: list[dict[str, Any]] = Field(default_factory=list)
    selected_evidence: list[dict[str, Any]] = Field(default_factory=list)
    covered_requirements: list[str] = Field(default_factory=list)
    missing_requirements: list[str] = Field(default_factory=list)

    draft_answer: str = ""
    revision_instructions: str = ""
    revision_count: int = 0
    final_answer: str = ""
    route: str = ""
