"""Schemas for the first-phase Agentic RAG loop."""

from typing import (
    Literal,
    Self,
)

from pydantic import (
    BaseModel,
    Field,
    model_validator,
)

UserRole = Literal["product_manager", "developer", "new_employee"]
RoleSource = Literal["profile", "explicit", "inferred", "hitl"]
RetrievalOperation = Literal["find", "list_resources", "stat"]
HydrationLevel = Literal["abstract", "overview", "full"]
RequirementPriority = Literal["required", "optional"]
RequirementEvidenceSource = Literal["knowledge_base", "user_context", "knowledge_and_context"]


class AnswerRequirement(BaseModel):
    """One answer obligation with explicit priority and evidence policy."""

    requirement_id: str = Field(min_length=1, max_length=30)
    description: str = Field(min_length=1, max_length=500)
    priority: RequirementPriority
    evidence_source: RequirementEvidenceSource


class IntentAnalysis(BaseModel):
    """Structured output produced by the intent analyzer."""

    intent: Literal[
        "fact_lookup",
        "procedure",
        "comparison",
        "troubleshooting",
        "summary",
        "analysis",
        "conversational",
    ]
    needs_retrieval: bool
    user_role: UserRole | None = None
    role_source: RoleSource | None = None
    role_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    role_evidence: list[str] = Field(default_factory=list, max_length=5)
    entities: list[str] = Field(default_factory=list, max_length=10)
    constraints: list[str] = Field(default_factory=list, max_length=10)
    answer_requirements: list[AnswerRequirement] = Field(default_factory=list, min_length=1, max_length=8)

    @model_validator(mode="after")
    def validate_answer_requirements(self) -> Self:
        """Require stable unique IDs and at least one required obligation."""
        requirement_ids = [item.requirement_id for item in self.answer_requirements]
        if len(requirement_ids) != len(set(requirement_ids)):
            raise ValueError("answer requirement IDs must be unique")
        if not any(item.priority == "required" for item in self.answer_requirements):
            raise ValueError("at least one answer requirement must be required")
        return self


class RetrievalTask(BaseModel):
    """A deterministic OpenViking retrieval operation."""

    task_id: str = Field(min_length=1, max_length=30)
    purpose: str = Field(min_length=1, max_length=300)
    operation: RetrievalOperation
    information_need: str = Field(min_length=1, max_length=1000)
    target_uri: str = Field(default="viking://resources", max_length=1000)
    query: str = Field(default="", max_length=2000)
    limit: int = Field(default=8, ge=1, le=10)
    recursive: bool = False
    node_limit: int = Field(default=100, ge=1, le=100)
    hydration_level: HydrationLevel = "full"


class RetrievalPlan(BaseModel):
    """Structured retrieval plan."""

    tasks: list[RetrievalTask] = Field(min_length=1, max_length=4)


class RewrittenQuery(BaseModel):
    """A rewritten semantic query for one retrieval task."""

    task_id: str
    query: str = Field(min_length=1, max_length=2000)


class QueryRewriteResult(BaseModel):
    """Structured output for all find-task rewrites."""

    queries: list[RewrittenQuery] = Field(default_factory=list, max_length=4)


class EvidenceAssessment(BaseModel):
    """Evidence coverage assessment."""

    required_sufficient: bool
    covered_required_ids: list[str] = Field(default_factory=list)
    missing_required_ids: list[str] = Field(default_factory=list)
    covered_optional_ids: list[str] = Field(default_factory=list)
    missing_optional_ids: list[str] = Field(default_factory=list)
    reason: str = Field(default="", max_length=1000)


class GroundednessAssessment(BaseModel):
    """Groundedness and citation verification output."""

    passed: bool
    action: Literal["pass", "revise", "retrieve"]
    unsupported_claims: list[str] = Field(default_factory=list)
    missing_required_ids: list[str] = Field(default_factory=list)
    missing_optional_ids: list[str] = Field(default_factory=list)
    revision_instructions: str = Field(default="", max_length=1000)
