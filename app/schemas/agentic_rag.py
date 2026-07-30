"""Schemas for the first-phase Agentic RAG loop."""

from typing import (
    Literal,
    Self,
)

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
)

UserRole = Literal["product_manager", "developer", "new_employee"]
RoleSource = Literal["profile", "explicit", "inferred", "hitl"]
RetrievalOperation = Literal["find", "grep"]
HydrationLevel = Literal["abstract", "overview", "full"]
RequirementPriority = Literal["required", "optional"]
RequirementEvidenceSource = Literal["knowledge_base", "user_context", "knowledge_and_context"]
MAX_REQUIRED_ANSWER_REQUIREMENTS = 1


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

    @field_validator("user_role", "role_source", mode="before")
    @classmethod
    def normalize_nullable_enum_strings(cls, value: object) -> object:
        """Normalize common model-rendered null strings before enum validation."""
        if isinstance(value, str) and value.strip().lower() in {
            "",
            "null",
            "none",
            "nil",
            "undefined",
        }:
            return None
        return value

    @model_validator(mode="after")
    def validate_answer_requirements(self) -> Self:
        """Keep exactly one user-requested obligation as the retrieval gate."""
        requirement_ids = [item.requirement_id for item in self.answer_requirements]
        if len(requirement_ids) != len(set(requirement_ids)):
            raise ValueError("answer requirement IDs must be unique")
        if not any(item.priority == "required" for item in self.answer_requirements):
            raise ValueError("at least one answer requirement must be required")

        # The evidence grader only blocks on required requirements.  Letting an
        # LLM split one question into multiple required items turns helpful
        # detail into a hard retrieval gate and causes unnecessary repair/HITL
        # loops.  The prompt asks the model to make the first required item a
        # single, complete statement of the user's explicit request; this is a
        # defensive invariant for malformed or legacy model output.
        required_seen = 0
        for requirement in self.answer_requirements:
            if requirement.priority != "required":
                continue
            required_seen += 1
            if required_seen > MAX_REQUIRED_ANSWER_REQUIREMENTS:
                requirement.priority = "optional"
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
    node_limit: int = Field(default=100, ge=1, le=100)
    hydration_level: HydrationLevel = "full"


class RewrittenQuery(BaseModel):
    """A rewritten semantic query for one retrieval task."""

    task_id: str
    query: str = Field(min_length=1, max_length=2000)


class QueryRewriteResult(BaseModel):
    """Structured output for all find-task rewrites."""

    queries: list[RewrittenQuery] = Field(default_factory=list, max_length=4)


class GrepKeywordResult(BaseModel):
    """Literal keywords used to build one bounded grep regex."""

    keywords: list[str] = Field(min_length=1, max_length=5)


class SystemScopeResult(BaseModel):
    """LLM selection of one verified root-level system candidate."""

    scope_confident: bool
    selected_uri: str | None = Field(default=None, max_length=1000)
    scoped_query: str = Field(min_length=1, max_length=2000)
    reason: str = Field(default="", max_length=500)

    @model_validator(mode="after")
    def validate_selected_scope(self) -> Self:
        """Require a selected candidate only for a confident match."""
        if self.scope_confident and not (self.selected_uri or "").strip():
            raise ValueError("selected_uri is required when scope_confident is true")
        if not self.scope_confident:
            self.selected_uri = None
        return self


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
