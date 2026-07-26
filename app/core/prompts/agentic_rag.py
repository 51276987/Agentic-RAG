"""Cached prompt templates for the Agentic RAG nodes."""

from pathlib import Path

_PROMPT_DIR = Path(__file__).with_name("agentic_rag_prompts")
_PROMPT_NAMES = (
    "intent_analyzer",
    "retrieval_planner",
    "query_rewrite",
    "evidence_grader",
    "answer_generator",
    "groundedness_verifier",
    "direct_answer",
)
AGENTIC_RAG_PROMPTS = {
    name: (_PROMPT_DIR / f"{name}.md").read_text(encoding="utf-8")
    for name in _PROMPT_NAMES
}


def get_agentic_rag_prompt(name: str) -> str:
    """Return a cached Agentic RAG prompt by node name."""
    try:
        return AGENTIC_RAG_PROMPTS[name]
    except KeyError as exc:
        raise ValueError(f"unknown Agentic RAG prompt: {name}") from exc
