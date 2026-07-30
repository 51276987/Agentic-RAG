"""Tests for resilient LLM structured-output parsing."""

import asyncio

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.config import RunnableConfig

from app.schemas.agentic_rag import EvidenceAssessment
from app.services.llm.service import (
    LLMService,
    StructuredOutputError,
    _structured_output_runnable,
)


class FakeChatModel:
    """Return a predefined message from the tool-bound runnable."""

    def __init__(self, response: AIMessage):
        """Store the response returned by the fake model."""
        self.response = response

    def bind_tools(self, _: list[type[EvidenceAssessment]]) -> RunnableLambda:
        """Mimic a chat model with tool binding support."""
        return RunnableLambda(lambda _messages: self.response)


class ConfigAwareRunnable:
    """Capture runtime tracing configuration passed to ``ainvoke``."""

    def __init__(self) -> None:
        """Initialize an empty captured config."""
        self.config = None

    async def ainvoke(self, _: list, config=None) -> AIMessage:
        """Return a response while recording LangChain runtime config."""
        self.config = config
        return AIMessage(content="summary")


def test_structured_output_accepts_plain_json_content() -> None:
    """Parse JSON content when an OpenAI-compatible model omits the tool call."""
    response = AIMessage(
        content=(
            '{"required_sufficient": true, "covered_required_ids": ["req_1"], '
            '"missing_required_ids": [], "covered_optional_ids": [], '
            '"missing_optional_ids": [], "reason": "证据完整"}'
        )
    )
    runnable = _structured_output_runnable(FakeChatModel(response), EvidenceAssessment)

    result = asyncio.run(runnable.ainvoke([]))

    assert isinstance(result, EvidenceAssessment)
    assert result.required_sufficient is True
    assert result.covered_required_ids == ["req_1"]


def test_structured_output_accepts_tool_call() -> None:
    """Keep parsing native tool calls when the provider emits one."""
    response = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "EvidenceAssessment",
                "args": {
                    "required_sufficient": False,
                    "covered_required_ids": [],
                    "missing_required_ids": ["req_1"],
                    "covered_optional_ids": [],
                    "missing_optional_ids": [],
                    "reason": "证据不完整",
                },
                "id": "assessment-1",
                "type": "tool_call",
            }
        ],
    )
    runnable = _structured_output_runnable(FakeChatModel(response), EvidenceAssessment)

    result = asyncio.run(runnable.ainvoke([]))

    assert isinstance(result, EvidenceAssessment)
    assert result.required_sufficient is False
    assert result.missing_required_ids == ["req_1"]


def test_structured_output_rejects_empty_response() -> None:
    """Raise a retryable domain error instead of returning None."""
    runnable = _structured_output_runnable(FakeChatModel(AIMessage(content="")), EvidenceAssessment)

    with pytest.raises(StructuredOutputError, match="EvidenceAssessment"):
        asyncio.run(runnable.ainvoke([]))


def test_invoke_with_retry_forwards_runnable_config() -> None:
    """Background compression traces must reach the underlying LLM runnable."""
    service = object.__new__(LLMService)
    runnable = ConfigAwareRunnable()
    config: RunnableConfig = {
        "run_name": "context-compression",
        "metadata": {"context_compression_job_id": "job-1"},
    }

    response = asyncio.run(
        service._invoke_with_retry(  # noqa: SLF001
            runnable,
            [],
            config,
        )
    )

    assert response.content == "summary"
    assert runnable.config == config
