"""This file contains the chat schema for the application."""

import re
from typing import Any
from typing import (
    List,
    Literal,
)

from pydantic import (
    BaseModel,
    Field,
    field_validator,
)

from app.schemas.base import BaseResponse


class Message(BaseModel):
    """Message model for chat endpoint.

    Attributes:
        role: The role of the message sender (user or assistant).
        content: The content of the message.
    """

    model_config = {"extra": "ignore"}

    role: Literal["user", "assistant", "system"] = Field(..., description="The role of the message sender")
    content: str = Field(..., description="The content of the message", min_length=1, max_length=3000)

    @field_validator("content")
    @classmethod
    def validate_content(cls, v: str) -> str:
        """Validate the message content.

        Args:
            v: The content to validate

        Returns:
            str: The validated content

        Raises:
            ValueError: If the content contains disallowed patterns
        """
        # Check for potentially harmful content
        if re.search(r"<script.*?>.*?</script>", v, re.IGNORECASE | re.DOTALL):
            raise ValueError("Content contains potentially harmful script tags")

        # Check for null bytes
        if "\0" in v:
            raise ValueError("Content contains null bytes")

        return v


class ChatHistoryMessage(Message):
    """A persisted chat message returned by the history and chat APIs.

    Input messages intentionally have a small size limit. Persisted assistant
    responses can legitimately be much longer, so they must not reuse that
    request-validation limit when being read from a LangGraph checkpoint.
    """

    content: str = Field(..., description="The complete persisted message content", min_length=1)


class ChatRequest(BaseModel):
    """Request model for chat endpoint.

    Attributes:
        messages: List of messages in the conversation.
    """

    messages: List[Message] = Field(
        ...,
        description="List of messages in the conversation",
        min_length=1,
    )


class ChatResponse(BaseResponse):
    """Response model for chat endpoint.

    Attributes:
        messages: List of messages in the conversation.
    """

    messages: List[ChatHistoryMessage] = Field(..., description="List of messages in the conversation")


class StreamDirectory(BaseModel):
    """A frontend-selectable OpenViking knowledge directory."""

    title: str = Field(..., description="Directory title shown to the user")
    uri: str = Field(..., description="Exact directory URI submitted on HITL resume")


class StreamOption(BaseModel):
    """A generic frontend-selectable HITL option."""

    title: str = Field(..., description="Option title shown to the user")
    value: str = Field(..., description="Fixed value submitted on HITL resume")


class StreamResponse(BaseResponse):
    """Response model for streaming chat endpoint.

    Attributes:
        event: The frontend-dispatchable stream event type.
        content: The content of a normal answer chunk.
        done: Whether the stream is complete.
    """

    event: Literal["message", "tool", "hitl", "done", "error"] = Field(
        default="message",
        description="Frontend-dispatchable event type",
    )
    content: str = Field(default="", description="The content of the current chunk")
    tool_name: str | None = Field(default=None, description="Stable node or external tool identifier")
    tool_kind: Literal["node", "openviking"] | None = Field(
        default=None,
        description="Whether this is a graph node transition or an OpenViking operation",
    )
    tool_status: Literal["started", "completed", "failed"] | None = Field(
        default=None,
        description="Current progress state of the tool event",
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Small frontend-displayable progress details; never raw evidence content",
    )
    hitl_type: Literal["question_clarification", "role_clarification"] | None = Field(
        default=None,
        description="HITL workflow type when event is hitl",
    )
    title: str | None = Field(default=None, description="HITL title shown to the user")
    directories: list[StreamDirectory] | None = Field(
        default=None,
        description="Knowledge directories available for query clarification",
    )
    options: list[StreamOption] | None = Field(
        default=None,
        description="Fixed options available for a generic HITL selection",
    )
    done: bool = Field(default=False, description="Whether the stream is complete")


class SessionTitle(BaseModel):
    """Structured output schema for session title generation."""

    title: str = Field(
        ...,
        min_length=1,
        max_length=60,
    )

    @field_validator("title")
    @classmethod
    def _normalize(cls, v: str) -> str:
        v = " ".join(v.split()).strip(" \"'`.,:;!?-")
        if not v:
            raise ValueError("empty title after normalization")
        return v
