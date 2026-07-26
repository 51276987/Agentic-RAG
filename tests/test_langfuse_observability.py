"""Tests for Langfuse trace metadata."""

from app.core.config import settings
from app.core.langgraph.graph import _trace_metadata


def test_trace_metadata_uses_langfuse_standard_fields() -> None:
    """Langfuse should group AgentLoop traces by session and user."""
    metadata = _trace_metadata(
        session_id="session-123",
        user_id="user-456",
        username="developer",
        mode="chat",
    )

    assert metadata["langfuse_session_id"] == "session-123"
    assert metadata["langfuse_user_id"] == "user-456"
    assert metadata["langfuse_tags"] == ["agentic-rag", settings.ENVIRONMENT.value, "chat"]
    assert metadata["agent_loop_version"] == "agentic-rag-v1"
