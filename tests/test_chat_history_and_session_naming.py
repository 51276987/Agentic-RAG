"""Regression tests for persisted history and automatic session titles."""

import pytest
from pydantic import ValidationError

from app.schemas.chat import ChatHistoryMessage, Message
from app.services.session_naming import _title_from_response_content


def test_persisted_history_accepts_long_assistant_response() -> None:
    """A long model answer must remain readable after checkpoint persistence."""
    content = "x" * 3_001

    history_message = ChatHistoryMessage(role="assistant", content=content)

    assert history_message.content == content
    with pytest.raises(ValidationError):
        Message(role="assistant", content=content)


def test_session_title_accepts_plain_text_model_response() -> None:
    """Qwen-style plain-text titles must not require a tool call or JSON object."""
    title = _title_from_response_content("非菜单场景业务变更处理")

    assert title.title == "非菜单场景业务变更处理"


def test_session_title_normalizes_and_limits_plain_text() -> None:
    """The fallback title remains compatible with the persisted title schema."""
    title = _title_from_response_content("  \n" + ("标题 " * 40))

    assert len(title.title) <= 60
    assert "\n" not in title.title
