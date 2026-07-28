"""Tests for frontend-facing SSE event payloads."""

import json

from app.api.v1.chatbot import (
    _format_stream_event,
    _stream_response_from_chunk,
)


def test_question_hitl_is_promoted_outside_content() -> None:
    """A query clarification interrupt should expose fixed directories at top level."""
    chunk = json.dumps(
        {
            "type": "question_clarification",
            "question": "请选择知识库目录并补充问题。",
            "system_options": [
                {
                    "label": "psLoss",
                    "value": "viking://resources/psLoss",
                }
            ],
        },
        ensure_ascii=False,
    )

    response = _stream_response_from_chunk(chunk)

    assert response.event == "hitl"
    assert response.hitl_type == "question_clarification"
    assert response.content == ""
    assert response.title == "请选择知识库目录并补充问题。"
    assert response.directories is not None
    assert response.directories[0].model_dump() == {
        "title": "psLoss",
        "uri": "viking://resources/psLoss",
    }
    assert _format_stream_event(response).startswith("hitl: ")


def test_role_hitl_is_promoted_to_fixed_options() -> None:
    """A role clarification interrupt should expose fixed options at top level."""
    chunk = json.dumps(
        {
            "type": "role_clarification",
            "question": "请选择身份。",
            "options": [{"label": "开发", "value": "developer"}],
        },
        ensure_ascii=False,
    )

    response = _stream_response_from_chunk(chunk)

    assert response.event == "hitl"
    assert response.hitl_type == "role_clarification"
    assert response.content == ""
    assert response.options is not None
    assert response.options[0].model_dump() == {
        "title": "开发",
        "value": "developer",
    }
    assert _format_stream_event(response).startswith("hitl: ")


def test_normal_chunk_remains_message_content() -> None:
    """A normal model chunk should retain the existing content contract."""
    response = _stream_response_from_chunk("普通回答 token")

    assert response.event == "message"
    assert response.content == "普通回答 token"
    assert response.hitl_type is None
    assert _format_stream_event(response).startswith("data: ")


def test_done_event_keeps_standard_data_prefix() -> None:
    """Only HITL events should use the custom hitl prefix."""
    response = _stream_response_from_chunk("固定回答")
    response.event = "done"
    response.done = True

    assert _format_stream_event(response).startswith("data: ")
