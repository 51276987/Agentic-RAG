"""Tests for completed graph stream fallback behavior."""

from app.core.langgraph.graph import _get_stream_fallback_answer


def test_completed_final_answer_is_used_when_stream_emitted_nothing() -> None:
    """HITL resume completion must not silently return only a done event."""
    answer = _get_stream_fallback_answer(
        {
            "final_answer": "恢复 HITL 后生成的最终回答",
            "route": "completed",
        },
        output_emitted=False,
    )

    assert answer == "恢复 HITL 后生成的最终回答"


def test_fallback_does_not_duplicate_existing_stream_output() -> None:
    """A normally streamed answer must not be emitted a second time."""
    answer = _get_stream_fallback_answer(
        {"final_answer": "已经流式发送的回答"},
        output_emitted=True,
    )

    assert answer is None


def test_empty_final_answer_has_no_fallback() -> None:
    """Missing final state should be handled as an explicit stream error."""
    answer = _get_stream_fallback_answer(
        {"final_answer": ""},
        output_emitted=False,
    )

    assert answer is None
