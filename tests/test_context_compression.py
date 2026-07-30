"""Tests for bounded asynchronous conversation-context compression."""

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)

from app.services.context_compression import (
    build_conversation_context,
    group_complete_conversation_turns,
)


def _history(turn_count: int, *, content_size: int = 1) -> list:
    """Build deterministic completed user/assistant turns."""
    messages = []
    for index in range(1, turn_count + 1):
        messages.extend(
            [
                HumanMessage(content=f"user-{index}-" + ("u" * content_size)),
                AIMessage(content=f"assistant-{index}-" + ("a" * content_size)),
            ]
        )
    return messages


def test_group_complete_turns_handles_hitl_and_ignores_dangling_user() -> None:
    """Consecutive HITL user messages belong to the same completed turn."""
    turns = group_complete_conversation_turns(
        [
            SystemMessage(content="ignored"),
            HumanMessage(content="original question"),
            HumanMessage(content="selected developer role"),
            AIMessage(content="final answer"),
            HumanMessage(content="next unanswered question"),
        ]
    )

    assert len(turns) == 1
    assert [message["role"] for message in turns[0].messages] == [
        "user",
        "user",
        "assistant",
    ]
    assert turns[0].turn_index == 1


def test_context_uses_two_older_summaries_and_latest_full_turn() -> None:
    """Only the last three completed turns are injected in chronological order."""
    turns = group_complete_conversation_turns(_history(4))
    summaries = {
        turns[1].source_hash: "summary for turn 2",
        turns[2].source_hash: "summary for turn 3",
        turns[3].source_hash: "available but latest remains full",
    }

    context = build_conversation_context(
        turns,
        summaries,
        recent_full_max_chars=1_000,
        history_hard_max_chars=10_000,
        summary_max_chars=500,
    )

    assert [item["turn_index"] for item in context] == [2, 3, 4, 4]
    assert [item["type"] for item in context] == [
        "conversation_summary",
        "conversation_summary",
        "user",
        "assistant",
    ]
    assert context[-1]["content"].startswith("assistant-4-")


def test_pending_summary_falls_back_to_full_history() -> None:
    """Missing/pending summaries retain raw messages while under the hard cap."""
    turns = group_complete_conversation_turns(_history(3))

    context = build_conversation_context(
        turns,
        {},
        recent_full_max_chars=1_000,
        history_hard_max_chars=10_000,
        summary_max_chars=500,
    )

    assert len(context) == 6
    assert all(item["compressed"] is False for item in context)
    assert [item["turn_index"] for item in context] == [1, 1, 2, 2, 3, 3]


def test_oversized_latest_turn_uses_completed_summary() -> None:
    """The newest turn switches from raw content to its summary over the soft limit."""
    turns = group_complete_conversation_turns(_history(1, content_size=100))
    context = build_conversation_context(
        turns,
        {turns[0].source_hash: "bounded latest summary"},
        recent_full_max_chars=50,
        history_hard_max_chars=1_000,
        summary_max_chars=500,
    )

    assert context == [
        {
            "type": "conversation_summary",
            "content": "bounded latest summary",
            "turn_index": 1,
            "compressed": True,
        }
    ]


def test_raw_fallback_never_exceeds_hard_history_limit() -> None:
    """Pending summaries may fall back to raw text but never without a cap."""
    turns = group_complete_conversation_turns(_history(3, content_size=200))
    context = build_conversation_context(
        turns,
        {},
        recent_full_max_chars=50,
        history_hard_max_chars=180,
        summary_max_chars=100,
    )

    assert sum(len(item["content"]) for item in context) <= 180
    assert context
    assert all(item["turn_index"] == 3 for item in context)
    assert any(item.get("truncated") for item in context)
