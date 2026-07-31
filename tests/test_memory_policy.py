"""Tests for long-term memory retrieval and persistence policy."""

import asyncio
from typing import Any

import app.services.memory as memory_module
from app.core.config import settings
from app.services.memory import (
    MemoryService,
    _format_relevant_memories,
    _select_key_memory_turn,
)


def test_memory_retrieval_keeps_only_near_cosine_distances() -> None:
    """Pgvector cosine distance is lower-is-better and weak matches are excluded."""
    result = _format_relevant_memories(
        {
            "results": [
                {"memory": "weak", "score": 0.62},
                {"memory": "nearest", "score": 0.12},
                {"memory": "relevant", "score": 0.31},
            ]
        },
        max_cosine_distance=0.35,
    )

    assert result.splitlines() == ["* nearest", "* relevant"]


def test_ordinary_knowledge_question_is_not_persisted() -> None:
    """A one-off knowledge lookup belongs to session history, not user memory."""
    selected = _select_key_memory_turn(
        [
            {"role": "user", "content": "请解释 ANN 模型的基本架构和应用场景"},
            {"role": "assistant", "content": "ANN 是一种人工神经网络模型。"},
        ]
    )

    assert selected == []


def test_durable_project_context_is_selected_for_memory() -> None:
    """Stable project and environment facts should remain available across sessions."""
    selected = _select_key_memory_turn(
        [
            {"role": "user", "content": "当前项目的生产环境统一使用 Qwen3.6，并将超时设置为 300 秒"},
            {"role": "assistant", "content": "已记录该生产环境模型和超时约定。"},
        ]
    )

    assert selected == [
        {"role": "user", "content": "当前项目的生产环境统一使用 Qwen3.6，并将超时设置为 300 秒"},
        {"role": "assistant", "content": "已记录该生产环境模型和超时约定。"},
    ]


class _FakeCache:
    async def get(self, _: str) -> None:
        return None

    async def set(self, _: str, __: str) -> None:
        return None


class _FakeMemory:
    def __init__(self) -> None:
        self.search_args: dict[str, Any] | None = None
        self.added_messages: list[dict[str, str]] | None = None

    async def search(self, **kwargs: Any) -> dict[str, list[dict[str, Any]]]:
        self.search_args = kwargs
        return {
            "results": [
                {"memory": "keep", "score": 0.2},
                {"memory": "drop", "score": 0.5},
            ]
        }

    async def add(self, messages: list[dict[str, str]], **_: Any) -> None:
        self.added_messages = messages


def test_service_search_uses_configured_limit_and_distance_filter(monkeypatch) -> None:
    """The service must not rely on mem0's 100-result and no-threshold defaults."""
    fake_memory = _FakeMemory()
    service = MemoryService()
    service._memory = fake_memory  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(memory_module, "cache_service", _FakeCache())
    monkeypatch.setattr(settings, "LONG_TERM_MEMORY_MAX_RESULTS", 5)
    monkeypatch.setattr(settings, "LONG_TERM_MEMORY_MAX_COSINE_DISTANCE", 0.35)

    result = asyncio.run(service.search("user-1", "当前项目模型配置"))

    assert fake_memory.search_args == {
        "user_id": "user-1",
        "query": "当前项目模型配置",
        "limit": 5,
    }
    assert result == "* keep"


def test_service_add_only_sends_selected_key_turn() -> None:
    """mem0 receives the latest durable turn rather than the entire checkpoint."""
    fake_memory = _FakeMemory()
    service = MemoryService()
    service._memory = fake_memory  # pyright: ignore[reportPrivateUsage]

    asyncio.run(
        service.add(
            "user-1",
            [
                {"role": "user", "content": "早期普通问题"},
                {"role": "assistant", "content": "早期普通回答"},
                {"role": "user", "content": "我的角色是开发者，以后请优先给出接口和排障步骤"},
                {"role": "assistant", "content": "已按开发者视角记录回答偏好。"},
            ],
        )
    )

    assert fake_memory.added_messages == [
        {"role": "user", "content": "我的角色是开发者，以后请优先给出接口和排障步骤"},
        {"role": "assistant", "content": "已按开发者视角记录回答偏好。"},
    ]
