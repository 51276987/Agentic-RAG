"""Tests for the OpenViking knowledge base API boundary."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from app.services.openviking.api import (
    OpenVikingKnowledgeAPI,
    _validate_resource_uri,
)


def test_resource_uri_accepts_only_knowledge_scopes() -> None:
    """Resource validation should reject non-knowledge OpenViking namespaces."""
    assert _validate_resource_uri("viking://resources/docs") == "viking://resources/docs"
    assert _validate_resource_uri("viking://user/resources/docs") == "viking://user/resources/docs"

    with pytest.raises(ValueError, match="仅允许访问"):
        _validate_resource_uri("viking://user/memories/preferences")

    with pytest.raises(ValueError, match="仅允许访问"):
        _validate_resource_uri("viking://skills/web-search")


def test_destructive_operations_reject_resource_root() -> None:
    """Root resource namespaces must never be valid destructive targets."""
    with pytest.raises(ValueError, match="根目录"):
        _validate_resource_uri("viking://resources", allow_root=False)

    with pytest.raises(ValueError, match="根目录"):
        _validate_resource_uri("viking://user/resources", allow_root=False)


def test_find_forces_resource_context_type() -> None:
    """Semantic retrieval should never search memories or skills."""
    client = OpenVikingKnowledgeAPI()
    request = AsyncMock(return_value={"resources": []})
    client._request = request  # pyright: ignore[reportPrivateUsage]

    result = asyncio.run(client.find("authentication", "viking://resources/docs", 5))

    assert result == {"resources": []}
    request.assert_awaited_once_with(
        "POST",
        "/api/v1/search/find",
        json_body={
            "query": "authentication",
            "target_uri": "viking://resources/docs",
            "context_type": ["resource"],
            "limit": 5,
        },
    )


def test_grep_uses_bounded_content_search_contract() -> None:
    """Keyword fallback should call the documented read-only grep endpoint."""
    client = OpenVikingKnowledgeAPI()
    request = AsyncMock(return_value={"matches": [], "count": 0})
    client._request = request  # pyright: ignore[reportPrivateUsage]

    result = asyncio.run(
        client.grep(
            "authentication|token",
            "viking://resources/docs",
            case_insensitive=True,
            node_limit=8,
            level_limit=10,
        )
    )

    assert result == {"matches": [], "count": 0}
    request.assert_awaited_once_with(
        "POST",
        "/api/v1/search/grep",
        json_body={
            "uri": "viking://resources/docs",
            "pattern": "authentication|token",
            "case_insensitive": True,
            "node_limit": 8,
            "level_limit": 10,
        },
    )


def test_delete_api_requires_explicit_confirmation() -> None:
    """The API must reject deletion until its caller completes HITL."""
    client = OpenVikingKnowledgeAPI()

    with pytest.raises(ValueError, match="HITL"):
        asyncio.run(client.delete("viking://resources/docs/old.md"))
