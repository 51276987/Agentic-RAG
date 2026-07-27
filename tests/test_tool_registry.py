"""Tests for the tools exposed to the agent."""

from app.core.langgraph.tools import tools


def test_internet_search_tool_is_not_registered() -> None:
    """The agent must not expose an internet search tool."""
    tool_names = {tool.name for tool in tools}

    assert "duckduckgo_results_json" not in tool_names
    assert all("search" not in name.lower() for name in tool_names)
