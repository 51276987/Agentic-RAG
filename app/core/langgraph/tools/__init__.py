"""LangGraph tools for enhanced language model capabilities.

This package contains custom tools that can be used with LangGraph to extend
the capabilities of language models. Internet search is intentionally not
registered.
"""

from langchain_core.tools.base import BaseTool

from .ask_human import ask_human

tools: list[BaseTool] = [ask_human]
