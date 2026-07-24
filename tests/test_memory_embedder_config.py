"""Tests for long-term memory embedder configuration."""

import pytest

from app.core.config import settings
from app.services.memory import MemoryService


def test_build_ollama_embedder_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ollama settings should map to the mem0 provider-specific fields."""
    monkeypatch.setattr(settings, "LONG_TERM_MEMORY_EMBEDDER_PROVIDER", "ollama")
    monkeypatch.setattr(settings, "LONG_TERM_MEMORY_EMBEDDER_MODEL", "nomic-embed-text:latest")
    monkeypatch.setattr(settings, "LONG_TERM_MEMORY_EMBEDDER_BASE_URL", "http://host.docker.internal:11434/")
    monkeypatch.setattr(settings, "LONG_TERM_MEMORY_EMBEDDER_DIMS", 768)

    config = MemoryService()._build_embedder_config()  # pyright: ignore[reportPrivateUsage]

    assert config == {
        "provider": "ollama",
        "config": {
            "model": "nomic-embed-text:latest",
            "embedding_dims": 768,
            "ollama_base_url": "http://host.docker.internal:11434",
        },
    }


def test_ollama_embedder_requires_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing Ollama endpoint should fail before mem0 initialization."""
    monkeypatch.setattr(settings, "LONG_TERM_MEMORY_EMBEDDER_PROVIDER", "ollama")
    monkeypatch.setattr(settings, "LONG_TERM_MEMORY_EMBEDDER_BASE_URL", "")

    with pytest.raises(ValueError, match="LONG_TERM_MEMORY_EMBEDDER_BASE_URL"):
        MemoryService()._build_embedder_config()  # pyright: ignore[reportPrivateUsage]
