"""Long-term memory service using mem0 and pgvector with optional cache layer."""

from typing import Any

from mem0 import AsyncMemory

from app.core.cache import (
    cache_key,
    cache_service,
)
from app.core.config import settings
from app.core.logging import logger


class MemoryService:
    """Service for managing long-term memory using mem0 and pgvector."""

    def __init__(self):
        """Initialize the memory service."""
        self._memory: AsyncMemory | None = None

    def _build_embedder_config(self) -> dict[str, Any]:
        """Build a mem0 embedder config for OpenAI-compatible or Ollama models."""
        provider = settings.LONG_TERM_MEMORY_EMBEDDER_PROVIDER
        if provider not in {"openai", "ollama"}:
            raise ValueError("LONG_TERM_MEMORY_EMBEDDER_PROVIDER 仅支持 openai 或 ollama")

        config: dict[str, Any] = {
            "model": settings.LONG_TERM_MEMORY_EMBEDDER_MODEL,
            "embedding_dims": settings.LONG_TERM_MEMORY_EMBEDDER_DIMS,
        }
        base_url = settings.LONG_TERM_MEMORY_EMBEDDER_BASE_URL.rstrip("/")
        if provider == "ollama":
            if not base_url:
                raise ValueError("Ollama embedder 需要配置 LONG_TERM_MEMORY_EMBEDDER_BASE_URL")
            config["ollama_base_url"] = base_url
        elif base_url:
            config["openai_base_url"] = base_url

        return {"provider": provider, "config": config}

    def _build_vector_store_config(self) -> dict[str, Any]:
        """Build pgvector config with dimensions matching the active embedder."""
        return {
            "provider": "pgvector",
            "config": {
                "collection_name": settings.LONG_TERM_MEMORY_COLLECTION_NAME,
                "embedding_model_dims": settings.LONG_TERM_MEMORY_EMBEDDER_DIMS,
                "dbname": settings.POSTGRES_DB,
                "user": settings.POSTGRES_USER,
                "password": settings.POSTGRES_PASSWORD,
                "host": settings.POSTGRES_HOST,
                "port": settings.POSTGRES_PORT,
            },
        }

    async def _get_memory(self) -> AsyncMemory:
        if self._memory is None:
            self._memory = await AsyncMemory.from_config(
                config_dict={
                    "vector_store": self._build_vector_store_config(),
                    "llm": {
                        "provider": "openai",
                        "config": {"model": settings.LONG_TERM_MEMORY_MODEL},
                    },
                    "embedder": self._build_embedder_config(),
                }
            )
        return self._memory

    async def initialize(self) -> None:
        """Pre-warm the mem0 AsyncMemory instance and its pgvector connection pool.

        Call once at startup so the first search() or add() doesn't pay the
        ~130ms from_config + pgvector.list_cols() cold-init cost.
        """
        await self._get_memory()
        logger.info(
            "memory_service_initialized",
            embedder_provider=settings.LONG_TERM_MEMORY_EMBEDDER_PROVIDER,
            embedder_model=settings.LONG_TERM_MEMORY_EMBEDDER_MODEL,
            embedding_dims=settings.LONG_TERM_MEMORY_EMBEDDER_DIMS,
            collection_name=settings.LONG_TERM_MEMORY_COLLECTION_NAME,
        )

    async def search(self, user_id: str | None, query: str) -> str:
        """Search relevant memories for a user.

        Checks cache first; on miss, queries mem0 and caches the result.

        Returns formatted memory string, or empty string on failure or when
        no user_id is supplied (anonymous sessions skip long-term memory
        rather than pooling under a shared partition).
        """
        if user_id is None:
            return ""
        try:
            # Check cache first
            key = cache_key("memory", str(user_id), query)
            cached = await cache_service.get(key)
            if cached is not None:
                logger.debug("memory_search_cache_hit", user_id=user_id)
                return cached

            memory = await self._get_memory()
            results = await memory.search(user_id=str(user_id), query=query)
            result = "\n".join([f"* {r['memory']}" for r in results["results"]])

            # Cache successful results
            if result:
                await cache_service.set(key, result)

            return result
        except Exception as e:
            logger.error("failed_to_get_relevant_memory", error=str(e), user_id=user_id, query=query)
            return ""

    async def add(self, user_id: str | None, messages: list[dict], metadata: dict | None = None) -> None:
        """Add messages to long-term memory for a user.

        No-op when ``user_id`` is ``None`` (see ``search`` for rationale).
        """
        if user_id is None:
            return
        try:
            memory = await self._get_memory()
            await memory.add(messages, user_id=str(user_id), metadata=metadata)
            logger.info("long_term_memory_updated_successfully", user_id=user_id)
        except Exception as e:
            logger.exception("failed_to_update_long_term_memory", user_id=user_id, error=str(e))


memory_service = MemoryService()
