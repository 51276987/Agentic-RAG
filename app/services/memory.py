"""Long-term memory service using mem0 and pgvector with optional cache layer."""

import re
from typing import Any

from mem0 import AsyncMemory

from app.core.cache import (
    cache_key,
    cache_service,
)
from app.core.config import settings
from app.core.logging import logger


_KEY_MEMORY_PATTERNS = (
    r"(?:请|帮我)?记住",
    r"(?:我是|我是一名|我的角色是|我的岗位是|我负责|我的职责是|我偏好|我习惯|我希望)",
    r"(?:本|当前|我们(?:的)?)(?:项目|系统|服务|环境|知识库|数据库|模型)",
    r"(?:生产|测试|开发)环境.*(?:使用|部署|配置|地址|端口|版本)",
    r"(?:技术栈|接口规范|编码规范|发布规范|项目约定|团队规范|长期规则|固定规则)",
    r"(?:默认|统一|后续).*(?:使用|采用|配置|规则|方式)",
)
_TRIVIAL_TURN_PATTERN = re.compile(r"^(?:你好|您好|在吗|谢谢|感谢|好的|知道了|再见|拜拜|test|测试)[!！。,.， ]*$", re.IGNORECASE)


def _message_text(message: dict[str, Any]) -> str:
    """Extract plain text from an OpenAI-style message without retaining blocks."""
    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return " ".join(
            str(block.get("text", "")).strip()
            for block in content
            if isinstance(block, dict) and block.get("type") in {"text", "output_text"}
        ).strip()
    return str(content).strip()


def _select_key_memory_turn(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Return only the latest durable user/assistant turn worth storing in mem0.

    Long-term memory is for stable user facts, project context, and durable
    working agreements. Ordinary knowledge questions and casual conversation
    remain in the session checkpoint instead of becoming user memories.
    """
    latest_user = next((item for item in reversed(messages) if item.get("role") == "user"), None)
    latest_assistant = next((item for item in reversed(messages) if item.get("role") == "assistant"), None)
    if latest_user is None or latest_assistant is None:
        return []

    user_text = _message_text(latest_user)
    assistant_text = _message_text(latest_assistant)
    if len(user_text) < 12 or not assistant_text or _TRIVIAL_TURN_PATTERN.fullmatch(user_text):
        return []
    if not any(re.search(pattern, user_text, re.IGNORECASE) for pattern in _KEY_MEMORY_PATTERNS):
        return []

    return [
        {"role": "user", "content": user_text[:1_500]},
        {"role": "assistant", "content": assistant_text[:3_000]},
    ]


def _format_relevant_memories(results: dict[str, Any], max_cosine_distance: float) -> str:
    """Keep only nearest pgvector memories and format them for prompt injection.

    mem0's pgvector adapter exposes ``score`` as cosine *distance* from the
    ``<=>`` operator, where a smaller value is more relevant.  It must not be
    passed to mem0's generic ``threshold`` argument because that implementation
    keeps scores greater than or equal to the threshold.
    """
    candidates: list[tuple[float, str]] = []
    for item in results.get("results", []):
        if not isinstance(item, dict):
            continue
        score = item.get("score")
        memory = item.get("memory")
        if isinstance(score, (int, float)) and isinstance(memory, str) and score <= max_cosine_distance:
            candidates.append((float(score), memory))
    candidates.sort(key=lambda item: item[0])
    return "\n".join(f"* {memory}" for _, memory in candidates)


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
            key = cache_key(
                "memory",
                str(user_id),
                query,
                str(settings.LONG_TERM_MEMORY_MAX_RESULTS),
                str(settings.LONG_TERM_MEMORY_MAX_COSINE_DISTANCE),
            )
            cached = await cache_service.get(key)
            if cached is not None:
                logger.debug("memory_search_cache_hit", user_id=user_id)
                return cached

            memory = await self._get_memory()
            results = await memory.search(
                user_id=str(user_id),
                query=query,
                limit=settings.LONG_TERM_MEMORY_MAX_RESULTS,
            )
            result = _format_relevant_memories(
                results,
                settings.LONG_TERM_MEMORY_MAX_COSINE_DISTANCE,
            )

            # Cache successful results
            if result:
                await cache_service.set(key, result)

            return result
        except Exception as e:
            logger.error("failed_to_get_relevant_memory", error=str(e), user_id=user_id, query=query)
            return ""

    async def add(self, user_id: str | None, messages: list[dict], metadata: dict | None = None) -> None:
        """Store only a durable latest conversation turn as long-term memory.

        No-op when ``user_id`` is ``None`` (see ``search`` for rationale).
        """
        if user_id is None:
            return
        key_turn = _select_key_memory_turn(messages)
        if not key_turn:
            logger.info("long_term_memory_skipped_non_key_turn", user_id=user_id)
            return
        try:
            memory = await self._get_memory()
            await memory.add(key_turn, user_id=str(user_id), metadata=metadata)
            logger.info("long_term_memory_updated_successfully", user_id=user_id)
        except Exception as e:
            logger.exception("failed_to_update_long_term_memory", user_id=user_id, error=str(e))


memory_service = MemoryService()
