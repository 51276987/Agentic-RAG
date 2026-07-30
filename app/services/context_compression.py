"""Asynchronous, persistent compression of recent conversation turns."""

import asyncio
import hashlib
import json
from contextlib import suppress
from dataclasses import dataclass
from datetime import (
    UTC,
    datetime,
    timedelta,
)
from typing import (
    Any,
    Iterable,
)
from uuid import (
    UUID,
    uuid4,
)

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.runnables.config import RunnableConfig
from sqlalchemy import (
    delete,
    select,
    text,
    update,
)
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlmodel import col

from app.core.config import settings
from app.core.logging import logger
from app.core.observability import get_langfuse_callback_handler
from app.core.prompts import CONTEXT_COMPRESSION_PROMPT
from app.models.context_compression_job import ContextCompressionJob
from app.services.llm import LLMService
from app.services.llm import llm_service
from app.utils import extract_text_content

HISTORY_TURN_COUNT = 3
_TRUNCATION_MARKER = "\n……（历史内容因上下文长度上限已截断）……\n"


@dataclass(frozen=True)
class ConversationTurn:
    """One completed logical turn and its stable source identity."""

    turn_index: int
    messages: tuple[dict[str, str], ...]
    source_hash: str
    char_count: int


def _message_role(message: Any) -> str | None:
    """Return a normalized user/assistant role for one message."""
    if isinstance(message, BaseMessage):
        if message.type == "human":
            return "user"
        if message.type == "ai":
            return "assistant"
        return None
    if not isinstance(message, dict):
        return None
    role = message.get("role") or message.get("type")
    if role in {"user", "human"}:
        return "user"
    if role in {"assistant", "ai"}:
        return "assistant"
    return None


def _normalized_message(message: Any) -> dict[str, str] | None:
    """Normalize a checkpoint message without retaining provider metadata."""
    role = _message_role(message)
    if role is None:
        return None
    if isinstance(message, BaseMessage):
        content = extract_text_content(message.content)
    else:
        raw_content = message.get("content", "")
        content = extract_text_content(raw_content) if isinstance(raw_content, (str, list)) else str(raw_content)
    content = content.strip()
    if not content:
        return None
    return {"role": role, "content": content}


def _turn_source_hash(messages: Iterable[dict[str, str]]) -> str:
    """Hash only normalized role/content/order so summaries cannot go stale."""
    canonical = json.dumps(
        list(messages),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def group_complete_conversation_turns(messages: Iterable[Any]) -> list[ConversationTurn]:
    """Group consecutive user messages followed by one assistant response."""
    turns: list[ConversationTurn] = []
    pending: list[dict[str, str]] = []

    for raw_message in messages:
        message = _normalized_message(raw_message)
        if message is None:
            continue
        if message["role"] == "user":
            pending.append(message)
            continue
        if not pending:
            continue

        pending.append(message)
        turn_messages = tuple(pending)
        turns.append(
            ConversationTurn(
                turn_index=len(turns) + 1,
                messages=turn_messages,
                source_hash=_turn_source_hash(turn_messages),
                char_count=sum(len(item["content"]) for item in turn_messages),
            )
        )
        pending = []

    return turns


def _truncate_text(value: str, max_chars: int) -> tuple[str, bool]:
    """Bound text deterministically while preserving both its start and end."""
    if max_chars <= 0:
        return "", bool(value)
    if len(value) <= max_chars:
        return value, False
    if max_chars <= len(_TRUNCATION_MARKER):
        return value[:max_chars], True
    content_budget = max_chars - len(_TRUNCATION_MARKER)
    head_chars = content_budget // 2
    tail_chars = content_budget - head_chars
    return (
        f"{value[:head_chars]}{_TRUNCATION_MARKER}{value[-tail_chars:]}",
        True,
    )


def _truncate_messages(
    messages: Iterable[dict[str, str]],
    max_chars: int,
) -> list[dict[str, str]]:
    """Bound a turn while retaining every available role boundary."""
    items = [dict(item) for item in messages]
    if not items or max_chars <= 0:
        return []
    if sum(len(item["content"]) for item in items) <= max_chars:
        return items

    bounded: list[dict[str, str]] = []
    remaining = max_chars
    for position, item in enumerate(items):
        remaining_items = len(items) - position
        item_budget = max(1, remaining // remaining_items)
        content, _ = _truncate_text(item["content"], item_budget)
        bounded.append({"role": item["role"], "content": content})
        remaining -= len(content)
    return bounded


def _raw_context_items(
    turn: ConversationTurn,
) -> list[dict[str, Any]]:
    """Represent a full turn for prompt-only context injection."""
    return [
        {
            "type": item["role"],
            "content": item["content"],
            "turn_index": turn.turn_index,
            "compressed": False,
        }
        for item in turn.messages
    ]


def _summary_context_item(
    turn: ConversationTurn,
    summary: str,
    summary_max_chars: int,
) -> list[dict[str, Any]]:
    """Represent one completed summary for prompt-only context injection."""
    bounded_summary, truncated = _truncate_text(summary.strip(), summary_max_chars)
    if not bounded_summary:
        return []
    item: dict[str, Any] = {
        "type": "conversation_summary",
        "content": bounded_summary,
        "turn_index": turn.turn_index,
        "compressed": True,
    }
    if truncated:
        item["truncated"] = True
    return [item]


def _bound_context_items(
    items: list[dict[str, Any]],
    max_chars: int,
) -> list[dict[str, Any]]:
    """Apply the hard history budget to one candidate turn."""
    if not items or max_chars <= 0:
        return []
    total_chars = sum(len(str(item.get("content", ""))) for item in items)
    if total_chars <= max_chars:
        return [dict(item) for item in items]

    remaining = max_chars
    bounded: list[dict[str, Any]] = []
    for position, item in enumerate(items):
        remaining_items = len(items) - position
        item_budget = max(1, remaining // remaining_items)
        content, truncated = _truncate_text(str(item.get("content", "")), item_budget)
        if not content:
            continue
        bounded_item = dict(item)
        bounded_item["content"] = content
        bounded_item["truncated"] = truncated
        bounded.append(bounded_item)
        remaining -= len(content)
    return bounded


def build_conversation_context(
    turns: list[ConversationTurn],
    completed_summaries: dict[str, str],
    *,
    recent_full_max_chars: int,
    history_hard_max_chars: int,
    summary_max_chars: int,
) -> list[dict[str, Any]]:
    """Project full checkpoint history into the bounded prompt context policy."""
    selected_turns = turns[-HISTORY_TURN_COUNT:]
    if not selected_turns or history_hard_max_chars <= 0:
        return []

    candidates: list[list[dict[str, Any]]] = []
    for position, turn in enumerate(selected_turns):
        is_most_recent = position == len(selected_turns) - 1
        summary = completed_summaries.get(turn.source_hash, "").strip()
        should_use_summary = bool(summary) and (not is_most_recent or turn.char_count > recent_full_max_chars)
        candidates.append(
            _summary_context_item(turn, summary, summary_max_chars) if should_use_summary else _raw_context_items(turn)
        )

    # Preserve the newest history first when pending summaries force raw
    # fallback beyond the hard cap.  Reversing again restores chronology.
    remaining = history_hard_max_chars
    selected_groups: list[list[dict[str, Any]]] = []
    for candidate in reversed(candidates):
        bounded = _bound_context_items(candidate, remaining)
        if not bounded:
            continue
        selected_groups.append(bounded)
        remaining -= sum(len(str(item["content"])) for item in bounded)
        if remaining <= 0:
            break

    return [item for group in reversed(selected_groups) for item in group]


class ContextCompressionService:
    """Persist, execute, and read asynchronous turn-summary jobs."""

    def __init__(self, model_service: LLMService):
        """Create the service around the shared retrying LLM client."""
        self.llm_service = model_service
        self._engine: AsyncEngine | None = None
        self._session_factory: async_sessionmaker[AsyncSession] | None = None
        self._worker_task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()
        self._wake_event = asyncio.Event()

    @property
    def initialized(self) -> bool:
        """Return whether persistent compression is available."""
        return self._session_factory is not None

    async def initialize(self) -> None:
        """Open a small async pool, recover stale jobs, and start the worker."""
        if not settings.CONTEXT_COMPRESSION_ENABLED or self.initialized:
            return

        database_url = URL.create(
            "postgresql+psycopg",
            username=settings.POSTGRES_USER,
            password=settings.POSTGRES_PASSWORD,
            host=settings.POSTGRES_HOST,
            port=settings.POSTGRES_PORT,
            database=settings.POSTGRES_DB,
        )
        engine = create_async_engine(
            database_url,
            pool_pre_ping=True,
            pool_size=max(1, settings.CONTEXT_COMPRESSION_DB_POOL_SIZE),
            max_overflow=1,
        )
        session_factory = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

        try:
            async with engine.connect() as connection:
                await connection.execute(text("SELECT 1"))
            self._engine = engine
            self._session_factory = session_factory
            await self._recover_stale_jobs()
        except Exception:
            await engine.dispose()
            logger.exception("context_compression_initialization_failed")
            raise

        self._stop_event.clear()
        self._worker_task = asyncio.create_task(
            self._worker_loop(),
            name="context-compression-worker",
        )
        logger.info(
            "context_compression_initialized",
            model=settings.CONTEXT_COMPRESSION_MODEL,
            pool_size=max(1, settings.CONTEXT_COMPRESSION_DB_POOL_SIZE),
        )

    async def close(self) -> None:
        """Stop the worker and release its database pool."""
        self._stop_event.set()
        self._wake_event.set()
        if self._worker_task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(self._worker_task), timeout=5)
            except asyncio.TimeoutError:
                self._worker_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self._worker_task
            self._worker_task = None
        if self._engine is not None:
            await self._engine.dispose()
        self._engine = None
        self._session_factory = None
        logger.info("context_compression_closed")

    async def prepare_context(
        self,
        *,
        session_id: str,
        thread_id: str,
        checkpoint_id: str | None,
        messages: Iterable[Any],
    ) -> list[dict[str, Any]]:
        """Enqueue relevant turns and return summaries or bounded raw fallback."""
        turns = group_complete_conversation_turns(messages)
        selected_turns = turns[-HISTORY_TURN_COUNT:]
        completed_summaries: dict[str, str] = {}

        if settings.CONTEXT_COMPRESSION_ENABLED and self.initialized and selected_turns:
            try:
                await self._enqueue_turns(
                    session_id=session_id,
                    thread_id=thread_id,
                    checkpoint_id=checkpoint_id,
                    turns=selected_turns,
                )
                completed_summaries = await self._get_completed_summaries(
                    session_id=session_id,
                    thread_id=thread_id,
                    turns=selected_turns,
                )
            except Exception:
                logger.exception(
                    "context_compression_prepare_failed_using_raw_fallback",
                    session_id=session_id,
                )

        return build_conversation_context(
            turns,
            completed_summaries,
            recent_full_max_chars=max(1, settings.CONTEXT_RECENT_FULL_MAX_CHARS),
            history_hard_max_chars=max(1, settings.CONTEXT_HISTORY_HARD_MAX_CHARS),
            summary_max_chars=max(1, settings.CONTEXT_COMPRESSION_SUMMARY_MAX_CHARS),
        )

    async def enqueue_recent_turns(
        self,
        *,
        session_id: str,
        thread_id: str,
        checkpoint_id: str | None,
        messages: Iterable[Any],
    ) -> None:
        """Persist summary work after a graph invocation has fully completed."""
        if not settings.CONTEXT_COMPRESSION_ENABLED or not self.initialized:
            return
        turns = group_complete_conversation_turns(messages)
        selected_turns = turns[-HISTORY_TURN_COUNT:]
        if not selected_turns:
            return
        try:
            await self._enqueue_turns(
                session_id=session_id,
                thread_id=thread_id,
                checkpoint_id=checkpoint_id,
                turns=selected_turns,
            )
        except Exception:
            logger.exception(
                "context_compression_enqueue_failed",
                session_id=session_id,
            )

    async def delete_session(self, session_id: str) -> None:
        """Delete compression state when the canonical chat history is cleared."""
        session_factory = self._session_factory
        if session_factory is None:
            return
        try:
            async with session_factory.begin() as database_session:
                await database_session.execute(
                    delete(ContextCompressionJob).where(col(ContextCompressionJob.session_id) == session_id)
                )
            logger.info(
                "context_compression_session_deleted",
                session_id=session_id,
            )
        except Exception:
            logger.exception(
                "context_compression_session_delete_failed",
                session_id=session_id,
            )
            raise

    async def _recover_stale_jobs(self) -> None:
        """Return jobs abandoned by a stopped process to the pending queue."""
        session_factory = self._require_session_factory()
        cutoff = datetime.now(UTC) - timedelta(seconds=max(1, settings.CONTEXT_COMPRESSION_STALE_SECONDS))
        async with session_factory.begin() as database_session:
            await database_session.execute(
                update(ContextCompressionJob)
                .where(
                    col(ContextCompressionJob.status) == "running",
                    col(ContextCompressionJob.updated_at) < cutoff,
                )
                .values(
                    status="pending",
                    started_at=None,
                    available_at=datetime.now(UTC),
                    updated_at=datetime.now(UTC),
                    error_message="recovered after worker restart",
                )
            )

    async def _enqueue_turns(
        self,
        *,
        session_id: str,
        thread_id: str,
        checkpoint_id: str | None,
        turns: list[ConversationTurn],
    ) -> None:
        """Idempotently persist jobs before waking the background worker."""
        session_factory = self._require_session_factory()
        now = datetime.now(UTC)
        rows: list[dict[str, Any]] = []
        source_max_chars = max(1, settings.CONTEXT_COMPRESSION_SOURCE_MAX_CHARS)
        for turn in turns:
            source_messages = _truncate_messages(turn.messages, source_max_chars)
            rows.append(
                {
                    "id": uuid4(),
                    "session_id": session_id,
                    "thread_id": thread_id,
                    "turn_index": turn.turn_index,
                    "source_hash": turn.source_hash,
                    "prompt_version": settings.CONTEXT_COMPRESSION_PROMPT_VERSION,
                    "source_checkpoint_id": checkpoint_id,
                    "source_messages": source_messages,
                    "source_message_count": len(turn.messages),
                    "source_char_count": turn.char_count,
                    "status": "pending",
                    "attempt_count": 0,
                    "available_at": now,
                    "created_at": now,
                    "updated_at": now,
                }
            )

        statement = (
            postgresql_insert(ContextCompressionJob)
            .values(rows)
            .on_conflict_do_nothing(
                constraint="uq_context_compression_job_source",
            )
        )
        async with session_factory.begin() as database_session:
            result = await database_session.execute(statement)
        if result.rowcount:
            logger.info(
                "context_compression_jobs_enqueued",
                session_id=session_id,
                job_count=result.rowcount,
            )
        self._wake_event.set()

    async def _get_completed_summaries(
        self,
        *,
        session_id: str,
        thread_id: str,
        turns: list[ConversationTurn],
    ) -> dict[str, str]:
        """Return only summaries whose exact normalized source still matches."""
        session_factory = self._require_session_factory()
        hashes = [turn.source_hash for turn in turns]
        statement = select(ContextCompressionJob).where(
            col(ContextCompressionJob.session_id) == session_id,
            col(ContextCompressionJob.thread_id) == thread_id,
            col(ContextCompressionJob.prompt_version) == settings.CONTEXT_COMPRESSION_PROMPT_VERSION,
            col(ContextCompressionJob.source_hash).in_(hashes),
            col(ContextCompressionJob.status) == "completed",
        )
        async with session_factory() as database_session:
            result = await database_session.execute(statement)
            jobs = result.scalars().all()
        return {job.source_hash: job.summary_text for job in jobs if job.summary_text and job.summary_text.strip()}

    async def _claim_next_job(self) -> ContextCompressionJob | None:
        """Atomically claim one due job without blocking other app workers."""
        session_factory = self._require_session_factory()
        async with session_factory.begin() as database_session:
            result = await database_session.execute(
                select(ContextCompressionJob)
                .where(
                    col(ContextCompressionJob.status) == "pending",
                    col(ContextCompressionJob.available_at) <= datetime.now(UTC),
                )
                .order_by(
                    col(ContextCompressionJob.available_at),
                    col(ContextCompressionJob.created_at),
                )
                .with_for_update(skip_locked=True)
                .limit(1)
            )
            job = result.scalar_one_or_none()
            if job is None:
                return None
            now = datetime.now(UTC)
            job.status = "running"
            job.started_at = now
            job.updated_at = now
            job.attempt_count += 1
            database_session.add(job)
        return job

    async def _mark_completed(self, job_id: UUID, summary: str) -> None:
        """Persist a successful summary and discard its duplicate source snapshot."""
        session_factory = self._require_session_factory()
        now = datetime.now(UTC)
        async with session_factory.begin() as database_session:
            await database_session.execute(
                update(ContextCompressionJob)
                .where(col(ContextCompressionJob.id) == job_id)
                .values(
                    status="completed",
                    summary_text=summary,
                    source_messages=None,
                    completed_at=now,
                    updated_at=now,
                    error_message=None,
                )
            )

    async def _mark_failed(self, job_id: UUID, error: Exception) -> None:
        """Persist a terminal worker failure without affecting chat requests."""
        session_factory = self._require_session_factory()
        async with session_factory.begin() as database_session:
            await database_session.execute(
                update(ContextCompressionJob)
                .where(col(ContextCompressionJob.id) == job_id)
                .values(
                    status="failed",
                    updated_at=datetime.now(UTC),
                    error_message=str(error)[:2000],
                )
            )

    async def _process_job(self, job: ContextCompressionJob) -> None:
        """Generate one summary outside the database transaction."""
        try:
            if not job.source_messages:
                raise ValueError("context compression source messages are empty")

            callback = get_langfuse_callback_handler()
            callbacks: list[BaseCallbackHandler] = [callback] if callback is not None else []
            runnable_config: RunnableConfig = {
                "callbacks": callbacks,
                "run_name": "context-compression",
                "tags": ["context-compression", settings.ENVIRONMENT.value],
                "metadata": {
                    "session_id": job.session_id,
                    "langfuse_session_id": job.session_id,
                    "context_compression_job_id": str(job.id),
                    "turn_index": job.turn_index,
                    "prompt_version": job.prompt_version,
                },
            }
            response = await self.llm_service.call(
                [
                    SystemMessage(content=CONTEXT_COMPRESSION_PROMPT),
                    HumanMessage(
                        content=json.dumps(
                            {
                                "turn_index": job.turn_index,
                                "messages": job.source_messages,
                            },
                            ensure_ascii=False,
                        )
                    ),
                ],
                model_name=settings.CONTEXT_COMPRESSION_MODEL,
                runnable_config=runnable_config,
                max_tokens=max(32, settings.CONTEXT_COMPRESSION_SUMMARY_MAX_TOKENS),
                temperature=0.1,
            )
            summary = extract_text_content(response.content).strip()
            if not summary:
                raise ValueError("context compression model returned empty content")
            summary, _ = _truncate_text(
                summary,
                max(1, settings.CONTEXT_COMPRESSION_SUMMARY_MAX_CHARS),
            )
            await self._mark_completed(job.id, summary)
            logger.info(
                "context_compression_job_completed",
                job_id=str(job.id),
                session_id=job.session_id,
                turn_index=job.turn_index,
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            logger.exception(
                "context_compression_job_failed",
                job_id=str(job.id),
                session_id=job.session_id,
                turn_index=job.turn_index,
            )
            try:
                await self._mark_failed(job.id, error)
            except Exception:
                logger.exception(
                    "context_compression_failure_persist_failed",
                    job_id=str(job.id),
                )

    async def _worker_loop(self) -> None:
        """Continuously execute persisted work and survive transient DB errors."""
        while not self._stop_event.is_set():
            self._wake_event.clear()
            try:
                job = await self._claim_next_job()
                if job is not None:
                    await self._process_job(job)
                    continue
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("context_compression_worker_iteration_failed")

            try:
                await asyncio.wait_for(
                    self._wake_event.wait(),
                    timeout=max(0.1, settings.CONTEXT_COMPRESSION_POLL_SECONDS),
                )
            except asyncio.TimeoutError:
                continue

    def _require_session_factory(self) -> async_sessionmaker[AsyncSession]:
        """Return the initialized session factory or fail internally."""
        if self._session_factory is None:
            raise RuntimeError("context compression service is not initialized")
        return self._session_factory


context_compression_service = ContextCompressionService(llm_service)
