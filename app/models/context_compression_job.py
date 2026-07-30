"""Persistent jobs for asynchronous conversation-context compression."""

from datetime import (
    UTC,
    datetime,
)
from typing import Any
from uuid import (
    UUID,
    uuid4,
)

from sqlalchemy import (
    CheckConstraint,
    Column,
    DateTime,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import (
    Field,
    SQLModel,
)


class ContextCompressionJob(SQLModel, table=True):
    """One asynchronous summary job for one completed conversation turn."""

    __tablename__ = "context_compression_job"  # pyright: ignore[reportAssignmentType]
    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'running', 'completed', 'failed')",
            name="ck_context_compression_job_status",
        ),
        CheckConstraint(
            "attempt_count >= 0",
            name="ck_context_compression_job_attempt_count",
        ),
        CheckConstraint(
            "source_message_count > 0",
            name="ck_context_compression_job_message_count",
        ),
        CheckConstraint(
            "source_char_count >= 0",
            name="ck_context_compression_job_char_count",
        ),
        UniqueConstraint(
            "session_id",
            "source_hash",
            "prompt_version",
            name="uq_context_compression_job_source",
        ),
        Index(
            "ix_context_compression_job_session_turn",
            "session_id",
            "turn_index",
        ),
        Index(
            "ix_context_compression_job_pending",
            "available_at",
            "created_at",
            postgresql_where=text("status = 'pending'"),
        ),
    )

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    session_id: str = Field(foreign_key="session.id", ondelete="CASCADE")
    thread_id: str
    turn_index: int
    source_hash: str = Field(max_length=64)
    prompt_version: int = Field(
        default=1,
        sa_column=Column(Integer, nullable=False, server_default=text("1")),
    )
    source_checkpoint_id: str | None = Field(default=None)
    source_messages: list[dict[str, Any]] | None = Field(
        default=None,
        sa_column=Column(JSONB, nullable=True),
    )
    source_message_count: int
    source_char_count: int
    status: str = Field(
        default="pending",
        sa_column=Column(
            String(20),
            nullable=False,
            server_default=text("'pending'"),
        ),
    )
    summary_text: str | None = Field(
        default=None,
        sa_column=Column(Text, nullable=True),
    )
    attempt_count: int = Field(
        default=0,
        sa_column=Column(Integer, nullable=False, server_default=text("0")),
    )
    available_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_column=Column(
            DateTime(timezone=True),
            nullable=False,
            server_default=text("CURRENT_TIMESTAMP"),
        ),
    )
    started_at: datetime | None = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), nullable=True),
    )
    completed_at: datetime | None = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), nullable=True),
    )
    error_message: str | None = Field(
        default=None,
        sa_column=Column(Text, nullable=True),
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_column=Column(
            DateTime(timezone=True),
            nullable=False,
            server_default=text("CURRENT_TIMESTAMP"),
        ),
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_column=Column(
            DateTime(timezone=True),
            nullable=False,
            server_default=text("CURRENT_TIMESTAMP"),
        ),
    )
