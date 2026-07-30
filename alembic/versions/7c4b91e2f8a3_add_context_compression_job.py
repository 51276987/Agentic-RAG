"""Add asynchronous context compression jobs.

Revision ID: 7c4b91e2f8a3
Revises: b25d38b0cd7c
Create Date: 2026-07-30

"""

from typing import (
    Sequence,
    Union,
)

import sqlalchemy as sa
import sqlmodel
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "7c4b91e2f8a3"  # pragma: allowlist secret
down_revision: Union[str, Sequence[str], None] = "b25d38b0cd7c"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create the persistent context-compression job table."""
    op.create_table(
        "context_compression_job",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("session_id", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("thread_id", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("turn_index", sa.Integer(), nullable=False),
        sa.Column("source_hash", sa.String(length=64), nullable=False),
        sa.Column("prompt_version", sa.Integer(), server_default=sa.text("1"), nullable=False),
        sa.Column(
            "source_checkpoint_id",
            sqlmodel.sql.sqltypes.AutoString(),
            nullable=True,
        ),
        sa.Column(
            "source_messages",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column("source_message_count", sa.Integer(), nullable=False),
        sa.Column("source_char_count", sa.Integer(), nullable=False),
        sa.Column(
            "status",
            sa.String(length=20),
            server_default=sa.text("'pending'"),
            nullable=False,
        ),
        sa.Column("summary_text", sa.Text(), nullable=True),
        sa.Column(
            "attempt_count",
            sa.Integer(),
            server_default=sa.text("0"),
            nullable=False,
        ),
        sa.Column(
            "available_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "attempt_count >= 0",
            name="ck_context_compression_job_attempt_count",
        ),
        sa.CheckConstraint(
            "source_char_count >= 0",
            name="ck_context_compression_job_char_count",
        ),
        sa.CheckConstraint(
            "source_message_count > 0",
            name="ck_context_compression_job_message_count",
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'running', 'completed', 'failed')",
            name="ck_context_compression_job_status",
        ),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["session.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "session_id",
            "source_hash",
            "prompt_version",
            name="uq_context_compression_job_source",
        ),
    )
    op.create_index(
        "ix_context_compression_job_session_turn",
        "context_compression_job",
        ["session_id", "turn_index"],
        unique=False,
    )
    op.create_index(
        "ix_context_compression_job_pending",
        "context_compression_job",
        ["available_at", "created_at"],
        unique=False,
        postgresql_where=sa.text("status = 'pending'"),
    )


def downgrade() -> None:
    """Remove context-compression jobs."""
    op.drop_index(
        "ix_context_compression_job_pending",
        table_name="context_compression_job",
        postgresql_where=sa.text("status = 'pending'"),
    )
    op.drop_index(
        "ix_context_compression_job_session_turn",
        table_name="context_compression_job",
    )
    op.drop_table("context_compression_job")
