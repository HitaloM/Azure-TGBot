"""Initial database

Revision ID: d22cf9e89625
Revises:
Create Date: 2025-04-15 19:50:01.349179

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d22cf9e89625"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "conversation",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.BigInteger(), nullable=False),
        sa.Column("user_message", sa.Text(), nullable=False),
        sa.Column("bot_response", sa.Text(), nullable=False),
        sa.Column("timestamp", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("conversation", schema=None) as batch_op:
        batch_op.create_index(
            "idx_conversation_user_timestamp", ["user_id", "timestamp"], unique=False
        )
        batch_op.create_index(batch_op.f("ix_conversation_timestamp"), ["timestamp"], unique=False)
        batch_op.create_index(batch_op.f("ix_conversation_user_id"), ["user_id"], unique=False)

    op.create_table(
        "whitelist",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("chat_id", sa.BigInteger(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("whitelist", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_whitelist_chat_id"), ["chat_id"], unique=True)


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("whitelist", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_whitelist_chat_id"))

    op.drop_table("whitelist")
    with op.batch_alter_table("conversation", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_conversation_user_id"))
        batch_op.drop_index(batch_op.f("ix_conversation_timestamp"))
        batch_op.drop_index("idx_conversation_user_timestamp")

    op.drop_table("conversation")
