"""add_chat_id_to_conversation

Revision ID: 23246dd846d2
Revises: d22cf9e89625
Create Date: 2025-06-10 20:03:11.303001

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "23246dd846d2"
down_revision: str | None = "d22cf9e89625"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    with op.batch_alter_table("conversation", schema=None) as batch_op:
        batch_op.add_column(sa.Column("chat_id", sa.BigInteger(), nullable=True))

    op.execute("UPDATE conversation SET chat_id = user_id WHERE chat_id IS NULL")

    with op.batch_alter_table("conversation", schema=None) as batch_op:
        batch_op.alter_column("chat_id", nullable=False)
        batch_op.create_index("ix_conversation_chat_id", ["chat_id"])
        batch_op.drop_index("idx_conversation_user_timestamp")
        batch_op.create_index(
            "idx_conversation_user_chat_timestamp", ["user_id", "chat_id", "timestamp"]
        )
        batch_op.create_index("idx_conversation_chat_timestamp", ["chat_id", "timestamp"])


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("conversation", schema=None) as batch_op:
        batch_op.drop_index("idx_conversation_chat_timestamp")
        batch_op.drop_index("idx_conversation_user_chat_timestamp")
        batch_op.drop_index("ix_conversation_chat_id")
        batch_op.create_index("idx_conversation_user_timestamp", ["user_id", "timestamp"])
        batch_op.drop_column("chat_id")
