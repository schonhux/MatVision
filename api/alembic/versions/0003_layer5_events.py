"""layer 5: event detection, corrections, and coaching tone

Revision ID: 0003
Revises: 0002
Create Date: 2026-08-06
"""

from alembic import op
import sqlalchemy as sa

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "matches",
        sa.Column("coach_tone", sa.String(), nullable=False, server_default="balanced"),
    )

    with op.batch_alter_table("events") as batch:
        batch.add_column(sa.Column("peak_ms", sa.Integer(), nullable=True))
        batch.add_column(sa.Column("confidence", sa.Float(), nullable=True))
        batch.add_column(sa.Column("measurements", sa.JSON(), server_default="{}"))
        batch.add_column(
            sa.Column("review_status", sa.String(), nullable=False, server_default="confirmed")
        )

    op.create_table(
        "corrections",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("event_id", sa.String(), sa.ForeignKey("events.id"), nullable=False),
        sa.Column("corrected_by", sa.String(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("field", sa.String(), nullable=False),
        sa.Column("old_value", sa.JSON(), nullable=True),
        sa.Column("new_value", sa.JSON(), nullable=True),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("use_for_training", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_corrections_event_id", "corrections", ["event_id"])


def downgrade() -> None:
    op.drop_table("corrections")
    with op.batch_alter_table("events") as batch:
        batch.drop_column("review_status")
        batch.drop_column("measurements")
        batch.drop_column("confidence")
        batch.drop_column("peak_ms")
    op.drop_column("matches", "coach_tone")
