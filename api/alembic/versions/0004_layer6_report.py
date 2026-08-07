"""layer 6: match stats cache, observations, and grounded reports

Revision ID: 0004
Revises: 0003
Create Date: 2026-08-06
"""

from alembic import op
import sqlalchemy as sa

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "matches",
        sa.Column("stats_summary", sa.JSON(), nullable=False, server_default="{}"),
    )

    op.create_table(
        "observations",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("match_id", sa.String(), sa.ForeignKey("matches.id"), nullable=False),
        sa.Column("type", sa.String(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("evidence_event_ids", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("stats", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("source", sa.String(), nullable=False, server_default="model:rules-v1"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_observations_match_id", "observations", ["match_id"])

    op.create_table(
        "reports",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("match_id", sa.String(), sa.ForeignKey("matches.id"), nullable=False),
        sa.Column("content", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("model_version", sa.String(), nullable=False),
        sa.Column("coach_tone", sa.String(), nullable=False),
        sa.Column("ratings", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_reports_match_id", "reports", ["match_id"], unique=True)


def downgrade() -> None:
    op.drop_index("ix_reports_match_id", table_name="reports")
    op.drop_table("reports")
    op.drop_index("ix_observations_match_id", table_name="observations")
    op.drop_table("observations")
    op.drop_column("matches", "stats_summary")
