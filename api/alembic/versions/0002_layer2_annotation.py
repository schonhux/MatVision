"""layer 2: annotation fields, state segments, match athletes

Revision ID: 0002
Revises: 0001
Create Date: 2026-08-06

Adds the schema needed by the annotation system:
  - Event gains Level-1/Level-2 label fields plus an annotator reference
  - state_segments: hand-labeled position spans (Layer 4's model writes here too)
  - match_athletes: which wrestler is which, plus the seed bbox from click-to-identify
  - matches gains venue + annotation_complete (both required for leakage-safe splits)
"""
from alembic import op
import sqlalchemy as sa

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None

match_state_enum = sa.Enum(
    "neutral", "top", "bottom", "scramble", "stopped", name="matchstate"
)


def upgrade() -> None:
    # --- matches: split metadata -------------------------------------------
    op.add_column("matches", sa.Column("venue", sa.String(), nullable=True))
    op.add_column(
        "matches",
        sa.Column("annotation_complete", sa.Boolean(), nullable=False, server_default=sa.false()),
    )

    # --- events: annotation labels ------------------------------------------
    # batch_alter_table is required for the FK: SQLite has no ALTER-for-constraints,
    # so alembic emulates it with copy-and-move. On PostgreSQL this compiles to a
    # plain ALTER, so the same migration works on both — important because tests run
    # against SQLite while production runs Postgres.
    with op.batch_alter_table("events") as batch:
        for col in [
            sa.Column("initiator", sa.String(), nullable=True),
            sa.Column("outcome", sa.String(), nullable=True),
            sa.Column("state_before", sa.String(), nullable=True),
            sa.Column("state_after", sa.String(), nullable=True),
            sa.Column("opponent_response", sa.String(), nullable=True),
            sa.Column("technique", sa.String(), nullable=True),
            sa.Column("detail", sa.JSON(), server_default="{}"),
            sa.Column("annotator_id", sa.String(), nullable=True),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        ]:
            batch.add_column(col)
        batch.create_foreign_key("fk_events_annotator", "users", ["annotator_id"], ["id"])

    # --- state_segments -------------------------------------------------------
    match_state_enum.create(op.get_bind(), checkfirst=True)
    op.create_table(
        "state_segments",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("match_id", sa.String(), sa.ForeignKey("matches.id"), nullable=False),
        sa.Column("state", match_state_enum, nullable=False),
        sa.Column("start_ms", sa.Integer(), nullable=False),
        sa.Column("end_ms", sa.Integer(), nullable=False),
        sa.Column("controlling", sa.String(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("source", sa.String(), server_default="human"),
        sa.Column("annotator_id", sa.String(), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_state_segments_match_id", "state_segments", ["match_id"])

    # --- match_athletes -------------------------------------------------------
    op.create_table(
        "match_athletes",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("match_id", sa.String(), sa.ForeignKey("matches.id"), nullable=False),
        sa.Column("role", sa.String(), nullable=False),
        sa.Column("athlete_name", sa.String(), nullable=True),
        sa.Column("singlet_color", sa.String(), nullable=True),
        sa.Column("seed_frame_ms", sa.Integer(), nullable=True),
        sa.Column("seed_bbox", sa.JSON(), server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_match_athletes_match_id", "match_athletes", ["match_id"])


def downgrade() -> None:
    op.drop_table("match_athletes")
    op.drop_table("state_segments")
    match_state_enum.drop(op.get_bind(), checkfirst=True)

    with op.batch_alter_table("events") as batch:
        batch.drop_constraint("fk_events_annotator", type_="foreignkey")
        for name in [
            "updated_at", "annotator_id", "detail", "technique",
            "opponent_response", "state_after", "state_before", "outcome", "initiator",
        ]:
            batch.drop_column(name)

    with op.batch_alter_table("matches") as batch:
        batch.drop_column("annotation_complete")
        batch.drop_column("venue")
