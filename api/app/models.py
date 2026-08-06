import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import String, DateTime, ForeignKey, Enum, Integer, JSON, Text, Float, Boolean
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    email: Mapped[str] = mapped_column(String, unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    matches: Mapped[list["Match"]] = relationship(back_populates="owner", cascade="all, delete-orphan")


class MatchStatus(str, enum.Enum):
    UPLOADED = "uploaded"
    VALIDATING = "validating"
    TRANSCODING = "transcoding"
    TRACKING = "tracking"
    EXTRACTING_POSE = "extracting_pose"
    CLASSIFYING_STATES = "classifying_states"
    DETECTING_EVENTS = "detecting_events"
    GENERATING_INSIGHTS = "generating_insights"
    COMPLETE = "complete"
    FAILED = "failed"


class Match(Base):
    __tablename__ = "matches"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)

    title: Mapped[str] = mapped_column(String, nullable=False, default="Untitled match")
    style: Mapped[str] = mapped_column(String, default="folkstyle")
    status: Mapped[MatchStatus] = mapped_column(
        Enum(MatchStatus), default=MatchStatus.UPLOADED, nullable=False
    )

    video_keys: Mapped[dict] = mapped_column(JSON, default=dict)

    duration_seconds: Mapped[float | None] = mapped_column(Integer, nullable=True)
    meta: Mapped[dict] = mapped_column(JSON, default=dict)

    # Layer 2: recorded so the dataset can be split without leakage. Two matches from
    # the same venue share mat, lighting, and camera position — putting one in train
    # and one in test inflates scores. See ml/datasets/splits.py.
    venue: Mapped[str | None] = mapped_column(String, nullable=True)
    # Marks a match as fully labeled and eligible for export into the training set.
    annotation_complete: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    owner: Mapped["User"] = relationship(back_populates="matches")
    jobs: Mapped[list["Job"]] = relationship(back_populates="match", cascade="all, delete-orphan")
    events: Mapped[list["Event"]] = relationship(back_populates="match", cascade="all, delete-orphan")
    state_segments: Mapped[list["StateSegment"]] = relationship(
        back_populates="match", cascade="all, delete-orphan"
    )
    athletes: Mapped[list["MatchAthlete"]] = relationship(
        back_populates="match", cascade="all, delete-orphan"
    )


class JobStageStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    match_id: Mapped[str] = mapped_column(ForeignKey("matches.id"), nullable=False, index=True)

    stage: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[JobStageStatus] = mapped_column(
        Enum(JobStageStatus), default=JobStageStatus.PENDING, nullable=False
    )

    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    artifacts: Mapped[dict] = mapped_column(JSON, default=dict)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    match: Mapped["Match"] = relationship(back_populates="jobs")


class Event(Base):
    __tablename__ = "events"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    match_id: Mapped[str] = mapped_column(ForeignKey("matches.id"), nullable=False, index=True)

    type: Mapped[str] = mapped_column(String, nullable=False)
    start_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    end_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    source: Mapped[str] = mapped_column(String, default="human")

    # --- Layer 2 annotation fields ---------------------------------------
    # Level 1 labels (see PROJECT_GUIDE.md Section 8, Layer 2 "annotation levels").
    # 'user' | 'opponent' | None — who initiated the action.
    initiator: Mapped[str | None] = mapped_column(String, nullable=True)
    # 'successful' | 'failed' | 'countered' | 'stalemate' | None
    outcome: Mapped[str | None] = mapped_column(String, nullable=True)

    # Level 2 labels — match state on either side of the event, opponent's response,
    # and the technique family (single_leg, double_leg, high_crotch, ...).
    state_before: Mapped[str | None] = mapped_column(String, nullable=True)
    state_after: Mapped[str | None] = mapped_column(String, nullable=True)
    opponent_response: Mapped[str | None] = mapped_column(String, nullable=True)
    technique: Mapped[str | None] = mapped_column(String, nullable=True)

    # Free-form Level 3 detail (setup type, entry quality, finish direction, ...)
    # kept as JSON so the schema doesn't churn while the label taxonomy is still
    # being figured out through actual labeling work.
    detail: Mapped[dict] = mapped_column(JSON, default=dict)

    # Who labeled this, for reviewer-agreement tracking. Null for events created
    # before Layer 2 or by an automated source.
    annotator_id: Mapped[str | None] = mapped_column(ForeignKey("users.id"), nullable=True)

    clip_key: Mapped[str | None] = mapped_column(String, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    match: Mapped["Match"] = relationship(back_populates="events")


class MatchState(str, enum.Enum):
    """The broad position states from PROJECT_GUIDE.md Layer 4. Labeled by hand in
    Layer 2; predicted by a model in Layer 4 against these exact same labels.
    """
    NEUTRAL = "neutral"
    TOP = "top"
    BOTTOM = "bottom"
    SCRAMBLE = "scramble"
    STOPPED = "stopped"


class StateSegment(Base):
    """A contiguous span of the match in a single position state. Layer 2 creates
    these by hand (source='human'); Layer 4's classifier writes them with
    source='model:<version>' so the two are directly comparable for evaluation.
    """
    __tablename__ = "state_segments"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    match_id: Mapped[str] = mapped_column(ForeignKey("matches.id"), nullable=False, index=True)

    state: Mapped[MatchState] = mapped_column(Enum(MatchState), nullable=False)
    start_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    end_ms: Mapped[int] = mapped_column(Integer, nullable=False)

    # For top/bottom, which athlete is on top. 'user' | 'opponent' | None.
    controlling: Mapped[str | None] = mapped_column(String, nullable=True)

    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)  # model only
    source: Mapped[str] = mapped_column(String, default="human")
    annotator_id: Mapped[str | None] = mapped_column(ForeignKey("users.id"), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    match: Mapped["Match"] = relationship(back_populates="state_segments")


class MatchAthlete(Base):
    """Identifies which wrestler is which within a match — the 'click your body in
    an early frame' step from PROJECT_GUIDE.md. Stores the seed bounding box the
    user clicked, which Layer 3's tracker uses to bind a track ID to an identity.

    Also carries the metadata that makes leakage-safe dataset splitting possible
    (athlete_name, venue): see ml/datasets/splits.py — you cannot split by athlete
    if you never recorded who was wrestling.
    """
    __tablename__ = "match_athletes"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    match_id: Mapped[str] = mapped_column(ForeignKey("matches.id"), nullable=False, index=True)

    # 'user' | 'opponent'
    role: Mapped[str] = mapped_column(String, nullable=False)
    athlete_name: Mapped[str | None] = mapped_column(String, nullable=True)
    singlet_color: Mapped[str | None] = mapped_column(String, nullable=True)

    # The frame + box the user clicked to identify this athlete.
    seed_frame_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    seed_bbox: Mapped[dict] = mapped_column(JSON, default=dict)  # {x1,y1,x2,y2} normalized 0-1

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    match: Mapped["Match"] = relationship(back_populates="athletes")


PIPELINE_STAGES: list[str] = [
    "validate",
    "transcode",
    # Layer 3+: "detect_track", "pose", "features"
    # Layer 4:  "states"
    # Layer 5:  "events", "consolidate", "clips"
    # Layer 6:  "stats", "observations", "report"
]
