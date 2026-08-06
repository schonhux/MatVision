import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import String, DateTime, ForeignKey, Enum, Integer, JSON, Text
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

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    owner: Mapped["User"] = relationship(back_populates="matches")
    jobs: Mapped[list["Job"]] = relationship(back_populates="match", cascade="all, delete-orphan")
    events: Mapped[list["Event"]] = relationship(back_populates="match", cascade="all, delete-orphan")


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

    clip_key: Mapped[str | None] = mapped_column(String, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    match: Mapped["Match"] = relationship(back_populates="events")


PIPELINE_STAGES: list[str] = [
    "validate",
    "transcode",
    # Layer 3+: "detect_track", "pose", "features"
    # Layer 4:  "states"
    # Layer 5:  "events", "consolidate", "clips"
    # Layer 6:  "stats", "observations", "report"
]
