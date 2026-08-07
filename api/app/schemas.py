from datetime import datetime
from typing import Literal

from pydantic import BaseModel, EmailStr, Field, model_validator

# --- Auth -------------------------------------------------------------------

class SignupRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    id: str
    email: str
    created_at: datetime

    model_config = {"from_attributes": True}


# --- Matches ------------------------------------------------------------------

class MatchCreateRequest(BaseModel):
    title: str = "Untitled match"
    filename: str
    content_type: str = "video/mp4"
    size_bytes: int
    style: str = "folkstyle"


class PresignedUploadResponse(BaseModel):
    match_id: str
    upload_url: str
    upload_fields: dict[str, str] = {}  # for POST-style presigned uploads
    object_key: str


class MatchResponse(BaseModel):
    id: str
    title: str
    style: str
    status: str
    duration_seconds: float | None
    video_keys: dict
    venue: str | None = None
    annotation_complete: bool = False
    coach_tone: str = "balanced"
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class UploadCompleteRequest(BaseModel):
    match_id: str


CoachTone = Literal["balanced", "hard", "extreme"]


class MatchSettingsUpdate(BaseModel):
    coach_tone: CoachTone


# --- Jobs -----------------------------------------------------------------

class JobResponse(BaseModel):
    id: str
    match_id: str
    stage: str
    status: str
    started_at: datetime | None
    finished_at: datetime | None
    error: str | None

    model_config = {"from_attributes": True}


# --- Events ---------------------------------------------------------------

# Controlled vocabularies for annotation. Kept as Literal (not free strings) so a
# typo becomes a 422 at label time rather than a silently-corrupt training example
# discovered weeks later. See PROJECT_GUIDE.md Layer 2 "annotation levels".
Initiator = Literal["user", "opponent"]
Outcome = Literal["successful", "failed", "countered", "stalemate"]
StateName = Literal["neutral", "top", "bottom", "scramble", "stopped"]


class EventCreateRequest(BaseModel):
    type: str = Field(min_length=1)
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    note: str | None = None

    # Level 1
    initiator: Initiator | None = None
    outcome: Outcome | None = None
    # Level 2
    state_before: StateName | None = None
    state_after: StateName | None = None
    opponent_response: str | None = None
    technique: str | None = None
    # Level 3 (free-form while the taxonomy is still being worked out)
    detail: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def end_after_start(self):
        if self.end_ms <= self.start_ms:
            raise ValueError("end_ms must be greater than start_ms")
        return self


class EventUpdateRequest(BaseModel):
    """Partial update — used by the annotation console for boundary editing and
    relabeling. Every field optional; only what's sent gets changed.
    """
    type: str | None = Field(default=None, min_length=1)
    start_ms: int | None = Field(default=None, ge=0)
    end_ms: int | None = Field(default=None, ge=0)
    note: str | None = None
    initiator: Initiator | None = None
    outcome: Outcome | None = None
    state_before: StateName | None = None
    state_after: StateName | None = None
    opponent_response: str | None = None
    technique: str | None = None
    detail: dict | None = None
    reason: str | None = Field(default=None, max_length=500)
    use_for_training: bool = True

    @model_validator(mode="after")
    def end_after_start_if_both_present(self):
        if self.start_ms is not None and self.end_ms is not None:
            if self.end_ms <= self.start_ms:
                raise ValueError("end_ms must be greater than start_ms")
        return self


class EventResponse(BaseModel):
    id: str
    match_id: str
    type: str
    start_ms: int
    peak_ms: int | None
    end_ms: int
    note: str | None
    source: str
    confidence: float | None
    measurements: dict
    review_status: str
    initiator: str | None
    outcome: str | None
    state_before: str | None
    state_after: str | None
    opponent_response: str | None
    technique: str | None
    detail: dict
    annotator_id: str | None
    clip_key: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class EventReviewRequest(BaseModel):
    status: Literal["confirmed", "rejected"]
    reason: str | None = Field(default=None, max_length=500)
    use_for_training: bool = True


class CorrectionResponse(BaseModel):
    id: str
    event_id: str
    corrected_by: str
    field: str
    old_value: object | None
    new_value: object | None
    reason: str | None
    use_for_training: bool
    created_at: datetime

    model_config = {"from_attributes": True}


# --- State segments (Layer 2 hand-labeling / Layer 4 model output) -----------

class StateSegmentCreateRequest(BaseModel):
    state: StateName
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    controlling: Initiator | None = None

    @model_validator(mode="after")
    def validate_segment(self):
        if self.end_ms <= self.start_ms:
            raise ValueError("end_ms must be greater than start_ms")
        # top/bottom are meaningless without knowing who's on top.
        if self.state in ("top", "bottom") and self.controlling is None:
            raise ValueError(f"state '{self.state}' requires 'controlling' to be set")
        return self


class StateSegmentUpdateRequest(BaseModel):
    state: StateName | None = None
    start_ms: int | None = Field(default=None, ge=0)
    end_ms: int | None = Field(default=None, ge=0)
    controlling: Initiator | None = None

    @model_validator(mode="after")
    def end_after_start_if_both_present(self):
        if self.start_ms is not None and self.end_ms is not None:
            if self.end_ms <= self.start_ms:
                raise ValueError("end_ms must be greater than start_ms")
        return self


class StateSegmentResponse(BaseModel):
    id: str
    match_id: str
    state: str
    start_ms: int
    end_ms: int
    controlling: str | None
    confidence: float | None
    source: str
    annotator_id: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class StateSummaryResponse(BaseModel):
    source: str | None
    segment_count: int
    total_duration_ms: int
    duration_ms_by_state: dict[str, int]
    percentage_by_state: dict[str, float]
    mean_confidence: float | None
    low_confidence_count: int


# --- Athlete identification ---------------------------------------------------

class BBox(BaseModel):
    """Normalized 0-1 coordinates so the seed box stays valid regardless of what
    resolution the video is played or analyzed at.
    """
    x1: float = Field(ge=0.0, le=1.0)
    y1: float = Field(ge=0.0, le=1.0)
    x2: float = Field(ge=0.0, le=1.0)
    y2: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def positive_area(self):
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("bbox must have positive width and height")
        return self


class MatchAthleteRequest(BaseModel):
    role: Initiator
    athlete_name: str | None = None
    singlet_color: str | None = None
    seed_frame_ms: int | None = Field(default=None, ge=0)
    seed_bbox: BBox | None = None


class MatchAthleteResponse(BaseModel):
    id: str
    match_id: str
    role: str
    athlete_name: str | None
    singlet_color: str | None
    seed_frame_ms: int | None
    seed_bbox: dict
    created_at: datetime

    model_config = {"from_attributes": True}


# --- Match annotation metadata -------------------------------------------------

class MatchAnnotationUpdate(BaseModel):
    venue: str | None = None
    annotation_complete: bool | None = None


# --- Layer 6: stats, observations, report --------------------------------------

class AthleteMatchStats(BaseModel):
    shot_attempts: int
    takedowns: int
    defended_shots: int
    conversion_rate: float | None
    escapes: int
    takedowns_conceded: int


class MatchStatsResponse(BaseModel):
    total_duration_ms: int
    duration_ms_by_state: dict[str, int]
    control_time_ms: dict[str, int]
    scramble_count: int
    longest_scramble_ms: int
    restarts: int
    by_athlete: dict[str, AthleteMatchStats]


class ObservationResponse(BaseModel):
    id: str
    match_id: str
    type: str
    summary: str
    evidence_event_ids: list[str]
    stats: dict
    source: str
    created_at: datetime

    model_config = {"from_attributes": True}


ReportStatementKind = Literal["observation", "interpretation"]


class ReportStatement(BaseModel):
    text: str
    kind: ReportStatementKind = "observation"
    evidence_event_ids: list[str] = Field(default_factory=list)


class ReportPriority(BaseModel):
    text: str
    evidence_event_ids: list[str] = Field(default_factory=list)


class ReportContent(BaseModel):
    summary: str = ""
    statements: list[ReportStatement] = Field(default_factory=list)
    priority: ReportPriority | None = None
    dropped_statement_count: int = 0


class ReportResponse(BaseModel):
    id: str
    match_id: str
    content: ReportContent
    model_version: str
    coach_tone: str
    ratings: dict
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ReportRatingRequest(BaseModel):
    evidence_validity: int = Field(ge=1, le=5)
    usefulness: int | None = Field(default=None, ge=1, le=5)
    note: str | None = Field(default=None, max_length=1000)
