from datetime import datetime

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
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class UploadCompleteRequest(BaseModel):
    match_id: str


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

class EventCreateRequest(BaseModel):
    type: str = Field(min_length=1)
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    note: str | None = None

    @model_validator(mode="after")
    def end_after_start(self):
        if self.end_ms <= self.start_ms:
            raise ValueError("end_ms must be greater than start_ms")
        return self


class EventResponse(BaseModel):
    id: str
    match_id: str
    type: str
    start_ms: int
    end_ms: int
    note: str | None
    source: str
    clip_key: str | None
    created_at: datetime

    model_config = {"from_attributes": True}
