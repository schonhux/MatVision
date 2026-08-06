from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import User, Match, MatchStatus, Job, JobStageStatus, PIPELINE_STAGES
from app.schemas import (
    MatchCreateRequest,
    PresignedUploadResponse,
    MatchResponse,
    UploadCompleteRequest,
)
from app.deps import get_current_user
from app import storage
from app.config import settings

router = APIRouter(prefix="/matches", tags=["matches"])


@router.post("", response_model=PresignedUploadResponse, status_code=201)
def create_match(
    payload: MatchCreateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    ext = "." + payload.filename.rsplit(".", 1)[-1].lower() if "." in payload.filename else ""
    if ext not in settings.allowed_video_extensions:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Allowed: {settings.allowed_video_extensions}")
    if payload.size_bytes > settings.max_upload_bytes:
        raise HTTPException(400, f"File exceeds max size of {settings.max_upload_bytes} bytes")

    match = Match(user_id=current_user.id, title=payload.title, style=payload.style)
    db.add(match)
    db.commit()
    db.refresh(match)

    key = storage.object_key(current_user.id, match.id, "original", f"video{ext}")
    match.video_keys = {"original": key}
    db.commit()

    storage.ensure_bucket()
    upload_url = storage.presigned_put_url(key, payload.content_type)

    return PresignedUploadResponse(match_id=match.id, upload_url=upload_url, object_key=key)


@router.post("/{match_id}/complete", response_model=MatchResponse)
def complete_upload(
    match_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    match = _get_owned_match(db, match_id, current_user)

    original_key = match.video_keys.get("original")
    if not original_key or not storage.object_exists(original_key):
        raise HTTPException(400, "Upload not found in object storage yet")

    match.status = MatchStatus.VALIDATING
    db.commit()

    for stage in PIPELINE_STAGES:
        db.add(Job(match_id=match.id, stage=stage, status=JobStageStatus.PENDING))
    db.commit()

    from app.queue import enqueue_pipeline
    enqueue_pipeline(match.id)

    db.refresh(match)
    return match


@router.get("", response_model=list[MatchResponse])
def list_matches(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    return (
        db.query(Match)
        .filter(Match.user_id == current_user.id)
        .order_by(Match.created_at.desc())
        .all()
    )


@router.get("/{match_id}", response_model=MatchResponse)
def get_match(match_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    return _get_owned_match(db, match_id, current_user)


@router.get("/{match_id}/video-url")
def get_video_url(match_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    match = _get_owned_match(db, match_id, current_user)
    key = match.video_keys.get("analysis_720p") or match.video_keys.get("original")
    if not key:
        raise HTTPException(404, "No video available for this match yet")
    return {"url": storage.presigned_get_url(key), "source": "analysis_720p" if match.video_keys.get("analysis_720p") else "original"}


def _get_owned_match(db: Session, match_id: str, current_user: User) -> Match:
    match = db.get(Match, match_id)
    if match is None or match.user_id != current_user.id:
        raise HTTPException(404, "Match not found")
    return match
