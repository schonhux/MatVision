from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import User, Event
from app.schemas import EventCreateRequest, EventResponse
from app.deps import get_current_user
from app.routers.matches import _get_owned_match
from app import storage

router = APIRouter(prefix="/matches/{match_id}/events", tags=["events"])


@router.post("", response_model=EventResponse, status_code=201)
def create_event(
    match_id: str,
    payload: EventCreateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _get_owned_match(db, match_id, current_user)
    event = Event(match_id=match_id, source="human", **payload.model_dump())
    db.add(event)
    db.commit()
    db.refresh(event)
    return event


@router.get("", response_model=list[EventResponse])
def list_events(
    match_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _get_owned_match(db, match_id, current_user)
    return (
        db.query(Event)
        .filter(Event.match_id == match_id)
        .order_by(Event.start_ms.asc())
        .all()
    )


@router.post("/{event_id}/cut-clip", response_model=EventResponse)
def cut_clip(
    match_id: str,
    event_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _get_owned_match(db, match_id, current_user)
    event = db.get(Event, event_id)
    if event is None or event.match_id != match_id:
        raise HTTPException(404, "Event not found")

    from app.queue import enqueue_clip_cut
    enqueue_clip_cut(match_id, event_id)

    db.refresh(event)
    return event


@router.get("/{event_id}/clip-url")
def get_clip_url(
    match_id: str,
    event_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _get_owned_match(db, match_id, current_user)
    event = db.get(Event, event_id)
    if event is None or event.match_id != match_id or not event.clip_key:
        raise HTTPException(404, "Clip not cut yet")
    return {"url": storage.presigned_get_url(event.clip_key)}
