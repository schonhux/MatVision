from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import User, Event
from app.schemas import EventCreateRequest, EventUpdateRequest, EventResponse
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
    event = Event(
        match_id=match_id,
        source="human",
        annotator_id=current_user.id,
        **payload.model_dump(),
    )
    db.add(event)
    db.commit()
    db.refresh(event)
    return event


@router.patch("/{event_id}", response_model=EventResponse)
def update_event(
    match_id: str,
    event_id: str,
    payload: EventUpdateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Partial update — powers boundary editing and relabeling in the annotation
    console. Only fields present in the request body are changed.
    """
    _get_owned_match(db, match_id, current_user)
    event = _get_event_in_match(db, event_id, match_id)

    updates = payload.model_dump(exclude_unset=True)

    # A partial update can still produce an invalid range by changing only one
    # endpoint, so re-validate against the merged result rather than the payload.
    new_start = updates.get("start_ms", event.start_ms)
    new_end = updates.get("end_ms", event.end_ms)
    if new_end <= new_start:
        raise HTTPException(422, "end_ms must be greater than start_ms")

    for field, value in updates.items():
        setattr(event, field, value)
    event.annotator_id = current_user.id

    db.commit()
    db.refresh(event)
    return event


@router.delete("/{event_id}", status_code=204)
def delete_event(
    match_id: str,
    event_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _get_owned_match(db, match_id, current_user)
    event = _get_event_in_match(db, event_id, match_id)
    db.delete(event)
    db.commit()


def _get_event_in_match(db: Session, event_id: str, match_id: str) -> Event:
    """Always look up an event *scoped to its match* — fetching by id alone would
    let a caller touch an event belonging to a different match.
    """
    event = db.get(Event, event_id)
    if event is None or event.match_id != match_id:
        raise HTTPException(404, "Event not found")
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
