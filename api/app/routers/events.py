from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app import storage
from app.database import get_db
from app.deps import get_current_user
from app.models import Correction, Event, User
from app.routers.matches import _get_owned_match
from app.schemas import (
    CorrectionResponse,
    EventCreateRequest,
    EventResponse,
    EventReviewRequest,
    EventUpdateRequest,
)

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
        review_status="confirmed",
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
    reason = updates.pop("reason", None)
    use_for_training = updates.pop("use_for_training", True)

    # A partial update can still produce an invalid range by changing only one
    # endpoint, so re-validate against the merged result rather than the payload.
    new_start = updates.get("start_ms", event.start_ms)
    new_end = updates.get("end_ms", event.end_ms)
    if new_end <= new_start:
        raise HTTPException(422, "end_ms must be greater than start_ms")
    if event.peak_ms is not None and not new_start <= event.peak_ms <= new_end:
        updates["peak_ms"] = (new_start + new_end) // 2

    if event.source.startswith("model:"):
        changed = False
        for field, value in updates.items():
            old_value = getattr(event, field)
            if old_value != value:
                changed = True
                db.add(Correction(
                    event_id=event.id,
                    corrected_by=current_user.id,
                    field=field,
                    old_value=old_value,
                    new_value=value,
                    reason=reason,
                    use_for_training=use_for_training,
                ))
        if changed:
            event.review_status = "corrected"

    for field, value in updates.items():
        setattr(event, field, value)
    if "start_ms" in updates or "end_ms" in updates:
        event.clip_key = None
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
    if event.source.startswith("model:"):
        raise HTTPException(409, "Reject model events instead of deleting them")
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
    source: Literal["all", "human", "model", "preferred"] = Query("all"),
    include_rejected: bool = Query(False),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _get_owned_match(db, match_id, current_user)
    query = db.query(Event).filter(Event.match_id == match_id)
    if not include_rejected:
        query = query.filter(Event.review_status != "rejected")
    if source == "human":
        query = query.filter(Event.source == "human")
    elif source == "model":
        query = query.filter(Event.source.like("model:%"))
    elif source == "preferred":
        has_model = db.query(Event.id).filter(
            Event.match_id == match_id,
            Event.source.like("model:%"),
            Event.review_status != "rejected",
        ).first()
        query = query.filter(Event.source.like("model:%") if has_model else Event.source == "human")
    return query.order_by(Event.start_ms.asc()).all()


@router.post("/{event_id}/review", response_model=EventResponse)
def review_event(
    match_id: str,
    event_id: str,
    payload: EventReviewRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _get_owned_match(db, match_id, current_user)
    event = _get_event_in_match(db, event_id, match_id)
    if not event.source.startswith("model:"):
        raise HTTPException(409, "Human events do not require model review")

    if event.review_status != payload.status:
        db.add(Correction(
            event_id=event.id,
            corrected_by=current_user.id,
            field="review_status",
            old_value=event.review_status,
            new_value=payload.status,
            reason=payload.reason,
            use_for_training=payload.use_for_training,
        ))
    event.review_status = payload.status
    event.annotator_id = current_user.id
    db.commit()
    db.refresh(event)
    return event


@router.get("/{event_id}/corrections", response_model=list[CorrectionResponse])
def list_event_corrections(
    match_id: str,
    event_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _get_owned_match(db, match_id, current_user)
    event = _get_event_in_match(db, event_id, match_id)
    return (
        db.query(Correction)
        .filter(Correction.event_id == event.id)
        .order_by(Correction.created_at.asc())
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
