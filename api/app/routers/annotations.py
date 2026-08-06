"""
Layer 2 annotation endpoints: match-state segments, athlete identification, and
match-level annotation metadata (venue, completion flag).

These are what turn the platform into the dataset-building tool described in
PROJECT_GUIDE.md Layer 2 — "the product is the dataset tool." Layer 4's state
classifier is trained against exactly the state_segments produced here.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import User, StateSegment, MatchAthlete, MatchState
from app.schemas import (
    StateSegmentCreateRequest,
    StateSegmentUpdateRequest,
    StateSegmentResponse,
    MatchAthleteRequest,
    MatchAthleteResponse,
    MatchAnnotationUpdate,
    MatchResponse,
)
from app.deps import get_current_user
from app.routers.matches import _get_owned_match

router = APIRouter(prefix="/matches/{match_id}", tags=["annotations"])


# --- Match-state segments -------------------------------------------------

@router.post("/states", response_model=StateSegmentResponse, status_code=201)
def create_state_segment(
    match_id: str,
    payload: StateSegmentCreateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _get_owned_match(db, match_id, current_user)
    _reject_overlap(db, match_id, payload.start_ms, payload.end_ms)

    segment = StateSegment(
        match_id=match_id,
        state=MatchState(payload.state),
        start_ms=payload.start_ms,
        end_ms=payload.end_ms,
        controlling=payload.controlling,
        source="human",
        annotator_id=current_user.id,
    )
    db.add(segment)
    db.commit()
    db.refresh(segment)
    return segment


@router.get("/states", response_model=list[StateSegmentResponse])
def list_state_segments(
    match_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _get_owned_match(db, match_id, current_user)
    return (
        db.query(StateSegment)
        .filter(StateSegment.match_id == match_id)
        .order_by(StateSegment.start_ms.asc())
        .all()
    )


@router.patch("/states/{segment_id}", response_model=StateSegmentResponse)
def update_state_segment(
    match_id: str,
    segment_id: str,
    payload: StateSegmentUpdateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _get_owned_match(db, match_id, current_user)
    segment = _get_segment_in_match(db, segment_id, match_id)

    updates = payload.model_dump(exclude_unset=True)
    new_start = updates.get("start_ms", segment.start_ms)
    new_end = updates.get("end_ms", segment.end_ms)
    if new_end <= new_start:
        raise HTTPException(422, "end_ms must be greater than start_ms")

    new_state = updates.get("state", segment.state.value if segment.state else None)
    new_controlling = updates.get("controlling", segment.controlling)
    if new_state in ("top", "bottom") and new_controlling is None:
        raise HTTPException(422, f"state '{new_state}' requires 'controlling' to be set")

    _reject_overlap(db, match_id, new_start, new_end, exclude_id=segment_id)

    for field, value in updates.items():
        setattr(segment, field, MatchState(value) if field == "state" else value)
    segment.annotator_id = current_user.id

    db.commit()
    db.refresh(segment)
    return segment


@router.delete("/states/{segment_id}", status_code=204)
def delete_state_segment(
    match_id: str,
    segment_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _get_owned_match(db, match_id, current_user)
    segment = _get_segment_in_match(db, segment_id, match_id)
    db.delete(segment)
    db.commit()


# --- Athlete identification -----------------------------------------------

@router.put("/athletes", response_model=MatchAthleteResponse)
def set_match_athlete(
    match_id: str,
    payload: MatchAthleteRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Upsert by role — a match has exactly one 'user' and one 'opponent', so
    re-identifying replaces rather than accumulating duplicates.
    """
    _get_owned_match(db, match_id, current_user)

    athlete = (
        db.query(MatchAthlete)
        .filter(MatchAthlete.match_id == match_id, MatchAthlete.role == payload.role)
        .first()
    )
    if athlete is None:
        athlete = MatchAthlete(match_id=match_id, role=payload.role)
        db.add(athlete)

    athlete.athlete_name = payload.athlete_name
    athlete.singlet_color = payload.singlet_color
    athlete.seed_frame_ms = payload.seed_frame_ms
    athlete.seed_bbox = payload.seed_bbox.model_dump() if payload.seed_bbox else {}

    db.commit()
    db.refresh(athlete)
    return athlete


@router.get("/athletes", response_model=list[MatchAthleteResponse])
def list_match_athletes(
    match_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _get_owned_match(db, match_id, current_user)
    return (
        db.query(MatchAthlete)
        .filter(MatchAthlete.match_id == match_id)
        .order_by(MatchAthlete.role.asc())
        .all()
    )


# --- Match annotation metadata --------------------------------------------

@router.patch("/annotation", response_model=MatchResponse)
def update_match_annotation(
    match_id: str,
    payload: MatchAnnotationUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Set venue (needed for leakage-safe splitting) and mark annotation complete.

    Marking complete requires the match to actually be annotated — otherwise an
    empty match could silently enter the training set and quietly degrade the
    dataset.
    """
    match = _get_owned_match(db, match_id, current_user)
    updates = payload.model_dump(exclude_unset=True)

    if updates.get("annotation_complete") is True:
        problems = _annotation_readiness_problems(db, match_id)
        if problems:
            raise HTTPException(422, f"Cannot mark complete: {'; '.join(problems)}")

    for field, value in updates.items():
        setattr(match, field, value)

    db.commit()
    db.refresh(match)
    return match


def _annotation_readiness_problems(db: Session, match_id: str) -> list[str]:
    problems = []
    athlete_roles = {
        a.role for a in db.query(MatchAthlete).filter(MatchAthlete.match_id == match_id).all()
    }
    if "user" not in athlete_roles:
        problems.append("the 'user' athlete has not been identified")
    if not db.query(StateSegment).filter(StateSegment.match_id == match_id).first():
        problems.append("no match-state segments have been labeled")
    return problems


# --- helpers ---------------------------------------------------------------

def _get_segment_in_match(db: Session, segment_id: str, match_id: str) -> StateSegment:
    segment = db.get(StateSegment, segment_id)
    if segment is None or segment.match_id != match_id:
        raise HTTPException(404, "State segment not found")
    return segment


def _reject_overlap(
    db: Session,
    match_id: str,
    start_ms: int,
    end_ms: int,
    exclude_id: str | None = None,
) -> None:
    """A wrestler can't be in two positions at once, so hand-labeled state segments
    must not overlap. Catching this at label time keeps the Layer 4 training data
    coherent — overlapping ground truth would make the state classifier's targets
    ambiguous.
    """
    query = db.query(StateSegment).filter(
        StateSegment.match_id == match_id,
        StateSegment.source == "human",
        StateSegment.start_ms < end_ms,
        StateSegment.end_ms > start_ms,
    )
    if exclude_id:
        query = query.filter(StateSegment.id != exclude_id)

    conflict = query.first()
    if conflict:
        raise HTTPException(
            409,
            f"Overlaps existing '{conflict.state.value}' segment "
            f"({conflict.start_ms}-{conflict.end_ms}ms)",
        )
