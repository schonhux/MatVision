"""State annotations, athlete identification, and annotation metadata."""

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.deps import get_current_user
from app.models import MatchAthlete, MatchState, StateSegment, User
from app.routers.matches import _get_owned_match
from app.schemas import (
    MatchAnnotationUpdate,
    MatchAthleteRequest,
    MatchAthleteResponse,
    MatchResponse,
    StateSegmentCreateRequest,
    StateSegmentResponse,
    StateSegmentUpdateRequest,
    StateSummaryResponse,
)

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
    source: Literal["all", "human", "model", "preferred"] = Query("all"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _get_owned_match(db, match_id, current_user)
    query = db.query(StateSegment).filter(StateSegment.match_id == match_id)
    if source == "human":
        query = query.filter(StateSegment.source == "human")
    elif source == "model":
        query = query.filter(StateSegment.source.like("model:%"))
    elif source == "preferred":
        has_model = db.query(StateSegment.id).filter(
            StateSegment.match_id == match_id,
            StateSegment.source.like("model:%"),
        ).first()
        query = query.filter(
            StateSegment.source.like("model:%") if has_model else StateSegment.source == "human"
        )
    return query.order_by(StateSegment.start_ms.asc()).all()


@router.get("/states/summary", response_model=StateSummaryResponse)
def state_summary(
    match_id: str,
    source: Literal["human", "model", "preferred"] = Query("preferred"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _get_owned_match(db, match_id, current_user)
    segments = list_state_segments(match_id, source, db, current_user)
    duration_by_state = {state.value: 0 for state in MatchState}
    for segment in segments:
        duration_by_state[segment.state.value] += segment.end_ms - segment.start_ms

    total = sum(duration_by_state.values())
    confidences = [segment.confidence for segment in segments if segment.confidence is not None]
    return {
        "source": segments[0].source if segments else None,
        "segment_count": len(segments),
        "total_duration_ms": total,
        "duration_ms_by_state": duration_by_state,
        "percentage_by_state": {
            state: round(duration / total * 100, 1) if total else 0.0
            for state, duration in duration_by_state.items()
        },
        "mean_confidence": round(sum(confidences) / len(confidences), 4) if confidences else None,
        "low_confidence_count": sum(value < 0.55 for value in confidences),
    }


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
    _require_human_segment(segment)

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
    _require_human_segment(segment)
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
    if not db.query(StateSegment).filter(
        StateSegment.match_id == match_id,
        StateSegment.source == "human",
    ).first():
        problems.append("no match-state segments have been labeled")
    return problems


# --- helpers ---------------------------------------------------------------

def _get_segment_in_match(db: Session, segment_id: str, match_id: str) -> StateSegment:
    segment = db.get(StateSegment, segment_id)
    if segment is None or segment.match_id != match_id:
        raise HTTPException(404, "State segment not found")
    return segment


def _require_human_segment(segment: StateSegment) -> None:
    if segment.source != "human":
        raise HTTPException(409, "Model predictions are read-only")


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
