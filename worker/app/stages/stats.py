from __future__ import annotations

from sqlalchemy.orm import Session

from app.models import Event, Match, StateSegment
from app.stages.base import StageError


def run(match: Match, db: Session) -> dict:
    from ml.reporting.stats import compute_match_stats

    events = _preferred_events(db, match.id)
    states = _preferred_states(db, match.id)
    if not states:
        raise StageError("No state segments available to compute match stats")

    stats = compute_match_stats(
        [_event_dict(event) for event in events],
        [_state_dict(segment) for segment in states],
    )
    match.stats_summary = stats
    db.commit()

    return {
        "total_duration_ms": stats["total_duration_ms"],
        "event_count": len(events),
        "state_segment_count": len(states),
    }


def _preferred_events(db: Session, match_id: str) -> list[Event]:
    model_events = (
        db.query(Event)
        .filter(
            Event.match_id == match_id,
            Event.source.like("model:%"),
            Event.review_status != "rejected",
        )
        .order_by(Event.start_ms)
        .all()
    )
    if model_events:
        return model_events
    return (
        db.query(Event)
        .filter(Event.match_id == match_id, Event.source == "human")
        .order_by(Event.start_ms)
        .all()
    )


def _preferred_states(db: Session, match_id: str) -> list[StateSegment]:
    model_states = (
        db.query(StateSegment)
        .filter(StateSegment.match_id == match_id, StateSegment.source.like("model:%"))
        .order_by(StateSegment.start_ms)
        .all()
    )
    if model_states:
        return model_states
    return (
        db.query(StateSegment)
        .filter(StateSegment.match_id == match_id, StateSegment.source == "human")
        .order_by(StateSegment.start_ms)
        .all()
    )


def _event_dict(event: Event) -> dict:
    return {
        "id": event.id,
        "type": event.type,
        "start_ms": event.start_ms,
        "peak_ms": event.peak_ms,
        "end_ms": event.end_ms,
        "initiator": event.initiator,
        "outcome": event.outcome,
        "confidence": event.confidence,
        "measurements": event.measurements or {},
    }


def _state_dict(segment: StateSegment) -> dict:
    return {
        "state": segment.state.value,
        "start_ms": segment.start_ms,
        "end_ms": segment.end_ms,
        "controlling": segment.controlling,
    }
