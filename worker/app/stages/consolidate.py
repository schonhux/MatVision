from __future__ import annotations

import json
import tempfile
from pathlib import Path

from sqlalchemy.orm import Session

from app import storage
from app.models import Event, Match
from app.stages.base import StageError


def run(match: Match, db: Session) -> dict:
    from ml.events.consolidation import consolidate_events, temporal_iou
    from ml.events.rules import EventCandidate

    candidate_key = storage.object_key(
        match.user_id, match.id, "artifacts", "event_candidates.json"
    )
    if not storage.object_exists(candidate_key):
        raise StageError("event_candidates.json missing - the events stage must run first")

    with tempfile.TemporaryDirectory() as tmp:
        local_candidates = str(Path(tmp) / "event_candidates.json")
        storage.download_to_path(candidate_key, local_candidates)
        raw = json.loads(Path(local_candidates).read_text(encoding="utf-8"))

    candidates = [EventCandidate.from_dict(item) for item in raw]
    consolidated = consolidate_events(candidates)
    reviewed = (
        db.query(Event)
        .filter(
            Event.match_id == match.id,
            Event.source.like("model:%"),
            Event.review_status != "unreviewed",
        )
        .all()
    )
    protected = [_event_candidate(event) for event in reviewed]

    db.query(Event).filter(
        Event.match_id == match.id,
        Event.source.like("model:%"),
        Event.review_status == "unreviewed",
    ).delete(synchronize_session=False)

    created = []
    for candidate in consolidated:
        already_reviewed = any(
            candidate.type == existing.type
            and candidate.initiator == existing.initiator
            and (temporal_iou(candidate, existing) >= 0.2 or abs(candidate.peak_ms - existing.peak_ms) < 1200)
            for existing in protected
        )
        if already_reviewed:
            continue
        event = Event(
            match_id=match.id,
            type=candidate.type,
            start_ms=candidate.start_ms,
            peak_ms=candidate.peak_ms,
            end_ms=candidate.end_ms,
            source="model:rules-v1",
            confidence=candidate.confidence,
            measurements=candidate.measurements,
            review_status="unreviewed",
            initiator=candidate.initiator,
            outcome=candidate.outcome,
            state_before=candidate.state_before,
            state_after=candidate.state_after,
        )
        db.add(event)
        created.append(event)
    db.commit()

    counts: dict[str, int] = {}
    for event in created:
        counts[event.type] = counts.get(event.type, 0) + 1
    return {
        "raw_candidate_count": len(candidates),
        "consolidated_count": len(consolidated),
        "created_count": len(created),
        "preserved_reviewed_count": len(reviewed),
        "events_by_type": counts,
        "source": "model:rules-v1",
    }


def _event_candidate(event: Event):
    from ml.events.rules import EventCandidate

    peak_ms = event.peak_ms if event.peak_ms is not None else (event.start_ms + event.end_ms) // 2
    return EventCandidate(
        type=event.type,
        start_ms=event.start_ms,
        peak_ms=peak_ms,
        end_ms=event.end_ms,
        initiator=event.initiator,
        outcome=event.outcome,
        confidence=event.confidence or 0.0,
        state_before=event.state_before,
        state_after=event.state_after,
        measurements=event.measurements or {},
    )
