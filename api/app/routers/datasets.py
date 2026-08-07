"""
Dataset export — turns hand-labeled matches into a training-ready file with
leakage-safe split tags.

This closes the Layer 2 loop: the annotation console produces labels, and this
endpoint packages them for Layer 4/5 model training. The split assignment is
computed by ml/datasets/splits.py and then *independently re-verified* before the
export is returned, so a leaky dataset can never silently ship.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

# ml/ lives at the repo root, outside the api package.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from app.database import get_db
from app.deps import get_current_user
from app.models import (
    Correction,
    Event,
    Match,
    MatchAthlete,
    StateSegment,
    User,
)
from ml.datasets.splits import (
    MatchRecord,
    assign_splits,
    split_summary,
    verify_no_leakage,
)

router = APIRouter(prefix="/datasets", tags=["datasets"])

DATASET_SCHEMA_VERSION = "1.1"


@router.get("/export")
def export_dataset(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    only_complete: bool = Query(True, description="Only include matches marked annotation_complete"),
    seed: int = Query(42, description="Split seed — change to reshuffle deterministically"),
    train_ratio: float = Query(0.7, gt=0, lt=1),
    val_ratio: float = Query(0.15, gt=0, lt=1),
):
    """Export this user's labeled matches as a schema-validated dataset.

    The response is deliberately self-describing (schema_version, split config,
    per-match counts, leakage verification result) so an exported file is
    interpretable months later without needing this code — a dataset card in JSON
    form, per PROJECT_GUIDE.md's docs/dataset-card.md.
    """
    test_ratio = round(1.0 - train_ratio - val_ratio, 10)
    if test_ratio <= 0:
        raise HTTPException(422, "train_ratio + val_ratio must be less than 1.0")

    query = db.query(Match).filter(Match.user_id == current_user.id)
    if only_complete:
        query = query.filter(Match.annotation_complete.is_(True))
    matches = query.order_by(Match.created_at.asc()).all()

    if not matches:
        raise HTTPException(
            404,
            "No annotated matches to export. Label at least one match and mark it "
            "complete first.",
        )

    # Build split records from athletes + venue.
    records = []
    for m in matches:
        athletes = db.query(MatchAthlete).filter(MatchAthlete.match_id == m.id).all()
        names = [a.athlete_name for a in athletes if a.athlete_name]
        records.append(MatchRecord(match_id=m.id, athletes=names, venue=m.venue))

    assignment = assign_splits(
        records, ratios=(train_ratio, val_ratio, test_ratio), seed=seed
    )

    # Independent re-check. If this ever returns violations, refuse to export —
    # a silently leaky dataset is worse than no dataset.
    violations = verify_no_leakage(records, assignment)
    if violations:
        raise HTTPException(
            500,
            f"Refusing to export: split verification failed with {len(violations)} "
            f"violation(s): {violations[:5]}",
        )

    exported_matches = []
    for m in matches:
        events = db.query(Event).filter(Event.match_id == m.id).order_by(Event.start_ms).all()
        event_ids = [event.id for event in events]
        corrections = (
            db.query(Correction)
            .filter(Correction.event_id.in_(event_ids))
            .order_by(Correction.created_at)
            .all()
            if event_ids else []
        )
        corrections_by_event: dict[str, list[Correction]] = {}
        for correction in corrections:
            corrections_by_event.setdefault(correction.event_id, []).append(correction)
        states = (
            db.query(StateSegment)
            .filter(StateSegment.match_id == m.id, StateSegment.source == "human")
            .order_by(StateSegment.start_ms)
            .all()
        )
        athletes = db.query(MatchAthlete).filter(MatchAthlete.match_id == m.id).all()

        exported_matches.append({
            "match_id": m.id,
            "split": assignment[m.id],
            "title": m.title,
            "style": m.style,
            "venue": m.venue,
            "duration_seconds": m.duration_seconds,
            "athletes": [
                {
                    "role": a.role,
                    "name": a.athlete_name,
                    "singlet_color": a.singlet_color,
                    "seed_frame_ms": a.seed_frame_ms,
                    "seed_bbox": a.seed_bbox,
                }
                for a in athletes
            ],
            "state_segments": [
                {
                    "state": s.state.value,
                    "start_ms": s.start_ms,
                    "end_ms": s.end_ms,
                    "controlling": s.controlling,
                }
                for s in states
            ],
            "events": [
                {
                    "event_id": e.id,
                    "type": e.type,
                    "start_ms": e.start_ms,
                    "peak_ms": e.peak_ms,
                    "end_ms": e.end_ms,
                    "initiator": e.initiator,
                    "outcome": e.outcome,
                    "state_before": e.state_before,
                    "state_after": e.state_after,
                    "opponent_response": e.opponent_response,
                    "technique": e.technique,
                    "detail": e.detail,
                    "source": e.source,
                    "confidence": e.confidence,
                    "measurements": e.measurements,
                    "review_status": e.review_status,
                    "corrections": [
                        {
                            "field": correction.field,
                            "old_value": correction.old_value,
                            "new_value": correction.new_value,
                            "reason": correction.reason,
                            "use_for_training": correction.use_for_training,
                        }
                        for correction in corrections_by_event.get(e.id, [])
                    ],
                }
                for e in events
            ],
        })

    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "split_config": {
            "seed": seed,
            "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
            "grouped_by": ["match_id", "athlete", "venue"],
            "leakage_verified": True,
        },
        "summary": {
            "match_count": len(matches),
            "split_counts": split_summary(assignment),
            "event_count": sum(len(m["events"]) for m in exported_matches),
            "correction_count": sum(
                len(event["corrections"])
                for match in exported_matches
                for event in match["events"]
            ),
            "state_segment_count": sum(len(m["state_segments"]) for m in exported_matches),
        },
        "matches": exported_matches,
    }


@router.get("/stats")
def dataset_stats(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Progress toward a usable dataset — how much labeling is actually done.
    Drives the annotation console's progress display so labeling effort is visible
    rather than guessed at.
    """
    matches = db.query(Match).filter(Match.user_id == current_user.id).all()
    complete = [m for m in matches if m.annotation_complete]

    total_events = 0
    total_states = 0
    total_corrections = 0
    labeled_duration_ms = 0
    events_by_type: dict[str, int] = {}
    states_by_type: dict[str, int] = {}

    for m in matches:
        events = db.query(Event).filter(Event.match_id == m.id).all()
        states = (
            db.query(StateSegment)
            .filter(StateSegment.match_id == m.id, StateSegment.source == "human")
            .all()
        )
        total_events += len(events)
        event_ids = [event.id for event in events]
        if event_ids:
            total_corrections += db.query(Correction).filter(
                Correction.event_id.in_(event_ids),
                Correction.use_for_training.is_(True),
            ).count()
        total_states += len(states)
        for e in events:
            events_by_type[e.type] = events_by_type.get(e.type, 0) + 1
        for s in states:
            key = s.state.value
            states_by_type[key] = states_by_type.get(key, 0) + 1
            labeled_duration_ms += s.end_ms - s.start_ms

    return {
        "total_matches": len(matches),
        "annotated_matches": len(complete),
        "total_events": total_events,
        "total_state_segments": total_states,
        "total_corrections": total_corrections,
        "labeled_minutes": round(labeled_duration_ms / 60000, 1),
        "events_by_type": events_by_type,
        "states_by_type": states_by_type,
        # BUILD_PLAN.md M2 gate: 5 matches labeled by Oct 1.
        "m2_gate": {"target_matches": 5, "current": len(complete), "met": len(complete) >= 5},
    }
