from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
from sqlalchemy.orm import Session

from app import storage
from app.models import Match, StateSegment
from app.stages.base import StageError


def run(match: Match, db: Session) -> dict:
    from ml.events.rules import detect_events

    features_key = storage.object_key(match.user_id, match.id, "artifacts", "features.parquet")
    if not storage.object_exists(features_key):
        raise StageError("features.parquet missing - the features stage must run first")

    states = _preferred_states(db, match.id)
    if not states:
        raise StageError("No state segments available for event detection")

    with tempfile.TemporaryDirectory() as tmp:
        local_features = str(Path(tmp) / "features.parquet")
        local_candidates = str(Path(tmp) / "event_candidates.json")
        storage.download_to_path(features_key, local_features)
        features = pd.read_parquet(local_features)
        candidates = detect_events(features, [_state_dict(segment) for segment in states])
        Path(local_candidates).write_text(
            json.dumps([candidate.to_dict() for candidate in candidates], indent=2),
            encoding="utf-8",
        )
        candidate_key = storage.object_key(
            match.user_id, match.id, "artifacts", "event_candidates.json"
        )
        storage.upload_from_path(local_candidates, candidate_key, content_type="application/json")

    counts: dict[str, int] = {}
    for candidate in candidates:
        counts[candidate.type] = counts.get(candidate.type, 0) + 1
    return {
        "candidate_key": candidate_key,
        "candidate_count": len(candidates),
        "candidates_by_type": counts,
        "state_source": states[0].source,
        "rules_version": "rules-v1",
    }


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


def _state_dict(segment: StateSegment) -> dict:
    return {
        "state": segment.state.value,
        "start_ms": segment.start_ms,
        "end_ms": segment.end_ms,
        "controlling": segment.controlling,
        "confidence": segment.confidence,
    }
