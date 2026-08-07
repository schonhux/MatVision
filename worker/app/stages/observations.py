from __future__ import annotations

from sqlalchemy.orm import Session

from app.models import Match, Observation
from app.stages.base import StageError
from app.stages.stats import _event_dict, _preferred_events


def run(match: Match, db: Session) -> dict:
    from ml.reporting.observations import detect_observations

    if not match.stats_summary:
        raise StageError("stats_summary missing - the stats stage must run first")

    events = _preferred_events(db, match.id)
    observations = detect_observations(
        [_event_dict(event) for event in events], match.stats_summary
    )

    db.query(Observation).filter(
        Observation.match_id == match.id,
        Observation.source.like("model:%"),
    ).delete(synchronize_session=False)

    db.add_all([
        Observation(
            match_id=match.id,
            type=observation["type"],
            summary=observation["summary"],
            evidence_event_ids=observation.get("evidence_event_ids", []),
            stats=observation.get("stats", {}),
            source="model:rules-v1",
        )
        for observation in observations
    ])
    db.commit()

    counts: dict[str, int] = {}
    for observation in observations:
        counts[observation["type"]] = counts.get(observation["type"], 0) + 1

    return {"observation_count": len(observations), "observations_by_type": counts}
