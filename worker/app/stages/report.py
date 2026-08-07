from __future__ import annotations

from sqlalchemy.orm import Session

from app.config import settings
from app.models import Match, Observation, Report
from app.stages.base import StageError
from app.stages.stats import _event_dict, _preferred_events


def run(match: Match, db: Session) -> dict:
    from ml.reporting.evidence import build_evidence_json
    from ml.reporting.llm import ReportGenerationError, generate_report_content, validate_report

    if not match.stats_summary:
        raise StageError("stats_summary missing - the stats stage must run first")

    events = _preferred_events(db, match.id)
    if not events:
        raise StageError("No events available to build a report")

    observations = (
        db.query(Observation)
        .filter(Observation.match_id == match.id)
        .order_by(Observation.created_at.asc())
        .all()
    )

    event_dicts = [_event_dict(event) for event in events]
    evidence = build_evidence_json(
        {"title": match.title, "style": match.style, "duration_seconds": match.duration_seconds},
        event_dicts,
        match.stats_summary,
        [
            {
                "type": observation.type,
                "summary": observation.summary,
                "evidence_event_ids": observation.evidence_event_ids or [],
            }
            for observation in observations
        ],
    )

    try:
        raw_content = generate_report_content(
            evidence, match.coach_tone, settings.anthropic_api_key, settings.anthropic_model
        )
    except ReportGenerationError as exc:
        raise StageError(str(exc)) from exc

    valid_event_ids = {event["id"] for event in event_dicts}
    content = validate_report(raw_content, valid_event_ids)

    report = db.query(Report).filter(Report.match_id == match.id).first()
    if report is None:
        report = Report(match_id=match.id)
        db.add(report)

    report.content = content
    report.model_version = settings.anthropic_model
    report.coach_tone = match.coach_tone
    db.commit()

    return {
        "statement_count": len(content["statements"]),
        "dropped_statement_count": content["dropped_statement_count"],
        "has_priority": content["priority"] is not None,
        "coach_tone": match.coach_tone,
        "model_version": settings.anthropic_model,
    }
