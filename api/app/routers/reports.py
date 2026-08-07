"""Layer 6: match stats, rule-detected observations, and the grounded coach report.

The heavy lifting (computing stats, detecting patterns, calling the LLM, and
validating its output) all happens in the worker's stats/observations/report
pipeline stages — this router only ever reads what those stages already
persisted, plus lets the athlete trigger a re-run (e.g. after correcting events
in the annotation console) and rate the result.
"""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.deps import get_current_user
from app.models import Job, JobStageStatus, Observation, Report, User
from app.routers.matches import _get_owned_match
from app.schemas import (
    MatchStatsResponse,
    ObservationResponse,
    ReportRatingRequest,
    ReportResponse,
)

router = APIRouter(prefix="/matches/{match_id}", tags=["reports"])

REPORT_STAGES = ("stats", "observations", "report")


@router.get("/stats", response_model=MatchStatsResponse)
def get_match_stats(
    match_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    match = _get_owned_match(db, match_id, current_user)
    if not match.stats_summary:
        raise HTTPException(404, "Stats have not been computed for this match yet")
    return match.stats_summary


@router.get("/observations", response_model=list[ObservationResponse])
def list_observations(
    match_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _get_owned_match(db, match_id, current_user)
    return (
        db.query(Observation)
        .filter(Observation.match_id == match_id)
        .order_by(Observation.created_at.asc())
        .all()
    )


@router.get("/report", response_model=ReportResponse)
def get_report(
    match_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _get_owned_match(db, match_id, current_user)
    report = db.query(Report).filter(Report.match_id == match_id).first()
    if report is None:
        raise HTTPException(404, "Report has not been generated for this match yet")
    return report


@router.post("/report/rating", response_model=ReportResponse)
def rate_report(
    match_id: str,
    payload: ReportRatingRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _get_owned_match(db, match_id, current_user)
    report = db.query(Report).filter(Report.match_id == match_id).first()
    if report is None:
        raise HTTPException(404, "Report has not been generated for this match yet")

    report.ratings = {
        **report.ratings,
        "evidence_validity": payload.evidence_validity,
        "usefulness": payload.usefulness,
        "note": payload.note,
        "rated_at": datetime.now(timezone.utc).isoformat(),
    }
    db.commit()
    db.refresh(report)
    return report


@router.post("/report/regenerate", status_code=202)
def regenerate_report(
    match_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Re-runs stats -> observations -> report without repeating the CV/state
    pipeline. Use after correcting events in the annotation console, or after
    switching coach_tone and wanting a fresh pass in that voice.
    """
    _get_owned_match(db, match_id, current_user)
    jobs = (
        db.query(Job)
        .filter(Job.match_id == match_id, Job.stage.in_(REPORT_STAGES))
        .all()
    )
    if not jobs:
        raise HTTPException(409, "This match has no report pipeline stages to re-run yet")

    for job in jobs:
        job.status = JobStageStatus.PENDING
        job.started_at = None
        job.finished_at = None
        job.error = None
    db.commit()

    from app.queue import enqueue_pipeline
    enqueue_pipeline(match_id)

    return {"status": "regenerating", "stages": REPORT_STAGES}
