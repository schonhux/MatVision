import importlib
import traceback
from datetime import datetime, timezone, timedelta

from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models import Match, Job, JobStageStatus, PIPELINE_STAGES
from app.stages.base import StageError

STUCK_RUNNING_TIMEOUT_MINUTES = 30


def _stage_module(stage_name: str):
    return importlib.import_module(f"app.stages.{stage_name}")


def _as_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def reap_stuck_running_jobs(match_id: str, db: Session) -> None:
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=STUCK_RUNNING_TIMEOUT_MINUTES)
    stuck = (
        db.query(Job)
        .filter(Job.match_id == match_id, Job.status == JobStageStatus.RUNNING)
        .all()
    )
    for job in stuck:
        started = _as_utc(job.started_at) or cutoff  # missing started_at -> treat as stuck
        if started < cutoff:
            job.status = JobStageStatus.PENDING
            job.started_at = None
    db.commit()


def run_all_stages(match_id: str) -> None:
    db = SessionLocal()
    try:
        match = db.get(Match, match_id)
        if match is None:
            return  # match was deleted out from under us; nothing to do

        reap_stuck_running_jobs(match_id, db)

        jobs_by_stage = {
            j.stage: j
            for j in db.query(Job).filter(Job.match_id == match_id).all()
        }

        for stage_name in PIPELINE_STAGES:
            job = jobs_by_stage.get(stage_name)
            if job is None:
                continue  # stage not registered for this match (shouldn't happen)

            if job.status == JobStageStatus.COMPLETE:
                continue  # already done — this is the resume-skip behavior
            if job.status == JobStageStatus.RUNNING:
                return  # another run owns this stage right now; don't double-process

            job.status = JobStageStatus.RUNNING
            job.started_at = datetime.now(timezone.utc)
            job.error = None
            db.commit()

            try:
                stage_module = _stage_module(stage_name)
                artifacts = stage_module.run(match, db)
                job.status = JobStageStatus.COMPLETE
                job.finished_at = datetime.now(timezone.utc)
                job.artifacts = artifacts or {}
                db.commit()
            except StageError as e:
                _fail_job(db, job, str(e))
                return
            except Exception as e:  # noqa: BLE001
                _fail_job(db, job, f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1000:]}")
                return

        match.status = _final_match_status(db, match_id)
        db.commit()
    finally:
        db.close()


def _fail_job(db: Session, job: Job, message: str) -> None:
    job.status = JobStageStatus.FAILED
    job.finished_at = datetime.now(timezone.utc)
    job.error = message
    match = db.get(Match, job.match_id)
    if match is not None:
        from app.models import MatchStatus
        match.status = MatchStatus.FAILED
    db.commit()


def _final_match_status(db: Session, match_id: str):
    from app.models import MatchStatus
    return MatchStatus.COMPLETE
