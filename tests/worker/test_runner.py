"""
Resumability — the Layer 1 acceptance criterion from BUILD_PLAN.md: 'a killed
worker resumes from the last completed stage.' Uses fake stage modules (no ffmpeg,
no real video) to deterministically control success/failure and count calls,
proving the orchestration logic itself is correct independent of any one stage's
implementation. See tests/worker/test_stages.py for the real ffmpeg stages.
"""

import sys
import types
from datetime import datetime, timezone, timedelta


def _install_fake_stages(should_fail_b: dict):
    calls = {"stage_a": 0, "stage_b": 0, "stage_c": 0}

    mod_a = types.ModuleType("app.stages.stage_a")
    mod_a.run = lambda match, db: (calls.__setitem__("stage_a", calls["stage_a"] + 1), {"ok": True})[1]
    sys.modules["app.stages.stage_a"] = mod_a

    def stage_b_run(match, db):
        calls["stage_b"] += 1
        if should_fail_b["value"]:
            from app.stages.base import StageError
            raise StageError("simulated failure")
        return {"ok": True}

    mod_b = types.ModuleType("app.stages.stage_b")
    mod_b.run = stage_b_run
    sys.modules["app.stages.stage_b"] = mod_b

    mod_c = types.ModuleType("app.stages.stage_c")
    mod_c.run = lambda match, db: (calls.__setitem__("stage_c", calls["stage_c"] + 1), {"ok": True})[1]
    sys.modules["app.stages.stage_c"] = mod_c

    return calls


def _seed_match_and_jobs(match_id="m1"):
    from app.database import SessionLocal
    from app import models

    db = SessionLocal()
    match = models.Match(id=match_id, user_id="u1", title="T")
    db.add(match)
    db.commit()
    for stage in models.PIPELINE_STAGES:
        db.add(models.Job(match_id=match_id, stage=stage, status=models.JobStageStatus.PENDING))
    db.commit()
    db.close()


def test_pipeline_stops_on_failure_and_does_not_run_later_stages(merged_worker_app):
    from app import models
    models.PIPELINE_STAGES.clear()
    models.PIPELINE_STAGES.extend(["stage_a", "stage_b", "stage_c"])
    calls = _install_fake_stages({"value": True})
    _seed_match_and_jobs()

    from app.stages.runner import run_all_stages
    from app.database import SessionLocal

    run_all_stages("m1")

    db = SessionLocal()
    jobs = {j.stage: j.status.value for j in db.query(models.Job).filter(models.Job.match_id == "m1").all()}
    assert jobs == {"stage_a": "complete", "stage_b": "failed", "stage_c": "pending"}
    assert calls == {"stage_a": 1, "stage_b": 1, "stage_c": 0}
    assert db.get(models.Match, "m1").status == models.MatchStatus.FAILED


def test_resume_skips_completed_stages_and_retries_failed_one(merged_worker_app):
    from app import models
    models.PIPELINE_STAGES.clear()
    models.PIPELINE_STAGES.extend(["stage_a", "stage_b", "stage_c"])
    should_fail = {"value": True}
    calls = _install_fake_stages(should_fail)
    _seed_match_and_jobs()

    from app.stages.runner import run_all_stages
    from app.database import SessionLocal

    run_all_stages("m1")  # first run: stage_b fails
    should_fail["value"] = False
    run_all_stages("m1")  # resume: should skip stage_a, retry+pass stage_b, run stage_c

    db = SessionLocal()
    jobs = {j.stage: j.status.value for j in db.query(models.Job).filter(models.Job.match_id == "m1").all()}
    assert jobs == {"stage_a": "complete", "stage_b": "complete", "stage_c": "complete"}
    assert calls["stage_a"] == 1, "completed stage was re-run on resume — resumability is broken"
    assert calls["stage_b"] == 2
    assert calls["stage_c"] == 1
    assert db.get(models.Match, "m1").status == models.MatchStatus.COMPLETE


def test_running_stage_is_not_double_processed(merged_worker_app):
    """If a stage is already RUNNING (e.g. another process/thread owns it), a
    second call to run_all_stages must not re-execute it or move past it.
    """
    from app import models
    models.PIPELINE_STAGES.clear()
    models.PIPELINE_STAGES.extend(["stage_a", "stage_b", "stage_c"])
    calls = _install_fake_stages({"value": False})
    _seed_match_and_jobs()

    from app.database import SessionLocal
    db = SessionLocal()
    job_b = db.query(models.Job).filter(models.Job.match_id == "m1", models.Job.stage == "stage_b").first()
    job_b.status = models.JobStageStatus.RUNNING
    job_b.started_at = datetime.now(timezone.utc)  # recent — not stuck, genuinely in-flight
    db.commit()
    db.close()

    from app.stages.runner import run_all_stages
    run_all_stages("m1")

    db = SessionLocal()
    jobs = {j.stage: j.status.value for j in db.query(models.Job).filter(models.Job.match_id == "m1").all()}
    assert jobs["stage_a"] == "complete"  # ran fine, comes before the running stage
    assert jobs["stage_b"] == "running"   # left alone
    assert jobs["stage_c"] == "pending"   # never reached
    assert calls["stage_b"] == 0
    assert calls["stage_c"] == 0


def test_stuck_running_job_is_reaped_and_retried(merged_worker_app):
    """A job RUNNING for far longer than plausible means the worker that owned it
    crashed. reap_stuck_running_jobs() must reset it so run_all_stages can retry it
    within the same call — this is the actual crash-recovery path.
    """
    from app import models
    models.PIPELINE_STAGES.clear()
    models.PIPELINE_STAGES.extend(["stage_a", "stage_b", "stage_c"])
    calls = _install_fake_stages({"value": False})
    _seed_match_and_jobs()

    from app.database import SessionLocal
    db = SessionLocal()
    job_b = db.query(models.Job).filter(models.Job.match_id == "m1", models.Job.stage == "stage_b").first()
    job_b.status = models.JobStageStatus.RUNNING
    job_b.started_at = datetime.now(timezone.utc) - timedelta(minutes=45)  # implausibly old
    db.commit()
    db.close()

    from app.stages.runner import run_all_stages
    run_all_stages("m1")

    db = SessionLocal()
    jobs = {j.stage: j.status.value for j in db.query(models.Job).filter(models.Job.match_id == "m1").all()}
    assert jobs == {"stage_a": "complete", "stage_b": "complete", "stage_c": "complete"}
    assert calls["stage_b"] == 1  # reaped, then actually executed
