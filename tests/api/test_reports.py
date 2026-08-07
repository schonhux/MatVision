def _make_match(client, headers):
    resp = client.post(
        "/matches",
        json={"title": "x", "filename": "match.mp4", "content_type": "video/mp4", "size_bytes": 1000},
        headers=headers,
    )
    return resp.json()["match_id"]


def _patch_upload(monkeypatch, uploaded=True):
    import app.storage as storage_module
    monkeypatch.setattr(storage_module, "ensure_bucket", lambda: None)
    monkeypatch.setattr(storage_module, "presigned_put_url", lambda *a, **kw: "https://fake/x")
    monkeypatch.setattr(storage_module, "object_exists", lambda key: uploaded)


def test_stats_404_before_computed(signed_up_user, monkeypatch):
    client, headers = signed_up_user
    _patch_upload(monkeypatch)
    match_id = _make_match(client, headers)

    resp = client.get(f"/matches/{match_id}/stats", headers=headers)
    assert resp.status_code == 404


def test_stats_returned_once_cached(signed_up_user, monkeypatch):
    client, headers = signed_up_user
    _patch_upload(monkeypatch)
    match_id = _make_match(client, headers)

    from app.database import SessionLocal
    from app.models import Match

    db = SessionLocal()
    match = db.get(Match, match_id)
    match.stats_summary = {
        "total_duration_ms": 6000,
        "duration_ms_by_state": {"neutral": 1000, "top": 0, "bottom": 5000, "scramble": 0, "stopped": 0},
        "control_time_ms": {"user": 0, "opponent": 5000},
        "scramble_count": 0,
        "longest_scramble_ms": 0,
        "restarts": 0,
        "by_athlete": {
            "user": {"shot_attempts": 0, "takedowns": 0, "defended_shots": 0, "conversion_rate": None, "escapes": 0, "takedowns_conceded": 1},
            "opponent": {"shot_attempts": 1, "takedowns": 1, "defended_shots": 0, "conversion_rate": 1.0, "escapes": 0, "takedowns_conceded": 0},
        },
    }
    db.commit()
    db.close()

    resp = client.get(f"/matches/{match_id}/stats", headers=headers)
    assert resp.status_code == 200
    assert resp.json()["by_athlete"]["opponent"]["takedowns"] == 1


def test_observations_list_scoped_to_match(signed_up_user, monkeypatch):
    client, headers = signed_up_user
    _patch_upload(monkeypatch)
    match_id = _make_match(client, headers)

    from app.database import SessionLocal
    from app.models import Observation

    db = SessionLocal()
    db.add(Observation(
        match_id=match_id, type="low_conversion", summary="test",
        evidence_event_ids=["e1"], stats={}, source="model:rules-v1",
    ))
    db.commit()
    db.close()

    resp = client.get(f"/matches/{match_id}/observations", headers=headers)
    assert resp.status_code == 200
    assert len(resp.json()) == 1
    assert resp.json()[0]["type"] == "low_conversion"


def test_report_404_before_generated(signed_up_user, monkeypatch):
    client, headers = signed_up_user
    _patch_upload(monkeypatch)
    match_id = _make_match(client, headers)

    resp = client.get(f"/matches/{match_id}/report", headers=headers)
    assert resp.status_code == 404


def _seed_report(match_id):
    from app.database import SessionLocal
    from app.models import Report

    db = SessionLocal()
    report = Report(
        match_id=match_id,
        content={
            "summary": "Overview.",
            "statements": [{"text": "grounded", "kind": "observation", "evidence_event_ids": ["e1"]}],
            "priority": {"text": "priority", "evidence_event_ids": ["e1"]},
            "dropped_statement_count": 1,
        },
        model_version="claude-sonnet-5",
        coach_tone="balanced",
        ratings={},
    )
    db.add(report)
    db.commit()
    db.close()


def test_get_report_returns_validated_content(signed_up_user, monkeypatch):
    client, headers = signed_up_user
    _patch_upload(monkeypatch)
    match_id = _make_match(client, headers)
    _seed_report(match_id)

    resp = client.get(f"/matches/{match_id}/report", headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["content"]["dropped_statement_count"] == 1
    assert body["content"]["priority"]["text"] == "priority"


def test_rate_report(signed_up_user, monkeypatch):
    client, headers = signed_up_user
    _patch_upload(monkeypatch)
    match_id = _make_match(client, headers)
    _seed_report(match_id)

    resp = client.post(
        f"/matches/{match_id}/report/rating",
        json={"evidence_validity": 4, "usefulness": 5, "note": "solid"},
        headers=headers,
    )
    assert resp.status_code == 200
    assert resp.json()["ratings"]["evidence_validity"] == 4
    assert resp.json()["ratings"]["usefulness"] == 5


def test_rate_report_rejects_out_of_range(signed_up_user, monkeypatch):
    client, headers = signed_up_user
    _patch_upload(monkeypatch)
    match_id = _make_match(client, headers)
    _seed_report(match_id)

    resp = client.post(
        f"/matches/{match_id}/report/rating",
        json={"evidence_validity": 9},
        headers=headers,
    )
    assert resp.status_code == 422


def test_regenerate_report_resets_stage_jobs_and_enqueues(signed_up_user, monkeypatch):
    client, headers = signed_up_user
    _patch_upload(monkeypatch)

    import app.queue as queue_module
    enqueued = []
    monkeypatch.setattr(queue_module, "enqueue_pipeline", lambda match_id: enqueued.append((match_id,)))

    match_id = _make_match(client, headers)
    complete_resp = client.post(f"/matches/{match_id}/complete", headers=headers)
    assert complete_resp.status_code == 200, complete_resp.text
    enqueued.clear()  # only assert on the regenerate call below

    from app.database import SessionLocal
    from app.models import Job, JobStageStatus

    db = SessionLocal()
    for job in db.query(Job).filter(Job.match_id == match_id, Job.stage.in_(["stats", "observations", "report"])):
        job.status = JobStageStatus.COMPLETE
    db.commit()
    db.close()

    resp = client.post(f"/matches/{match_id}/report/regenerate", headers=headers)
    assert resp.status_code == 202
    assert enqueued == [(match_id,)]

    db = SessionLocal()
    reset_jobs = db.query(Job).filter(Job.match_id == match_id, Job.stage.in_(["stats", "observations", "report"])).all()
    assert all(job.status == JobStageStatus.PENDING for job in reset_jobs)


def test_reports_scoped_to_owner(client, monkeypatch):
    _patch_upload(monkeypatch)

    resp1 = client.post("/auth/signup", json={"email": "a@test.com", "password": "hunter2222"})
    headers1 = {"Authorization": f"Bearer {resp1.json()['access_token']}"}
    resp2 = client.post("/auth/signup", json={"email": "b@test.com", "password": "hunter2222"})
    headers2 = {"Authorization": f"Bearer {resp2.json()['access_token']}"}

    match_id = _make_match(client, headers1)
    _seed_report(match_id)

    resp = client.get(f"/matches/{match_id}/report", headers=headers2)
    assert resp.status_code == 404
