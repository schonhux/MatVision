"""
Matches CRUD + presigned upload flow. Storage (boto3/MinIO) and the queue
(dramatiq/redis) are mocked here — this suite tests API/DB logic, not
infrastructure integration. Real storage+queue integration is exercised by
`docker compose up` on the Mac per BUILD_PLAN.md's [sandbox] vs [Mac] split.
"""

import sys


def _patch_storage_and_queue(monkeypatch, uploaded_keys=None):
    """Mocks the two external dependencies matches.py touches: object storage
    and the job queue. Returns a dict recording what was enqueued, for assertions.
    """
    import app.storage as storage_module

    uploaded_keys = uploaded_keys if uploaded_keys is not None else set()

    monkeypatch.setattr(storage_module, "ensure_bucket", lambda: None)
    monkeypatch.setattr(storage_module, "presigned_put_url", lambda key, ct, **kw: f"https://fake-s3/{key}")
    monkeypatch.setattr(storage_module, "presigned_get_url", lambda key, **kw: f"https://fake-s3-get/{key}")
    monkeypatch.setattr(storage_module, "object_exists", lambda key: key in uploaded_keys)
    # matches.py calls `storage.presigned_put_url` etc via `from app import storage` then `storage.x`,
    # so patching the module's attributes (above) is what actually takes effect at call time.

    enqueued = {"pipeline": [], "clip": []}
    monkeypatch.setattr(
        sys.modules.setdefault("app.queue", __import__("app.queue", fromlist=["x"])),
        "enqueue_pipeline",
        lambda match_id: enqueued["pipeline"].append(match_id),
    )
    monkeypatch.setattr(
        sys.modules["app.queue"], "enqueue_clip_cut",
        lambda match_id, event_id: enqueued["clip"].append((match_id, event_id)),
    )
    return uploaded_keys, enqueued


def test_create_match_returns_presigned_upload(signed_up_user, monkeypatch):
    client, headers = signed_up_user
    _patch_storage_and_queue(monkeypatch)

    resp = client.post(
        "/matches",
        json={"title": "Semifinal", "filename": "match.mp4", "content_type": "video/mp4", "size_bytes": 1000},
        headers=headers,
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["upload_url"].startswith("https://fake-s3/")
    assert "match_id" in body
    assert body["object_key"].endswith("video.mp4")


def test_create_match_rejects_bad_extension(signed_up_user, monkeypatch):
    client, headers = signed_up_user
    _patch_storage_and_queue(monkeypatch)
    resp = client.post(
        "/matches",
        json={"title": "x", "filename": "match.exe", "content_type": "video/mp4", "size_bytes": 1000},
        headers=headers,
    )
    assert resp.status_code == 400


def test_create_match_rejects_oversized_file(signed_up_user, monkeypatch):
    client, headers = signed_up_user
    _patch_storage_and_queue(monkeypatch)
    resp = client.post(
        "/matches",
        json={
            "title": "x", "filename": "match.mp4", "content_type": "video/mp4",
            "size_bytes": 2 * 1024 * 1024 * 1024,  # 2GB > 1GB limit
        },
        headers=headers,
    )
    assert resp.status_code == 400


def test_complete_upload_creates_jobs_and_enqueues(signed_up_user, monkeypatch):
    client, headers = signed_up_user
    # Patch BEFORE the first /matches call — create_match() itself calls
    # storage.ensure_bucket(), which would otherwise hit real (nonexistent) S3.
    uploaded_keys, enqueued = _patch_storage_and_queue(monkeypatch, uploaded_keys=set())

    create_resp = client.post(
        "/matches",
        json={"title": "x", "filename": "match.mp4", "content_type": "video/mp4", "size_bytes": 1000},
        headers=headers,
    )
    match_id = create_resp.json()["match_id"]
    object_key = create_resp.json()["object_key"]
    uploaded_keys.add(object_key)  # simulate the browser's direct upload having landed

    resp = client.post(f"/matches/{match_id}/complete", headers=headers)
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "validating"
    assert enqueued["pipeline"] == [match_id]

    jobs_resp = client.get(f"/matches/{match_id}/jobs", headers=headers)
    stages = [j["stage"] for j in jobs_resp.json()]

    # Assert against PIPELINE_STAGES rather than a hardcoded list — every layer adds
    # stages, and this test should verify "a job row exists per stage, in order",
    # not re-encode the pipeline definition (which would break on every layer).
    from app.models import PIPELINE_STAGES
    assert stages == PIPELINE_STAGES
    assert all(j["status"] == "pending" for j in jobs_resp.json())


def test_complete_upload_fails_if_object_not_in_storage(signed_up_user, monkeypatch):
    client, headers = signed_up_user
    _patch_storage_and_queue(monkeypatch, uploaded_keys=set())  # nothing ever "uploaded"

    create_resp = client.post(
        "/matches",
        json={"title": "x", "filename": "match.mp4", "content_type": "video/mp4", "size_bytes": 1000},
        headers=headers,
    )
    match_id = create_resp.json()["match_id"]

    resp = client.post(f"/matches/{match_id}/complete", headers=headers)
    assert resp.status_code == 400


def test_matches_are_scoped_to_owner(client, monkeypatch):
    _patch_storage_and_queue(monkeypatch)

    r1 = client.post("/auth/signup", json={"email": "a@test.com", "password": "hunter2222"})
    r2 = client.post("/auth/signup", json={"email": "b@test.com", "password": "hunter2222"})
    headers_a = {"Authorization": f"Bearer {r1.json()['access_token']}"}
    headers_b = {"Authorization": f"Bearer {r2.json()['access_token']}"}

    create_resp = client.post(
        "/matches",
        json={"title": "A's match", "filename": "match.mp4", "content_type": "video/mp4", "size_bytes": 1000},
        headers=headers_a,
    )
    match_id = create_resp.json()["match_id"]

    # Owner can see it
    assert client.get(f"/matches/{match_id}", headers=headers_a).status_code == 200
    # User B cannot — must 404, not leak a 403 that would confirm the match exists
    assert client.get(f"/matches/{match_id}", headers=headers_b).status_code == 404

    # List endpoint scoping
    assert len(client.get("/matches", headers=headers_a).json()) == 1
    assert len(client.get("/matches", headers=headers_b).json()) == 0


def test_video_url_uses_analysis_copy_when_available(signed_up_user, monkeypatch):
    client, headers = signed_up_user
    _patch_storage_and_queue(monkeypatch)

    create_resp = client.post(
        "/matches",
        json={"title": "x", "filename": "match.mp4", "content_type": "video/mp4", "size_bytes": 1000},
        headers=headers,
    )
    match_id = create_resp.json()["match_id"]

    resp = client.get(f"/matches/{match_id}/video-url", headers=headers)
    assert resp.status_code == 200
    assert resp.json()["source"] == "original"  # no analysis_720p yet


def test_match_coach_tone_setting(signed_up_user, monkeypatch):
    client, headers = signed_up_user
    _patch_storage_and_queue(monkeypatch)
    created = client.post(
        "/matches",
        json={"title": "x", "filename": "match.mp4", "content_type": "video/mp4", "size_bytes": 1000},
        headers=headers,
    ).json()
    match_id = created["match_id"]

    updated = client.patch(
        f"/matches/{match_id}/settings",
        json={"coach_tone": "extreme"},
        headers=headers,
    )
    assert updated.status_code == 200
    assert updated.json()["coach_tone"] == "extreme"

    invalid = client.patch(
        f"/matches/{match_id}/settings",
        json={"coach_tone": "unhinged"},
        headers=headers,
    )
    assert invalid.status_code == 422
