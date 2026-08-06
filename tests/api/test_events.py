def _patch_queue(monkeypatch):
    import app.queue as queue_module
    enqueued = []
    monkeypatch.setattr(
        queue_module,
        "_cut_event_clip_stub",
        type("S", (), {"send": staticmethod(lambda *a: enqueued.append(a))}),
    )
    return enqueued


def _make_match(client, headers):
    resp = client.post(
        "/matches",
        json={"title": "x", "filename": "match.mp4", "content_type": "video/mp4", "size_bytes": 1000},
        headers=headers,
    )
    return resp.json()["match_id"]


def test_create_and_list_events(signed_up_user, monkeypatch):
    client, headers = signed_up_user
    import app.storage as storage_module
    monkeypatch.setattr(storage_module, "ensure_bucket", lambda: None)
    monkeypatch.setattr(storage_module, "presigned_put_url", lambda *a, **kw: "https://fake/x")

    match_id = _make_match(client, headers)

    resp = client.post(
        f"/matches/{match_id}/events",
        json={"type": "shot_attempt", "start_ms": 1000, "end_ms": 3000, "note": "good setup"},
        headers=headers,
    )
    assert resp.status_code == 201, resp.text
    assert resp.json()["source"] == "human"

    list_resp = client.get(f"/matches/{match_id}/events", headers=headers)
    assert len(list_resp.json()) == 1
    assert list_resp.json()[0]["type"] == "shot_attempt"


def test_events_ordered_by_start_time(signed_up_user, monkeypatch):
    client, headers = signed_up_user
    import app.storage as storage_module
    monkeypatch.setattr(storage_module, "ensure_bucket", lambda: None)
    monkeypatch.setattr(storage_module, "presigned_put_url", lambda *a, **kw: "https://fake/x")
    match_id = _make_match(client, headers)

    client.post(f"/matches/{match_id}/events", json={"type": "b", "start_ms": 5000, "end_ms": 6000}, headers=headers)
    client.post(f"/matches/{match_id}/events", json={"type": "a", "start_ms": 1000, "end_ms": 2000}, headers=headers)

    events = client.get(f"/matches/{match_id}/events", headers=headers).json()
    assert [e["type"] for e in events] == ["a", "b"]


def test_events_scoped_to_match_owner(client, monkeypatch):
    import app.storage as storage_module
    monkeypatch.setattr(storage_module, "ensure_bucket", lambda: None)
    monkeypatch.setattr(storage_module, "presigned_put_url", lambda *a, **kw: "https://fake/x")

    r1 = client.post("/auth/signup", json={"email": "a@test.com", "password": "hunter2222"})
    r2 = client.post("/auth/signup", json={"email": "b@test.com", "password": "hunter2222"})
    headers_a = {"Authorization": f"Bearer {r1.json()['access_token']}"}
    headers_b = {"Authorization": f"Bearer {r2.json()['access_token']}"}

    match_id = _make_match(client, headers_a)

    resp = client.get(f"/matches/{match_id}/events", headers=headers_b)
    assert resp.status_code == 404


def test_event_rejects_invalid_time_range(signed_up_user, monkeypatch):
    client, headers = signed_up_user
    import app.storage as storage_module
    monkeypatch.setattr(storage_module, "ensure_bucket", lambda: None)
    monkeypatch.setattr(storage_module, "presigned_put_url", lambda *a, **kw: "https://fake/x")
    match_id = _make_match(client, headers)

    resp = client.post(
        f"/matches/{match_id}/events",
        json={"type": "shot_attempt", "start_ms": 3000, "end_ms": 1000},
        headers=headers,
    )
    assert resp.status_code == 422


def test_cut_clip_rejects_event_from_other_match(signed_up_user, monkeypatch):
    client, headers = signed_up_user
    import app.storage as storage_module
    monkeypatch.setattr(storage_module, "ensure_bucket", lambda: None)
    monkeypatch.setattr(storage_module, "presigned_put_url", lambda *a, **kw: "https://fake/x")
    enqueued = _patch_queue(monkeypatch)

    match_a = _make_match(client, headers)
    match_b = _make_match(client, headers)
    event_resp = client.post(
        f"/matches/{match_a}/events",
        json={"type": "shot_attempt", "start_ms": 1000, "end_ms": 3000},
        headers=headers,
    )
    event_id = event_resp.json()["id"]

    resp = client.post(f"/matches/{match_b}/events/{event_id}/cut-clip", headers=headers)
    assert resp.status_code == 404
    assert enqueued == []
