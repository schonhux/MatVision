"""Auth flow — see BUILD_PLAN.md Layer 1 'how we verify': pytest auth flow."""


def test_signup_creates_user_and_returns_token(client):
    resp = client.post("/auth/signup", json={"email": "a@test.com", "password": "hunter2222"})
    assert resp.status_code == 201
    assert "access_token" in resp.json()


def test_signup_duplicate_email_rejected(client):
    client.post("/auth/signup", json={"email": "a@test.com", "password": "hunter2222"})
    resp = client.post("/auth/signup", json={"email": "a@test.com", "password": "different1"})
    assert resp.status_code == 400


def test_signup_short_password_rejected(client):
    resp = client.post("/auth/signup", json={"email": "a@test.com", "password": "short"})
    assert resp.status_code == 422  # pydantic min_length validation


def test_login_with_correct_credentials(client):
    client.post("/auth/signup", json={"email": "a@test.com", "password": "hunter2222"})
    resp = client.post("/auth/login", data={"username": "a@test.com", "password": "hunter2222"})
    assert resp.status_code == 200
    assert "access_token" in resp.json()


def test_login_with_wrong_password_rejected(client):
    client.post("/auth/signup", json={"email": "a@test.com", "password": "hunter2222"})
    resp = client.post("/auth/login", data={"username": "a@test.com", "password": "WRONG"})
    assert resp.status_code == 401


def test_login_unknown_email_rejected(client):
    resp = client.post("/auth/login", data={"username": "nobody@test.com", "password": "whatever1"})
    assert resp.status_code == 401


def test_me_requires_auth(client):
    resp = client.get("/auth/me")
    assert resp.status_code == 401


def test_me_rejects_garbage_token(client):
    resp = client.get("/auth/me", headers={"Authorization": "Bearer not-a-real-token"})
    assert resp.status_code == 401


def test_me_returns_current_user(signed_up_user):
    client, headers = signed_up_user
    resp = client.get("/auth/me", headers=headers)
    assert resp.status_code == 200
    assert resp.json()["email"] == "wrestler@test.com"
