def test_token_login_success(client):
    response = client.post(
        "/v1/auth/token",
        data={"username": "demo", "password": "testpass"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["token_type"] == "bearer"
    assert body["access_token"]
    assert body["refresh_token"]
    assert body["expires_in"] == 3600


def test_json_login_accepts_email_field(client):
    response = client.post(
        "/v1/auth/login",
        json={"email": "demo", "password": "testpass"},
    )
    assert response.status_code == 200
    assert response.json()["access_token"]


def test_login_rejects_bad_password(client):
    response = client.post(
        "/v1/auth/login",
        json={"username": "demo", "password": "wrong"},
    )
    assert response.status_code == 401


def test_me_requires_auth(client):
    response = client.get("/v1/auth/me")
    assert response.status_code == 401


def test_me_with_access_token(client, auth_headers):
    response = client.get("/v1/auth/me", headers=auth_headers)
    assert response.status_code == 200
    assert response.json() == {"username": "demo"}


def test_refresh_rotates_token(client, tokens):
    first_refresh = tokens["refresh_token"]
    response = client.post("/v1/auth/refresh", json={"refresh_token": first_refresh})
    assert response.status_code == 200
    body = response.json()
    assert body["access_token"] != tokens["access_token"]
    assert body["refresh_token"] != first_refresh

    replay = client.post("/v1/auth/refresh", json={"refresh_token": first_refresh})
    assert replay.status_code == 401


def test_logout_revokes_refresh_token(client, tokens):
    response = client.post("/v1/auth/logout", json={"refresh_token": tokens["refresh_token"]})
    assert response.status_code == 204

    replay = client.post(
        "/v1/auth/refresh",
        json={"refresh_token": tokens["refresh_token"]},
    )
    assert replay.status_code == 401


def test_access_token_rejected_as_refresh(client, tokens):
    response = client.post(
        "/v1/auth/refresh",
        json={"refresh_token": tokens["access_token"]},
    )
    assert response.status_code == 401
