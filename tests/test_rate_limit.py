from app.core.limiter import limiter


def test_login_succeeds_when_rate_limit_enabled(client):
    limiter.reset()
    limiter.enabled = True
    try:
        response = client.post(
            "/v1/auth/login",
            json={"username": "demo", "password": "testpass"},
        )
        assert response.status_code == 200, response.text
        assert response.json()["access_token"]
        assert "X-RateLimit-Limit" in response.headers
    finally:
        limiter.enabled = False
        limiter.reset()


def test_login_rate_limit(client):
    limiter.reset()
    limiter.enabled = True
    try:
        for _ in range(5):
            response = client.post(
                "/v1/auth/login",
                json={"username": "demo", "password": "wrong"},
            )
            assert response.status_code == 401

        blocked = client.post(
            "/v1/auth/login",
            json={"username": "demo", "password": "wrong"},
        )
        assert blocked.status_code == 429
    finally:
        limiter.enabled = False
        limiter.reset()
