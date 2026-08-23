def test_login_preflight_allows_expo_web_origin(client):
    response = client.options(
        "/v1/auth/login",
        headers={
            "Origin": "http://localhost:8082",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:8082"
