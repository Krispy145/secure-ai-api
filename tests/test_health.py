def test_root_redirects_to_docs(client):
    response = client.get("/", follow_redirects=False)
    assert response.status_code in {307, 302}
    assert response.headers["location"] == "/docs"


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_ping(client):
    response = client.get("/ping")
    assert response.status_code == 200
    assert response.json()["message"] == "pong"
