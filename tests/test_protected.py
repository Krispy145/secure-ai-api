def test_phishing_samples_require_auth(client):
    response = client.get("/v1/predict/samples")
    assert response.status_code == 401


def test_phishing_samples_with_token(client, auth_headers):
    response = client.get("/v1/predict/samples", headers=auth_headers)
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body, list)
    assert len(body) >= 1
    assert {"id", "url", "label", "score"} <= body[0].keys()


def test_phishing_predict_with_token(client, auth_headers):
    response = client.post(
        "/v1/predict/phishing",
        headers=auth_headers,
        json={"url": "https://www.google.com"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["input_url"] == "https://www.google.com"
    assert body["prediction"] in {"phishing", "legitimate"}


def test_convenience_phishing_path_is_protected(client):
    response = client.get("/phishing/samples")
    assert response.status_code == 401


def test_rag_query_requires_auth(client):
    response = client.post("/v1/rag/query", json={"query": "What is rate limiting?"})
    assert response.status_code == 401


def test_rag_query_with_token(client, auth_headers):
    response = client.post(
        "/v1/rag/query",
        headers=auth_headers,
        json={"query": "What is rate limiting?"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "What is rate limiting?"
    assert isinstance(body["response"], str)
    assert body["response"]


def test_rag_query_returns_sample_phishing_context(client, auth_headers):
    response = client.post(
        "/v1/rag/query",
        headers=auth_headers,
        json={"query": "what is phishing?"},
    )
    assert response.status_code == 200
    body = response.json()
    assert "phishing" in body["response"].lower()
    assert "configure document ingestion" not in body["response"].lower()
