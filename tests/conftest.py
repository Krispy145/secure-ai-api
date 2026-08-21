# isort: skip_file
# flake8: noqa: E402
import os

os.environ["SECRET_KEY"] = "test-secret-key-not-for-production-use"
os.environ["DEMO_USERNAME"] = "demo"
os.environ["DEMO_PASSWORD"] = "testpass"
os.environ["RATE_LIMIT_ENABLED"] = "false"
os.environ["CORS_ORIGINS"] = "http://localhost:5173"

import pytest
from fastapi.testclient import TestClient

from app.core.limiter import limiter
from app.core.users import clear_refresh_tokens
from app.main import app

limiter.enabled = False


@pytest.fixture
def client():
    clear_refresh_tokens()
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def tokens(client):
    response = client.post(
        "/v1/auth/token",
        data={"username": "demo", "password": "testpass"},
    )
    assert response.status_code == 200, response.text
    return response.json()


@pytest.fixture
def auth_headers(tokens):
    return {"Authorization": f"Bearer {tokens['access_token']}"}
