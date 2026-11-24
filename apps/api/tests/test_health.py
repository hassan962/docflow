from fastapi.testclient import TestClient

from docflow_api.main import create_app


def test_healthz_never_needs_the_database() -> None:
    client = TestClient(create_app())
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_openapi_is_served() -> None:
    client = TestClient(create_app())
    assert client.get("/openapi.json").json()["info"]["title"] == "DocFlow API"
