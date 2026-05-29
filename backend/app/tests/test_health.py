"""Tests for ``GET /healthz``.

The route reports liveness + readiness in the response body and always
returns 200; readiness is conveyed by ``model_loaded``.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.tests.conftest import FakePredictorService


def test_healthz_reports_ready_when_service_present(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_version"] == "test-v0"
    assert body["api_version"]
    assert "timestamp" in body


def test_healthz_reports_loading_when_service_missing(
    client_without_service: TestClient,
) -> None:
    response = client_without_service.get("/healthz")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "loading"
    assert body["model_loaded"] is False


def test_healthz_echoes_request_id_header(client: TestClient) -> None:
    response = client.get("/healthz", headers={"x-request-id": "deadbeef"})
    assert response.status_code == 200
    assert response.headers.get("x-request-id") == "deadbeef"


def test_healthz_generates_request_id_when_absent(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    rid = response.headers.get("x-request-id")
    assert rid and len(rid) >= 16


def test_healthz_uses_overridden_model_version(
    build_client,
) -> None:
    service = FakePredictorService(model_version="v9.9.9")
    with build_client(service) as test_client:
        body = test_client.get("/healthz").json()
        assert body["model_version"] == "v9.9.9"
