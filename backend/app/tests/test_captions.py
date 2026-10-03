"""Tests for ``POST /v1/captions``.

Covers the route's status-code contract end-to-end:

* 200 — happy path, typed ``CaptionResponse`` body
* 400 — empty file upload
* 413 — payload above ``max_upload_bytes``
* 415 — disallowed content type
* 422 — bytes that the predictor cannot decode (synthetic)
* 503 — predictor not yet loaded
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from starlette.datastructures import UploadFile

from app.tests.conftest import FakePredictorService


def _image_field(payload: bytes, content_type: str = "image/jpeg", name: str = "a.jpg"):
    return {"image": (name, payload, content_type)}


def test_captions_happy_path_returns_typed_response(
    client: TestClient, fake_service: FakePredictorService
) -> None:
    response = client.post("/v1/captions", files=_image_field(b"\xff\xd8stub"))
    assert response.status_code == 200

    body = response.json()
    assert body["caption"] == "a test caption"
    assert body["model_version"] == "test-v0"
    assert body["decode_strategy"] == "greedy"
    assert body["latency_ms"] == 1.23
    assert body["request_id"]

    # Service actually received the upload payload.
    assert fake_service.calls == [b"\xff\xd8stub"]


def test_captions_request_id_matches_response_header(client: TestClient) -> None:
    response = client.post(
        "/v1/captions",
        files=_image_field(b"\xff\xd8stub"),
        headers={"x-request-id": "trace-123"},
    )
    assert response.status_code == 200
    assert response.headers.get("x-request-id") == "trace-123"
    assert response.json()["request_id"] == "trace-123"


def test_captions_rejects_unsupported_content_type(client: TestClient) -> None:
    response = client.post(
        "/v1/captions",
        files=_image_field(b"hello", content_type="text/plain", name="a.txt"),
    )
    assert response.status_code == 415
    assert "Unsupported content type" in response.json()["detail"]


def test_captions_rejects_empty_upload(client: TestClient) -> None:
    response = client.post("/v1/captions", files=_image_field(b""))
    assert response.status_code == 400
    assert "Empty" in response.json()["detail"]


def test_captions_rejects_oversize_upload(client: TestClient) -> None:
    # Fake service.max_upload_bytes = 1024
    response = client.post("/v1/captions", files=_image_field(b"x" * 2048))
    assert response.status_code == 413
    assert "limit" in response.json()["detail"].lower()


def test_captions_oversize_upload_reads_at_most_limit_plus_one(
    client: TestClient,
    fake_service: FakePredictorService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # TASK-006: an oversized upload must not be pulled into memory in full.
    limit = fake_service.max_upload_bytes
    reads: list[tuple[int, int]] = []  # (requested size, bytes returned)
    original_read = UploadFile.read

    async def spy_read(self: UploadFile, size: int = -1) -> bytes:
        data = await original_read(self, size)
        reads.append((size, len(data)))
        return data

    monkeypatch.setattr(UploadFile, "read", spy_read)

    response = client.post("/v1/captions", files=_image_field(b"x" * (limit * 10)))

    assert response.status_code == 413
    assert reads, "route never read the upload"
    assert all(size >= 0 for size, _ in reads), f"unbounded read() call: {reads}"
    assert sum(n for _, n in reads) <= limit + 1
    assert fake_service.calls == []


def test_captions_accepts_upload_exactly_at_limit(
    client: TestClient, fake_service: FakePredictorService
) -> None:
    payload = b"x" * fake_service.max_upload_bytes
    response = client.post("/v1/captions", files=_image_field(payload))
    assert response.status_code == 200
    assert fake_service.calls == [payload]


def test_captions_returns_422_on_decode_failure(build_client) -> None:
    bad_service = FakePredictorService(raise_decode_error=True)
    with build_client(bad_service) as test_client:
        response = test_client.post("/v1/captions", files=_image_field(b"\xff\xd8junk"))
    assert response.status_code == 422
    assert "synthetic decode failure" in response.json()["detail"]


def test_captions_returns_503_when_predictor_not_loaded(
    client_without_service: TestClient,
) -> None:
    response = client_without_service.post("/v1/captions", files=_image_field(b"\xff\xd8stub"))
    assert response.status_code == 503
    assert "not ready" in response.json()["detail"].lower()
