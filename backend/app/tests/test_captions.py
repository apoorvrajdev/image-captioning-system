"""Tests for ``POST /v1/captions``.

Covers the route's status-code contract end-to-end:

* 200 — happy path, typed ``CaptionResponse`` body
* 400 — empty file upload
* 413 — payload above ``max_upload_bytes``, or image dimensions above the limits
* 415 — disallowed content type
* 422 — bytes that the predictor cannot decode (synthetic, or no readable image header)
* 503 — predictor not yet loaded
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from starlette.datastructures import UploadFile

from app.tests.conftest import FakePredictorService
from app.tests.image_bytes import encode, png_header


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


def test_captions_rejects_webp_before_reading_it(
    client: TestClient, fake_service: FakePredictorService
) -> None:
    """TF 2.15 can't decode WebP, so it's refused up front instead of failing as a 422."""
    response = client.post(
        "/v1/captions",
        files=_image_field(
            b"RIFF\x1a\x00\x00\x00WEBPVP8 ", content_type="image/webp", name="a.webp"
        ),
    )
    assert response.status_code == 415
    assert "Unsupported content type" in response.json()["detail"]
    assert fake_service.calls == []


def test_captions_without_a_strategy_field_still_decodes_greedily(
    client: TestClient, fake_service: FakePredictorService
) -> None:
    """Clients that don't send the field get exactly the old behaviour (TASK-034)."""
    response = client.post("/v1/captions", files=_image_field(b"\xff\xd8stub"))
    assert response.status_code == 200
    assert response.json()["decode_strategy"] == "greedy"
    assert fake_service.strategies == [None]


@pytest.mark.parametrize("strategy", ["greedy", "beam"])
def test_captions_passes_the_requested_strategy_through(
    client: TestClient, fake_service: FakePredictorService, strategy: str
) -> None:
    response = client.post(
        "/v1/captions",
        files=_image_field(b"\xff\xd8stub"),
        data={"decode_strategy": strategy},
    )
    assert response.status_code == 200
    assert response.json()["decode_strategy"] == strategy
    assert fake_service.strategies == [strategy]
    assert fake_service.calls == [b"\xff\xd8stub"]


def test_captions_treats_an_empty_strategy_field_as_omitted(
    client: TestClient, fake_service: FakePredictorService
) -> None:
    """An empty form value is what a blank HTML control sends; it means the default."""
    response = client.post(
        "/v1/captions", files=_image_field(b"\xff\xd8stub"), data={"decode_strategy": ""}
    )
    assert response.status_code == 200
    assert response.json()["decode_strategy"] == "greedy"
    assert fake_service.strategies == [None]


@pytest.mark.parametrize("strategy", ["sampling", "BEAM", "greedy "])
def test_captions_rejects_an_unknown_strategy_before_reading_the_image(
    client: TestClient, fake_service: FakePredictorService, strategy: str
) -> None:
    response = client.post(
        "/v1/captions",
        files=_image_field(b"\xff\xd8stub"),
        data={"decode_strategy": strategy},
    )
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail.startswith(f"Unsupported decode_strategy: {strategy!r}.")
    assert "['greedy', 'beam']" in detail
    assert fake_service.calls == []


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


@pytest.mark.parametrize(("fmt", "content_type"), [("JPEG", "image/jpeg"), ("PNG", "image/png")])
def test_captions_accepts_a_real_image_within_the_dimension_limits(
    build_client, fmt: str, content_type: str
) -> None:
    service = FakePredictorService(check_dimensions=True, max_upload_bytes=64 * 1024)
    payload = encode(fmt)
    with build_client(service) as test_client:
        response = test_client.post("/v1/captions", files=_image_field(payload, content_type))
    assert response.status_code == 200
    assert service.calls == [payload]


def test_captions_returns_413_when_image_dimensions_exceed_the_limits(build_client) -> None:
    # A 10000x5000 header in under 100 bytes: far below the byte limit.
    service = FakePredictorService(check_dimensions=True)
    payload = png_header(10_000, 5_000)
    assert len(payload) < service.max_upload_bytes
    with build_client(service) as test_client:
        response = test_client.post(
            "/v1/captions", files=_image_field(payload, "image/png", "big.png")
        )
    assert response.status_code == 413
    assert response.json()["detail"] == (
        "Image dimensions exceed the limit of 16384 pixels per side "
        "and 40,000,000 pixels in total."
    )


def test_captions_returns_422_for_bytes_without_an_image_header(build_client) -> None:
    service = FakePredictorService(check_dimensions=True)
    with build_client(service) as test_client:
        response = test_client.post("/v1/captions", files=_image_field(b"\xff\xd8junk"))
    assert response.status_code == 422
    assert response.json()["detail"] == (
        "Could not decode image bytes: expected a JPEG, PNG or BMP image."
    )


def test_captions_returns_503_when_predictor_not_loaded(
    client_without_service: TestClient,
) -> None:
    response = client_without_service.post("/v1/captions", files=_image_field(b"\xff\xd8stub"))
    assert response.status_code == 503
    assert "not ready" in response.json()["detail"].lower()
