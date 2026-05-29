"""Shared fixtures for the backend test suite.

These tests deliberately avoid loading TensorFlow or any real model.
The route layer depends on ``PredictorService`` only through duck-typed
attributes (``model_version``, ``decode_strategy``, ``max_upload_bytes``,
``caption_image_bytes``), so a small fake stands in cleanly and keeps the
whole suite under one second.

We also bypass the FastAPI lifespan entirely. The lifespan builds a real
``CaptionPredictor`` from disk, which requires weights, a tokenizer, and a
TF graph build. Tests build a fresh ``FastAPI`` instance, wire the same
router and middleware, and stash the fake service directly on
``app.state.predictor_service`` — the exact slot the lifespan would have
populated in production.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes import router
from app.core.config import BackendSettings
from app.core.logging import RequestContextMiddleware, configure_app_logging
from app.utils.image import ImageDecodeError

configure_app_logging()


class FakePredictorService:
    """Duck-typed stand-in for ``PredictorService``."""

    def __init__(
        self,
        *,
        caption: str = "a test caption",
        latency_ms: float = 1.23,
        decode_strategy: str = "greedy",
        model_version: str = "test-v0",
        max_upload_bytes: int = 1024,
        raise_decode_error: bool = False,
    ) -> None:
        self.model_version = model_version
        self.decode_strategy = decode_strategy
        self.max_upload_bytes = max_upload_bytes
        self._caption = caption
        self._latency_ms = latency_ms
        self._raise = raise_decode_error
        self.calls: list[bytes] = []

    async def caption_image_bytes(self, image_bytes: bytes) -> tuple[str, float]:
        self.calls.append(image_bytes)
        if self._raise:
            raise ImageDecodeError("synthetic decode failure")
        return self._caption, self._latency_ms


def _build_app(service: FakePredictorService | None) -> FastAPI:
    app = FastAPI()
    app.state.backend_settings = BackendSettings()
    app.state.predictor_service = service
    app.add_middleware(RequestContextMiddleware)
    app.include_router(router)
    return app


@pytest.fixture
def fake_service() -> FakePredictorService:
    return FakePredictorService()


@pytest.fixture
def client(fake_service: FakePredictorService) -> Iterator[TestClient]:
    with TestClient(_build_app(fake_service)) as test_client:
        yield test_client


@pytest.fixture
def client_without_service() -> Iterator[TestClient]:
    with TestClient(_build_app(None)) as test_client:
        yield test_client


@pytest.fixture
def build_client() -> Callable[[FakePredictorService | None], TestClient]:
    def _make(service: FakePredictorService | None) -> TestClient:
        return TestClient(_build_app(service))

    return _make
