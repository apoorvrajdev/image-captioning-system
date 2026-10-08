"""Tests for ``BodySizeLimitMiddleware`` on ``POST /v1/captions``.

An oversized body must get the route's 413 before the multipart parser reads
it, whether or not the client declares ``Content-Length``. The app is driven
through raw ASGI, with the body handed over in 64 KiB chunks as uvicorn does,
so each test can count how much of the body the app actually read.
"""

from __future__ import annotations

import json
import tempfile
import weakref
from collections.abc import Callable
from typing import Any

import anyio
import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.types import Message

from app.api.routes import router
from app.core.body_limit import MULTIPART_OVERHEAD_BYTES, BodySizeLimitMiddleware
from app.core.config import BackendSettings
from app.core.logging import REQUEST_ID_HEADER, RequestContextMiddleware
from app.tests.conftest import FakePredictorService

CHUNK = 64 * 1024
BOUNDARY = b"test-boundary"
ORIGIN = "http://localhost:5173"


def _build_app(service: FakePredictorService) -> FastAPI:
    app = FastAPI()
    app.state.backend_settings = BackendSettings()
    app.state.predictor_service = service
    # Same order as create_app: the cap innermost, then CORS, then request context.
    app.add_middleware(BodySizeLimitMiddleware, max_upload_bytes=service.max_upload_bytes)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[ORIGIN],
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        allow_credentials=False,
    )
    app.add_middleware(RequestContextMiddleware)
    app.include_router(router)
    return app


def _multipart(image: bytes) -> bytes:
    return (
        b"--" + BOUNDARY + b"\r\n"
        b'Content-Disposition: form-data; name="image"; filename="a.jpg"\r\n'
        b"Content-Type: image/jpeg\r\n\r\n" + image + b"\r\n--" + BOUNDARY + b"--\r\n"
    )


def _post(
    app: FastAPI,
    body: bytes,
    *,
    declare_length: bool,
    after: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """POST ``body`` to ``/v1/captions``; return the status, JSON, headers and bytes read.

    ``after`` runs once the response is sent, inside the same event loop.
    """
    chunks = [body[i : i + CHUNK] for i in range(0, len(body), CHUNK)]
    result: dict[str, Any] = {"read": 0, "body": b""}
    next_chunk = 0

    async def receive() -> Message:
        nonlocal next_chunk
        if next_chunk == len(chunks):
            await anyio.sleep_forever()  # a live connection with nothing more to send
        chunk = chunks[next_chunk]
        next_chunk += 1
        result["read"] += len(chunk)
        return {"type": "http.request", "body": chunk, "more_body": next_chunk < len(chunks)}

    async def send(message: Message) -> None:
        if message["type"] == "http.response.start":
            result["status"] = message["status"]
            result["headers"] = {k.decode(): v.decode() for k, v in message["headers"]}
        elif message["type"] == "http.response.body":
            result["body"] += message.get("body", b"")

    headers = [
        (b"content-type", b"multipart/form-data; boundary=" + BOUNDARY),
        (b"origin", ORIGIN.encode()),
    ]
    if declare_length:
        headers.append((b"content-length", str(len(body)).encode()))
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/captions",
        "raw_path": b"/v1/captions",
        "query_string": b"",
        "root_path": "",
        "headers": headers,
        "client": ("127.0.0.1", 50000),
        "server": ("testserver", 80),
    }

    async def run() -> None:
        with anyio.fail_after(10):
            await app(scope, receive, send)
        if after is not None:
            after()

    anyio.run(run)
    result["json"] = json.loads(result["body"])
    return result


def test_declared_oversize_body_is_refused_before_it_is_read() -> None:
    service = FakePredictorService()
    body = _multipart(b"x" * (service.max_upload_bytes + 2 * MULTIPART_OVERHEAD_BYTES))

    result = _post(_build_app(service), body, declare_length=True)

    assert result["status"] == 413
    assert result["json"] == {
        "detail": f"Image exceeds the {service.max_upload_bytes}-byte upload limit."
    }
    assert result["headers"].get(REQUEST_ID_HEADER)
    assert result["headers"].get("access-control-allow-origin") == ORIGIN
    assert result["read"] == 0
    assert service.calls == []


def test_undeclared_oversize_body_stops_at_the_cap() -> None:
    service = FakePredictorService()
    cap = service.max_upload_bytes + MULTIPART_OVERHEAD_BYTES
    body = _multipart(b"x" * (cap * 4))

    result = _post(_build_app(service), body, declare_length=False)

    assert result["status"] == 413
    assert "limit" in result["json"]["detail"]
    assert result["read"] <= cap + CHUNK
    assert service.calls == []


@pytest.mark.parametrize(("extra_bytes", "status"), [(0, 200), (1, 413)])
def test_body_under_the_cap_reaches_the_route(extra_bytes: int, status: int) -> None:
    # Under the cap the middleware is transparent: the route's exact limit decides.
    service = FakePredictorService()
    image = b"x" * (service.max_upload_bytes + extra_bytes)
    body = _multipart(image)

    result = _post(_build_app(service), body, declare_length=True)

    assert result["status"] == status
    assert result["read"] == len(body)
    if status == 200:
        assert service.calls == [image]
    else:
        # The route's 413 and the middleware's 413 read the same.
        assert result["json"] == {
            "detail": f"Image exceeds the {service.max_upload_bytes}-byte upload limit."
        }
        assert service.calls == []


def test_refused_upload_leaves_no_temp_file_open(monkeypatch: pytest.MonkeyPatch) -> None:
    # A file part past 1 MiB is spooled to disk. Starlette doesn't close it when the cap
    # interrupts the parse, so it must be released as soon as the request ends. anyio
    # before 4.14.2 kept it referenced from an idle worker thread.
    spooled: weakref.WeakSet[tempfile.SpooledTemporaryFile[bytes]] = weakref.WeakSet()
    rollovers: list[int] = []
    original_init = tempfile.SpooledTemporaryFile.__init__
    original_rollover = tempfile.SpooledTemporaryFile.rollover

    def tracking_init(
        self: tempfile.SpooledTemporaryFile[bytes], *args: Any, **kwargs: Any
    ) -> None:
        original_init(self, *args, **kwargs)
        spooled.add(self)

    def tracking_rollover(self: tempfile.SpooledTemporaryFile[bytes]) -> None:
        rollovers.append(1)
        original_rollover(self)

    monkeypatch.setattr(tempfile.SpooledTemporaryFile, "__init__", tracking_init)
    monkeypatch.setattr(tempfile.SpooledTemporaryFile, "rollover", tracking_rollover)
    service = FakePredictorService(max_upload_bytes=2 * 1024 * 1024)
    body = _multipart(b"x" * (service.max_upload_bytes * 2))
    still_open: list[int] = []

    result = _post(
        _build_app(service),
        body,
        declare_length=False,
        after=lambda: still_open.append(sum(not f.closed for f in spooled)),
    )

    assert result["status"] == 413
    assert rollovers, "the file part never reached disk, so this test proves nothing"
    assert still_open == [0]
