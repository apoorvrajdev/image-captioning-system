"""Request-body size cap, enforced while the body is still arriving.

FastAPI parses a multipart form before any route or dependency runs, and
Starlette's parser reads the whole body, spooling file parts to a temporary
file past 1 MiB. The route's own size check therefore runs only after an
oversized upload has been received in full. This middleware rejects such a
body first: as soon as the declared ``Content-Length``, or the bytes received
so far, exceed the cap.

It raises the same 413 ``HTTPException`` the route uses, from inside
``receive()``. FastAPI re-raises an ``HTTPException`` that occurs while it
reads the body, so the response keeps the route's ``{"detail": ...}`` shape
and passes through the CORS and request-id middleware like any other error.
"""

from __future__ import annotations

from starlette.exceptions import HTTPException
from starlette.status import HTTP_413_CONTENT_TOO_LARGE
from starlette.types import ASGIApp, Message, Receive, Scope, Send

# Room for the multipart framing around the image: boundary lines and the
# part's headers. The route still enforces the exact image-size limit.
MULTIPART_OVERHEAD_BYTES = 64 * 1024


class BodySizeLimitMiddleware:
    """Reject request bodies larger than ``max_upload_bytes`` plus framing."""

    def __init__(self, app: ASGIApp, *, max_upload_bytes: int) -> None:
        self.app = app
        self.max_upload_bytes = max_upload_bytes
        self.max_body_bytes = max_upload_bytes + MULTIPART_OVERHEAD_BYTES

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        declared = _declared_length(scope)
        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            if declared is not None and declared > self.max_body_bytes:
                raise self._too_large()
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_body_bytes:
                    raise self._too_large()
            return message

        await self.app(scope, limited_receive, send)

    def _too_large(self) -> HTTPException:
        return HTTPException(
            status_code=HTTP_413_CONTENT_TOO_LARGE,
            detail=f"Image exceeds the {self.max_upload_bytes}-byte upload limit.",
        )


def _declared_length(scope: Scope) -> int | None:
    """Return the request's ``Content-Length``, or ``None`` if absent or invalid."""
    for name, value in scope["headers"]:
        if name == b"content-length":
            try:
                return int(value)
            except ValueError:
                return None
    return None
