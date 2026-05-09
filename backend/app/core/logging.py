"""HTTP-layer logging glue.

The ML package already configures structlog (``captioning.utils.logging``).
The FastAPI process has two extra needs on top of that:

1. **Request correlation** — every log line emitted while handling a
   request should carry the same ``request_id`` so logs can be grouped.
   We bind it once via ``structlog.contextvars`` so any ``log.info(...)``
   downstream automatically inherits it without threading the id through
   function signatures.

2. **Access logs as structured events** — uvicorn's default access log is
   a plain string. Re-emitting one structured ``request_finished`` event
   per request keeps the log stream homogeneous and indexable.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from captioning.utils.logging import configure_logging, get_logger

log = get_logger(__name__)

REQUEST_ID_HEADER = "x-request-id"


def configure_app_logging() -> None:
    """Initialise structlog for the FastAPI process.

    Idempotent — delegates to the ML package's ``configure_logging`` so dev
    gets pretty colourised output and ``APP_ENV=production`` flips to JSON.
    """
    configure_logging()


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Bind a request id to structlog and log start/finish events.

    The id comes from the inbound ``x-request-id`` header when present
    (so an upstream gateway can stitch traces), or a fresh ``uuid4`` hex
    otherwise. Either way it's echoed back on the response.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex

        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
        )

        start = time.perf_counter()
        log.info("request_started")
        try:
            response = await call_next(request)
        except Exception:
            duration_ms = (time.perf_counter() - start) * 1000
            log.exception("request_failed", duration_ms=round(duration_ms, 2))
            raise

        duration_ms = (time.perf_counter() - start) * 1000
        log.info(
            "request_finished",
            status=response.status_code,
            duration_ms=round(duration_ms, 2),
        )
        response.headers[REQUEST_ID_HEADER] = request_id
        return response


def current_request_id() -> str:
    """Return the request id bound to the current contextvars, or ``""``."""
    return str(structlog.contextvars.get_contextvars().get("request_id", ""))
