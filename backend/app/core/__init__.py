"""Core: backend settings and HTTP-layer logging glue."""

from app.core.config import BackendSettings, get_backend_settings
from app.core.logging import (
    REQUEST_ID_HEADER,
    RequestContextMiddleware,
    configure_app_logging,
    current_request_id,
)

__all__ = [
    "REQUEST_ID_HEADER",
    "BackendSettings",
    "RequestContextMiddleware",
    "configure_app_logging",
    "current_request_id",
    "get_backend_settings",
]
