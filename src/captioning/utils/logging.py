"""Structured logging setup.

Why structlog instead of stdlib `logging`?
    * Logs are *data*, not strings. structlog emits dicts that grafana/Datadog/
      Better Stack can index without regex parsing.
    * The same code path produces colourised pretty logs in dev and JSON logs
      in prod, controlled by ``APP_ENV``. Grep the same fields in either mode.
    * Bound context (request IDs, model versions) propagates automatically.

Usage:
    >>> from captioning.utils.logging import configure_logging, get_logger
    >>> configure_logging()
    >>> log = get_logger(__name__)
    >>> log.info("training started", epoch=1, batch_size=64)
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

import structlog

_CONFIGURED = False


def _resolve_level(level: str | int | None) -> int:
    """Coerce a log-level argument (or env default) to a numeric level.

    Why this helper exists:
        ``logging.getLevelName`` is *bidirectional* — it returns ``int`` for
        known names and ``str`` for unknown ones (e.g. ``"Level FOO"``). That
        union return type defeats type narrowing and would be passed straight
        through to ``structlog.make_filtering_bound_logger``, which requires
        ``int``. We resolve once here, fall back to ``INFO`` on unknown
        names, and return a guaranteed ``int``.
    """
    if level is None:
        level = os.environ.get("LOG_LEVEL", "INFO")
    if isinstance(level, int):
        return level
    resolved = logging.getLevelName(level.upper())
    return resolved if isinstance(resolved, int) else logging.INFO


def configure_logging(level: str | int | None = None, json_logs: bool | None = None) -> None:
    """Initialise structlog. Idempotent — calling twice has no effect.

    Args:
        level: Log level name (``"INFO"``) or numeric value. Defaults to env
            ``LOG_LEVEL`` or ``INFO``.
        json_logs: If True, render JSON; if False, render pretty colourised.
            Defaults to True when ``APP_ENV=production``, else False.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    level_int = _resolve_level(level)
    if json_logs is None:
        json_logs = os.environ.get("APP_ENV", "development").lower() == "production"

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level_int,
    )

    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        timestamper,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    renderer: Any = (
        structlog.processors.JSONRenderer()
        if json_logs
        else structlog.dev.ConsoleRenderer(colors=True)
    )

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level_int),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _CONFIGURED = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a logger bound to ``name`` (typically ``__name__``)."""
    if not _CONFIGURED:
        configure_logging()
    return structlog.get_logger(name)
