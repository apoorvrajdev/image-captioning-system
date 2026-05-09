"""Pydantic request/response models for the captioning API.

Schemas live separately from routes so the OpenAPI spec is stable even
when handler logic changes. Every field is annotated with an example so
``/docs`` is self-explanatory to anyone reviewing the portfolio.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    """Liveness + readiness payload for ``GET /healthz``."""

    status: str = Field(..., description="``ok`` once the predictor is loaded.")
    model_loaded: bool = Field(..., description="True after weights + tokenizer are in memory.")
    model_version: str = Field(..., description="Semantic version of the served model.")
    api_version: str = Field(..., description="Backend release version.")
    timestamp: datetime = Field(..., description="Server time the response was built (UTC).")

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "status": "ok",
                "model_loaded": True,
                "model_version": "v1.0.0",
                "api_version": "0.1.0",
                "timestamp": "2026-05-09T12:00:00Z",
            }
        },
    )


class CaptionResponse(BaseModel):
    """Successful response from ``POST /v1/captions``."""

    caption: str = Field(..., description="Generated caption text (without start/end tokens).")
    model_version: str = Field(..., description="Model version that produced this caption.")
    decode_strategy: str = Field(..., description="Decoding strategy used (e.g. ``greedy``).")
    latency_ms: float = Field(..., description="Inference time in milliseconds.")
    request_id: str = Field(..., description="Correlation id; matches the ``x-request-id`` header.")

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "caption": "a man riding a surfboard on a wave",
                "model_version": "v1.0.0",
                "decode_strategy": "greedy",
                "latency_ms": 187.42,
                "request_id": "8f1c2e3b4d5a4f8e9b0c1d2e3f4a5b6c",
            }
        },
    )


class ErrorResponse(BaseModel):
    """Uniform error envelope returned by every non-2xx status."""

    detail: str = Field(..., description="Human-readable error message.")
    request_id: str = Field(default="", description="Correlation id for log lookup.")
