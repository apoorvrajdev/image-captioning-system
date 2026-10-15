"""HTTP routes: ``/healthz`` and ``/v1/captions``.

Routes are intentionally thin: validate inputs, delegate to the
``PredictorService``, shape the response. No model code, no TF imports.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status

from app.core.config import BackendSettings
from app.core.logging import current_request_id
from app.schemas.caption import CaptionResponse, ErrorResponse, HealthResponse
from app.services.predictor_service import (
    DECODE_STRATEGIES,
    PredictorBusyError,
    PredictorService,
)
from app.utils.image import ALLOWED_CONTENT_TYPES, ImageDecodeError, ImageTooLargeError
from captioning.utils import get_logger

log = get_logger(__name__)

router = APIRouter()

# The overload 503. One caption took 0.8-0.9 s of inference in the deploy smoke
# tests, so a slot usually frees within a couple of seconds.
BUSY_DETAIL = "The server is busy with other captions. Retry in a few seconds."
BUSY_RETRY_AFTER_SECONDS = 2


async def get_predictor_service(request: Request) -> PredictorService:
    """Resolve the singleton ``PredictorService`` from app state.

    Returns 503 instead of crashing if the lifespan hasn't finished loading
    weights yet (which can happen if ``/v1/captions`` is hit during a
    rolling restart). Async, so FastAPI resolves it on the event loop rather
    than in a worker thread that busy inference might be holding.
    """
    service: PredictorService | None = getattr(request.app.state, "predictor_service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Predictor is not ready yet.",
        )
    return service


@router.get(
    "/healthz",
    response_model=HealthResponse,
    tags=["health"],
    summary="Liveness + readiness probe",
)
async def healthz(request: Request) -> HealthResponse:
    """Return readiness state. Always 200 — readiness is conveyed by ``model_loaded``.

    Runs entirely on the event loop, with no sync dependency, so it never
    waits for a worker thread that inference might be holding.
    """
    settings: BackendSettings = request.app.state.backend_settings
    service: PredictorService | None = getattr(request.app.state, "predictor_service", None)
    return HealthResponse(
        status="ok" if service is not None else "loading",
        model_loaded=service is not None,
        model_version=service.model_version if service is not None else settings.model_version,
        api_version=settings.api_version,
        timestamp=datetime.now(timezone.utc),
    )


@router.post(
    "/v1/captions",
    response_model=CaptionResponse,
    tags=["captions"],
    status_code=status.HTTP_200_OK,
    summary="Generate a caption for an uploaded image",
    responses={
        400: {"model": ErrorResponse, "description": "Empty upload."},
        413: {
            "model": ErrorResponse,
            "description": "Image exceeds the upload size or pixel-dimension limit.",
        },
        415: {"model": ErrorResponse, "description": "Unsupported image content type."},
        422: {
            "model": ErrorResponse,
            "description": "Image bytes could not be decoded, or `decode_strategy` is unknown.",
        },
        503: {
            "model": ErrorResponse,
            "description": "Predictor not ready, or every inference slot is busy (with `Retry-After`).",
        },
    },
)
async def caption_image(
    image: UploadFile = File(
        ...,
        description="Image file to caption. Allowed: JPEG, PNG, BMP.",
    ),
    decode_strategy: str | None = Form(
        None,
        description=(
            "Decoding strategy for this request: `greedy` (the default) or `beam`. "
            "Beam search uses the server's configured width and penalties."
        ),
    ),
    service: PredictorService = Depends(get_predictor_service),
) -> CaptionResponse:
    """Accept a multipart image upload and return a generated caption."""
    if decode_strategy is not None and decode_strategy not in DECODE_STRATEGIES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                f"Unsupported decode_strategy: {decode_strategy!r}. "
                f"Allowed: {list(DECODE_STRATEGIES)}."
            ),
        )
    if image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported content type: {image.content_type!r}. "
                f"Allowed: {sorted(ALLOWED_CONTENT_TYPES)}."
            ),
        )

    # Read at most one byte past the limit: enough to detect an oversized
    # upload without pulling the whole file into memory.
    payload = await image.read(service.max_upload_bytes + 1)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty file upload.",
        )
    if len(payload) > service.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"Image exceeds the {service.max_upload_bytes}-byte upload limit.",
        )

    try:
        caption, latency_ms = await service.caption_image_bytes(
            payload, decode_strategy=decode_strategy
        )
    except PredictorBusyError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=BUSY_DETAIL,
            headers={"Retry-After": str(BUSY_RETRY_AFTER_SECONDS)},
        ) from exc
    except ImageTooLargeError as exc:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=str(exc),
        ) from exc
    except ImageDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc

    return CaptionResponse(
        caption=caption,
        model_version=service.model_version,
        decode_strategy=decode_strategy or service.decode_strategy,
        latency_ms=round(latency_ms, 2),
        request_id=current_request_id(),
    )
