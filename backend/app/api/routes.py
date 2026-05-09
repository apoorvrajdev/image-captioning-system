"""HTTP routes: ``/healthz`` and ``/v1/captions``.

Routes are intentionally thin: validate inputs, delegate to the
``PredictorService``, shape the response. No model code, no TF imports.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status

from app.core.config import BackendSettings, get_backend_settings
from app.core.logging import current_request_id
from app.schemas.caption import CaptionResponse, ErrorResponse, HealthResponse
from app.services.predictor_service import PredictorService
from app.utils.image import ALLOWED_CONTENT_TYPES, ImageDecodeError
from captioning.utils import get_logger

log = get_logger(__name__)

router = APIRouter()


def get_predictor_service(request: Request) -> PredictorService:
    """Resolve the singleton ``PredictorService`` from app state.

    Returns 503 instead of crashing if the lifespan hasn't finished loading
    weights yet (which can happen if ``/v1/captions`` is hit during a
    rolling restart).
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
async def healthz(
    request: Request,
    settings: BackendSettings = Depends(get_backend_settings),
) -> HealthResponse:
    """Return readiness state. Always 200 — readiness is conveyed by ``model_loaded``."""
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
        413: {"model": ErrorResponse, "description": "Image exceeds size limit."},
        415: {"model": ErrorResponse, "description": "Unsupported image content type."},
        422: {"model": ErrorResponse, "description": "Image bytes could not be decoded."},
        503: {"model": ErrorResponse, "description": "Predictor not ready."},
    },
)
async def caption_image(
    image: UploadFile = File(
        ...,
        description="Image file to caption. Allowed: JPEG, PNG, WebP, BMP.",
    ),
    service: PredictorService = Depends(get_predictor_service),
) -> CaptionResponse:
    """Accept a multipart image upload and return a generated caption."""
    if image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported content type: {image.content_type!r}. "
                f"Allowed: {sorted(ALLOWED_CONTENT_TYPES)}."
            ),
        )

    payload = await image.read()
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty file upload.",
        )
    if len(payload) > service.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(f"Image is {len(payload)} bytes; limit is {service.max_upload_bytes}."),
        )

    try:
        caption, latency_ms = await service.caption_image_bytes(payload)
    except ImageDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    return CaptionResponse(
        caption=caption,
        model_version=service.model_version,
        decode_strategy=service.decode_strategy,
        latency_ms=round(latency_ms, 2),
        request_id=current_request_id(),
    )
